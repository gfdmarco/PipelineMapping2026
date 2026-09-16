import rclpy
from rclpy.node import Node

import numpy as np

from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration

from std_msgs.msg import Float32,Float32MultiArray,MultiArrayDimension, Bool

from . import carNode

CONE_MERGE_DIST = 1.5

# profundidade/lateral grande (lidar esparso, intrinsecos da camera) e
# poluem o mapa com fantasmas. O carro passa perto de todos os cones ao
# longo da volta, entao alcance curto nao perde cobertura
MAX_MAP_RANGE = 12.0         # m

# As deteccoes [lateral, frente] saem no frame do SENSOR (Lidar1 esta
# 0.45 m a frente do centro do carro, ver settings.json do FSDS). Sem
# compensar, cada cone mapeado para o mundo desloca 0.45 m na direcao do
# heading -- o mesmo cone visto com headings diferentes (antes/depois de
# uma curva) vira dois cones no mapa
SENSOR_FWD_OFFSET = 0.45     # m

# Ruido de medicao (cresce com a distancia), igual ao do SLAM
MEAS_STD_BASE = 0.15         # m
MEAS_STD_PER_M = 0.03        # m por metro

# Associacao
GATE_MIN = 1.2               # m
GATE_MAX = 2.4               # m
DRIFT_STD_RATE = 0.02        # m/s de deriva esperada desde a ultima vista
DRIFT_AGE_CAP = 60.0         # s

# Gestao de landmarks
CONFIRM_HITS = 3             # deteccoes para confirmar um cone
TENTATIVE_TIMEOUT = 3.0      # s sem rever um tentativo -> descarta
MERGE_RADIUS = 1.5           # m: cones da mesma cor mais perto que isso fundem
MERGE_PERIOD = 2.0           # s entre passadas incrementais de fusao

# Janela de cones locais (formato da percepcao)
LOCAL_AHEAD = 22.0
LOCAL_BEHIND = 2.0
LOCAL_LATERAL = 10.0

# Linha de largada/chegada: cones laranja (2 = grande, 3 = pequeno) que
# o FSDS coloca na largada. Servem de marco fisico para fechar a volta
ORANGE_IDS = (2, 3)
START_ANCHOR_RADIUS = 20.0   # m da origem do mapa onde procurar a linha
START_ANCHOR_MIN_CONES = 2

class ConeMapper(Node):
    def __init__(self, car_node):
        super().__init__('ConeMapper')

        self.global_cones: list[dict] = []   # [{x, y, class_id}]
        self.car_node = car_node

    def globalizer(self, local_cones):
        # Tenta pegar TF base_link → odom - CORRIGIDO PARA MAP por Gabriel(o frame de referência da odometria)
        try:
            pose = self.car_node.get_pose("map")
            if pose is None:
                return
            tx = pose[0]
            ty = pose[1]
            yaw = pose[2]
        except:
            tx, ty = self.path_x[-1], self.path_y[-1]
            # talvez essa linha esteja atrapalhando o set do referencial: yaw = 0.0

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        for cone in local_cones:
            lat = cone[0]
            fwd = cone[1]
            cls = cone[2]
            mx = tx + fwd * cos_y - lat * sin_y
            my = ty + fwd * sin_y + lat * cos_y
            self._merge_cone(mx, my, cls)   # sem precisar inverter nada aqui

        return self.global_cones

    def _merge_cone(self, x, y, class_id):
        for c in self.global_cones:
            if np.hypot(x - c['x'], y - c['y']) < CONE_MERGE_DIST and c['class_id'] == class_id:
                n = c['n']
                c['x'] = (c['x'] * n + x) / (n + 1)
                c['y'] = (c['y'] * n + y) / (n + 1)
                c['n'] += 1
                return
        self.global_cones.append({'x': x, 'y': y, 'class_id': class_id, 'n': 1})


    def update(self, detections, now):
        """Integra as deteccoes do frame ([lateral, frente, classId]) no mapa."""
        detections = np.asarray(detections)
        if detections.ndim != 2 or len(detections) == 0:
            self._prune(now)
            return

        ranges = np.linalg.norm(detections[:, :2].astype(float), axis=1)
        near = ranges <= MAX_MAP_RANGE
        if not near.any():
            self._prune(now)
            return
        detections = detections[near].astype(float)
        ranges = ranges[near]

        # frame do sensor -> frame do carro (so para o mapa: o planner
        # local continua recebendo as deteccoes cruas, como sempre)
        detections[:, 1] += self.sensor_fwd_offset

        world = self.car.get_pose("map")

        # processa do mais perto para o mais longe (medicao mais precisa
        # ganha a disputa) e bloqueia cada landmark a uma deteccao por frame
        order = np.argsort(ranges)
        used = set()
        for i in order:
            z = world[i]
            cls = int(detections[i, 2])
            meas_var = (MEAS_STD_BASE + MEAS_STD_PER_M * ranges[i]) ** 2

            idx = np.flatnonzero(self.cls == cls)
            idx = idx[~np.isin(idx, list(used))] if used else idx

            best = -1
            if len(idx):
                d = np.linalg.norm(self.pos[idx] - z, axis=1)
                j = int(np.argmin(d))
                age = min(now - self.seen[idx[j]], DRIFT_AGE_CAP)
                S = self.var[idx[j]] + meas_var + (DRIFT_STD_RATE * age) ** 2
                gate = np.clip(3.0 * np.sqrt(S), GATE_MIN, GATE_MAX)
                if d[j] < gate:
                    best = int(idx[j])

            if best < 0:
                self.pos = np.vstack([self.pos, z])
                self.var = np.append(self.var, meas_var)
                self.cls = np.append(self.cls, cls)
                self.hits = np.append(self.hits, 1)
                self.seen = np.append(self.seen, now)
                used.add(len(self.var) - 1)
                continue

            gain = self.var[best] / (self.var[best] + meas_var)
            self.pos[best] += gain * (z - self.pos[best])
            self.var[best] *= (1.0 - gain)
            self.hits[best] += 1
            self.seen[best] = now
            used.add(best)

        self._prune(now)
        if now - self.last_merge > MERGE_PERIOD:
            self.last_merge = now
            self.mergeDuplicates()

    