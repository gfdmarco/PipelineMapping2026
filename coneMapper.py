"""Mapa global persistente de cones (frame `map`).

Recebe os cones do frame ([lateral, frente, classe], no frame base_link),
transforma para o mundo com a pose do SLAM e associa a landmarks ja
existentes por vizinho mais proximo:

  * associacao POR POSICAO (nao por cor): um cone com a cor trocada pelo
    YOLO num frame nao vira um cone duplicado; a cor final e o voto
    majoritario das deteccoes.
  * posicao = media ponderada por 1/sigma^2, com sigma crescendo com a
    distancia (deteccao de longe pesa menos).
  * so deteccoes ate MAX_MAP_RANGE entram no mapa (de longe o erro de
    distancia explode e gera fantasmas).
  * `confirmed()` devolve so cones vistos >= CONFIRM_HITS vezes: e isso
    que o planejador global usa (fantasma de 1 frame nao entra no Delaunay).

Classe pura, sem ROS (roda igual no CarNode e no simulador 2D).
"""
import numpy as np

from .frames import car_to_world

ASSOC_DIST = 1.2         # m: deteccao a menos disso de um landmark = mesmo cone
MAX_MAP_RANGE = 12.0     # m: deteccoes mais longe nao entram no mapa
MEAS_STD_BASE = 0.10     # m
MEAS_STD_PER_M = 0.03    # m por metro de distancia
CONFIRM_HITS = 3         # deteccoes para o cone entrar no mapa "confirmado"
N_CLASSES = 4            # 0 azul, 1 amarelo, 2 laranja grande, 3 laranja


class ConeMapper:
    def __init__(self, assoc_dist=ASSOC_DIST, max_range=MAX_MAP_RANGE,
                 confirm_hits=CONFIRM_HITS):
        self.assoc_dist = assoc_dist
        self.max_range = max_range
        self.confirm_hits = confirm_hits
        self.pos = np.empty((0, 2))          # (N,2) x, y no frame map
        self.wsum = np.empty(0)              # soma dos pesos 1/sigma^2
        self.votes = np.empty((0, N_CLASSES))
        self.hits = np.empty(0, dtype=int)
        self.first_t = np.empty(0)
        self.last_t = np.empty(0)
        self.first_lap = np.empty(0, dtype=int)
        self.frozen = False                  # True = so localiza, nao edita

    # ------------------------------------------------------------------
    def update(self, local_cones, pose, t=0.0, lap=0):
        """Integra um frame. Retorna estatisticas do frame (telemetria)."""
        stats = {"n_in": 0, "n_far": 0, "n_new": 0, "n_assoc": 0}
        cones = np.asarray(local_cones, dtype=float)
        if cones.ndim != 2 or len(cones) == 0 or pose is None:
            return stats
        stats["n_in"] = len(cones)

        rng = np.hypot(cones[:, 0], cones[:, 1])
        near = rng <= self.max_range
        stats["n_far"] = int((~near).sum())
        if self.frozen or not near.any():
            return stats
        cones, rng = cones[near], rng[near]

        world = car_to_world(cones[:, :2], pose)
        cls = np.clip(cones[:, 2].astype(int), 0, N_CLASSES - 1)
        w = 1.0 / (MEAS_STD_BASE + MEAS_STD_PER_M * rng) ** 2

        used = set()
        for i in np.argsort(rng):            # mais perto (mais preciso) primeiro
            z = world[i]
            j = -1
            if len(self.pos):
                d = np.hypot(self.pos[:, 0] - z[0], self.pos[:, 1] - z[1])
                if used:
                    d[list(used)] = np.inf   # 1 deteccao por landmark por frame
                k = int(np.argmin(d))
                if d[k] < self.assoc_dist:
                    j = k
            if j < 0:
                self._add(z, w[i], cls[i], t, lap)
                used.add(len(self.pos) - 1)
                stats["n_new"] += 1
            else:
                W = self.wsum[j] + w[i]
                self.pos[j] = (self.pos[j] * self.wsum[j] + z * w[i]) / W
                self.wsum[j] = W
                self.votes[j, cls[i]] += 1
                self.hits[j] += 1
                self.last_t[j] = t
                used.add(j)
                stats["n_assoc"] += 1
        return stats

    def _add(self, z, w, c, t, lap):
        self.pos = np.vstack([self.pos, z])
        self.wsum = np.append(self.wsum, w)
        v = np.zeros((1, N_CLASSES)); v[0, c] = 1
        self.votes = np.vstack([self.votes, v])
        self.hits = np.append(self.hits, 1)
        self.first_t = np.append(self.first_t, t)
        self.last_t = np.append(self.last_t, t)
        self.first_lap = np.append(self.first_lap, lap)

    # ------------------------------------------------------------------
    @property
    def classes(self):
        return self.votes.argmax(axis=1) if len(self.votes) else np.empty(0, dtype=int)

    def class_purity(self):
        """Fracao de votos na classe vencedora (1.0 = YOLO sempre concordou)."""
        if not len(self.votes):
            return np.empty(0)
        return self.votes.max(axis=1) / np.maximum(self.votes.sum(axis=1), 1)

    def confirmed(self, min_hits=None):
        """(N,3) [x, y, classe] dos cones confirmados (usado pelo planejador)."""
        m = self.hits >= (self.confirm_hits if min_hits is None else min_hits)
        if not m.any():
            return np.empty((0, 3))
        return np.column_stack([self.pos[m], self.classes[m]])

    def counts(self):
        cls = self.classes
        conf = self.hits >= self.confirm_hits
        return {
            "map_total": int(len(self.pos)),
            "map_confirmed": int(conf.sum()),
            "map_blue": int(((cls == 0) & conf).sum()),
            "map_yellow": int(((cls == 1) & conf).sum()),
            "map_orange": int(((cls >= 2) & conf).sum()),
            "map_tentative": int((~conf).sum()),
        }

    def snapshot(self):
        """Mapa completo para a telemetria (dashboard compara com o GT)."""
        cls, pur = self.classes, self.class_purity()
        return [[round(float(self.pos[i, 0]), 3), round(float(self.pos[i, 1]), 3),
                 int(cls[i]), int(self.hits[i]), round(float(pur[i]), 2),
                 int(self.first_lap[i])] for i in range(len(self.pos))]
