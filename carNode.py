"""CarNode (versao FSDS): casca ROS2 + FSDS em volta de core.Mission.

Fluxo por frame (disparado pela CHEGADA dos cones, como no carro real):
    /lidar_node/cones [lateral, frente, classe] (base_link)
      -> pose map (TF map->base_link, SLAM) e pose odom (TF odom->base_link)
      -> Mission.step(): mapa global, detector de volta, EXPLORE/TRACK,
         rota local ou janela da trajetoria global, pure pursuit + PID
      -> client.setCarControls (FSDS)

Toda a logica fica em core.py (sem ROS), que o simulador 2D tambem roda.
Telemetria: <MAPPING_RUN_DIR>/carNode.jsonl (ver telemetry.py).

Parametros ROS principais:
    cones_topic              (/lidar_node/cones)  saida do main.py (bridge)
    use_internal_perception  (False) True = YOLO proprio (modo antigo)
    min_distance_traveled_m / position_threshold_m / heading_threshold_deg /
    rearm_distance_m         detector de volta
    cones_timeout_s          (0.5) sem cones novos por mais que isso -> freia
"""
import os
import sys

import numpy as np
import math
import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray
from tf2_ros import Buffer, TransformListener, TransformException
from cartographer_ros_msgs.srv import TrajectoryQuery   # vem do source do workspace SLAM

from .core import Mission, TRACK
from .frames import yaw_from_quat, wrap_angle
from .telemetry import Telemetry
from .coneMapper import ConeMapper

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)
import fsds  # noqa: E402

INTERNAL_PERCEPTION_HZ = 20
TRACK_CONTROL_HZ = 20        # laco de controle do TRACK (nao espera os cones)
REPROJECT_MIN_MOVE = 0.3     # m: correcao do SLAM menor que isso nao refaz o mapa
TRACK_STALE_S = 2.0          # s sem cones no TRACK antes do watchdog frear


class CarNode(Node):
    def __init__(self):
        super().__init__("CarNode")
        p = self.declare_parameter
        p('cones_topic', '/lidar_node/cones')
        # 'posearray' = fsdsCones (PoseArray com stamp de captura, classe em z):
        # o mapa usa a TF map->base_link NO STAMP (sem erro de latencia).
        # 'array' = contrato do carro real (Float32MultiArray, sem stamp).
        p('cones_msg', 'posearray')
        p('cones_posearray_topic', 'fsdsCones')
        p('use_internal_perception', False)
        p('min_distance_traveled_m', 30.0)
        p('position_threshold_m', 6.0)
        p('heading_threshold_deg', 30.0)
        p('rearm_distance_m', 12.0)
        p('cones_timeout_s', 0.5)
        p('freeze_map_in_track', False)
        g = lambda n: self.get_parameter(n).value

        self.tele = Telemetry("carNode", logger=self.get_logger())
        self.mission = Mission(self.tele, lap_kwargs=dict(
            min_distance=g('min_distance_traveled_m'),
            pos_threshold=g('position_threshold_m'),
            heading_threshold_deg=g('heading_threshold_deg'),
            rearm_dist=g('rearm_distance_m')),
            freeze_map_in_track=g('freeze_map_in_track'))
        self.cones_timeout = g('cones_timeout_s')

        try:
            self.client = fsds.FSDSClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)
        except Exception as e:
            raise RuntimeError(f"Unable to connect to simulator.\n{e}")

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.controlEnabled = True
        self.create_subscription(Bool, '/enable_control', self.authorization, 10)
        self.lap_pub = self.create_publisher(Bool, '/mapping/lap_closed', 10)
        self.mode_pub = self.create_publisher(String, '/mapping/mode', 10)
        self.map_pub = self.create_publisher(Float32MultiArray, '/mapping/global_cones', 10)
        self.traj_pub = self.create_publisher(Path, '/mapping/global_path', 10)

        self.last_time = None
        self.last_cones_wall = None
        self.published_laps = 0
        self.published_traj = None

        if g('use_internal_perception'):
            from . import perception          # so importa YOLO se for usar
            self.perception = perception.Perception(self.client)
            self.create_timer(1.0 / INTERNAL_PERCEPTION_HZ, self._internal_perception)
        elif g('cones_msg') == 'posearray':
            self.create_subscription(PoseArray, g('cones_posearray_topic'), self._cb_cones_pa, 10)
        else:
            self.create_subscription(Float32MultiArray, g('cones_topic'), self._cb_cones, 10)

        self.create_timer(0.1, self._watchdog)
        # TRACK: rota = trajetoria global + pose (TF), nao depende da percepcao
        # (~4 Hz). Um timer proprio comanda o carro a TRACK_CONTROL_HZ.
        self.mission.track_timer = True
        self.last_ctl_time = None
        self.create_timer(1.0 / TRACK_CONTROL_HZ, self._track_control)
        self.create_timer(1.0, self._publish_map)
        self.tele.event("run_start", cones_topic=g('cones_topic'), cones_msg=g('cones_msg'),
                        internal_perception=g('use_internal_perception'),
                        min_distance=g('min_distance_traveled_m'),
                        pos_threshold=g('position_threshold_m'),
                        heading_threshold_deg=g('heading_threshold_deg'),
                        rearm_dist=g('rearm_distance_m'))
        # Vamos usar este array para salvar as detecções de cones a fim de corrigir as posições quando tiver loop closure
        self.detections = []

        # consulta da trajetoria + reprojecao do mapa ficam num grupo PROPRIO
        # (outra thread do MultiThreadedExecutor): com milhares de nos, ler a
        # resposta e refazer o mapa leva centenas de ms e, na mesma thread,
        # travava os cones e o controle (cones_stale a cada 2-4 s no FSDS)
        self.lock = threading.RLock()
        self.cg_slam = MutuallyExclusiveCallbackGroup()
        self.traj_cli = self.create_client(TrajectoryQuery, '/trajectory_query',
                                           callback_group=self.cg_slam)
        self.query_pending = False
        self.last_nodes = None
        self.create_timer(2.0, self._ask_trajectory, callback_group=self.cg_slam)

    # ------------------------------------------------------------------
    def authorization(self, msg):
        self.controlEnabled = msg.data
        self.tele.event("control_enabled", value=bool(msg.data))

    def _cb_cones(self, msg):
        flat = np.asarray(msg.data, dtype=float)
        cones = flat.reshape(-1, 3) if flat.size % 3 == 0 and flat.size else np.empty((0, 3))
        cones = cones[np.isfinite(cones).all(axis=1)]
        self.process(cones)

    def _cb_cones_pa(self, msg):
        cones = np.array([[-p.position.y, p.position.x, round(p.position.z)] for p in msg.poses],
                         dtype=float).reshape(-1, 3)
        stamp = rclpy.time.Time.from_msg(msg.header.stamp)
        if len(cones):
            # (horario em segundos [float], cones (N,3) no frame do carro, volta)
            self.detections.append((stamp.nanoseconds * 1e-9, cones.copy(),
                                    self.mission.lap_count))
        self.process(cones, self.get_pose("map", stamp))

    def _internal_perception(self):
        [image] = self.client.simGetImages(
            [fsds.ImageRequest(camera_name='ZED_RGB', image_type=fsds.ImageType.Scene,
                               pixels_as_float=False, compress=True)], vehicle_name='FSCar')
        cones = self.perception.detectCones(image)
        self.process(np.asarray(cones, dtype=float).reshape(-1, 3) if np.size(cones) else np.empty((0, 3)))

    def process(self, cones, pose_meas=None):
        now = self.get_clock().now()
        dt = 0.0 if self.last_time is None else (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        self.last_cones_wall = now

        pose_map = self.get_pose("map")
        pose_odom = self.get_pose("odom")
        speed = self.client.getCarState().speed
        with self.lock:
            steer, thr, brk = self.mission.step(cones, pose_map, pose_odom, speed, dt,
                                                self.tele.clock(), pose_meas=pose_meas)
        # no TRACK quem comanda e o _track_control; aqui so se caiu na rota local
        track_owned = self.mission.mode == 'TRACK' and self.mission.last_source == 'global'
        if self.controlEnabled and not track_owned:
            self.client.setCarControls(fsds.CarControls(throttle=thr, steering=steer, brake=brk))
        self._publish_events()

    def _track_control(self):
        """Controle a TRACK_CONTROL_HZ no TRACK: pose da TF + trajetoria global."""
        # no TRACK a rota nao depende dos cones: so pose (TF) + trajetoria
        if self.mission.mode != 'TRACK':
            self.last_ctl_time = None
            return
        now = self.get_clock().now()
        dt = 0.0 if self.last_ctl_time is None else (now - self.last_ctl_time).nanoseconds * 1e-9
        self.last_ctl_time = now
        pose_map = self.get_pose("map")
        speed = self.client.getCarState().speed
        with self.lock:
            cmd = self.mission.control_step(pose_map, speed, dt, self.tele.clock())
        if cmd is None:
            return                      # fallback: o step() dos cones comanda
        steer, thr, brk = cmd
        if self.controlEnabled:
            self.client.setCarControls(fsds.CarControls(throttle=thr, steering=steer, brake=brk))
        self._publish_events()

    def _cones_stale(self):
        if self.last_cones_wall is None:
            return True
        age = (self.get_clock().now() - self.last_cones_wall).nanoseconds * 1e-9
        return age > self.cones_timeout

    def _watchdog(self):
        """Percepcao parou (bridge caiu / YOLO travou): freia em vez de seguir o ultimo comando."""
        if self.last_cones_wall is None:
            return
        age = (self.get_clock().now() - self.last_cones_wall).nanoseconds * 1e-9
        if age > self.cones_timeout:
            self.tele.periodic("cones_stale", 1.0, age_s=age, mode=self.mission.mode)
            # no TRACK o controle segue pela pose; so freia se a percepcao
            # sumiu de vez (bridge caiu -> a odometria/TF tambem para)
            if self.mission.mode == 'TRACK' and age < TRACK_STALE_S:
                return
            if self.controlEnabled:
                self.client.setCarControls(fsds.CarControls(throttle=0.0, steering=0.0, brake=0.5))

    def _publish_events(self):
        m = self.mission
        if m.lap_count != self.published_laps:
            self.published_laps = m.lap_count
            self.lap_pub.publish(Bool(data=True))
        if m.trajectory is not None and m.trajectory is not self.published_traj:
            self.published_traj = m.trajectory
            path = Path()
            path.header.frame_id = 'map'
            path.header.stamp = self.get_clock().now().to_msg()
            for x, y in m.trajectory.points:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x, ps.pose.position.y = float(x), float(y)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.traj_pub.publish(path)

    def _publish_map(self):
        self.mode_pub.publish(String(data=self.mission.mode))
        cones = self.mission.mapper.confirmed()
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="cones", size=len(cones), stride=3 * len(cones)),
                          MultiArrayDimension(label="cone", size=3, stride=3)]
        msg.data = cones.astype(np.float32).flatten().tolist()
        self.map_pub.publish(msg)

    def get_pose(self, reference_frame, stamp=None):
        """(x, y, yaw) de base_link em reference_frame (no stamp, ou a mais
        recente), ou None. Nao bloqueia."""
        try:
            tf = self.tf_buffer.lookup_transform(reference_frame, 'base_link',
                                                 stamp if stamp is not None else rclpy.time.Time())
        except TransformException:
            return None
        q = tf.transform.rotation
        return (tf.transform.translation.x, tf.transform.translation.y,
                yaw_from_quat(q.x, q.y, q.z, q.w))

    def destroy_node(self):
        self.tele.event("run_end", laps=self.mission.lap_count, mode=self.mission.mode)
        self.tele.close()
        super().destroy_node()

    def _ask_trajectory(self):
        if self.query_pending or not self.traj_cli.service_is_ready():
            return
        self.query_pending = True
        fut = self.traj_cli.call_async(TrajectoryQuery.Request(trajectory_id=0))
        fut.add_done_callback(self._on_trajectory)

    def _on_trajectory(self, fut):
        self.query_pending = False
        try:
            poses = fut.result().trajectory          # lista de PoseStamped, frame map
        except Exception as e:
            self.tele.periodic("traj_query_fail", 5.0, err=str(e))
            return
        if len(poses) < 2:
            return
        t   = np.array([p.header.stamp.sec + p.header.stamp.nanosec * 1e-9 for p in poses])
        x   = np.array([p.pose.position.x for p in poses])
        y   = np.array([p.pose.position.y for p in poses])
        yaw = np.array([yaw_from_quat(p.pose.orientation.x, p.pose.orientation.y,
                                      p.pose.orientation.z, p.pose.orientation.w) for p in poses])

        # quanto os nos ANTIGOS (presentes nas duas consultas) se moveram.
        # Casa pelo horario: o mesmo no volta sempre com o mesmo stamp.
        moved = 0.0
        if self.last_nodes is not None:
            t0, x0, y0 = self.last_nodes
            _, i_new, i_old = np.intersect1d(np.round(t, 6), np.round(t0, 6),
                                             return_indices=True)
            if len(i_new):
                moved = float(np.max(np.hypot(x[i_new] - x0[i_old], y[i_new] - y0[i_old])))
        self.last_nodes = (t, x, y)

        if moved > REPROJECT_MIN_MOVE:              # UMA correcao, nao uma por no
            self._reproject(t, x, y, yaw, moved)

    def _reproject(self, t, x, y, yaw, moved):
        # parte pesada FORA do lock (roda na thread do cg_slam enquanto os
        # cones e o controle seguem): refaz o mapa com as deteccoes ate agora
        snap = list(self.detections)
        new_map = ConeMapper()
        for (td, cones, lap) in snap:
            new_map.update(cones, self.pose_at(td, t, x, y, yaw), td, lap)
        with self.lock:
            # deteccoes que chegaram durante o replay
            for (td, cones, lap) in self.detections[len(snap):]:
                new_map.update(cones, self.pose_at(td, t, x, y, yaw), td, lap)
            new_map.frozen = self.mission.mapper.frozen   # respeita freeze_map_in_track
            self.mission.mapper = new_map
            now = self.tele.clock()                       # mesmo relogio que o Mission usa
            self.tele.event("map_reprojected", max_move_m=moved, n_nodes=len(t),
                            n_frames=len(self.detections), **new_map.counts())
            if self.mission.mode == TRACK:
                self.mission._try_build(now, reason="loop_closure")

    @staticmethod
    def pose_at(tq, t, x, y, yaw):
        i = np.clip(np.searchsorted(t, tq), 1, len(t) - 1)   # t[i-1] <= tq <= t[i]
        a = np.clip((tq - t[i-1]) / (t[i] - t[i-1]), 0.0, 1.0)
        dyaw = wrap_angle(yaw[i] - yaw[i-1])
        return (x[i-1] + a * (x[i] - x[i-1]),
                y[i-1] + a * (y[i] - y[i-1]),
                wrap_angle(yaw[i-1] + a * dyaw))


def main():
    rclpy.init()
    node = CarNode()
    # varias threads: o grupo do SLAM (consulta + reprojecao) nao trava o
    # grupo padrao (cones, controle a 20 Hz, watchdog, publicacoes)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.client.setCarControls(fsds.CarControls(throttle=0.0, steering=0.0, brake=1.0))
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()