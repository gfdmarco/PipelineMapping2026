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
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import Bool, Float32MultiArray, MultiArrayDimension, String
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray
from tf2_ros import Buffer, TransformListener, TransformException

from .core import Mission
from .frames import yaw_from_quat
from .telemetry import Telemetry

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)
import fsds  # noqa: E402

INTERNAL_PERCEPTION_HZ = 20


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
        self.create_timer(1.0, self._publish_map)
        self.tele.event("run_start", cones_topic=g('cones_topic'), cones_msg=g('cones_msg'),
                        internal_perception=g('use_internal_perception'),
                        min_distance=g('min_distance_traveled_m'),
                        pos_threshold=g('position_threshold_m'),
                        heading_threshold_deg=g('heading_threshold_deg'),
                        rearm_dist=g('rearm_distance_m'))

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
        steer, thr, brk = self.mission.step(cones, pose_map, pose_odom, speed, dt,
                                            self.tele.clock(), pose_meas=pose_meas)
        if self.controlEnabled:
            self.client.setCarControls(fsds.CarControls(throttle=thr, steering=steer, brake=brk))
        self._publish_events()

    def _watchdog(self):
        """Percepcao parou (bridge caiu / YOLO travou): freia em vez de seguir o ultimo comando."""
        if self.last_cones_wall is None:
            return
        age = (self.get_clock().now() - self.last_cones_wall).nanoseconds * 1e-9
        if age > self.cones_timeout:
            self.tele.periodic("cones_stale", 1.0, age_s=age)
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


def main():
    rclpy.init()
    node = CarNode()
    try:
        rclpy.spin(node)
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
