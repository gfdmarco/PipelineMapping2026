import rclpy
from rclpy.node import Node
import os,sys
from std_msgs.msg import Float32,Float32MultiArray,MultiArrayDimension,Bool
import numpy as np
from . import perception, pathPlanning, control, coneMapper, globalPlanning

import math

from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration

fsds_lib_path = os.path.join(os.path.expanduser("~"), "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import fsds

IMAGE_RECEIVING_FREQUENCY = 60
DATA_PUBLISH_FREQUENCY = 60

MIN_DISTANCE = 15.0

# Liga o plot matplotlib ao vivo para validar a rota (desligar em producao)
ENABLE_PLOT = False

class CarNode(Node):
    def __init__(self):
        super().__init__("CarNode")

        try:
            self.client = fsds.FSDSClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)

        except Exception as e:
            raise RuntimeError(f"Unable to connect to simulator.\n{e}")

        self.perception = perception.Perception(self.client)
        self.local_planner = pathPlanning.PathPlanner()
        self.global_planner = globalPlanning.GlobalPlanner()
        self.controller = control.VehicleController()
        self.visualizer = pathPlanning.PathVisualizer() if ENABLE_PLOT else None
        # trecho novo:
        self.coneMapper = coneMapper.ConeMapper(self)

        self.controlEnabled = True
        self.lastControlTime = self.get_clock().now()
        self.create_subscription(Bool,'/enable_control',self.authorization,10)

        self.cones = np.array([])
        self.path = np.empty((0, 2))

        # esboço de salvar estado pra detectar fechamento de volta
        self.declare_parameter('min_distance_traveled_m', 40.0)
        self.declare_parameter('position_threshold_m', 15.0)
        self.declare_parameter('heading_threshold_deg', 20.0)
        self.declare_parameter('check_period_s', 0.2)
        # falta salvar a rota ainda com write_state - IMPORTANTE PRO SLAM
        self.declare_parameter('pbstream_path', '/tmp/racing_map.pbstream')

        self.check_period = self.get_parameter('check_period_s').value
        self.create_timer(self.check_period, self.check_lap_closure)

        self.min_distance = self.get_parameter('min_distance_traveled_m').value
        self.pos_threshold = self.get_parameter('position_threshold_m').value
        self.heading_threshold = math.radians(
            self.get_parameter('heading_threshold_deg').value
        )
        self.pbstream_path = self.get_parameter('pbstream_path').value

        self.lap_closed = False 
        self.last_odom_pose = None 
        self.current_pose = None
        self.laps = 0

        self.start_pose = None
        self.distance_traveled = 0.0

        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.event_pub = self.create_publisher(Bool, '/mapping/lap_closed', 10)
        
        # Publisher temporario para testar a saida do detectCones
        self.conesPublisher = self.create_publisher(Float32MultiArray, "cones", 10)

        # salvando todos os cones
        self.all_cones = []
        # recebendo o gatilho de mudança
        self.event_sub = self.create_subscription(Bool, '/mapping/lap_closed', self.trackManager, 10)

        self.create_timer(1.0/DATA_PUBLISH_FREQUENCY,self.input_process)
  
    def authorization(self,msg):
        self.controlEnabled = msg.data
        state = "ON" if msg.data else "OFF"
        self.get_logger().info(f"Permission: {state}")

    def input_process(self):
        [image] = self.client.simGetImages(
            [fsds.ImageRequest(camera_name='ZED_RGB',
                            image_type=fsds.ImageType.Scene,
                            pixels_as_float=False,
                            compress=True)],
            vehicle_name='FSCar'
        )
        self.cones = self.perception.detectCones(image)
        #trecho novo:
        for cone in self.cones:
            self.all_cones.append(cone)
        if not self.lap_closed:
            self.path = self.local_planner.planPath(self.cones)
        self.publish_cones(self.cones)
        if(self.controlEnabled):
            self.control_step()

        if self.visualizer is not None:
            self.visualizer.update(self.cones, self.path, self.local_planner)
            
        self.coneMapper.globalizer(self.cones)

    def control_step(self):
        now = self.get_clock().now()
        dt = (now - self.lastControlTime).nanoseconds * 1e-9
        self.lastControlTime = now

        speed = self.client.getCarState().speed
        steering, throttle, brake = self.controller.compute(self.path, speed, dt)

        self.client.setCarControls(
            fsds.CarControls(throttle=throttle, steering=steering, brake=brake))

    def publish_cones(self, cones):
        msg = Float32MultiArray()

        cones = np.asarray(cones, dtype=np.float32)
        rows = cones.shape[0] if cones.ndim == 2 else 0
        cols = cones.shape[1] if cones.ndim == 2 else 0

        msg.layout.dim = [
            MultiArrayDimension(label="cones", size=rows, stride=rows * cols),
            MultiArrayDimension(label="cone", size=cols, stride=cols),
        ]
        msg.data = cones.flatten().tolist()

        self.conesPublisher.publish(msg)

    def get_pose(self, reference_frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                reference_frame,
                'base_link',
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
        except TransformException as e:
            self.get_logger().warn(
                f"Sem TF {reference_frame} -> base_link: {e}"
            )
            return None

        x = tf.transform.translation.x
        y = tf.transform.translation.y
        q = tf.transform.rotation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        )

        return x, y, yaw

    def check_lap_closure(self): 
        self.get_logger().info("CHECK LAP EXECUTOU")

        if self.lap_closed:
            return 
        
        map_pose = self.get_pose("map")
        odom_pose = self.get_pose("odom")
        
        if map_pose is None or odom_pose is None:
            self.get_logger().warn("Sem pose map -> base_link")
            return 

        map_x, map_y, map_yaw = map_pose
        odom_x, odom_y, _ = odom_pose

        if self.start_pose is None:
            self.start_pose = odom_pose 
            self.last_odom_pose = odom_pose
            self.distance_traveled = 0.0
            self.get_logger().info(f'Pose de partida registrada: x={map_x:.2f} y={map_y:.2f} yaw={math.degrees(map_yaw):.1f}')
            return
        
        last_odom_x, last_odom_y, _ = self.last_odom_pose

        delta_distance = math.hypot(
            odom_x - last_odom_x,
            odom_y - last_odom_y
        )

        self.distance_traveled += delta_distance

        self.last_odom_pose = odom_pose

        if self.distance_traveled < self.min_distance:
            return

        sx, sy, syaw = self.start_pose
        dist_to_start = math.hypot(odom_x - sx, odom_y - sy)

        heading_diff = abs(math.atan2(
            math.sin(odom_pose[2] - syaw), math.cos(odom_pose[2] - syaw)
        ))

        self.get_logger().info(
        f'percorrido={self.distance_traveled:.1f}m | '
        f'dist_inicio={dist_to_start:.2f}m | '
        f'heading={math.degrees(heading_diff):.1f}deg')
        
        if dist_to_start > self.pos_threshold:
            return
        if heading_diff > self.heading_threshold:
            return

        self.lap_closed = True
        self.get_logger().info(
            f'Fechamento de volta detectado! distância percorrida='
            f'{self.distance_traveled:.1f}m, dist. até a partida='
            f'{dist_to_start:.2f}m, diff. de orientação='
            f'{math.degrees(heading_diff):.1f}°'
        )

        self.event_pub.publish(Bool(data=True))
        #trecho novo
        self.laps += 1

    def try_build_trajectory(self, cones_arr):
        # primeira trajetoria com agressividade 0 (linha central pura,
        # velocidade minima): o mapa recem-fechado ainda e o menos
        # confiavel, e a entrada no modo TRACK ja troca o controlador --
        # a agressividade sobe a cada volta completada no TRACK
        lap = max(1, self.laps)
        aggressiveness = min(1.0, globalPlanning.AGGRESSIVENESS_PER_LAP
                                * (lap - 1))
        debug = {}

        trajectory = self.global_planner.buildTrajectory(self, cones_arr, aggressiveness, debug)

        self.get_logger().info(
            f"trajetoria global pronta (volta {self.laps}): "
            f"{len(trajectory.points)} pontos, agressividade "
            f"{aggressiveness:.2f}, v max {trajectory.speeds.max():.1f} m/s")

    def trackManager(self, msg):
        if msg.data:
            cones_arr = np.array([[c['x'], c['y'], c['class_id']]
                                for c in self.coneMapper.global_cones])
            self.try_build_trajectory(cones_arr)

        
def main():
    rclpy.init()
    node = CarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()