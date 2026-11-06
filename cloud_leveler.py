"""cloud_leveler.py -- nivela a nuvem do LiDAR pelo plano do chao antes do Cartographer.

Le a nuvem CRUA (in_topic: fsdsLidar3D no FSDS; o topico do LiDAR no carro --
este no nao sabe qual) e publica em out_topic a mesma nuvem, no mesmo frame e
com o mesmo carimbo, com o z de cada ponto trocado pela altura acima do plano
do chao ajustado naquela nuvem (menos lidar_z, para a TF base_link->lidar_link
continuar valendo). Algoritmo: ground_filter.level_cloud (o mesmo que o bridge
do FSDS ja usava e foi validado offline).

Por que: arfagem/rolagem do carro (acelerar, frear, curva) ou terreno inclinado
levantam o chao distante para dentro da faixa min_z/max_z do Cartographer; o
scan matching se agarra a esses arcos e o mapa gira (10/10 no FSDS: erro de
dezenas de metros sem nivelar, < 1 m nivelando).

Parametros:
  in_topic   (fsdsLidar3D)           nuvem crua (sensor_msgs/PointCloud2, x/y/z float32)
  out_topic  (/mapping/points_level) nuvem nivelada (so x, y, z)
  lidar_z    (0.55)                  altura do LiDAR acima do chao (m) -- MEDIR no carro
  max_range  (0.0)                   > 0 descarta pontos alem disso antes do ajuste
No carro: conferir que o frame da nuvem tem z para cima (FLU) e que o LiDAR
esta montado quase nivelado (o ajuste aceita ate ~6 graus; alem disso a nuvem
sai sem nivelar e o evento 'level_fail' aparece na telemetria).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

from .ground_filter import level_cloud
from .telemetry import Telemetry

_NP = {PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64}


def cloud_to_xyz(msg):
    """PointCloud2 qualquer (com campos x, y, z) -> (N, 3) float64 sem NaN."""
    f = {fl.name: fl for fl in msg.fields}
    if not all(k in f for k in 'xyz'):
        raise ValueError("nuvem sem campos x/y/z")
    n = msg.width * msg.height
    if n == 0:
        return np.empty((0, 3))
    dt = np.dtype({'names': list('xyz'),
                   'formats': [_NP[f[k].datatype] for k in 'xyz'],
                   'offsets': [f[k].offset for k in 'xyz'],
                   'itemsize': msg.point_step})
    if msg.is_bigendian:
        dt = dt.newbyteorder('>')
    raw = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    pts = np.stack([raw['x'], raw['y'], raw['z']], axis=1).astype(np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


def xyz_to_cloud(points, header):
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(points)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate('xyz')]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12 * len(points)
    msg.is_dense = True
    msg.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    return msg


class CloudLeveler(Node):
    def __init__(self):
        super().__init__('cloud_leveler')
        self.declare_parameter('in_topic', 'fsdsLidar3D')
        self.declare_parameter('out_topic', '/mapping/points_level')
        self.declare_parameter('lidar_z', 0.55)
        self.declare_parameter('max_range', 0.0)
        g = lambda k: self.get_parameter(k).value
        self.lidar_z, self.max_range = float(g('lidar_z')), float(g('max_range'))
        self.tele = Telemetry("cloudLeveler", logger=self.get_logger())
        self.pub = self.create_publisher(PointCloud2, g('out_topic'), 10)
        # QoS reliable: casa com o bridge do FSDS. Se o driver do LiDAR do carro
        # publicar best effort, esta assinatura precisa ser best effort tambem.
        self.create_subscription(PointCloud2, g('in_topic'), self.cb, 10)
        self.n = self.fails = 0
        self.tele.event("run_start", in_topic=g('in_topic'), out_topic=g('out_topic'),
                        lidar_z=self.lidar_z, max_range=self.max_range)

    def cb(self, msg):
        try:
            pts = cloud_to_xyz(msg)
        except Exception as e:
            self.tele.periodic("warn", 5.0, msg=f"nuvem invalida: {e}")
            return
        if self.max_range > 0 and len(pts):
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < self.max_range]
        out, info = level_cloud(pts, self.lidar_z)
        self.n += 1
        if not info['ok']:
            self.fails += 1
            self.tele.periodic("level_fail", 2.0, n_pts=len(pts), fails=self.fails, total=self.n)
        self.tele.periodic("level", 1.0, pitch_deg=info['pitch_deg'], roll_deg=info['roll_deg'],
                           n_ground=info['n_ground'], ok=info['ok'], n_pts=len(pts))
        self.pub.publish(xyz_to_cloud(out, msg.header))

    def destroy_node(self):
        self.tele.event("run_end", clouds=self.n, fails=self.fails)
        self.tele.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = CloudLeveler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
