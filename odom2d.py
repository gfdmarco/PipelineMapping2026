"""odom2d.py -- odometria 2D para o Cartographer, a partir da INS.

Le /imu/odometry (SBG Ellipse2-D real ou sbg_mock -- este no nao sabe qual) e
republica em /mapping/odom_2d a MESMA mensagem com z = 0 e orientacao so com
o yaw. O Cartographer assina /mapping/odom_2d no lugar do /imu/odometry.

Por que (run 20261023_111513, replay offline do Cartographer com o bag):
  odometria da INS como vem (roll/pitch/z)  -> erro do SLAM 17.6 m (mediana), mapa girado 27 graus
  so z = 0                                  -> 25.9 m   (z nao e o problema)
  roll/pitch suavizados (media de 1 s)      -> 12.2 m   (nao e o ruido)
  pitch constante (sem variacao)            ->  0.57 m
  so yaw, z = 0 (este no)                   ->  0.48 m
O Cartographer 2D, sem IMU, usa a VARIACAO de atitude da odometria para
estimar a gravidade e "nivelar" a nuvem. O carro rola alguns decimos de grau
nas curvas -> ele inclina a nuvem -> o chao entra na faixa min_z/max_z ->
o scan matching gira o mapa. A nuvem que o Cartographer recebe ja vem
nivelada (no FSDS pelo ground_filter do bridge), entao a atitude da INS so
atrapalha. A pose do carro (TF odom->base_link) nao muda: so o que vai para o
Cartographer.

CONVERGENCIA UTM (06/11): o driver do SBG publica a POSICAO em grade UTM
(easting/northing) mas o YAW em relacao ao norte VERDADEIRO. Em Campinas
(zona 23, meridiano central 45W) os dois diferem pela convergencia meridiana
    gamma = atan(tan(lon - lon0) * sin(lat)) = +0.801 grau
(mesma formula do driver; o sbg_mock loga "convergencia +0.801 graus").
Sem corrigir, o carro parece andar 0.8 grau "de lado" e o mapa gira.
Somamos gamma ao yaw para ele ficar no mesmo referencial (grade) da posicao.
Sinal conferido nos runs 20261023_115613 e 20261106_131508: nas retas,
(direcao do movimento pelas posicoes) - yaw = +0.82 / +0.48 grau, e no replay
do Cartographer +0.8 reduziu o erro ao vivo alinhado (0.77 -> 0.49 m e
1.09 -> 0.69 m); -0.8 piorou (2.01 m e 1.55 m).
Outro local/pista: recalcular pela formula acima com lat/lon do /imu/nav_sat_fix.

Parametros: in_topic (/imu/odometry), out_topic (/mapping/odom_2d),
            yaw_offset_deg (+0.801: convergencia UTM em Campinas; 0 desliga)
"""
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class Odom2D(Node):
    def __init__(self):
        super().__init__('odom2d')
        self.declare_parameter('in_topic', '/imu/odometry')
        self.declare_parameter('out_topic', '/mapping/odom_2d')
        self.declare_parameter('yaw_offset_deg', 0.801)
        self.yaw_offset = math.radians(self.get_parameter('yaw_offset_deg').value)
        self.get_logger().info(f"correcao de yaw (convergencia UTM): "
                               f"{math.degrees(self.yaw_offset):+.3f} graus")
        self.pub = self.create_publisher(Odometry, self.get_parameter('out_topic').value, 50)
        self.create_subscription(Odometry, self.get_parameter('in_topic').value, self.cb, 50)

    def cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        yaw += self.yaw_offset          # norte verdadeiro -> norte da grade UTM (igual a posicao)
        msg.pose.pose.position.z = 0.0
        q.x = q.y = 0.0
        q.z, q.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        # covariancias de z/roll/pitch nao fazem sentido em 2D
        for i in (14, 21, 28):
            msg.pose.covariance[i] = 0.0
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = Odom2D()
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
