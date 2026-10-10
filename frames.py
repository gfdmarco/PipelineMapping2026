"""Conversoes de referencial UNICAS do pipeline (fonte dos bugs de eixo/sinal).

Convencoes (valem para TODO o pipeline):
  * Local (carro):  [lateral, frente]  -- lateral POSITIVO = DIREITA,
    frente positivo = a frente. E o formato da percepcao, do PathPlanner,
    do controlador e do topico /lidar_node/cones.
  * Mundo (map/odom): REP-103 / ENU -- x, y em metros, y POSITIVO =
    ESQUERDA, yaw anti-horario a partir de +x.
  * O FSDS ja entrega tudo em ENU (docs/coordinate-frames.md e o bridge
    oficial nao trocam sinal nenhum). NAO negar y vindo do FSDS.

Teste rapido mental: carro na origem, yaw = 0 (olhando para +x). Um cone
1 m a DIREITA e 5 m a frente ([lateral=+1, frente=5]) fica no mundo em
(x=5, y=-1).
"""
import numpy as np


def car_to_world(local, pose):
    """local (N,2+) [lateral, frente, ...] -> (N,2) [x, y] no mundo."""
    local = np.asarray(local, dtype=float)
    if local.size == 0:
        return np.empty((0, 2))
    local = np.atleast_2d(local)
    tx, ty, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    lat, fwd = local[:, 0], local[:, 1]
    # no frame do carro em REP-103: x_c = frente, y_c = -lateral
    x = tx + fwd * c + lat * s
    y = ty + fwd * s - lat * c
    return np.column_stack([x, y])


def world_to_car(world, pose):
    """world (N,2) [x, y] -> (N,2) [lateral, frente] no frame do carro."""
    world = np.asarray(world, dtype=float)
    if world.size == 0:
        return np.empty((0, 2))
    world = np.atleast_2d(world)
    tx, ty, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = world[:, 0] - tx, world[:, 1] - ty
    fwd = c * dx + s * dy
    left = -s * dx + c * dy
    return np.column_stack([-left, fwd])


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat(x, y, z, w):
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
