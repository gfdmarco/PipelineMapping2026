"""Remocao de chao por plano ajustado a cada nuvem (sem ROS, testavel offline).

(Copia de simulator_comunication/ground_filter.py: o bridge usa para a
percepcao interna, o cloud_leveler usa para o Cartographer -- no FSDS e no carro.)

Por que: o Lidar1 do FSDS tem 4 feixes (-5..+5 graus) a 0.55 m; os dois de
baixo batem no chao a ~6 m e ~19 m. Quando o carro arfa (acelera/freia) ou o
terreno inclina, 0.3-0.5 grau ja levanta o chao a 19 m em 10-15 cm -- e esses
pontos entram na faixa min_z/max_z do Cartographer como se fossem obstaculos.
Sao arcos que andam junto com o carro: o casamento de scans se agarra a eles e
a pose do SLAM explode (reproduzido offline: erro de 42 m de mediana com o
Cartographer real e odometria perfeita; sem esses pontos, < 1 m).

Como: o chao e ~98% da nuvem. Ajusta z = a*x + b*y + c (frame do LiDAR) por
minimos quadrados com rejeicao iterativa e troca o z de cada ponto pela ALTURA
acima desse plano (menos a altura do LiDAR, para a TF base_link->lidar_link
continuar valendo). Resultado: chao em z_base ~ 0 qualquer que seja a arfagem.
"""
import numpy as np


def level_cloud(points, lidar_z, tol=(0.15, 0.06, 0.03)):
    """points (N,3) no frame do LiDAR -> (nuvem nivelada, info).

    info = dict(pitch_deg, roll_deg, n_ground, ok). Se o ajuste falhar
    (poucos pontos de chao), devolve a nuvem original com ok=False.
    """
    pts = np.asarray(points, dtype=np.float64)
    info = {"pitch_deg": 0.0, "roll_deg": 0.0, "n_ground": 0, "ok": False}
    if len(pts) < 50:
        return points, info
    cand = pts[np.abs(pts[:, 2] + lidar_z) < 0.4]
    coef = np.array([0.0, 0.0, -lidar_z])
    for t in tol:
        if len(cand) < 30:
            return points, info
        A = np.c_[cand[:, 0], cand[:, 1], np.ones(len(cand))]
        coef, *_ = np.linalg.lstsq(A, cand[:, 2], rcond=None)
        res = cand[:, 2] - A @ coef
        cand = cand[np.abs(res) < t]
    a, b, c = coef
    # sanidade: inclinacao ate ~6 graus e altura perto da nominal
    if abs(a) > 0.1 or abs(b) > 0.1 or abs(c + lidar_z) > 0.2:
        return points, info
    h = (pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c)) / np.sqrt(1 + a * a + b * b)
    out = pts.copy()
    out[:, 2] = h - lidar_z
    info.update(pitch_deg=float(np.degrees(np.arctan(a))), roll_deg=float(np.degrees(np.arctan(b))),
                n_ground=int(len(cand)), ok=True)
    return out.astype(np.float32), info
