import numpy as np

# Parametros do veiculo (FSCar do FSDS)
WHEELBASE = 1.55         # m, distancia entre eixos
MAX_STEER_DEG = 25.0     # graus de estercamento quando steering = 1.0 (docs FSDS)

# Pure pursuit (lateral)
LOOKAHEAD_GAIN = 0.6     # s   -> lookahead cresce com a velocidade
MIN_LOOKAHEAD = 4.0      # m
MAX_LOOKAHEAD = 5.0      # m
STRAIGHT_LOOKAHEAD = 9.0 # m   -> lookahead usado quando a rota a frente e reta
STRAIGHT_TOLERANCE = 0.45 # m  -> desvio medio da corda para considerar reta
STRAIGHT_HORIZON = 12.0  # m   -> janela a frente usada nessa medicao
STRAIGHT_MAX_ANGLE = 12.0 # graus -> reta so vale alinhada ao heading do carro

STEERING_TAU = 0.1  # s   -> filtro passa-baixa do comando de steering

# PID (longitudinal)
TARGET_SPEED = 4      # m/s
KP = 0.3
KI = 0.05
KD = 0.00
MAX_THROTTLE = 0.3
MAX_BRAKE = 0.5


class PID:
    """PID classico com saturacao da integral (anti-windup)."""

    def __init__(self, kp, ki, kd, integral_limit=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error, dt):
        if dt <= 0.0:
            return self.kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral,
                                -self.integral_limit, self.integral_limit)

        derivative = 0.0
        if self.prev_error is not None:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative


class PurePursuit:
    """Controle lateral por pure pursuit.

    Recebe a rota no referencial do carro como [lateral, frente] (mesmo
    formato do PathPlanner: lateral +direita, frente +a frente, carro na
    origem apontando para +frente).

    computeSteering retorna o comando de steering do FSDS em [-1, 1]
    (+1 = direita, -1 = esquerda, convencao da API python / fullLap).
    """

    def __init__(self, wheelbase=WHEELBASE, max_steer_deg=MAX_STEER_DEG,
                 lookahead_gain=LOOKAHEAD_GAIN,
                 min_lookahead=MIN_LOOKAHEAD, max_lookahead=MAX_LOOKAHEAD,
                 straight_lookahead=STRAIGHT_LOOKAHEAD,
                 straight_tolerance=STRAIGHT_TOLERANCE,
                 straight_horizon=STRAIGHT_HORIZON,
                 straight_max_angle=STRAIGHT_MAX_ANGLE):
        self.wheelbase = wheelbase
        self.max_steer_rad = np.deg2rad(max_steer_deg)
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.straight_lookahead = straight_lookahead
        self.straight_tolerance = straight_tolerance
        self.straight_horizon = straight_horizon
        self.straight_max_angle_rad = np.deg2rad(straight_max_angle)

    def computeSteering(self, path, speed):
        path = np.asarray(path, dtype=np.float64)
        if path.ndim != 2 or len(path) < 2:
            return None

        lookahead = np.clip(self.lookahead_gain * speed,
                            self.min_lookahead, self.max_lookahead)

        # em trecho reto alonga o lookahead: o steering fica bem menos
        # sensivel ao ruido da percepcao (zigue-zague), sem afetar curvas
        straight = self._straightFactor(path)
        lookahead += straight * max(0.0, self.straight_lookahead - lookahead)

        target, dist = self._lookaheadPoint(path, lookahead)
        if target is None or dist < 1e-3:
            return None

        # angulo entre o heading (+frente) e o ponto alvo; positivo = direita
        alpha = np.arctan2(target[0], target[1])

        # geometria do pure pursuit: angulo de esterconamento das rodas
        delta = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), dist)

        return float(np.clip(delta / self.max_steer_rad, -1.0, 1.0))

    def _lookaheadPoint(self, path, lookahead):
        """Primeiro ponto da rota a distancia `lookahead` do carro.

        Interpola no segmento que cruza o circulo de lookahead; se a rota
        inteira for mais curta, usa o ultimo ponto.
        """
        dists = np.linalg.norm(path, axis=1)

        for i in range(1, len(path)):
            if dists[i] < lookahead:
                continue

            # segmento path[i-1] -> path[i] cruza o circulo: acha o ponto
            # exato resolvendo ||a + t*d||^2 = lookahead^2 para t em [0, 1]
            a, d = path[i - 1], path[i] - path[i - 1]
            dd = d @ d
            if dd < 1e-12:
                return path[i], dists[i]

            t_half = -(a @ d) / dd
            disc = t_half ** 2 - (a @ a - lookahead ** 2) / dd
            if disc < 0.0:
                return path[i], dists[i]

            t = np.clip(t_half + np.sqrt(disc), 0.0, 1.0)
            target = a + t * d
            return target, float(np.linalg.norm(target))

        # rota mais curta que o lookahead: mira no ultimo waypoint
        return path[-1], float(dists[-1])

    def _straightFactor(self, path):
        """Quao reta e a rota a frente: 0 = curvando, 1 = reta.

        Usa o desvio medio COM SinAL dos waypoints em relacao a corda do
        trecho dentro do horizonte: numa curva todos desviam para o mesmo
        lado (media alta); o ruido da percepcao se cancela (media ~0).
        Ignora path[0], que e a posicao do carro, para medir a geometria
        da pista e nao o offset do carro nela.
        """
        pts = path[1:]
        pts = pts[np.linalg.norm(pts, axis=1) <= self.straight_horizon]
        if len(pts) < 3:
            return 0.0

        chord = pts[-1] - pts[0]
        length = np.linalg.norm(chord)
        if length < 1e-6:
            return 0.0

        # na saida de curva a reta a frente ainda esta angulada em relacao
        # ao nariz do carro: o lookahead precisa continuar curto ate o
        # carro terminar de virar, senao ele relaxa cedo e sai por fora
        misalign = abs(np.arctan2(chord[0], chord[1]))
        align = np.clip(1.0 - misalign / self.straight_max_angle_rad, 0.0, 1.0)
        if align <= 0.0:
            return 0.0

        deviation = abs(np.mean(np.cross(pts[1:-1] - pts[0], chord / length)))
        return float(align *
                     np.clip(1.0 - deviation / self.straight_tolerance, 0.0, 1.0))


class VehicleController:
    """Junta o controle lateral (pure pursuit) e longitudinal (PID).

    compute(path, speed, dt) -> (steering, throttle, brake), prontos para
    o fsds.CarControls. Sem rota valida, mantem o ultimo steering e freia
    suavemente.
    """

    def __init__(self, target_speed=TARGET_SPEED,
                 max_throttle=MAX_THROTTLE, max_brake=MAX_BRAKE):
        self.target_speed = target_speed
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.lateral = PurePursuit()
        self.longitudinal = PID(KP, KI, KD)
        self.last_steering = 0.0

    def compute(self, path, speed, dt):
        steering = self.lateral.computeSteering(path, speed)

        if steering is None:
            # sem rota: segura o ultimo steering e desacelera
            self.longitudinal.reset()
            return self.last_steering, 0.0, 0.3 * self.max_brake

        # filtro passa-baixa de 1a ordem: suaviza o ruido frame a frame da
        # percepcao sem atrasar de forma relevante a entrada nas curvas
        if dt > 0.0:
            alpha = min(1.0, dt / STEERING_TAU)
            steering = self.last_steering + (steering - self.last_steering) * alpha

        self.last_steering = steering

        u = self.longitudinal.update(self.target_speed - speed, dt)
        throttle = float(np.clip(u, 0.0, self.max_throttle))
        brake = float(np.clip(-u, 0.0, self.max_brake))

        return steering, throttle, brake
