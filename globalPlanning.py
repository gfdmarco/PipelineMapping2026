import numpy as np
from scipy.spatial import Delaunay, QhullError
from .pathPlanning import BLUE_ID, YELLOW_ID, MAX_EDGE_LENGTH
from .frames import world_to_car

# Construcao da linha central global
RESAMPLE_STEP = 1.0      # m   -> espacamento dos pontos da trajetoria
MAX_CHAIN_STEP = 6.0     # m   -> salto maximo entre midpoints ao ordenar
CLOSE_LOOP_DIST = 10.0   # m   -> dist ultimo->primeiro para fechar o laco (8 ate 06/11)
MIN_LOOP_LENGTH = 60.0   # m   -> laco mais curto que isso e um falso ciclo
                         #        local de midpoints, nao a pista
MIDPOINT_MERGE = 0.8     # m   -> midpoints mais proximos que isso viram um so
CHAIN_STARTS = 8         #     -> tentativas de encadeamento (inicios diferentes)
MIN_TURN_COS = 0.5       #     -> proximo passo nao pode dobrar mais que ~60
                         #        graus em relacao ao passo anterior (-0.3 ate 06/11)
# Correcoes de 06/11 (GA offline nos mapas dos runs 152737/153358/153930: trajetoria
# valida em 56% -> 94% dos mapas reais). Na pista C o laco nao fechava NEM com os
# cones perfeitos do GT: buraco de 6.6 m sem par azul/amarelo logo apos a largada.
BRIDGE_STEP = 17.0       # m   -> sem midpoint a < MAX_CHAIN_STEP, aceita um ate aqui...
BRIDGE_COS = 0.88        #     -> ...se estiver quase em linha reta (atravessa buracos)
MIDPOINT_MIN_CLEAR = 1.3 # m   -> midpoint mais perto que isso de um cone e aresta
                         #        espuria (cone duplicado/fantasma, diagonal em curva)
MIN_TRAJ_CLEARANCE = 0.9 # m   -> trajetoria com ponto mais perto que isso de um cone
                         #        e rejeitada (o Mission mantem a anterior e tenta de novo)

# Otimizacao da linha (suavizacao com restricao de corredor)
SMOOTH_ITERATIONS = 300
SMOOTH_ALPHA = 0.2
CAR_HALF_WIDTH = 0.7     # m
SAFETY_MARGIN = 0.9      # m   -> distancia minima ate o cone alem do carro (0.5 ate 17/10, depois 0.7)
MIN_SMOOTH_CORRIDOR = 0.4  # m -> mesmo com agressividade 0 a linha pode se
                           #      mover ate isso: tira o ziguezague da
                           #      corrente de midpoints sem buscar a linha
                           #      de corrida

# Perfil de velocidade
MAX_LAT_ACC = 6.0        # m/s^2 -> aceleracao lateral maxima admitida
MAX_LON_ACC = 4.0        # m/s^2 -> aceleracao para frente
MAX_LON_DEC = 6.0        # m/s^2 -> frenagem
MIN_SPEED = 4.5          # m/s
MAX_SPEED = 12.0         # m/s   -> teto com agressividade maxima

# Progressao por volta: comeca conservador e libera a cada volta completada.
# aggressiveness 0 = linha central e MIN_SPEED; 1 = linha otimizada e MAX_SPEED
AGGRESSIVENESS_PER_LAP = 0.25


class GlobalTrajectory:
    """Trajetoria global fechada: pontos (N,2) no mundo + velocidade alvo."""

    DIRECTION_HORIZON = 6.0   # m de trajetoria usados para decidir o sentido
    DIRECTION_MARGIN = 1.0    # m de vantagem exigida para INVERTER o sentido

    def __init__(self, points, speeds, aggressiveness=0.0):
        self.points = np.asarray(points, dtype=float)
        self.speeds = np.asarray(speeds, dtype=float)
        self.aggressiveness = aggressiveness
        seg = np.linalg.norm(np.diff(np.vstack([self.points, self.points[:1]]), axis=0), axis=1)
        self.length = float(seg.sum())
        self._step = None     # sentido de percurso decidido (com histerese)

    def localWindow(self, pose, window=30.0):
        """Trecho a frente do carro em [lateral, frente] + velocidade alvo.

        pose = (x, y, yaw) do carro no frame map. Acha o ponto mais
        proximo e devolve a janela seguinte (dando a volta no laco),
        pronta para o controlador. O sentido de percurso dos pontos e
        arbitrario (vem do encadeamento do mapa): a janela segue na
        direcao que aponta junto com o carro, decidida pela MEDIA de ~6 m
        de trajetoria e com histerese (so inverte com vantagem clara).

        Retorna (path (M,2) [lateral, frente], v_alvo, idx_mais_proximo,
        erro_lateral_m) ou None sem pose.
        """
        if pose is None or len(self.points) < 3:
            return None
        local = world_to_car(self.points, pose)       # [lateral(+dir), frente]
        dist = np.hypot(local[:, 0], local[:, 1])
        nearest = int(np.argmin(dist))
        n = len(self.points)

        k = max(2, int(self.DIRECTION_HORIZON / RESAMPLE_STEP))
        off = np.arange(1, k + 1)
        fwd_pos = float(np.mean(local[(nearest + off) % n, 1]))
        fwd_neg = float(np.mean(local[(nearest - off) % n, 1]))

        step = 1 if fwd_pos >= fwd_neg else -1
        if (self._step is not None and step != self._step
                and abs(fwd_pos - fwd_neg) < self.DIRECTION_MARGIN):
            step = self._step          # evidencia fraca: mantem o sentido
        self._step = step

        n_points = max(2, int(window / RESAMPLE_STEP))
        idx = (nearest + step * np.arange(n_points)) % n
        path = np.vstack([[0.0, 0.0], local[idx]])
        return path, float(self.speeds[nearest]), nearest, float(dist[nearest])


class GlobalPlanner:
    """Constroi a trajetoria global a partir do mapa de cones do SLAM.

    1. Linha central por Delaunay no mapa inteiro (mesma logica do
       planejador local, mas no mundo e fechando o laco da pista).
    2. Otimizacao: suavizacao iterativa da linha restrita ao corredor
       entre os cones (aproxima a linha de curvatura minima).
    3. Perfil de velocidade pela curvatura, com limites de aceleracao
       lateral e longitudinal (passadas de frenagem e acelaracao).

    `aggressiveness` em [0, 1] dosa o quanto da otimizacao e da
    velocidade maxima e usado -- cresce a cada volta completada.
    """

    def buildTrajectory(self, cones_world, aggressiveness, debug=None):
        """Retorna GlobalTrajectory ou None (motivo em debug['fail'])."""
        if debug is None:
            debug = {}
        cones_world = np.asarray(cones_world)
        debug['cones'] = len(cones_world)
        if len(cones_world) < 6:
            debug['fail'] = 'poucos_cones'
            return None

        midpoints = self._trackMidpoints(cones_world)
        debug['midpoints'] = len(midpoints)
        if len(midpoints) < 4:
            debug['fail'] = 'poucos_midpoints'
            return None

        centerline, closed = self._chainLoop(midpoints)
        debug['encadeados'] = len(centerline)
        debug['gap_fechamento_m'] = (round(float(np.linalg.norm(
            centerline[-1] - centerline[0])), 1) if len(centerline) else None)
        if not closed:
            debug['fail'] = 'laco_nao_fechou'
            return None

        centerline = self._resample(centerline)
        line = self._optimizeLine(centerline, cones_world, aggressiveness)
        bc = cones_world[cones_world[:, 2].astype(int) <= YELLOW_ID, :2]
        clr = float(np.min(np.linalg.norm(line[:, None, :] - bc[None, :, :], axis=2)))
        debug['folga_min_m'] = round(clr, 2)
        if clr < MIN_TRAJ_CLEARANCE:
            # mapa torto/duplicado gera linha colada num cone: melhor manter a anterior
            debug['fail'] = 'folga_baixa'
            return None
        speeds = self._speedProfile(line, aggressiveness)
        traj = GlobalTrajectory(line, speeds, aggressiveness)
        debug['n_points'] = len(line)
        debug['length_m'] = round(traj.length, 1)
        debug['v_max'] = round(float(speeds.max()), 2)
        debug['v_min'] = round(float(speeds.min()), 2)
        debug['offset_max_m'] = round(float(np.max(np.min(np.linalg.norm(
            line[:, None, :] - centerline[None, :, :], axis=2), axis=1))), 2)
        debug['kappa_max'] = round(float(self._curvature(line).max()), 3)
        return traj

    def _trackMidpoints(self, cones):
        # so azul/amarelo entram na triangulacao: os cones laranja da
        # largada ficam ENTRE as fileiras e roubam arestas azul<->amarelo
        # do Delaunay, abrindo um buraco de midpoints exatamente na linha
        # de chegada -- onde o laco precisa fechar
        ids = cones[:, 2].astype(int)
        cones = cones[(ids == BLUE_ID) | (ids == YELLOW_ID)]
        if len(cones) < 3:
            return np.empty((0, 2))
        xy = cones[:, :2]
        ids = cones[:, 2].astype(int)
        try:
            tri = Delaunay(xy)
        except QhullError:
            return np.empty((0, 2))

        midpoints, seen = [], set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                if {ids[a], ids[b]} != {BLUE_ID, YELLOW_ID}:
                    continue
                if np.linalg.norm(xy[a] - xy[b]) > MAX_EDGE_LENGTH:
                    continue
                midpoints.append((xy[a] + xy[b]) / 2.0)
        if not midpoints:
            return np.empty((0, 2))
        mp = np.array(midpoints)
        if MIDPOINT_MIN_CLEAR > 0:
            d = np.min(np.linalg.norm(mp[:, None, :] - xy[None, :, :], axis=2), axis=1)
            mp = mp[d >= MIDPOINT_MIN_CLEAR]
        return mp

    def _chainLoop(self, midpoints):
        """Ordena os midpoints em um laco fechado, de forma robusta a ruido.

        Tres defesas em relacao ao vizinho-mais-proximo ingenuo:
          1. Midpoints quase coincidentes (arestas de Delaunay vizinhas)
             sao fundidos antes, para a corrente nao ziguezaguear neles.
          2. O encadeamento nao pode dobrar para tras (MIN_TURN_COS):
             num cruzamento de midpoints ruidosos ele segue o fluxo da
             pista em vez de voltar pelo outro lado.
          3. Tenta varios pontos de partida e fica com a melhor corrente
             (fechada primeiro, mais comprida depois) -- o indice 0 pode
             cair justamente numa regiao mal mapeada.
        """
        midpoints = self._mergeMidpoints(midpoints)
        n = len(midpoints)
        if n < 4:
            return midpoints, False

        starts = np.unique(np.linspace(0, n - 1, CHAIN_STARTS).astype(int))
        best_chain, best_closed, best_len = midpoints[:0], False, -1.0

        for start in starts:
            chain = self._chainFrom(midpoints, int(start))
            length = float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1)))
            closed = (len(chain) >= 4
                      and length > MIN_LOOP_LENGTH
                      and np.linalg.norm(chain[-1] - chain[0]) < CLOSE_LOOP_DIST)
            if (closed, length) > (best_closed, best_len):
                best_chain, best_closed, best_len = chain, closed, length

        return best_chain, best_closed

    def _chainFrom(self, midpoints, start):
        """Encadeia por vizinho mais proximo com continuidade de direcao."""
        used = np.zeros(len(midpoints), dtype=bool)
        used[start] = True
        ordered = [start]
        direction = None

        while not used.all():
            current = midpoints[ordered[-1]]
            cand = np.flatnonzero(~used)
            delta = midpoints[cand] - current
            dist = np.linalg.norm(delta, axis=1)

            ok = dist < MAX_CHAIN_STEP
            if direction is not None:
                cos = (delta @ direction) / np.maximum(dist, 1e-9)
                forward = ok & (cos > MIN_TURN_COS)
                # so dobra para tras se nao houver caminho seguindo em frente
                if forward.any():
                    ok = forward
                if not ok.any() and BRIDGE_STEP > 0:
                    # ponte: buraco de midpoints (largada, trecho mal mapeado)
                    ok = (dist < BRIDGE_STEP) & (cos > BRIDGE_COS)
            if not ok.any():
                break

            nxt = cand[ok][int(np.argmin(dist[ok]))]
            step = midpoints[nxt] - current
            norm = np.linalg.norm(step)
            if norm > 1e-9:
                direction = step / norm
            ordered.append(int(nxt))
            used[nxt] = True

        return midpoints[ordered]

    def _mergeMidpoints(self, midpoints, radius=MIDPOINT_MERGE):
        """Funde midpoints quase coincidentes (media simples)."""
        merged, counts = [], []
        for p in midpoints:
            for k in range(len(merged)):
                if np.linalg.norm(merged[k] - p) < radius:
                    merged[k] = (merged[k] * counts[k] + p) / (counts[k] + 1)
                    counts[k] += 1
                    break
            else:
                merged.append(p.copy())
                counts.append(1)
        return np.array(merged) if merged else midpoints

    def _resample(self, points, step=RESAMPLE_STEP):
        """Reamostra o laco fechado com espacamento uniforme."""
        loop = np.vstack([points, points[:1]])
        seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]

        si = np.arange(0.0, total, step)
        x = np.interp(si, s, loop[:, 0])
        y = np.interp(si, s, loop[:, 1])
        return np.column_stack([x, y])

    def _corridorHalfWidth(self, points, cones):
        """Quanto cada ponto pode se afastar da linha central com seguranca."""
        blue = cones[cones[:, 2].astype(int) == BLUE_ID, :2]
        yellow = cones[cones[:, 2].astype(int) == YELLOW_ID, :2]
        margin = CAR_HALF_WIDTH + SAFETY_MARGIN

        half = np.full(len(points), 0.0)
        for i, p in enumerate(points):
            d_blue = np.min(np.linalg.norm(blue - p, axis=1)) if len(blue) else np.inf
            d_yel = np.min(np.linalg.norm(yellow - p, axis=1)) if len(yellow) else np.inf
            half[i] = max(0.0, min(d_blue, d_yel) - margin)
        return half

    def _optimizeLine(self, centerline, cones, aggressiveness):
        """Suavizacao iterativa restrita ao corredor (linha de menor curvatura).

        Cada ponto e puxado para a media dos vizinhos (reduz curvatura) e o
        deslocamento e limitado SO NA DIRECAO NORMAL a linha central (antes
        era um disco em volta do ponto: os pontos escorregavam ao longo da
        pista, embolavam e geravam picos de curvatura de raio < 1 m que o
        perfil de velocidade transformava em freadas a 1.5 m/s). No fim a
        linha e reamostrada com passo uniforme. Com aggressiveness 0 ainda
        suaviza dentro de MIN_SMOOTH_CORRIDOR (tira o ziguezague).
        """
        half = self._corridorHalfWidth(centerline, cones)
        corridor = np.minimum(half, np.maximum(half * aggressiveness,
                                               MIN_SMOOTH_CORRIDOR))
        if corridor.max() <= 0.0:
            return centerline
        tang = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
        tang /= np.maximum(np.linalg.norm(tang, axis=1), 1e-9)[:, None]
        normal = np.column_stack([-tang[:, 1], tang[:, 0]])

        line = centerline.copy()
        for _ in range(SMOOTH_ITERATIONS):
            target = 0.5 * (np.roll(line, 1, axis=0) + np.roll(line, -1, axis=0))
            line = line + SMOOTH_ALPHA * (target - line)
            off = np.einsum('ij,ij->i', line - centerline, normal)
            off = np.clip(off, -corridor, corridor)
            line = centerline + normal * off[:, None]
        return self._resample(line)

    def _curvature(self, points):
        """Curvatura discreta (1/raio) por tres pontos consecutivos."""
        prev_p = np.roll(points, 1, axis=0)
        next_p = np.roll(points, -1, axis=0)

        a = np.linalg.norm(points - prev_p, axis=1)
        b = np.linalg.norm(next_p - points, axis=1)
        c = np.linalg.norm(next_p - prev_p, axis=1)

        cross = ((points[:, 0] - prev_p[:, 0]) * (next_p[:, 1] - prev_p[:, 1])
                 - (points[:, 1] - prev_p[:, 1]) * (next_p[:, 0] - prev_p[:, 0]))
        area2 = np.abs(cross)
        denom = np.maximum(a * b * c, 1e-9)
        return 2.0 * area2 / denom

    def _speedProfile(self, points, aggressiveness):
        """Velocidade alvo por ponto: limite lateral + frenagem/aceleracao."""
        v_cap = MIN_SPEED + aggressiveness * (MAX_SPEED - MIN_SPEED)

        kappa = self._curvature(points)
        v = np.sqrt(MAX_LAT_ACC / np.maximum(kappa, 1e-6))
        v = np.clip(v, 0.0, v_cap)

        seg = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
        n = len(v)

        # passada para tras (duas voltas no laco): respeita a frenagem
        for i in range(2 * n - 1, -1, -1):
            j, k = i % n, (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[k] ** 2 + 2.0 * MAX_LON_DEC * seg[j]))

        # passada para frente: respeita a aceleracao disponivel
        for i in range(2 * n):
            j, k = i % n, (i + 1) % n
            v[k] = min(v[k], np.sqrt(v[j] ** 2 + 2.0 * MAX_LON_ACC * seg[j]))

        return v
