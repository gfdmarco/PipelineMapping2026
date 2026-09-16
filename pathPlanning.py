import numpy as np
from scipy.spatial import Delaunay, QhullError
import matplotlib.pyplot as plt

# Class ids do modelo YOLO 16_01.pt:
# 0 = blue_cone (esquerda), 1 = yellow_cone (direita),
# 2 = large_orange_cone, 3 = orange_cone
BLUE_ID = 0
YELLOW_ID = 1
ORANGE_ID = 2

MAX_EDGE_LENGTH = 7.0   # m  -> descarta links azul<->amarelo longos demais
MAX_STEP_LENGTH = 6.0   # m  -> corta o caminho se o proximo ponto pular demais


class PathPlanner:
    """Planejador de rota (linha central) via triangulacao de Delaunay.

    Recebe os cones no referencial do carro como [lateral, frente, classId],
    o mesmo formato produzido por Perception.detectCones (lateral +direita,
    frente +a frente). Cones azuis (id 0) ficam a esquerda, amarelos (id 1)
    a direita.

    planPath retorna a linha central como um array ordenado de waypoints
    [lateral, frente], comecando no carro (0, 0).
    """

    def __init__(self, blue_id=BLUE_ID, yellow_id=YELLOW_ID,
                 max_edge_length=MAX_EDGE_LENGTH, max_step_length=MAX_STEP_LENGTH):
        self.blue_id = blue_id
        self.yellow_id = yellow_id
        self.max_edge_length = max_edge_length
        self.max_step_length = max_step_length

        # estado de debug, preenchido por planPath para o visualizador
        self.points = np.empty((0, 2))            # xy dos cones triangulados
        self.simplices = np.empty((0, 3), dtype=int)
        self.midpoints = np.empty((0, 2))

    def planPath(self, cones):
        self.points = np.empty((0, 2))
        self.simplices = np.empty((0, 3), dtype=int)
        self.midpoints = np.empty((0, 2))

        if cones is None or len(cones) < 3:
            return np.empty((0, 2))

        cones = np.asarray(cones, dtype=np.float64)
        xy = cones[:, :2]                  # (lateral, frente)
        ids = cones[:, 2].astype(int)

        try:
            tri = Delaunay(xy)
        except QhullError:
            # pontos colineares / degenerados
            return np.empty((0, 2))

        self.points = xy
        self.simplices = tri.simplices

        midpoints = self._trackMidpoints(xy, ids, tri.simplices)
        if midpoints.size == 0:
            return np.empty((0, 2))

        self.midpoints = midpoints
        return self._orderWaypoints(midpoints)

    def _trackMidpoints(self, xy, ids, simplices):
        """Midpoint de cada aresta unica que liga um cone azul a um amarelo."""
        midpoints = []
        seen = set()
        for simplex in simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)

                if {ids[a], ids[b]} != {self.blue_id, self.yellow_id}:
                    continue
                if np.linalg.norm(xy[a] - xy[b]) > self.max_edge_length:
                    continue
                midpoints.append((xy[a] + xy[b]) / 2.0)

        return np.array(midpoints) if midpoints else np.empty((0, 2))

    def _orderWaypoints(self, midpoints):
        """Ordena os midpoints por vizinho mais proximo, partindo do carro."""
        remaining = list(range(len(midpoints)))
        ordered = []
        current = np.array([0.0, 0.0])     # posicao do carro

        while remaining:
            nxt = min(remaining,
                      key=lambda i: np.linalg.norm(midpoints[i] - current))
            dist = np.linalg.norm(midpoints[nxt] - current)
            if ordered and dist > self.max_step_length:
                break
            ordered.append(nxt)
            current = midpoints[nxt]
            remaining.remove(nxt)

        path = midpoints[ordered]
        # prepende a posicao do carro para o caminho comecar embaixo do carro
        return np.vstack(([0.0, 0.0], path))


class PathVisualizer:
    """Plot ao vivo (vista de cima) dos cones, da triangulacao e da rota.

    Carro aponta para cima: frente no eixo Y, lateral no eixo X.
    """

    def __init__(self, view_range=35.0):
        self.view_range = view_range
        plt.ion()
        # Impede a janela de roubar o foco a cada plt.pause() (backend Qt)
        plt.rcParams['figure.raise_window'] = False
        self.fig, self.ax = plt.subplots(figsize=(6, 8))

    def update(self, cones, path, planner=None):
        ax = self.ax
        ax.clear()

        cones = np.asarray(cones)
        if cones.ndim == 2 and len(cones):
            lat, fwd = cones[:, 0], cones[:, 1]
            ids = cones[:, 2].astype(int)
            blue = ids == BLUE_ID
            yellow = ids == YELLOW_ID
            other = ~(blue | yellow)
            ax.scatter(lat[blue], fwd[blue], c='blue', s=45, label='azul (E)')
            ax.scatter(lat[yellow], fwd[yellow], c='gold', s=45,
                       edgecolors='k', linewidths=0.4, label='amarelo (D)')
            if other.any():
                ax.scatter(lat[other], fwd[other], c='orange', s=45,
                           label='laranja')

        # arestas da triangulacao (debug)
        if planner is not None and len(planner.simplices):
            pts = planner.points
            for s in planner.simplices:
                loop = pts[[s[0], s[1], s[2], s[0]]]
                ax.plot(loop[:, 0], loop[:, 1], color='lightgray',
                        lw=0.6, zorder=0)

        # midpoints candidatos
        if planner is not None and len(planner.midpoints):
            mp = planner.midpoints
            ax.scatter(mp[:, 0], mp[:, 1], c='green', s=18, marker='x',
                       label='midpoints')

        # rota planejada
        path = np.asarray(path)
        if path.ndim == 2 and len(path):
            ax.plot(path[:, 0], path[:, 1], '-o', color='red', lw=2,
                    ms=4, label='rota')

        # carro
        ax.scatter([0], [0], c='black', marker='^', s=130, label='carro')

        ax.set_xlim(-self.view_range / 2, self.view_range / 2)
        ax.set_ylim(-2, self.view_range)
        ax.set_xlabel('lateral [m]  (+direita)')
        ax.set_ylabel('frente [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

