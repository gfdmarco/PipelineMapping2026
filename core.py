"""Nucleo do pipeline de Mapeamento SEM dependencia de ROS/FSDS.

O CarNode (ROS + FSDS) e o simulador 2D chamam exatamente este codigo,
entao o que o simulador valida e o que roda no carro.

    LapDetector  -> conta voltas (frame odom, continuo, rearma a cada volta)
    Mission      -> maquina de estados EXPLORE -> TRACK:
                    EXPLORE: rota local reativa (Delaunay no frame do carro)
                             + construcao do mapa global de cones
                    volta 1 fechada: monta a trajetoria global com o mapa
                             confirmado; sucesso -> TRACK, falha -> continua
                             EXPLORE e tenta de novo a cada RETRY_PERIOD
                    TRACK:   segue a trajetoria global (localWindow) com o
                             perfil de velocidade; a cada volta reconstroi
                             com agressividade maior. Se sair mais de
                             TRACK_MAX_XTRACK da trajetoria (localizacao
                             ruim / mapa errado) cai para a rota local
                             naquele frame (evento track_fallback).

Entradas por frame (step): cones locais [lateral, frente, classe], pose
map (SLAM) e pose odom, velocidade, dt, t. Saida: steering, throttle,
brake. Toda decisao relevante vira evento de telemetria.
"""
import math
import time

import numpy as np

from . import pathPlanning, control, globalPlanning
from .coneMapper import ConeMapper
from .frames import wrap_angle, car_to_world

EXPLORE, TRACK = "EXPLORE", "TRACK"

RETRY_PERIOD = 2.0         # s entre tentativas de montar a trajetoria
TRACK_MAX_XTRACK = 3.0     # m: longe disso da trajetoria -> rota local

LARGE_ORANGE_ID = 2
GATE_RANGE = 6.0        # m: laranja grande a menos disso conta
GATE_MIN_CONES = 2      # no mesmo quadro

class LapDetector:
    """Fechamento de volta pela volta a pose de largada (frame odom).

    Rearme: depois de cada volta, so pode fechar a proxima depois de
    (a) percorrer min_distance e (b) ter se afastado mais que rearm_dist
    da largada. Assim funciona para N voltas e nao dispara duas vezes
    enquanto o carro ainda esta dentro do raio.
    """

    def __init__(self, min_distance=30.0, pos_threshold=6.0,
                 heading_threshold_deg=30.0, rearm_dist=12.0):
        self.min_distance = min_distance
        self.pos_threshold = pos_threshold
        self.heading_threshold = math.radians(heading_threshold_deg)
        self.rearm_dist = rearm_dist
        self.start_pose = None
        self.last_pose = None
        self.dist_lap = 0.0        # distancia na volta atual
        self.dist_total = 0.0
        self.armed = False
        self.laps = 0
        self.lap_start_t = None
        self.last = {}             # ultimo estado (telemetria)
        self.line_xy = None        # centro do portao laranja (linha de chegada)
        self.last_step = 0.0
        self.near_line = False
        self.gate_left = None      # m que faltam ate a linha, desde que o portao foi visto

    def update(self, pose_odom, t, cones=None):
        """Retorna dict do evento quando fecha uma volta, senao None."""
        if pose_odom is None:
            return None
        x, y, yaw = pose_odom
        if self.start_pose is None:
            self.start_pose = pose_odom
            self.last_pose = pose_odom
            self.lap_start_t = t
            return None

        self.last_step = math.hypot(x - self.last_pose[0], y - self.last_pose[1])
        self.dist_lap += self.last_step
        self.last_pose = pose_odom
        sx, sy, syaw = self.start_pose
        d_start = math.hypot(x - sx, y - sy)
        dh = abs(wrap_angle(yaw - syaw))
        if d_start > self.rearm_dist and self.dist_lap > self.min_distance:
            self.armed = True
        self.last = {"dist_lap": self.dist_lap, "d_start": d_start,
                    "heading_diff_deg": math.degrees(dh), "armed": self.armed}

        if not self.armed:
            self.gate_left = None
            return None

        # portao laranja visto a frente: guarda quanto falta ate ele e so
        # fecha a volta depois de percorrer essa distancia (na LINHA, nao
        # quando o portao aparece a ~6 m)
        step_d = self.last_step
        if self.gate_left is not None:
            self.gate_left -= step_d
        if cones is not None and len(cones):
            c = np.asarray(cones, dtype=float).reshape(-1, 3)
            o = c[c[:, 2] == LARGE_ORANGE_ID]
            near = o[(o[:, 1] > 0) & (np.linalg.norm(o[:, :2], axis=1) < GATE_RANGE)]
            if len(near) >= GATE_MIN_CONES:
                self.gate_left = float(np.mean(near[:, 1]))
        gate = self.gate_left is not None and self.gate_left <= 0.0

        # reserva: pela pose, quando passa a linha (ou a largada, se a linha
        # ainda nao foi vista) olhando na direcao da largada
        if self.line_xy is not None:
            lx, ly = self.line_xy
            d_line = math.hypot(x - lx, y - ly)
            passed = (x - lx) * math.cos(syaw) + (y - ly) * math.sin(syaw) >= 0.0
            by_pose = d_line < self.pos_threshold and passed and dh < self.heading_threshold
        else:
            by_pose = d_start < self.pos_threshold and dh < self.heading_threshold
        self.near_line = (d_line if self.line_xy is not None else d_start) < self.pos_threshold
        if not (gate or by_pose):
            return None

        self.laps += 1
        ev = {"lap": self.laps, "lap_time": t - self.lap_start_t,
            "lap_dist": self.dist_lap, "d_start": d_start,
            "heading_diff_deg": math.degrees(dh),
            "trigger": "laranja" if gate else "pose",
            "pose_odom": list(pose_odom)}
        self.dist_total += self.dist_lap
        self.dist_lap = 0.0
        self.armed = False
        self.gate_left = None
        self.lap_start_t = t
        return ev


class ProgressCounter:
    """Conta voltas no modo TRACK pelo PROGRESSO ao longo da trajetoria global.

    O detector por "voltar a pose de largada" (LapDetector) depende da pose
    absoluta: com deriva maior que pos_threshold ele simplesmente para de
    contar (visto no simulador: 1 volta detectada em 8). Aqui conta-se o
    avanco do indice mais proximo da trajetoria (localizacao relativa ao
    proprio mapa): uma volta = percorrer os N pontos do laco, cruzando o
    ponto da trajetoria mais proximo da largada.
    """

    def __init__(self, traj, nearest, start_xy, t):
        self.n = len(traj.points)
        d = np.hypot(traj.points[:, 0] - start_xy[0], traj.points[:, 1] - start_xy[1])
        self.start_idx = int(np.argmin(d))
        self.last = nearest
        self.u = 0.0                         # avanco acumulado (indices)
        self.step = traj._step or 1
        ahead = ((self.start_idx - nearest) * self.step) % self.n
        # largada logo a frente (< 25% do laco) = esse cruzamento ja foi
        # contado pelo LapDetector (que fecha ~pos_threshold antes)
        self.next = ahead + (self.n if ahead < 0.25 * self.n else 0)

    def update(self, nearest, step):
        if step:
            self.step = step
        d = nearest - self.last
        d = (d + self.n // 2) % self.n - self.n // 2       # menor arco
        self.u += d * self.step
        self.last = nearest
        if self.u >= self.next:
            self.next += self.n
            return True
        return False

    def rebase(self, traj, nearest):
        """Troca a trajetoria SEM perder o progresso da volta.

        O progresso esta em indices da trajetoria antiga; converte para a
        nova pela razao entre o numero de pontos (as duas cobrem a mesma
        pista, reamostradas com passo uniforme) e reancora o indice atual.
        """
        n_new = len(traj.points)
        if n_new <= 0:
            return
        scale = n_new / float(self.n)
        self.u *= scale
        self.next *= scale
        self.n = n_new
        self.last = nearest


class Mission:
    def __init__(self, tele, lap_kwargs=None, freeze_map_in_track=False, freeze_after_laps=0):
        self.tele = tele
        self.local_planner = pathPlanning.PathPlanner()
        self.global_planner = globalPlanning.GlobalPlanner()
        self.controller = control.VehicleController()
        self.mapper = ConeMapper()
        self.laps = LapDetector(**(lap_kwargs or {}))
        self.freeze_map_in_track = freeze_map_in_track
        # Regra (06/11): o mapa de cones congela ao FECHAR a volta N (0 = nunca).
        # O mapa melhora ate a volta 3 (erro alinhado 0.65 -> 0.36 -> 0.20 m no
        # run 20261106_131508); depois disso so acumula risco (fantasmas
        # temporarios, falso loop closure tardio, custo do reproject crescendo).
        self.freeze_after_laps = freeze_after_laps
        # lap_count no instante do congelamento: deteccoes com lap >= isto
        # (gravadas pelo CarNode) ficam FORA do mapa, inclusive no reproject
        self.frozen_at_lap = None
        # trajetoria percorrida (x, y, yaw no map, a cada ~0.5 m): usada para a
        # cor geometrica dos cones (ConeMapper.apply_side) ao montar a trajetoria
        self.pose_hist = []

        self.mode = EXPLORE
        self.trajectory = None
        self.progress = None
        self.lap_count = 0
        self.last_lap_t = None
        self.start_xy = None                 # largada no frame map
        self.lap1_closed = False
        self.build_attempts = 0
        self.last_build_try = -1e9
        self.path = np.empty((0, 2))
        self.frame = 0
        # True quando quem chama roda control_step() num timer proprio no
        # TRACK (CarNode/sim2d): o step() dos cones deixa de comandar o
        # carro nesse modo, salvo quando cai para a rota local (fallback)
        self.track_timer = False
        self.gate_xy = None                  # centro do portao laranja no frame map
        self.rebase_progress = False
        self.last_source = "local"
        self.last_cmd = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    def step(self, cones, pose_map, pose_odom, speed, dt, t, pose_meas=None):
        """pose_meas: pose map no INSTANTE DA CAPTURA dos cones (TF no stamp
        da mensagem). Sem ela o mapa usa a pose atual e herda a latencia da
        percepcao: 100 ms a 4 m/s = 0.4 m de erro ao longo da pista."""
        self.frame += 1
        cones = np.asarray(cones, dtype=float).reshape(-1, 3) if np.size(cones) else np.empty((0, 3))
        lap_now = self.lap_count          # voltas completas ate aqui
        if self.last_lap_t is None:
            self.last_lap_t = t
        if self.start_xy is None and pose_map is not None:
            self.start_xy = (pose_map[0], pose_map[1])
        if pose_map is not None and (not self.pose_hist or
                                     math.hypot(pose_map[0] - self.pose_hist[-1][0],
                                                pose_map[1] - self.pose_hist[-1][1]) >= 0.5):
            self.pose_hist.append(tuple(pose_map))

        # 0) linha de chegada: centro do 1o portao laranja visto (largada)
        if self.gate_xy is None and len(cones):
            pg = pose_meas if pose_meas is not None else pose_map
            o = cones[cones[:, 2] == LARGE_ORANGE_ID]
            o = o[(o[:, 1] > 0) & (np.linalg.norm(o[:, :2], axis=1) < GATE_RANGE)]
            if pg is not None and len(o) >= GATE_MIN_CONES:
                w = car_to_world(o[:, :2], pg)
                self.gate_xy = (float(w[:, 0].mean()), float(w[:, 1].mean()))
                self.laps.line_xy = self.gate_xy
                self.tele.event("gate_found", xy=list(self.gate_xy), n=len(o))

        # 1) mapa global
        tm = time.perf_counter()
        mstats = self.mapper.update(cones, pose_meas if pose_meas is not None else pose_map,
                                    t, lap_now)
        map_ms = (time.perf_counter() - tm) * 1e3
        if pose_map is None:
            self.tele.periodic("tf_missing", 2.0, frame="map", mode=self.mode)

        # 2) volta. EXPLORE: volta a pose de largada (pose map do SLAM se
        #    houver, senao odom). TRACK: progresso na trajetoria (abaixo).
        # pose map: e o mesmo frame da linha de chegada (gate_xy)
        lap_ev = self.laps.update(pose_map if pose_map is not None else pose_odom, t, cones)
        if lap_ev is not None:
            if not self.lap1_closed:
                self._on_lap(lap_ev, t, method="pose_largada")
            else:
                self.tele.event("lap_check_pose", **lap_ev)   # so comparacao
        # chegando na linha da volta 1 (portao a frente ou perto da largada):
        # ja entra no TRACK para cruzar o portao pela trajetoria global; a
        # volta em si so conta no cruzamento da linha (acima)
        if (self.mode == EXPLORE and not self.lap1_closed and self.laps.armed
                and (self.laps.gate_left is not None or self.laps.near_line)
                and t - self.last_build_try > RETRY_PERIOD):
            self._try_build(t, reason="pre_lap1")
        if (self.lap1_closed and self.mode == EXPLORE
                and t - self.last_build_try > RETRY_PERIOD):
            self._try_build(t, reason="retry")

        # 3) rota
        tp = time.perf_counter()
        v_target, xtrack, source = None, None, "local"
        if self.mode == TRACK and self.trajectory is not None:
            path, v_target, xtrack = self._track_route(pose_map, t)
            if path is not None:
                self.path, source = path, "global"
        if source == "local":
            self.path = self.local_planner.planPath(cones)
        plan_ms = (time.perf_counter() - tp) * 1e3

        # 4) controle (no TRACK com timer, quem comanda e o control_step)
        if self.track_timer and source == "global":
            steer, thr, brk = self.last_cmd
        else:
            steer, thr, brk = self.controller.compute(self.path, speed, dt, v_target)
            self.last_cmd = (steer, thr, brk)
        self.last_source = source

        cls = cones[:, 2].astype(int) if len(cones) else np.empty(0, dtype=int)
        self.tele.event(
            "frame", lap=lap_now + 1, mode=self.mode, src_path=source,
            n=len(cones), nb=int((cls == 0).sum()), ny=int((cls == 1).sum()),
            no=int((cls >= 2).sum()),
            n_mid=len(self.local_planner.midpoints) if source == "local" else None,
            path_n=len(self.path), path_len=_path_len(self.path),
            plan_ms=plan_ms, map_ms=map_ms,
            steer=steer, thr=thr, brk=brk, v=speed,
            v_tgt=self.controller.last_target, la=getattr(self.controller.lateral, "last_lookahead", None),
            no_path=self.controller.last_no_path, xtrack=xtrack,
            pm=list(pose_map) if pose_map is not None else None,
            po=list(pose_odom) if pose_odom is not None else None,
            m_new=mstats["n_new"], m_assoc=mstats["n_assoc"], m_far=mstats["n_far"],
            **self.mapper.counts())
        self.tele.periodic("lap_check", 1.0, **self.laps.last)
        return steer, thr, brk

    # ------------------------------------------------------------------
    def _track_route(self, pose_map, t):
        """TRACK: janela da trajetoria global + contagem de voltas por
        progresso. Retorna (path, v_target, xtrack); path None = fallback."""
        res = self.trajectory.localWindow(pose_map)
        if res is None:
            self.tele.periodic("track_fallback", 1.0, reason="sem_pose_map")
            return None, None, None
        path, v_target, nearest, xtrack = res
        if self.progress is None:
            # a volta fecha na LINHA (portao laranja), nao na pose de largada
            line = self.gate_xy or self.start_xy or (0.0, 0.0)
            self.progress = ProgressCounter(self.trajectory, nearest, line, t)
            self.rebase_progress = False
        elif self.rebase_progress:
            # so limpa a flag quando o rebase acontece de fato: o _on_lap
            # abaixo reconstroi a trajetoria e liga a flag de novo
            self.rebase_progress = False
            self.progress.rebase(self.trajectory, nearest)
        elif self.progress.update(nearest, self.trajectory._step):
            self._on_lap({"lap": self.lap_count + 1, "lap_time": t - self.last_lap_t,
                          "lap_dist": self.trajectory.length}, t, method="progresso")
        if xtrack > TRACK_MAX_XTRACK:
            self.tele.periodic("track_fallback", 1.0, reason="xtrack", xtrack=xtrack)
            return None, None, xtrack
        return path, v_target, xtrack

    def control_step(self, pose_map, speed, dt, t):
        """Laco de controle rapido do TRACK (timer, sem esperar os cones).

        Retorna (steering, throttle, brake), ou None quando nao esta no
        TRACK ou a trajetoria nao serve (sem pose / longe dela): nesses
        casos quem comanda e o step() dos cones, com a rota local.
        """
        if self.mode != TRACK or self.trajectory is None:
            return None
        path, v_target, xtrack = self._track_route(pose_map, t)
        if path is None:
            return None
        self.path = path
        cmd = self.controller.compute(path, speed, dt, v_target)
        self.last_cmd = cmd
        self.tele.event("ctl", v=speed, v_tgt=self.controller.last_target,
                        steer=cmd[0], thr=cmd[1], brk=cmd[2], xtrack=xtrack,
                        pm=list(pose_map) if pose_map is not None else None)
        return cmd

    def _on_lap(self, ev, t, method):
        self.lap_count += 1
        ev = dict(ev, lap=self.lap_count, lap_time=t - self.last_lap_t)
        self.last_lap_t = t
        self.tele.event("lap_closed", mode=self.mode, method=method, **ev, **self.mapper.counts())
        self.tele.event("map_snapshot", lap=ev["lap"], cones=self.mapper.snapshot())
        if self.freeze_after_laps and self.lap_count >= self.freeze_after_laps and not self.mapper.frozen:
            self._freeze_map(t, reason=f"lap{self.lap_count}")
        if ev["lap"] == 1:
            self.lap1_closed = True
            self._try_build(t, reason="lap1")
        elif self.mode == TRACK:
            self._try_build(t, reason=f"lap{ev['lap']}")

    def _freeze_map(self, t, reason):
        self.mapper.frozen = True
        self.frozen_at_lap = self.lap_count
        self.tele.event("map_frozen", reason=reason, lap=self.lap_count, **self.mapper.counts())

    def _try_build(self, t, reason):
        self.last_build_try = t
        self.build_attempts += 1
        laps_done = self.lap_count
        # 1a trajetoria com agressividade 0 (linha central, v minima); sobe
        # AGGRESSIVENESS_PER_LAP a cada volta completada no TRACK
        aggr = min(1.0, globalPlanning.AGGRESSIVENESS_PER_LAP * max(0, laps_done - 1))
        # cor geometrica: se o reproject ja calculou o lado com as poses
        # OTIMIZADAS do Cartographer, mantem; senao usa as poses ao vivo (que
        # sao as mesmas com que este mapa foi montado)
        if not getattr(self.mapper, "side_from_slam", False):
            self.mapper.apply_side(self.pose_hist)
        cones = self.mapper.confirmed()
        debug = {}
        tb = time.perf_counter()
        traj = self.global_planner.buildTrajectory(cones, aggr, debug)
        build_ms = (time.perf_counter() - tb) * 1e3
        if traj is None:
            self.tele.event("traj_build_fail", reason=reason, attempt=self.build_attempts,
                            aggr=aggr, build_ms=build_ms, **debug)
            return False
        self.trajectory = traj
        # mantem o progresso da volta: reancora na nova trajetoria no proximo
        # frame (zerar aqui fazia o contador nunca fechar volta, porque a
        # trajetoria e reconstruida a cada loop closure)
        if self.progress is not None:
            self.rebase_progress = True
        self.tele.event("traj_build", reason=reason, attempt=self.build_attempts,
                        aggr=aggr, build_ms=build_ms, **debug)
        self.tele.event("traj_snapshot", lap=laps_done, aggr=aggr,
                        pts=np.round(traj.points, 2), v=np.round(traj.speeds, 2))
        if self.mode != TRACK:
            self.tele.event("mode_switch", frm=self.mode, to=TRACK, lap=laps_done,
                            attempt=self.build_attempts)
            self.mode = TRACK
            if self.freeze_map_in_track and not self.mapper.frozen:
                self._freeze_map(None, reason="enter_track")
        return True


def _path_len(path):
    path = np.asarray(path)
    if path.ndim != 2 or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
