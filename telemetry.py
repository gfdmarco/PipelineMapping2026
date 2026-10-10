"""Telemetria estruturada (JSONL) do pipeline de Mapeamento.

Cada processo escreve UM arquivo `<RUN_DIR>/<fonte>.jsonl`, uma linha JSON
por evento:

    {"t": 12.345, "src": "carNode", "ev": "lap_closed", "lap": 1, ...}

`t` = tempo de parede em segundos (epoch, time.time()) -- igual entre
processos, o analisador alinha tudo por ele -- ou o relogio simulado
quando o simulador 2D injeta `clock`. Todos os
processos de um mesmo run compartilham o RUN_DIR via variavel de
ambiente MAPPING_RUN_DIR (o run.sh cria e exporta). Sem ela, cria
~/Mapping2026/runs/<data_hora>.

Sem dependencia de ROS: o mesmo modulo roda no carro, no FSDS e no
simulador 2D, e o analisador (Tests/analysis/analyze_runs.py) le
qualquer um dos tres.

Uso:
    tele = Telemetry("carNode")
    tele.event("lap_closed", lap=1, lap_time=52.3)
    tele.periodic("lap_check", 1.0, dist=12.3)   # no maximo 1x por segundo
"""
import json
import math
import os
import time

import numpy as np

# Eventos que tambem vao para o terminal (ROS logger ou print) num
# formato curto e "grepavel":  [EV lap_closed] lap=1 lap_time=52.3
CONSOLE_EVENTS = {
    "run_start", "lap_closed", "mode_switch", "traj_build", "traj_build_fail",
    "loop_closure", "map_odom_jump", "tf_missing", "gt_track", "cone_hit",
    "off_track", "run_end", "warn", "lidar_check", "imu_check", "perception_stats",
    "slam_status", "referee", "lidar_mirror_check", "lidar_stats",
}


RUNS_ROOT = os.path.join(os.path.expanduser("~"), "Mapping2026", "runs")


def _default_run_dir():
    """Sem MAPPING_RUN_DIR: usa o run aberto pelo run.sh (arquivo CURRENT,
    se tiver menos de 6 h), para processos abertos em outros terminais
    cairem na mesma pasta. Senao cria uma pasta nova."""
    cur = os.path.join(RUNS_ROOT, "CURRENT")
    try:
        if time.time() - os.path.getmtime(cur) < 6 * 3600:
            d = open(cur).read().strip()
            if d:
                return d
    except OSError:
        pass
    return os.path.join(RUNS_ROOT, time.strftime("%Y%m%d_%H%M%S"))


def _clean(v):
    """Converte numpy/float nao-finito para algo serializavel."""
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return round(f, 4) if math.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


class Telemetry:
    def __init__(self, source, run_dir=None, clock=None, logger=None,
                 console=True, enabled=True):
        """
        source : nome do processo/arquivo (carNode, fsdsBridge, loopMonitor, sim2d)
        run_dir: pasta do run (default: $MAPPING_RUN_DIR ou ~/Mapping2026/runs/<ts>)
        clock  : funcao sem argumentos que retorna o tempo em s (sim 2D injeta
                 o relogio simulado); default = monotonic desde a criacao
        logger : objeto com .info(str) (ex.: node.get_logger()); default print
        """
        self.source = source
        self.enabled = enabled
        self.console = console
        self.logger = logger
        self._t0 = time.monotonic()
        self.clock = clock or time.time
        self._last_periodic = {}
        self._fh = None
        if not enabled:
            return
        self.run_dir = run_dir or os.environ.get("MAPPING_RUN_DIR") or _default_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)
        self.path = os.path.join(self.run_dir, f"{source}.jsonl")
        # line-buffered: se o processo morrer, o que ja foi escrito fica salvo
        self._fh = open(self.path, "a", buffering=1)

    # ------------------------------------------------------------------
    def event(self, ev, **fields):
        if not self.enabled or self._fh is None:   # desligado ou ja fechado
            return
        rec = {"t": round(float(self.clock()), 3), "src": self.source, "ev": ev}
        rec.update(_clean(fields))
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if self.console and ev in CONSOLE_EVENTS:
            self._print(ev, fields)

    def periodic(self, ev, period_s, **fields):
        """Igual a event(), mas no maximo uma vez a cada period_s segundos."""
        now = self.clock()
        last = self._last_periodic.get(ev)
        if last is not None and now - last < period_s:
            return False
        self._last_periodic[ev] = now
        self.event(ev, **fields)
        return True

    def _print(self, ev, fields):
        short = []
        for k, v in fields.items():
            v = _clean(v)
            if isinstance(v, (list, dict)) and len(json.dumps(v)) > 60:
                continue  # snapshots grandes so no arquivo
            short.append(f"{k}={v}")
        msg = f"[EV {ev}] " + " ".join(short)
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
