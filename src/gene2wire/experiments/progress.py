"""Worker-safe progress events, relayed by the notebook's parent process.

Each worker appends to its own JSONL file. The parent tails complete lines, so
Loky stdout forwarding and a multiprocessing manager are not required. Events
are diagnostics and never enter model selection or checkpoint fingerprints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from .io import jsonable


@dataclass(frozen=True)
class EventWriter:
    path: Path
    context: dict = field(default_factory=dict)

    def bind(self, **context):
        return EventWriter(self.path, {**self.context, **context})

    def __call__(self, event):
        row = jsonable({**self.context, **event, "timestamp": time.time()})
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


class ProgressRelay:
    """Relay detailed events while printing one compact heartbeat per minute.

    A progress unit is a complete model evaluation (selection plus final refit)
    for one dataset/rho, repetition, fold and observation/calibration scenario.
    Candidate fits are retained in the event log but are not separate units:
    their count can vary with strategy, warm-start reuse and optimizer retries.
    Scenario ``unit_complete`` events never increment the model-unit count.
    """

    _SCENARIO_KEYS = ("work_id", "dataset", "sharing_strength", "repetition",
                      "outer_fold", "analysis", "mechanism", "loss_rate",
                      "calibration_fraction", "calibration_spec")

    def __init__(self, directory, *, enabled=True, level="summary", interval=60.,
                 total_units=0, cached_units=0):
        if level not in ("summary", "model", "trial"):
            raise ValueError("progress_level must be 'summary', 'model' or 'trial'")
        if not 0 < interval <= 3600:
            raise ValueError("progress_interval must be positive and at most 3600 seconds")
        if total_units < 0 or cached_units < 0 or cached_units > total_units:
            raise ValueError("Progress counts must satisfy 0 <= cached_units <= total_units")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.enabled, self.level, self.interval = enabled, level, float(interval)
        self.total_units, self.done_units = total_units, cached_units
        self.cached_units = cached_units
        self.completed_models, self.failed_scenarios = set(), set()
        self.rows, self.offsets, self.active = [], {}, {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = self.last_heartbeat = time.monotonic()
        self.thread = None

    def writer(self, key, **context):
        return EventWriter(self.directory / f"{key}.jsonl", context)

    @staticmethod
    def label(row):
        bits = [str(row.get("dataset", ""))]
        for name, short in (("sharing_strength", "rho"), ("repetition", "rep"),
                            ("outer_fold", "fold"), ("loss_rate", "loss")):
            if row.get(name) is not None:
                bits.append(f"{short}={row[name]}")
        if row.get("analysis"):
            bits.append(str(row["analysis"]))
        if row.get("model"):
            bits.append(str(row["model"]))
        return " | ".join(bits)

    def _accept(self, row):
        self.rows.append(row)
        event, unit = row.get("event"), row.get("work_id")
        label = self.label(row)
        if event in ("model_start", "candidate_start", "refit_start", "scenario_start"):
            self.active[unit] = label
        if event == "model_complete" and row.get("model"):
            model_key = self._event_key(row, include_model=True)
            if model_key not in self.completed_models:
                self.completed_models.add(model_key)
                self.done_units += 1
            self.active.pop(unit, None)
        if event == "calibration_failed":
            self.failed_scenarios.add(self._event_key(row))
            self.active.pop(unit, None)
        if event == "unit_complete":
            self.active.pop(unit, None)
        if not self.enabled or self.level == "summary":
            return
        seconds = row.get("elapsed_seconds")
        elapsed = "" if seconds is None else f"; {seconds:.1f}s"
        if event == "candidate_inventory":
            print(f"[cache] {label}: {row.get('cached', 0)}/{row['total']} candidates reusable "
                  f"(disk={row.get('checkpoint_cached', 0)}, memory={row.get('memory_cached', 0)}); "
                  f"{row['pending']} need fitting", flush=True)
        elif event == "model_start":
            print(f"[start] {label}", flush=True)
        elif event == "model_complete":
            summary = row.get("summary") or {}
            details = ", ".join(f"{key}={summary[key]}" for key in
                                ("selected_structure", "rank", "shared_l2", "residual_l2",
                                 "final_converged") if key in summary)
            print(f"[done] {label}; {row.get('cache_status', 'complete')}{elapsed} "
                  f"{details}", flush=True)
        elif event == "calibration_failed":
            print(f"[failed calibration] {label}: {row.get('error')}", flush=True)
        elif event == "candidate_complete" and self.level == "trial":
            print(f"[trial] {label} {row.get('index')}/{row.get('total')}; "
                  f"{row.get('cache_status')}{elapsed}", flush=True)

    @classmethod
    def _event_key(cls, row, *, include_model=False):
        keys = cls._SCENARIO_KEYS + (("model",) if include_model else ())
        return json.dumps(jsonable({key: row.get(key) for key in keys}),
                          sort_keys=True, allow_nan=False)

    def _status(self, now, *, final=False, interrupted=False):
        minutes = max(0, int((now - self.started) // 60))
        label = "interrupted" if interrupted else "finished" if final else "running"
        local_time = datetime.fromtimestamp(time.time(), ZoneInfo("America/Los_Angeles"))
        message = (f"[{label} {minutes}min] finished units {self.done_units}/{self.total_units}, "
                   f"current time {local_time:%Y-%m-%d %H:%M:%S %Z}")
        if self.failed_scenarios:
            message += f"; failed calibration scenarios {len(self.failed_scenarios)}"
        print(message, flush=True)

    def heartbeat(self, now=None):
        """Print at most one line per interval; return whether a line was printed."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if not self.enabled or now - self.last_heartbeat < self.interval:
                return False
            self._status(now)
            self.last_heartbeat = now
            return True

    def drain(self):
        with self.lock:
            for path in sorted(self.directory.glob("*.jsonl")):
                offset = self.offsets.get(path, 0)
                with path.open("rb") as handle:
                    handle.seek(offset)
                    while True:
                        line = handle.readline()
                        if not line or not line.endswith(b"\n"):
                            break
                        self._accept(json.loads(line))
                        offset = handle.tell()
                self.offsets[path] = offset

    def _loop(self):
        while not self.stop.wait(.25):
            self.drain()
            self.heartbeat()

    def __enter__(self):
        if self.enabled:
            print(f"[cache] {self.cached_units}/{self.total_units} model evaluations reusable "
                  f"from verified complete scenario summaries; "
                  f"{self.total_units-self.cached_units} still to process "
                  "(may reuse model/candidate/refit caches).", flush=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.drain()
        if self.enabled:
            self._status(time.monotonic(), final=True, interrupted=bool(exc and exc[0]))
