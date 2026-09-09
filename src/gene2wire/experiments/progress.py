"""Worker-safe progress events, relayed by the notebook's parent process.

Each worker appends to its own JSONL file. The parent tails complete lines, so
Loky stdout forwarding and a multiprocessing manager are not required. Events
are diagnostics and never enter model selection or checkpoint fingerprints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time
import warnings
from zoneinfo import ZoneInfo

from .io import jsonable
from .workers import worker_capacity


@dataclass(frozen=True)
class EventWriter:
    path: Path
    context: dict = field(default_factory=dict)

    def bind(self, **context):
        return EventWriter(self.path, {**self.context, **context})

    def __call__(self, event):
        row = jsonable({**self.context, **event, "timestamp": time.time(), "pid": os.getpid()})
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
                 total_units=0, cached_units=0, worker_status=None,
                 worker_slots=0, requested_workers=0):
        if level not in ("summary", "model", "trial"):
            raise ValueError("progress_level must be 'summary', 'model' or 'trial'")
        if not 0 < interval <= 3600:
            raise ValueError("progress_interval must be positive and at most 3600 seconds")
        if total_units < 0 or cached_units < 0 or cached_units > total_units:
            raise ValueError("Progress counts must satisfy 0 <= cached_units <= total_units")
        if worker_slots < 0 or worker_slots > requested_workers:
            raise ValueError("Worker slots must satisfy 0 <= worker_slots <= requested_workers")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.enabled, self.level, self.interval = enabled, level, float(interval)
        self.total_units, self.done_units = total_units, cached_units
        self.cached_units = cached_units
        self.reused_fit_units = self.new_or_mixed_units = self.unknown_units = 0
        self.fit_activity, self.model_accounting = {}, []
        self.completed_models, self.failed_scenarios = set(), set()
        self.rows, self.offsets, self.active = [], {}, {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = self.last_heartbeat = time.monotonic()
        self.thread = None
        self.worker_status = worker_status
        self.worker_slots = int(worker_slots)
        self.capacity = worker_capacity(requested_workers)
        self.worker_activity = {}
        self.worker_status_failed = False

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
        if event in ("task_group_start", "task_group_complete") and row.get("pid") is not None:
            # A process can execute successive groups in different JSONL files.
            # File-name drain order must not resurrect a previously ended task.
            stamp = (float(row["timestamp"]), event == "task_group_complete")
            previous = self.worker_activity.get(row["pid"])
            if previous is None or stamp > previous[0]:
                self.worker_activity[row["pid"]] = (stamp, event == "task_group_start")
        label = self.label(row)
        if event in ("model_start", "candidate_start", "refit_start", "scenario_start"):
            self.active[unit] = label
        if event in ("candidate_complete", "refit_complete") and row.get("model"):
            model_key = self._event_key(row, include_model=True)
            activity = self.fit_activity.setdefault(model_key, set())
            status = row.get("cache_status")
            if status == "fitted":
                activity.add("new")
            elif status in ("checkpoint", "memory"):
                activity.add("reused")
            else:
                activity.add("unknown")
        if event == "model_complete" and row.get("model"):
            model_key = self._event_key(row, include_model=True)
            if model_key not in self.completed_models:
                self.completed_models.add(model_key)
                self.done_units += 1
                activity = self.fit_activity.get(model_key, set())
                # A final-refit cache hit does not imply its candidate search
                # was cached. Conversely, a rebuilt model wrapper can report
                # "fitted" although every candidate/refit was reused.
                if "new" in activity:
                    accounting = "new_or_mixed"
                    self.new_or_mixed_units += 1
                elif "unknown" not in activity and (
                    "reused" in activity or row.get("cache_status") in ("checkpoint", "memory")
                ):
                    accounting = "reused_fits"
                    self.reused_fit_units += 1
                else:
                    accounting = "unknown"
                    self.unknown_units += 1
                self.model_accounting.append({
                    **{key: row.get(key) for key in self._SCENARIO_KEYS},
                    "model": row["model"], "accounting": accounting})
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
        message += (f"; restored results {self.cached_units}, reused fits {self.reused_fit_units}, "
                    f"new/mixed {self.new_or_mixed_units}")
        if self.unknown_units:
            message += f", fit status unknown {self.unknown_units}"
        if self.failed_scenarios:
            message += f"; failed calibration scenarios {len(self.failed_scenarios)}"
        print(message, flush=True)

    def heartbeat(self, now=None):
        """Print at most one line per interval; return whether a line was printed."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if now - self.last_heartbeat < self.interval:
                return False
            if self.enabled:
                self._status(now)
            self._worker_status("running")
            self.last_heartbeat = now
            return self.enabled

    def _worker_status(self, phase):
        """Report occupied scheduling slots, never infer utilization from CPUs."""
        if self.worker_status is None or self.worker_status_failed:
            return
        occupied = (0 if phase in ("finished", "interrupted") else
                    sum(active for _, active in self.worker_activity.values()))
        # A late drain can briefly contain a finished PID and its replacement.
        # Occupancy is bounded by this experiment's explicitly scheduled pool.
        occupied = min(self.worker_slots, occupied)
        local_time = datetime.fromtimestamp(time.time(), ZoneInfo("America/Los_Angeles"))
        snapshot = {**self.capacity, "phase": phase, "worker_slots": self.worker_slots,
                    "occupied_workers": occupied,
                    "available_workers": self.worker_slots - occupied,
                    "finished_units": self.done_units, "total_units": self.total_units,
                    "local_time": f"{local_time:%Y-%m-%d %H:%M:%S %Z}"}
        try:
            self.worker_status(snapshot)
        except Exception as exc:
            # Rendering is optional diagnostics, not a training dependency.
            self.worker_status_failed = True
            warnings.warn(f"Worker status callback disabled: {exc}", RuntimeWarning, stacklevel=2)

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
            print(f"[cache] verified result summaries {self.cached_units}/{self.total_units}; "
                  f"remaining {self.total_units-self.cached_units} require cache checks or processing. "
                  "Model/candidate/refit reuse is counted during execution; unchecked does not mean uncached.",
                  flush=True)
        self._worker_status("running")
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.drain()
        if self.enabled:
            self._status(time.monotonic(), final=True, interrupted=bool(exc and exc[0]))
        self._worker_status("interrupted" if exc and exc[0] else "finished")
