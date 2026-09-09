"""Worker-safe progress events, relayed by the notebook's parent process.

Each worker appends to its own JSONL file. The parent tails complete lines, so
Loky stdout forwarding and a multiprocessing manager are not required. Events
are diagnostics and never enter model selection or checkpoint fingerprints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import threading
import time

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
    def __init__(self, directory, *, enabled=True, level="model", interval=30.,
                 total_units=0, cached_units=0):
        if level not in ("model", "trial"):
            raise ValueError("progress_level must be 'model' or 'trial'")
        if not 0 < interval <= 3600:
            raise ValueError("progress_interval must be positive and at most 3600 seconds")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.enabled, self.level, self.interval = enabled, level, float(interval)
        self.total_units, self.done_units = total_units, cached_units
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
        if event == "unit_complete":
            self.done_units += 1
            self.active.pop(unit, None)
        if not self.enabled:
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
        elif event == "unit_complete":
            print(f"[units] {self.done_units}/{self.total_units} complete; {label}{elapsed}", flush=True)
        elif event == "calibration_failed":
            print(f"[failed calibration] {label}: {row.get('error')}", flush=True)
        elif event == "candidate_complete" and self.level == "trial":
            print(f"[trial] {label} {row.get('index')}/{row.get('total')}; "
                  f"{row.get('cache_status')}{elapsed}", flush=True)

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
            now = time.monotonic()
            if self.enabled and now - self.last_heartbeat >= self.interval:
                with self.lock:
                    active = list(self.active.values())
                    print(f"[running {now-self.started:.0f}s] units {self.done_units}/{self.total_units}; "
                          f"active models/scenarios: {len(active)}", flush=True)
                    for label in active:
                        print(f"  {label}", flush=True)
                self.last_heartbeat = now

    def __enter__(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.drain()
