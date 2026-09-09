"""Notebook worker occupancy display; diagnostic state never controls scheduling."""
from __future__ import annotations

from html import escape
import os
import warnings

from joblib import cpu_count


def worker_capacity(requested_workers):
    """Detect the process CPU allowance, not free CPUs on the shared cluster.

    Joblib respects affinity and container CPU quotas. A valid SLURM per-task
    allocation provides an additional upper bound. These are diagnostic limits;
    this function does not reduce the requested process budget.
    """
    requested_workers = int(requested_workers)
    if requested_workers < 0:
        raise ValueError("requested_workers cannot be negative")
    process_limit = max(1, int(cpu_count()))
    try:
        slurm_limit = int(os.environ.get("SLURM_CPUS_PER_TASK", ""))
    except ValueError:
        slurm_limit = 0
    slurm_limit = slurm_limit if slurm_limit > 0 else None
    detected_limit = min(process_limit, slurm_limit) if slurm_limit else process_limit
    return {"requested_workers": requested_workers,
            "process_cpu_allowance": process_limit,
            "slurm_cpus_per_task": slurm_limit,
            "detected_cpu_allowance": detected_limit,
            "requested_exceeds_cpu_allowance": requested_workers > detected_limit}


def _format_worker_status(snapshot):
    phase = snapshot.get("phase", "preparing")
    requested = snapshot["requested_workers"]
    allowance = snapshot["detected_cpu_allowance"]
    slots = snapshot.get("worker_slots")
    if slots is None:
        occupancy = "Worker pool: waiting for task scheduling."
    else:
        used = snapshot["occupied_workers"]
        available = snapshot["available_workers"]
        occupancy = (f"Workers using {used}/{slots}; available {available}/{slots} "
                     "scheduled slots.")
        if slots == 0:
            occupancy += " No pending worker tasks."
    capacity = f"Requested: {requested}. Detectable CPU allowance: {allowance}."
    if snapshot.get("slurm_cpus_per_task") is not None:
        capacity += f" SLURM CPUs per task: {snapshot['slurm_cpus_per_task']}."
    if snapshot["requested_exceeds_cpu_allowance"]:
        capacity += " Requested workers exceed this allowance; scheduling is unchanged."
    timestamp = snapshot.get("local_time")
    state = phase.capitalize() + (f" · updated {timestamp}" if timestamp else "")
    explanation = ("Using means a worker is assigned a task, including fitting, cache checks and I/O. "
                   "Available means unused slots in this experiment's scheduled pool, "
                   "not free CPUs on the cluster. Updates follow the progress interval.")
    return (f"<div><strong>{escape(occupancy)}</strong><br>{escape(capacity)}"
            f"<br>{escape(state)}<br><small>{escape(explanation)}</small></div>")


class NotebookWorkerStatus:
    """Create one output cell and update it through the pipeline callback.

    Instantiate in a separate cell before running the experiment, then pass the
    instance as ``worker_status=worker_status``. IPython is optional for the core;
    outside a notebook the latest snapshot remains available in ``snapshot``.
    Display errors cannot interrupt a scientific fit.
    """

    def __init__(self, requested_workers):
        self.snapshot = {**worker_capacity(requested_workers), "phase": "preparing"}
        self._handle = self._html = None
        self._display_failed = False
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
            if get_ipython() is not None:
                self._html = HTML
                self._handle = display(HTML(_format_worker_status(self.snapshot)), display_id=True)
        except ImportError:
            pass
        except Exception as exc:
            self._disable_display(exc)

    def _disable_display(self, exc):
        self._handle = None
        if not self._display_failed:
            warnings.warn(f"Worker status display unavailable: {exc}", RuntimeWarning, stacklevel=2)
            self._display_failed = True

    def __call__(self, snapshot):
        self.snapshot = dict(snapshot)
        if self._handle is not None:
            try:
                self._handle.update(self._html(_format_worker_status(self.snapshot)))
            except Exception as exc:
                self._disable_display(exc)
