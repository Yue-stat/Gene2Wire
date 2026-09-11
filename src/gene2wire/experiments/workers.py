"""Notebook CPU allocation and worker activity, without claiming globally idle CPUs."""
from __future__ import annotations

from html import escape
import os
from pathlib import Path
import re
import warnings

from joblib import cpu_count


def _positive_integer(value):
    """Reject unset, malformed and nonpositive scheduler values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _cgroup_cpu_quota():
    """Best-effort Linux CPU bandwidth quota, in possibly fractional CPUs.

    Inspect the process cgroup and its ancestors, not just the cgroup mount root.
    The conventional mount paths also handle containers with a cgroup namespace.
    Joblib supplies a separate fallback if an unusual mount is not accessible.
    """
    roots = {Path("/sys/fs/cgroup"): {Path("/sys/fs/cgroup")},
             Path("/sys/fs/cgroup/cpu"): {Path("/sys/fs/cgroup/cpu")},
             Path("/sys/fs/cgroup/cpu,cpuacct"): {Path("/sys/fs/cgroup/cpu,cpuacct")}}
    membership = _read_text("/proc/self/cgroup") or ""
    for line in membership.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        # Ignore unusual parent-relative namespace paths. Root and joblib still
        # give safe fallback observations without walking outside the mount.
        relative = Path(relative.lstrip("/"))
        if ".." in relative.parts:
            continue
        candidates = ([Path("/sys/fs/cgroup")] if not controllers else
                      [Path("/sys/fs/cgroup/cpu"), Path("/sys/fs/cgroup/cpu,cpuacct")]
                      if "cpu" in controllers.split(",") else [])
        for root in candidates:
            current = root / relative
            while current != root:
                roots[root].add(current)
                current = current.parent
    quotas = []
    for paths in roots.values():
        for path in paths:
            value = _read_text(path / "cpu.max")
            if value:
                fields = value.split()
                if len(fields) == 2:
                    quota, period = map(_positive_integer, fields)
                    if quota and period:
                        quotas.append(quota / period)
            quota = _positive_integer(_read_text(path / "cpu.cfs_quota_us"))
            period = _positive_integer(_read_text(path / "cpu.cfs_period_us"))
            if quota and period:
                quotas.append(quota / period)
    return min(quotas) if quotas else None


def _scheduler_allocations():
    """Preserve allocation scope; a job allocation is not a per-task allocation."""
    allocations = []
    for variable, scope in (
        ("SLURM_CPUS_PER_TASK", "task"),
        ("SLURM_CPUS_ON_NODE", "node"),
        ("PBS_NUM_PPN", "node"),
        ("PBS_NP", "job"),
        ("NSLOTS", "job"),
    ):
        value = _positive_integer(os.environ.get(variable))
        if value:
            allocations.append({"variable": variable, "scope": scope, "cpus": value})
    # This can describe several nodes, for example 32(x2),16. A nonuniform
    # allocation does not identify which node hosts the notebook kernel.
    node_spec = os.environ.get("SLURM_JOB_CPUS_PER_NODE", "")
    parts = node_spec.split(",") if node_spec else []
    counts = []
    for part in parts:
        match = re.fullmatch(r"\s*([1-9]\d*)(?:\(x[1-9]\d*\))?\s*", part)
        if match is None:
            counts = []
            break
        counts.append(int(match.group(1)))
    if counts and len(set(counts)) == 1 and not any(
            row["variable"] == "SLURM_CPUS_ON_NODE" for row in allocations):
        allocations.append({"variable": "SLURM_JOB_CPUS_PER_NODE",
                            "scope": "node (uniform allocation)", "cpus": counts[0]})
    return allocations


def worker_capacity(requested_workers):
    """Report machine, kernel and scheduler capacities; never alter scheduling.

    CPU allowance is an allocation/affinity bound, not the number of globally
    idle CPUs. CPU bandwidth quotas can be fractional, unlike worker processes.
    ``process_cpu_allowance`` is joblib's integer detection retained for callers.
    """
    requested_workers = int(requested_workers)
    if requested_workers < 0:
        raise ValueError("requested_workers cannot be negative")
    machine_limit = _positive_integer(os.cpu_count())
    try:
        affinity_limit = _positive_integer(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError, NotImplementedError):
        affinity_limit = None
    try:
        process_limit = _positive_integer(cpu_count())
    except (OSError, ValueError, NotImplementedError):
        process_limit = None
    quota = _cgroup_cpu_quota()
    allocations = _scheduler_allocations()
    limits = [value for value in (machine_limit, affinity_limit, process_limit, quota)
              if value is not None]
    limits.extend(row["cpus"] for row in allocations)
    detected_limit = min(limits) if limits else None
    slurm_limit = _positive_integer(os.environ.get("SLURM_CPUS_PER_TASK"))
    return {"requested_workers": requested_workers,
            "machine_logical_cpus": machine_limit,
            "affinity_logical_cpus": affinity_limit,
            "process_cpu_allowance": process_limit,
            "cgroup_cpu_quota": quota,
            "scheduler_allocations": allocations,
            "slurm_cpus_per_task": slurm_limit,
            "detected_cpu_allowance": detected_limit,
            "globally_idle_cpus": None,
            "requested_exceeds_cpu_allowance": (
                requested_workers > detected_limit if detected_limit is not None else False)}


def _display_capacity(value):
    return "unknown" if value is None else f"{value:g}"


def _format_worker_status(snapshot):
    phase = snapshot.get("phase", "preparing")
    requested = snapshot["requested_workers"]
    allowance = snapshot.get("detected_cpu_allowance")
    slots = snapshot.get("worker_slots")
    if slots is None:
        occupancy = "Experiment workers running: waiting for task scheduling."
    else:
        used = snapshot["occupied_workers"]
        occupancy = f"Experiment workers running: {used} (pool size: {slots})."
        if slots == 0:
            occupancy += " No pending worker tasks."
    capacity = (f"Kernel CPU allowance: {_display_capacity(allowance)} CPU equivalents; "
                f"configured N_JOBS: {requested}.")
    sources = (f"Machine logical CPUs: {_display_capacity(snapshot.get('machine_logical_cpus'))}; "
               f"kernel CPU affinity: {_display_capacity(snapshot.get('affinity_logical_cpus'))}; "
               f"cgroup quota: {_display_capacity(snapshot.get('cgroup_cpu_quota'))}; "
               f"joblib process limit: {_display_capacity(snapshot.get('process_cpu_allowance'))}.")
    allocations = snapshot.get("scheduler_allocations", [])
    if allocations:
        sources += " Scheduler allocation: " + "; ".join(
            f"{row['cpus']} CPUs / {row['scope']} ({row['variable']})" for row in allocations) + "."
    else:
        sources += " Scheduler allocation: unknown."
    if snapshot.get("requested_exceeds_cpu_allowance"):
        capacity += " Requested workers exceed this allowance; scheduling is unchanged."
    timestamp = snapshot.get("local_time")
    state = phase.capitalize() + (f" · updated {timestamp}" if timestamp else "")
    explanation = ("Globally idle CPUs on this shared node: unknown. Allocation is capacity, "
                   "not free CPUs. Running counts only this experiment's workers assigned to tasks, "
                   "including fitting, cache checks and I/O. Updates follow the progress interval.")
    return (f"<div><strong>{escape(capacity)}</strong><br>{escape(sources)}<br>{escape(occupancy)}"
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
