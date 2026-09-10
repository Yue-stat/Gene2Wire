"""Worker occupancy must follow task lifetimes, not individual model messages."""
import json
from unittest.mock import patch

import pytest

from gene2wire.experiments.pipeline import _run_scenario_group
from gene2wire.experiments.progress import ProgressRelay
from gene2wire.experiments.workers import NotebookWorkerStatus, worker_capacity, _format_worker_status
from gene2wire.experiments.workers import _cgroup_cpu_quota, _scheduler_allocations


@pytest.fixture
def controlled_capacity(monkeypatch):
    """Tests must not inherit CPU quotas or scheduler variables from CI."""
    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NUM_PPN",
                     "PBS_NP", "NSLOTS", "SLURM_JOB_CPUS_PER_NODE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("gene2wire.experiments.workers.os.cpu_count", lambda: 64)
    monkeypatch.setattr("gene2wire.experiments.workers.os.sched_getaffinity",
                        lambda _: set(range(16)), raising=False)
    monkeypatch.setattr("gene2wire.experiments.workers._cgroup_cpu_quota", lambda: None)
    monkeypatch.setattr("gene2wire.experiments.workers.cpu_count", lambda: 16)


def test_cpu_allowance_is_diagnostic_and_not_cluster_availability(monkeypatch, controlled_capacity):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    with patch("gene2wire.experiments.workers.cpu_count", return_value=12):
        capacity = worker_capacity(32)
        assert capacity["requested_workers"] == 32
        assert capacity["process_cpu_allowance"] == 12
        assert capacity["detected_cpu_allowance"] == 8
        assert capacity["requested_exceeds_cpu_allowance"]
        # Affinity / cgroup limits can be stricter than the SLURM allocation.
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        assert worker_capacity(32)["detected_cpu_allowance"] == 12
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "invalid")
        assert worker_capacity(32)["slurm_cpus_per_task"] is None
    rendered = _format_worker_status({**capacity, "worker_slots": 15,
        "occupied_workers": 10, "available_workers": 5, "phase": "running"})
    assert "Experiment workers running: 10 (pool size: 15)" in rendered
    assert "configured N_JOBS: 32" in rendered and "allowance: 8" in rendered
    assert "Machine logical CPUs: 64" in rendered and "kernel CPU affinity: 16" in rendered
    assert "8 CPUs / task (SLURM_CPUS_PER_TASK)" in rendered
    assert "scheduling is unchanged" in rendered
    assert "Globally idle CPUs on this shared node: unknown" in rendered
    assert "available 5/15" not in rendered and "scheduled slots" not in rendered


def test_fractional_quota_is_not_reported_as_idle_worker_slots(monkeypatch, controlled_capacity):
    monkeypatch.setattr("gene2wire.experiments.workers._cgroup_cpu_quota", lambda: 1.5)
    capacity = worker_capacity(2)
    assert capacity["cgroup_cpu_quota"] == capacity["detected_cpu_allowance"] == 1.5
    assert capacity["globally_idle_cpus"] is None
    assert capacity["requested_exceeds_cpu_allowance"]
    rendered = _format_worker_status(capacity)
    assert "allowance: 1.5 CPU equivalents" in rendered
    assert "Scheduler allocation: unknown" in rendered


def test_scheduler_scopes_and_heterogeneous_node_allocation(monkeypatch, controlled_capacity):
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32(x2),16")
    assert _scheduler_allocations() == []  # Cannot know the notebook's node.
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32(x2),32")
    assert _scheduler_allocations()[0]["cpus"] == 32
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "12")
    assert _scheduler_allocations() == [
        {"variable": "SLURM_CPUS_ON_NODE", "scope": "node", "cpus": 12}]
    assert worker_capacity(8)["detected_cpu_allowance"] == 12
    monkeypatch.setenv("NSLOTS", "6")
    assert worker_capacity(8)["detected_cpu_allowance"] == 6
    assert {"variable": "NSLOTS", "scope": "job", "cpus": 6} in _scheduler_allocations()
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "0")
    monkeypatch.setenv("PBS_NP", "-1")
    monkeypatch.setenv("PBS_NUM_PPN", "garbage")
    assert len(_scheduler_allocations()) == 2


def test_unknown_capacity_is_explicit_and_does_not_stop_notebook(monkeypatch, controlled_capacity):
    monkeypatch.setattr("gene2wire.experiments.workers.os.cpu_count", lambda: None)
    def unavailable(*args):
        raise OSError("unavailable")
    monkeypatch.setattr("gene2wire.experiments.workers.os.sched_getaffinity", unavailable)
    monkeypatch.setattr("gene2wire.experiments.workers.cpu_count", unavailable)
    capacity = worker_capacity(32)
    assert capacity["detected_cpu_allowance"] is None
    assert not capacity["requested_exceeds_cpu_allowance"]
    assert "Kernel CPU allowance: unknown" in _format_worker_status(capacity)


def test_process_cgroup_quota_uses_strictest_ancestor(monkeypatch):
    files = {
        "/proc/self/cgroup": "0::/ondemand/session\n",
        "/sys/fs/cgroup/cpu.max": "max 100000",
        "/sys/fs/cgroup/ondemand/cpu.max": "150000 100000",
        "/sys/fs/cgroup/ondemand/session/cpu.max": "200000 100000",
    }
    monkeypatch.setattr("gene2wire.experiments.workers._read_text", lambda path: files.get(str(path)))
    assert _cgroup_cpu_quota() == 1.5


def test_v1_cgroup_quota_and_unlimited_fallback(monkeypatch):
    files = {
        "/proc/self/cgroup": "3:cpu,cpuacct:/job\n",
        "/sys/fs/cgroup/cpu,cpuacct/job/cpu.cfs_quota_us": "400000",
        "/sys/fs/cgroup/cpu,cpuacct/job/cpu.cfs_period_us": "100000",
    }
    monkeypatch.setattr("gene2wire.experiments.workers._read_text", lambda path: files.get(str(path)))
    assert _cgroup_cpu_quota() == 4
    files["/sys/fs/cgroup/cpu,cpuacct/job/cpu.cfs_quota_us"] = "-1"
    assert _cgroup_cpu_quota() is None
    files.clear()
    files["/proc/self/cgroup"] = "0::/../../outside\n"
    assert _cgroup_cpu_quota() is None


def test_latest_worker_event_wins_across_files_and_model_boundaries(tmp_path):
    snapshots = []
    relay = ProgressRelay(tmp_path, enabled=False, interval=60,
        worker_status=snapshots.append, worker_slots=2, requested_workers=32)
    relay.started = relay.last_heartbeat = 0.
    # File a is newer, but file z will be drained last. The old finish in z must
    # not mark this PID idle after it has already started a subsequent group.
    files = {"a": [{"event": "task_group_start", "pid": 7, "timestamp": 30.}],
             "z": [{"event": "task_group_start", "pid": 7, "timestamp": 10.},
                   {"event": "task_group_complete", "pid": 7, "timestamp": 20.}],
             "b": [{"event": "task_group_start", "pid": 8, "timestamp": 31.},
                   {"event": "model_complete", "pid": 8, "timestamp": 32.,
                    "work_id": "w", "model": "PU", "cache_status": "checkpoint"}]}
    for name, rows in files.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    relay.drain()
    relay.heartbeat(59.)
    assert snapshots == []
    relay.heartbeat(60.)
    assert snapshots[-1]["occupied_workers"] == 2
    assert snapshots[-1]["available_workers"] == 0
    assert snapshots[-1]["requested_workers"] == 32
    assert snapshots[-1]["worker_slots"] == 2
    relay.heartbeat(61.)
    assert len(snapshots) == 1
    (tmp_path / "c.jsonl").write_text(json.dumps(
        {"event": "task_group_complete", "pid": 8, "timestamp": 40.}) + "\n")
    relay.drain()
    relay.heartbeat(120.)
    assert snapshots[-1]["occupied_workers"] == 1
    assert snapshots[-1]["available_workers"] == 1
    relay._worker_status("finished")
    assert snapshots[-1]["occupied_workers"] == 0
    assert snapshots[-1]["available_workers"] == 2


def test_cached_run_never_invents_a_worker_pool(tmp_path):
    snapshots = []
    with ProgressRelay(tmp_path, enabled=False, worker_status=snapshots.append,
                       total_units=100, cached_units=100, requested_workers=32):
        pass
    assert [row["phase"] for row in snapshots] == ["running", "finished"]
    assert all(row["worker_slots"] == row["occupied_workers"] == row["available_workers"] == 0
               for row in snapshots)
    assert "No pending worker tasks" in _format_worker_status(snapshots[-1])


def test_group_finish_is_emitted_even_when_a_fit_raises():
    events = []
    tasks = [(0, None, 0, "unit", events.append, {"loss_rate": .8})]
    with patch("gene2wire.experiments.pipeline._run_checkpointed_fold", side_effect=ValueError("fit failed")):
        with pytest.raises(ValueError, match="fit failed"):
            _run_scenario_group(tasks, None, None, None, None, None)
    assert [row["event"] for row in events] == ["task_group_start", "task_group_complete"]


def test_optional_callback_failure_does_not_stop_training_progress(tmp_path):
    def failing_callback(snapshot):
        raise RuntimeError("closed notebook")
    with pytest.warns(RuntimeWarning, match="Worker status callback disabled"):
        with ProgressRelay(tmp_path, enabled=False, worker_status=failing_callback,
                           total_units=1, worker_slots=1, requested_workers=1) as relay:
            relay.writer("task")({"event": "model_complete", "model": "PU", "cache_status": "checkpoint"})
    assert relay.done_units == 1
    assert relay.worker_status_failed


def test_notebook_helper_can_be_called_without_ipython_display():
    with patch.dict("sys.modules", {"IPython": None}):
        status = NotebookWorkerStatus(32)
    snapshot = {**worker_capacity(32), "phase": "finished", "worker_slots": 2,
                "occupied_workers": 0, "available_workers": 2}
    status(snapshot)
    assert status.snapshot == snapshot
