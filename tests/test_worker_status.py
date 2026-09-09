"""Worker occupancy must follow task lifetimes, not individual model messages."""
import json
from unittest.mock import patch

import pytest

from gene2wire.experiments.pipeline import _run_scenario_group
from gene2wire.experiments.progress import ProgressRelay
from gene2wire.experiments.workers import NotebookWorkerStatus, worker_capacity, _format_worker_status


def test_cpu_allowance_is_diagnostic_and_not_cluster_availability(monkeypatch):
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
    assert "Workers using 10/15; available 5/15" in rendered
    assert "Requested: 32" in rendered and "allowance: 8" in rendered
    assert "scheduling is unchanged" in rendered
    assert "not free CPUs on the cluster" in rendered


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
