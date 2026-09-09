"""Joint selection diagnostics retain supervision and missing-evidence boundaries."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gene2wire.experiments.reporting import (
    compact_summaries, diagnostic_summaries, joint_selection_diagnostics,
)


def _unit(*, model="Reference+PU-Joint", rate=.8, fold=0, repetition=0,
          supervision="reference_plus_pu"):
    return dict(dataset="simulation", sharing_strength=.5, analysis="primary",
                mechanism="technical_sar", loss_rate=rate, calibration_fraction=.2,
                calibration_spec="correct", outer_fold=fold, repetition=repetition,
                model=model, supervision=supervision)


def test_joint_diagnostics_compare_only_own_converged_candidates_and_do_not_mutate():
    unit = _unit()
    tuning = pd.DataFrame([
        dict(**unit, kind="direct", validation_loss=.30, converged=True),
        dict(**unit, kind="lowrank", validation_loss=.25, converged="True"),
        dict(**unit, kind="joint", validation_loss=.01, converged=False),
        dict(**unit, kind="joint", validation_loss=.02, converged=None),
        dict(**unit, kind="joint", validation_loss=np.inf, converged=True),
        dict(**unit, kind="joint", validation_loss=.28, converged=True),
        dict(**{**unit, "model": "Reference+PU"}, kind="direct", validation_loss=.001,
             converged=True),
    ])
    selected = pd.DataFrame([dict(**unit, selected_structure="lowrank",
                                 validation_observed_log_loss=.25)])
    frames = {"tuning": tuning, "selected": selected}
    before = {name: frame.copy(deep=True) for name, frame in frames.items()}
    result = joint_selection_diagnostics(frames)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.recorded_candidates == 6 and row.eligible_candidates == 3
    assert row.convergence_unknown_candidates == 1
    assert row.direct_minimum_validation_loss == .30
    assert row.lowrank_minimum_validation_loss == .25
    assert row.joint_minimum_validation_loss == .28
    assert row.best_recorded_validation_loss == .25
    assert row.selected_family == "lowrank"
    assert row.direct_minus_selected_validation_loss == pytest.approx(.05)
    assert row.selected_minus_best_recorded_validation_loss == 0
    assert row.unit_scope == "fold_repetition"
    assert row.candidate_evidence_status == row.selection_evidence_status == "recorded"
    for name in frames:
        pd.testing.assert_frame_equal(frames[name], before[name])
    assert "joint_selection_diagnostics" in diagnostic_summaries(frames)
    assert "joint_selection_diagnostics" not in compact_summaries(frames)


def test_joint_diagnostics_separate_rates_folds_repetitions_and_supervision():
    units = [_unit(), _unit(rate=.6), _unit(fold=1), _unit(repetition=1),
             _unit(supervision="another_explicit_view"),
             _unit(model="PU-Joint", supervision="calibrated_pu"),
             _unit(model="Joint", supervision="observed")]
    tuning, selected = [], []
    for index, unit in enumerate(units):
        tuning.append(dict(**unit, kind="direct", validation_loss=.2 + index / 100,
                           converged=True))
        selected.append(dict(**unit, kind="direct",
                             validation_observed_log_loss=.2 + index / 100))
    result = joint_selection_diagnostics({"tuning": pd.DataFrame(tuning),
                                          "selected": pd.DataFrame(selected)})
    assert len(result) == len(units)
    assert (result.recorded_candidates == 1).all()
    assert (result.direct_minus_selected_validation_loss == 0).all()
    assert result.lowrank_minimum_validation_loss.isna().all()
    assert result.joint_minimum_validation_loss.isna().all()


def test_joint_diagnostics_unknown_legacy_metadata_is_not_eligible_or_selected():
    tuning = pd.DataFrame([dict(model="PU-Joint", validation_loss=.2)])
    result = joint_selection_diagnostics({"tuning": tuning})
    row = result.iloc[0]
    assert row.unit_scope == "recorded_context_only"
    assert row.eligible_candidates == 0
    assert row.convergence_unknown_candidates == row.unknown_family_candidates == 1
    assert np.isnan(row.best_recorded_validation_loss)
    assert np.isnan(row.direct_minus_selected_validation_loss)
    assert "convergence not recorded" in row.candidate_evidence_status
    assert "structure not recorded" in row.candidate_evidence_status
    assert row.selection_evidence_status == "selection not recorded"
    assert joint_selection_diagnostics({}).empty
    assert joint_selection_diagnostics({"tuning": pd.DataFrame({"kind": ["joint"]})}).empty


def test_joint_diagnostics_all_nonconverged_and_ambiguous_selections_are_explicit():
    unit = _unit()
    tuning = pd.DataFrame([dict(**unit, kind="direct", validation_loss=.2, converged=False)])
    choice = dict(**unit, kind="direct", validation_observed_log_loss=.2)
    result = joint_selection_diagnostics({"tuning": tuning,
                                          "selected": pd.DataFrame([choice, choice])})
    row = result.iloc[0]
    assert row.candidate_evidence_status == "no converged finite candidates"
    assert row.selection_evidence_status == "multiple selection rows for recorded unit"
    assert np.isnan(row.selected_validation_loss)
    assert np.isnan(row.direct_minimum_validation_loss)
    result = joint_selection_diagnostics({"tuning": tuning,
        "selected": pd.DataFrame([choice]).drop(columns="outer_fold")})
    assert result.iloc[0].selection_evidence_status == "selection unit metadata missing"


def test_joint_diagnostics_natural_nan_context_and_missing_selection_fields():
    unit = {**_unit(), "loss_rate": None, "sharing_strength": np.nan, "mechanism": "natural"}
    tuning = pd.DataFrame([dict(**unit, kind="direct", validation_loss=.2, converged=True)])
    selected = pd.DataFrame([dict(**unit, kind=pd.NA)])
    row = joint_selection_diagnostics({"tuning": tuning, "selected": selected}).iloc[0]
    assert row.selection_evidence_status == "selection fields missing"
    assert row.selected_family is None and np.isnan(row.selected_validation_loss)
    assert row.direct_minimum_validation_loss == .2
