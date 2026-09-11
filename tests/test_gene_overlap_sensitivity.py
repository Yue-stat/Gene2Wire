from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from gene2wire.experiments.datasets.simulation import generate_simulation
from gene2wire.experiments.gene_overlap import build_overlap_views
from gene2wire.experiments.gene_overlap_plotting import plot_overlap_results
from gene2wire.experiments.gene_overlap_sensitivity import (
    TECH_SAR_80_MODELS,
    prepare_tech_sar80_sensitivity,
)
from gene2wire.experiments.protocol import Settings


def _primary_views():
    dataset = generate_simulation(
        0, .5, seed=7, n_cells=60, n_targets=4, n_gene_features=6,
        n_location_features=0, n_target_features=2, n_slices=6,
    )
    return build_overlap_views(
        dataset, [0, 1], panel_size=3, n_repetitions=1,
        panel_design="crossed", strata=("slice",), include_controls=False, seed=7,
    )


def _primary_settings():
    return Settings(
        n_outer_folds=2, n_repetitions=1, n_jobs=1, loss_rates=(0.,),
        run_information_controls=False, run_random_forest=False,
        run_mechanism_controls=False, run_calibration_controls=False,
        run_qiao=False,
    )


def test_tech_sar80_plan_is_disabled_by_default_and_isolated_when_enabled():
    views = _primary_views()
    settings = _primary_settings()
    assert prepare_tech_sar80_sensitivity(views, settings) is None

    plan = prepare_tech_sar80_sensitivity(views, settings, enabled=True)
    assert settings.loss_rates == (0.,)
    assert plan.settings.loss_rates == (.8,)
    assert plan.export_name != "simulation_gene_overlap_0910"
    assert len(plan.views) == 2  # two overlap values, union only
    assert plan.expected_scenario_units == 4  # two views times two folds
    assert plan.expected_model_evaluations == 12
    for view in plan.views:
        assert view.metadata["experiment_context"]["arm"] == "union"
        assert view.metadata["experiment_context"]["observation_profile"] == (
            "technical_sar_80_sensitivity"
        )
        assert tuple(view.metadata["model_allowlist"]) == TECH_SAR_80_MODELS
        assert view.metadata["sensitivity_analysis"]["role"] == "secondary"
    assert all(
        "observation_profile" not in view.metadata["experiment_context"]
        for view in views
    )


def test_tech_sar80_plan_rejects_natural_data_and_mixed_primary_loss():
    views = _primary_views()
    natural = replace(views[0], natural_observed=views[0].reference.copy())
    with pytest.raises(ValueError, match="Natural|eligible"):
        prepare_tech_sar80_sensitivity((natural,), _primary_settings(), enabled=True)
    mixed = replace(_primary_settings(), loss_rates=(0., .8))
    with pytest.raises(ValueError, match="only loss_rate=0"):
        prepare_tech_sar80_sensitivity(views, mixed, enabled=True)
    with pytest.raises(ValueError, match="different export"):
        prepare_tech_sar80_sensitivity(
            views, _primary_settings(), enabled=True,
            export_name="same", primary_export_name="same",
        )


def _scenario_plot_tables():
    aggregate, contrasts = [], []
    scenarios = (("primary", 0., True), ("primary", .8, False))
    for analysis, loss_rate, include_controls in scenarios:
        common = {
            "dataset": "simulation", "sharing_strength": .5,
            "panel_design": "crossed", "analysis": analysis,
            "mechanism": "technical_sar", "loss_rate": loss_rate,
            "calibration_fraction": .2, "calibration_spec": "correct",
        }
        for overlap in (0., .5, 1.):
            for model, offset in (("PU", 0.), ("PU-MIRT", .01), ("PU-Joint", .02)):
                aggregate.append({
                    **common, "arm": "union", "model": model,
                    "actual_overlap": overlap, "macro_auprc": .3 + offset,
                    "macro_log_loss": .4 - offset, "macro_brier": .2 - offset,
                })
                if model != "PU":
                    for metric in ("macro_auprc", "macro_log_loss", "macro_brier"):
                        contrasts.append({
                            **common, "model": model, "metric": metric,
                            "actual_overlap": overlap,
                            "difference_vs_100pct_overlap": offset * (1 - overlap),
                        })
            if include_controls:
                for arm, offset in (("intersection", -.02), ("disjoint_coefficient", -.01),
                                    ("all_gene_oracle", .03)):
                    aggregate.append({
                        **common, "arm": arm, "model": "PU",
                        "actual_overlap": overlap, "macro_auprc": .3 + offset,
                        "macro_log_loss": .4 - offset, "macro_brier": .2 - offset,
                    })
    return pd.DataFrame(aggregate), pd.DataFrame(contrasts)


def test_plotting_separates_primary_and_tech_sar80_titles_and_files(tmp_path, monkeypatch):
    aggregate, contrasts = _scenario_plot_tables()
    artifacts = SimpleNamespace(
        tables={"aggregate": aggregate, "overlap_contrasts": contrasts},
        export_dir=Path(tmp_path), manifest={},
    )
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    paths = plot_overlap_results(artifacts, tmp_path, prefix="simulation", display=True)

    assert len(paths) == 5  # primary: methods/controls/R_M; sensitivity: methods/R_M
    assert len(paths) == len(set(paths))
    titles = [figure._suptitle.get_text() for figure in shown]
    assert sum("Primary: no added positive-label loss" in title for title in titles) == 3
    assert sum("Sensitivity: Technical-SAR; 80% added positive-label loss" in title
               for title in titles) == 2
    assert sum("sensitivity_technical_sar_loss80" in path.name for path in paths) == 2
    assert all(axis.get_title() for figure in shown for axis in figure.axes)
    for figure in shown:
        plt.close(figure)
