from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gene2wire.experiments.gene_overlap import rebuild_overlap_summaries
from gene2wire.experiments.gene_overlap_plotting import plot_overlap_results
from gene2wire.experiments.pipeline import Artifacts


def _per_repetition_two_rhos():
    rows = []
    for rho in (0.0, 0.5):
        for overlap in (0.0, 0.5, 1.0):
            for model, offset in (("PU", 0.0), ("PU-MIRT", 0.01), ("PU-Joint", 0.02)):
                rows.append({
                    "dataset": "simulation", "sharing_strength": rho,
                    "analysis": "primary", "mechanism": "technical_sar",
                    "loss_rate": 0.0, "calibration_fraction": 0.2,
                    "calibration_spec": "correct", "panel_design": "crossed",
                    "panel_size": 8, "probability_semantics": "reference",
                    "arm": "union", "model": model,
                    "actual_overlap": overlap, "repetition": 0,
                    "macro_auprc": 0.30 + offset + overlap * 0.01,
                    "macro_log_loss": 0.40 - offset - overlap * 0.01,
                    "macro_brier": 0.20 - offset - overlap * 0.005,
                })
    return pd.DataFrame(rows)


def test_multi_rho_overlap_contrast_is_scalar_and_persisted(tmp_path):
    artifacts = Artifacts(
        tables={"per_repetition": _per_repetition_two_rhos()},
        export_dir=Path(tmp_path),
        manifest={"table_files": ["per_repetition.csv"]},
    )
    rebuild_overlap_summaries(artifacts)
    contrasts = artifacts.tables["overlap_contrasts"]
    assert len(contrasts) == 2 * 3 * 2 * 3 * 1  # rho × overlap × candidate × metric × repetition
    assert set(contrasts["sharing_strength"]) == {0.0, 0.5}
    assert pd.api.types.is_numeric_dtype(contrasts["difference_vs_100pct_overlap"])
    assert contrasts["difference_vs_100pct_overlap"].map(np.isscalar).all()
    assert "overlap_contrasts.csv" in artifacts.manifest["table_files"]
    assert (Path(tmp_path) / "overlap_contrasts.csv").is_file()


def test_overlap_figures_are_separated_and_titled_by_rho(tmp_path, monkeypatch):
    aggregate_rows = []
    contrast_rows = []
    for rho in (0.0, 0.5):
        for overlap in (0.0, 0.5, 1.0):
            for model, offset in (("PU", 0.0), ("PU-MIRT", 0.01), ("PU-Joint", 0.02)):
                aggregate_rows.append({
                    "dataset": "simulation", "sharing_strength": rho,
                    "panel_design": "crossed", "arm": "union", "model": model,
                    "actual_overlap": overlap, "macro_auprc": 0.3 + offset,
                    "macro_log_loss": 0.4 - offset, "macro_brier": 0.2 - offset,
                })
                for metric in ("macro_auprc", "macro_log_loss", "macro_brier"):
                    contrast_rows.append({
                        "dataset": "simulation", "sharing_strength": rho,
                        "panel_design": "crossed", "model": model,
                        "metric": metric, "actual_overlap": overlap,
                        "difference_vs_100pct_overlap": offset,
                    })
            for arm, offset in (("intersection", -0.02), ("separate_panel", -0.01),
                                ("union", 0.0), ("all_gene_oracle", 0.03)):
                aggregate_rows.append({
                    "dataset": "simulation", "sharing_strength": rho,
                    "panel_design": "crossed", "arm": arm, "model": "PU",
                    "actual_overlap": overlap, "macro_auprc": 0.3 + offset,
                    "macro_log_loss": 0.4 - offset, "macro_brier": 0.2 - offset,
                })
    artifacts = SimpleNamespace(tables={
        "aggregate": pd.DataFrame(aggregate_rows),
        "overlap_contrasts": pd.DataFrame(contrast_rows),
    }, export_dir=Path(tmp_path), manifest={})
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    paths = plot_overlap_results(artifacts, tmp_path, prefix="simulation", display=True)
    assert len(paths) == 6  # primary, controls and R_M for each rho
    assert len(shown) == 6
    assert all(fig._suptitle is not None for fig in shown)
    assert {"0", "0.5"} <= {fig._suptitle.get_text().split("ρ=")[-1].split(" ")[0]
                              for fig in shown}
    assert all(axis.get_title() for fig in shown for axis in fig.axes)
    assert all(path.suffix == ".pdf" for path in paths)
    for fig in shown:
        plt.close(fig)
