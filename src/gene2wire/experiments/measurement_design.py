"""Outcome-blind measurement-degradation designs.

The utilities in this module define artificial assay panels and positive
censoring without depending on a fitted model.  A requested panel value is a
per-assay *coverage*, not the legacy fixed-budget pairwise-overlap parameter.
Three cyclic panels retain the complete item union at every supported coverage;
at coverage one, every panel contains every item.

Gene panels and target panels deliberately use the same generator.  Their
application differs: a gene-panel draw becomes an input observation mask,
whereas a target-panel draw is intersected with native assay availability and
therefore removes entries from the fitting loss rather than turning them into
negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit


MEASUREMENT_DESIGN_VERSION = "cyclic-union-v1"
CENSORING_VERSION = "assay-target-logistic-fold-local-v1"
DEFAULT_ASSAYS = ("A", "B", "C")


def _readonly(value, dtype=None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _labels(values: Sequence[object], *, name: str, minimum: int = 1) -> tuple[str, ...]:
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or len(array) < minimum or np.any(pd.isna(array)):
        raise ValueError(f"{name} must be a one-dimensional sequence of at least {minimum} labels")
    labels = tuple(map(str, array))
    if any(not value for value in labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{name} must contain unique, nonempty labels")
    return labels


def _seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _rows(rows: Sequence[int] | np.ndarray | None, n_cells: int) -> np.ndarray:
    if rows is None:
        return np.arange(n_cells, dtype=int)
    value = np.asarray(rows)
    if value.dtype == bool:
        if value.shape != (n_cells,):
            raise ValueError("A boolean rows mask must have length n_cells")
        value = np.flatnonzero(value)
    if value.ndim != 1 or (value.size and not np.issubdtype(value.dtype, np.integer)):
        raise ValueError("rows must contain integer row indices")
    value = value.astype(int)
    if (len(np.unique(value)) != len(value)
            or np.any(value < 0) or np.any(value >= n_cells)):
        raise ValueError("rows contains duplicate or out-of-range indices")
    return value


def _binary_matrix(value, *, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or (shape is not None and array.shape != shape):
        expected = "" if shape is None else f" with shape {shape}"
        raise ValueError(f"{name} must be a two-dimensional binary matrix{expected}")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError(f"{name} must be binary")
    return array.astype(bool, copy=True)


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _coverage_grid(values: Sequence[float], n_assays: int) -> tuple[float, ...]:
    coverages = tuple(float(value) for value in values)
    minimum = 1.0 / n_assays
    if (not coverages or len(set(coverages)) != len(coverages)
            or any(not np.isfinite(value) or value > 1.0
                   or value < minimum - 1e-12 for value in coverages)):
        raise ValueError(
            f"coverages must be distinct finite values in [{minimum:g}, 1] "
            "for a union-preserving equal-budget design"
        )
    return coverages


@dataclass(frozen=True)
class CyclicPanelDraw:
    """One equal-budget panel draw, with columns aligned to ``item_ids``."""

    requested_coverage: float
    panel_size: int
    assay_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    incidence: np.ndarray
    pairwise_intersections: np.ndarray
    pairwise_overlap: np.ndarray
    union_count: int
    duplicate_of_coverage: float | None = None

    def __post_init__(self) -> None:
        incidence = _binary_matrix(
            self.incidence, name="incidence",
            shape=(len(self.assay_ids), len(self.item_ids)),
        )
        intersections = np.asarray(self.pairwise_intersections)
        overlap = np.asarray(self.pairwise_overlap, dtype=float)
        square = (len(self.assay_ids), len(self.assay_ids))
        if intersections.shape != square or overlap.shape != square:
            raise ValueError("Pairwise matrices must align with assay_ids")
        if not np.all(np.isfinite(overlap)):
            raise ValueError("Pairwise overlap must be finite")
        if np.any(incidence.sum(axis=1) != int(self.panel_size)):
            raise ValueError("Every assay must contain panel_size items")
        union_count = int(incidence.any(axis=0).sum())
        if union_count != int(self.union_count):
            raise ValueError("union_count is inconsistent with incidence")
        object.__setattr__(self, "incidence", _readonly(incidence, bool))
        object.__setattr__(self, "pairwise_intersections", _readonly(intersections, int))
        object.__setattr__(self, "pairwise_overlap", _readonly(overlap, float))

    @property
    def actual_coverage(self) -> float:
        return self.panel_size / len(self.item_ids)

    @property
    def union_coverage(self) -> float:
        return self.union_count / len(self.item_ids)

    @property
    def mean_pairwise_overlap(self) -> float:
        upper = np.triu_indices(len(self.assay_ids), 1)
        return float(np.mean(self.pairwise_overlap[upper]))

    def panel_items(self, assay: str) -> tuple[str, ...]:
        try:
            row = self.assay_ids.index(str(assay))
        except ValueError as error:
            raise ValueError(f"Unknown assay {assay!r}") from error
        return tuple(item for item, present in zip(self.item_ids, self.incidence[row]) if present)


@dataclass(frozen=True)
class CyclicPanelFamily:
    """A common circular order and fixed assay starts across coverage levels."""

    assay_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    circular_order: tuple[str, ...]
    start_offsets: Mapping[str, int]
    draws: Mapping[float, CyclicPanelDraw]
    seed: int
    version: str = MEASUREMENT_DESIGN_VERSION

    def draw(self, requested_coverage: float) -> CyclicPanelDraw:
        matches = [draw for value, draw in self.draws.items()
                   if np.isclose(value, requested_coverage, rtol=0, atol=1e-12)]
        if len(matches) != 1:
            raise ValueError(f"Unknown requested coverage {requested_coverage!r}")
        return matches[0]


def draw_cyclic_panel_family(
    item_ids: Sequence[object],
    coverages: Sequence[float],
    *,
    seed: int,
    assay_ids: Sequence[object] = DEFAULT_ASSAYS,
) -> CyclicPanelFamily:
    """Create nested, equal-budget cyclic panels with a complete item union.

    The circular item order, global rotation and assignment of equally spaced
    starts to named assays are sampled once.  A panel is the first ``K`` items
    encountered from its fixed start, so lowering ``K`` can only remove items.
    ``K`` is never below ``ceil(D / A)``; this is precisely the bound needed to
    cover every item when the ``A`` starts partition the circle into balanced
    gaps.
    """
    seed = _seed(seed)
    assays = tuple(sorted(_labels(assay_ids, name="assay_ids", minimum=2)))
    items_input = _labels(item_ids, name="item_ids", minimum=len(assays))
    n_assays, n_items = len(assays), len(items_input)
    coverage_values = _coverage_grid(coverages, n_assays)

    order_seed, layout_seed = np.random.SeedSequence(seed).spawn(2)
    canonical_items = np.asarray(sorted(items_input), dtype=object)
    circular = tuple(map(str, np.random.default_rng(order_seed).permutation(canonical_items)))
    layout_rng = np.random.default_rng(layout_seed)
    rotation = int(layout_rng.integers(n_items))
    assay_cycle = tuple(map(str, layout_rng.permutation(np.asarray(assays, dtype=object))))
    base_starts = tuple((index * n_items) // n_assays for index in range(n_assays))
    starts = {
        assay: int((rotation + base_starts[index]) % n_items)
        for index, assay in enumerate(assay_cycle)
    }
    max_gap = max(
        (base_starts[(index + 1) % n_assays] - base_starts[index]) % n_items
        for index in range(n_assays)
    )
    minimum_panel_size = int(np.ceil(n_items / n_assays))
    if max_gap != minimum_panel_size:
        raise RuntimeError("Balanced cyclic starts do not attain the union bound")

    input_lookup = {item: index for index, item in enumerate(items_input)}
    seen_sizes: dict[int, float] = {}
    draws: dict[float, CyclicPanelDraw] = {}
    for requested in sorted(coverage_values, reverse=True):
        panel_size = max(minimum_panel_size, _round_half_up(requested * n_items))
        incidence = np.zeros((n_assays, n_items), dtype=bool)
        for assay_row, assay in enumerate(assays):
            start = starts[assay]
            selected = (circular[(start + step) % n_items] for step in range(panel_size))
            incidence[assay_row, [input_lookup[item] for item in selected]] = True
        union_count = int(incidence.any(axis=0).sum())
        if union_count != n_items:
            raise RuntimeError("Cyclic panel construction failed to preserve the item union")
        intersections = incidence.astype(int) @ incidence.astype(int).T
        overlap = intersections / panel_size
        draw = CyclicPanelDraw(
            requested_coverage=float(requested), panel_size=panel_size,
            assay_ids=assays, item_ids=items_input, incidence=incidence,
            pairwise_intersections=intersections, pairwise_overlap=overlap,
            union_count=union_count, duplicate_of_coverage=seen_sizes.get(panel_size),
        )
        if np.isclose(requested, 1.0, rtol=0, atol=1e-12) and not incidence.all():
            raise RuntimeError("Coverage one must expose every item in every assay")
        seen_sizes.setdefault(panel_size, float(requested))
        draws[float(requested)] = draw

    ordered_draws = sorted(draws.values(), key=lambda draw: draw.panel_size)
    for lower, higher in zip(ordered_draws, ordered_draws[1:]):
        if np.any(lower.incidence & ~higher.incidence):
            raise RuntimeError("Cyclic panels are not nested across coverage")
    return CyclicPanelFamily(
        assay_ids=assays, item_ids=items_input, circular_order=circular,
        start_offsets=dict(starts), draws=draws, seed=seed,
    )


def panel_design_tables(family: CyclicPanelFamily) -> dict[str, pd.DataFrame]:
    """Return exact condition, per-assay and pairwise panel diagnostics."""
    conditions, assays, pairs, multiplicities = [], [], [], []
    for requested in sorted(family.draws, reverse=True):
        draw = family.draws[requested]
        counts = draw.incidence.sum(axis=0)
        conditions.append({
            "requested_coverage": requested,
            "panel_size": draw.panel_size,
            "actual_coverage": draw.actual_coverage,
            "mean_pairwise_overlap": draw.mean_pairwise_overlap,
            "union_count": draw.union_count,
            "union_coverage": draw.union_coverage,
            "duplicate_of_coverage": draw.duplicate_of_coverage,
        })
        for row, assay in enumerate(draw.assay_ids):
            assays.append({
                "requested_coverage": requested, "assay": assay,
                "panel_size": int(draw.incidence[row].sum()),
                "actual_coverage": float(draw.incidence[row].mean()),
            })
        for first in range(len(draw.assay_ids)):
            for second in range(first + 1, len(draw.assay_ids)):
                pairs.append({
                    "requested_coverage": requested,
                    "assay_a": draw.assay_ids[first],
                    "assay_b": draw.assay_ids[second],
                    "intersection_count": int(draw.pairwise_intersections[first, second]),
                    "overlap_fraction": float(draw.pairwise_overlap[first, second]),
                })
        for number in range(1, len(draw.assay_ids) + 1):
            multiplicities.append({
                "requested_coverage": requested,
                "assay_multiplicity": number,
                "item_count": int(np.sum(counts == number)),
            })
    return {
        "conditions": pd.DataFrame(conditions),
        "assays": pd.DataFrame(assays),
        "pairs": pd.DataFrame(pairs),
        "multiplicity": pd.DataFrame(multiplicities),
    }


def assign_virtual_assays(
    cell_ids: Sequence[object],
    strata: Sequence[object],
    *,
    seed: int,
    assay_ids: Sequence[object] = DEFAULT_ASSAYS,
) -> np.ndarray:
    """Assign cells reproducibly and as evenly as possible within each stratum.

    Ordering uses hashes of stable cell IDs, not input row order.  Remainder
    assignments are rotated toward assays with the smallest accumulated count;
    this keeps the global allocation balanced without inspecting any outcome or
    expression value.
    """
    seed = _seed(seed)
    cells = _labels(cell_ids, name="cell_ids", minimum=1)
    assays = tuple(sorted(_labels(assay_ids, name="assay_ids", minimum=2)))
    values = np.asarray(strata, dtype=object)
    if values.shape != (len(cells),) or np.any(pd.isna(values)):
        raise ValueError("strata must be a cell-aligned vector without missing values")
    strata_labels = np.asarray(list(map(str, values)), dtype=object)
    if len(cells) < len(assays):
        raise ValueError("There must be at least one cell per virtual assay")

    result = np.empty(len(cells), dtype=object)
    totals = {assay: 0 for assay in assays}
    tie_order = tuple(map(str, np.random.default_rng(seed).permutation(
        np.asarray(assays, dtype=object))))
    tie_rank = {assay: rank for rank, assay in enumerate(tie_order)}
    for stratum in sorted(set(strata_labels), key=str):
        rows = np.flatnonzero(strata_labels == stratum)
        rows = np.asarray(sorted(rows, key=lambda row: (
            hashlib.sha256(f"{seed}\0{stratum}\0{cells[row]}".encode()).hexdigest(),
            cells[row],
        )), dtype=int)
        quotient, remainder = divmod(len(rows), len(assays))
        base = {assay: quotient for assay in assays}
        extras = sorted(assays, key=lambda assay: (totals[assay], tie_rank[assay]))[:remainder]
        counts = {assay: base[assay] + int(assay in extras) for assay in assays}
        cycle = sorted(assays, key=lambda assay: (totals[assay], tie_rank[assay]))
        cursor = 0
        for assay in cycle:
            take = counts[assay]
            result[rows[cursor:cursor + take]] = assay
            cursor += take
            totals[assay] += take
        if max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError("Within-stratum assay allocation is not balanced")
    if any(np.sum(result == assay) == 0 for assay in assays):
        raise ValueError("Balanced assignment did not place a cell in every virtual assay")
    return np.asarray(result, dtype=str)


def row_panel_mask(row_assays: Sequence[object], draw: CyclicPanelDraw) -> np.ndarray:
    """Expand an assay-by-item incidence matrix to aligned cell rows."""
    values = np.asarray(row_assays, dtype=object)
    if values.ndim != 1 or np.any(pd.isna(values)):
        raise ValueError("row_assays must be a one-dimensional vector without missing values")
    labels = np.asarray(list(map(str, values)), dtype=object)
    lookup = {assay: row for row, assay in enumerate(draw.assay_ids)}
    unknown = sorted(set(labels).difference(lookup), key=str)
    if unknown:
        raise ValueError(f"row_assays contains unknown assays: {unknown}")
    indices = np.asarray([lookup[label] for label in labels], dtype=int)
    return _readonly(draw.incidence[indices], bool)


def apply_target_panel(
    native_measured,
    row_assays: Sequence[object],
    draw: CyclicPanelDraw,
) -> np.ndarray:
    """Intersect an artificial target panel with native measurement support."""
    native = _binary_matrix(native_measured, name="native_measured")
    if native.shape[1] != len(draw.item_ids) or native.shape[0] != len(row_assays):
        raise ValueError("native_measured must align with rows and draw.item_ids")
    planned = row_panel_mask(row_assays, draw)
    return _readonly(native & planned, bool)


@dataclass(frozen=True)
class MeasurementSupportAudit:
    """Fold-local assay-by-item support after native and artificial masks."""

    assay_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    measurement_counts: np.ndarray
    effective_incidence: np.ndarray
    co_measurement_counts: np.ndarray
    per_assay_coverage: np.ndarray
    union_coverage: float
    unsupported_items: tuple[str, ...]
    single_assay_items: tuple[str, ...]
    graph_connected: bool
    n_components: int
    n_rows: int

    def __post_init__(self) -> None:
        a, d = len(self.assay_ids), len(self.item_ids)
        if np.asarray(self.measurement_counts).shape != (a, d):
            raise ValueError("measurement_counts must align with assays and items")
        if np.asarray(self.effective_incidence).shape != (a, d):
            raise ValueError("effective_incidence must align with assays and items")
        if np.asarray(self.co_measurement_counts).shape != (a, a):
            raise ValueError("co_measurement_counts must be square in assays")
        object.__setattr__(self, "measurement_counts", _readonly(self.measurement_counts, int))
        object.__setattr__(self, "effective_incidence", _readonly(self.effective_incidence, bool))
        object.__setattr__(self, "co_measurement_counts", _readonly(self.co_measurement_counts, int))
        object.__setattr__(self, "per_assay_coverage", _readonly(self.per_assay_coverage, float))


def _components(adjacency: np.ndarray) -> int:
    remaining = set(range(len(adjacency)))
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            neighbours = set(np.flatnonzero(adjacency[current])).intersection(remaining)
            remaining.difference_update(neighbours)
            stack.extend(neighbours)
    return components


def audit_measurement_support(
    measured,
    row_assays: Sequence[object],
    item_ids: Sequence[object],
    *,
    rows: Sequence[int] | np.ndarray | None = None,
    assay_ids: Sequence[object] = DEFAULT_ASSAYS,
) -> MeasurementSupportAudit:
    """Audit effective assay-item and cross-assay support on authorized rows."""
    matrix = _binary_matrix(measured, name="measured")
    items = _labels(item_ids, name="item_ids", minimum=1)
    assays = tuple(sorted(_labels(assay_ids, name="assay_ids", minimum=2)))
    if matrix.shape[1] != len(items) or matrix.shape[0] != len(row_assays):
        raise ValueError("measured must align with row_assays and item_ids")
    labels = np.asarray(list(map(str, np.asarray(row_assays, dtype=object))), dtype=object)
    if labels.shape != (len(matrix),) or not set(labels).issubset(assays):
        raise ValueError("row_assays must align with measured and use declared assays")
    selected = _rows(rows, len(matrix))
    counts = np.asarray([
        matrix[np.intersect1d(selected, np.flatnonzero(labels == assay), assume_unique=False)].sum(axis=0)
        for assay in assays
    ], dtype=int)
    effective = counts > 0
    co_measurement = effective.astype(int) @ effective.astype(int).T
    adjacency = co_measurement > 0
    np.fill_diagonal(adjacency, False)
    n_components = _components(adjacency)
    multiplicity = effective.sum(axis=0)
    return MeasurementSupportAudit(
        assay_ids=assays, item_ids=items, measurement_counts=counts,
        effective_incidence=effective, co_measurement_counts=co_measurement,
        per_assay_coverage=effective.mean(axis=1),
        union_coverage=float(np.mean(multiplicity > 0)),
        unsupported_items=tuple(item for item, number in zip(items, multiplicity) if number == 0),
        single_assay_items=tuple(item for item, number in zip(items, multiplicity) if number == 1),
        graph_connected=n_components == 1, n_components=n_components,
        n_rows=len(selected),
    )


@dataclass(frozen=True)
class CensoringDesign:
    """Common random numbers for assay-by-target positive censoring."""

    assay_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    assay_target_score: np.ndarray
    uniforms: np.ndarray
    heterogeneity_delta: float
    seed: int
    version: str = CENSORING_VERSION

    def __post_init__(self) -> None:
        score = np.asarray(self.assay_target_score, dtype=float)
        uniforms = np.asarray(self.uniforms, dtype=float)
        if score.shape != (len(self.assay_ids), len(self.target_ids)):
            raise ValueError("assay_target_score must align with assays and targets")
        if uniforms.ndim != 2 or uniforms.shape[1] != len(self.target_ids):
            raise ValueError("uniforms must have shape (n_cells, n_targets)")
        if (not np.all(np.isfinite(score)) or not np.all(np.isfinite(uniforms))
                or np.any(uniforms < 0) or np.any(uniforms >= 1)):
            raise ValueError("Censoring design arrays must be finite and uniforms in [0,1)")
        if not np.isfinite(self.heterogeneity_delta) or self.heterogeneity_delta < 0:
            raise ValueError("heterogeneity_delta must be finite and nonnegative")
        object.__setattr__(self, "assay_target_score", _readonly(score, float))
        object.__setattr__(self, "uniforms", _readonly(uniforms, float))


@dataclass(frozen=True)
class CensoringResult:
    observed: np.ndarray
    sensitivity: np.ndarray
    mechanism: str
    requested_retention: float
    expected_train_retention: float
    realized_train_retention: float
    normalizer: float | None
    train_positive_count: int
    train_retained_count: int
    target_normalizers: np.ndarray | None = None
    target_train_positive_counts: np.ndarray | None = None
    targets_without_train_positives: tuple[str, ...] = ()
    matched_to_train_retained_count: int | None = None
    version: str = CENSORING_VERSION

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed)
        sensitivity = np.asarray(self.sensitivity, dtype=float)
        if observed.shape != sensitivity.shape or observed.ndim != 2:
            raise ValueError("observed and sensitivity must be aligned matrices")
        if not np.all(np.isin(observed, [0, 1])) or not np.all(np.isfinite(sensitivity)):
            raise ValueError("Censoring result must contain binary observations and finite sensitivities")
        if np.any(sensitivity < 0) or np.any(sensitivity > 1):
            raise ValueError("Sensitivity must lie in [0,1]")
        object.__setattr__(self, "observed", _readonly(observed, bool))
        object.__setattr__(self, "sensitivity", _readonly(sensitivity, float))
        if self.target_normalizers is not None:
            normalizers = np.asarray(self.target_normalizers, dtype=float)
            if normalizers.shape != (observed.shape[1],) or not np.all(np.isfinite(normalizers)):
                raise ValueError("target_normalizers must be a finite target-aligned vector")
            object.__setattr__(self, "target_normalizers", _readonly(normalizers, float))
        if self.target_train_positive_counts is not None:
            counts = np.asarray(self.target_train_positive_counts)
            if (counts.shape != (observed.shape[1],)
                    or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0)):
                raise ValueError("target_train_positive_counts must be a nonnegative target-aligned vector")
            object.__setattr__(self, "target_train_positive_counts", _readonly(counts, int))

    @property
    def diagnostics(self) -> dict[str, object]:
        """JSON-safe fold-local censoring diagnostics."""
        return {
            "mechanism": self.mechanism,
            "requested_retention": self.requested_retention,
            "expected_train_retention": self.expected_train_retention,
            "realized_train_retention": self.realized_train_retention,
            "normalizer": self.normalizer,
            "target_normalizers": (None if self.target_normalizers is None
                                     else self.target_normalizers.tolist()),
            "target_train_positive_counts": (
                None if self.target_train_positive_counts is None
                else self.target_train_positive_counts.tolist()),
            "targets_without_train_positives": list(self.targets_without_train_positives),
            "train_positive_count": self.train_positive_count,
            "train_retained_count": self.train_retained_count,
            "matched_to_train_retained_count": self.matched_to_train_retained_count,
            "version": self.version,
        }


@dataclass(frozen=True)
class MeasurementObservationPlan:
    """Immutable, reusable assay-label/score/uniform censoring plan.

    ``generate`` estimates only the fold-specific normalization; the assay
    labels, heterogeneous surface and random uniforms stay fixed across folds,
    retention levels and models.  ``generate_pair`` additionally constructs a
    uniform arm with exactly the heterogeneous arm's realized training count.
    """

    row_assays: tuple[str, ...]
    design: CensoringDesign

    def __post_init__(self) -> None:
        labels = tuple(map(str, self.row_assays))
        if len(labels) != self.design.uniforms.shape[0]:
            raise ValueError("row_assays must align with the design's cell uniforms")
        if not set(labels).issubset(self.design.assay_ids):
            raise ValueError("row_assays contains an assay absent from the design")
        object.__setattr__(self, "row_assays", labels)

    @property
    def assay_ids(self) -> tuple[str, ...]:
        return self.design.assay_ids

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.design.target_ids

    @property
    def h(self) -> np.ndarray:
        return self.design.assay_target_score

    @property
    def u(self) -> np.ndarray:
        return self.design.uniforms

    def identity(self) -> dict[str, object]:
        """Return a JSON-safe identity including hashes of every random object."""
        def digest_array(value: np.ndarray) -> str:
            contiguous = np.ascontiguousarray(value)
            header = f"{contiguous.dtype.str}|{contiguous.shape}".encode()
            return hashlib.sha256(header + contiguous.tobytes()).hexdigest()

        payload = {
            "version": self.design.version,
            "seed": self.design.seed,
            "n_cells": len(self.row_assays),
            "assay_ids": list(self.assay_ids),
            "target_ids": list(self.target_ids),
            "heterogeneity_delta": self.design.heterogeneity_delta,
            "row_assays_sha256": hashlib.sha256(
                "\0".join(self.row_assays).encode("utf-8")
            ).hexdigest(),
            "h_sha256": digest_array(self.h),
            "u_sha256": digest_array(self.u),
        }
        payload["identity_sha256"] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        return payload

    def generate(
        self,
        reference,
        measured,
        train_rows: Sequence[int],
        *,
        retention: float,
        mechanism: str,
    ) -> CensoringResult:
        key = str(mechanism).lower().replace("-", "_")
        if key in {"heterogeneous", "assay_target", "assay_target_sar", "sar"}:
            return apply_heterogeneous_censoring(
                reference, measured, self.row_assays, train_rows,
                retention, self.design,
            )
        if key in {"matched_uniform", "matched_scar", "uniform", "scar"}:
            return apply_matched_uniform_censoring(
                reference, measured, self.row_assays, train_rows,
                retention, self.design,
            )
        raise ValueError(
            "mechanism must be 'heterogeneous' or 'matched_uniform' "
            "(aliases: assay_target/sar and uniform/scar)"
        )

    def generate_pair(
        self,
        reference,
        measured,
        train_rows: Sequence[int],
        *,
        retention: float,
    ) -> tuple[CensoringResult, CensoringResult]:
        return apply_censoring_pair(
            reference, measured, self.row_assays, train_rows,
            retention, self.design,
        )


def make_censoring_design(
    n_cells: int,
    target_ids: Sequence[object],
    *,
    seed: int,
    assay_ids: Sequence[object] = DEFAULT_ASSAYS,
    heterogeneity_delta: float = 1.0,
) -> CensoringDesign:
    """Draw one standardized assay-target surface and cell-target uniforms."""
    seed = _seed(seed)
    if isinstance(n_cells, (bool, np.bool_)) or not isinstance(n_cells, (int, np.integer)) or n_cells < 1:
        raise ValueError("n_cells must be a positive integer")
    assays = tuple(sorted(_labels(assay_ids, name="assay_ids", minimum=2)))
    targets = _labels(target_ids, name="target_ids", minimum=1)
    delta = float(heterogeneity_delta)
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("heterogeneity_delta must be finite and nonnegative")
    score_seed, uniform_seed = np.random.SeedSequence(seed).spawn(2)
    raw = np.random.default_rng(score_seed).normal(size=(len(assays), len(targets)))
    scale = float(raw.std())
    if scale < 1e-12:
        raise RuntimeError("Generated assay-target scores have zero variance")
    score = (raw - float(raw.mean())) / scale
    uniforms = np.random.default_rng(uniform_seed).uniform(size=(int(n_cells), len(targets)))
    return CensoringDesign(
        assay_ids=assays, target_ids=targets, assay_target_score=score,
        uniforms=uniforms, heterogeneity_delta=delta, seed=seed,
    )


def make_observation_plan(
    row_assays: Sequence[object],
    target_ids: Sequence[object],
    *,
    seed: int,
    assay_ids: Sequence[object] = DEFAULT_ASSAYS,
    heterogeneity_delta: float = 1.0,
) -> MeasurementObservationPlan:
    """Create a plan owning assay labels, standardized ``h`` and uniforms ``u``."""
    values = np.asarray(row_assays, dtype=object)
    if values.ndim != 1 or np.any(pd.isna(values)):
        raise ValueError("row_assays must be one-dimensional without missing values")
    labels = tuple(map(str, values))
    design = make_censoring_design(
        len(labels), target_ids, seed=seed, assay_ids=assay_ids,
        heterogeneity_delta=heterogeneity_delta,
    )
    return MeasurementObservationPlan(labels, design)


def _censoring_inputs(reference, measured, row_assays, train_rows, design):
    z = _binary_matrix(reference, name="reference")
    w = _binary_matrix(measured, name="measured", shape=z.shape)
    if z.shape != design.uniforms.shape:
        raise ValueError("reference/measured must align with the censoring design")
    labels = np.asarray(list(map(str, np.asarray(row_assays, dtype=object))), dtype=object)
    if labels.shape != (len(z),) or not set(labels).issubset(design.assay_ids):
        raise ValueError("row_assays must align with cells and use design assay IDs")
    train = _rows(train_rows, len(z))
    if not len(train):
        raise ValueError("train_rows cannot be empty")
    assay_lookup = {assay: row for row, assay in enumerate(design.assay_ids)}
    assay_index = np.asarray([assay_lookup[label] for label in labels], dtype=int)
    positives = z & w
    train_positives = positives[train]
    return z, w, train, assay_index, positives, train_positives


def _retention(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0 < result <= 1:
        raise ValueError("retention must lie in (0,1]")
    return result


def apply_heterogeneous_censoring(
    reference,
    measured,
    row_assays: Sequence[object],
    train_rows: Sequence[int],
    retention: float,
    design: CensoringDesign,
) -> CensoringResult:
    """Censor positives with fold-local assay-by-target logistic propensities."""
    retention = _retention(retention)
    _, _, train, assay_index, positives, train_positives = _censoring_inputs(
        reference, measured, row_assays, train_rows, design)
    n_positive = int(train_positives.sum())
    target_counts = train_positives.sum(axis=0).astype(int)
    missing_targets = tuple(
        target for target, count in zip(design.target_ids, target_counts) if count == 0
    )
    eta = design.heterogeneity_delta * design.assay_target_score[assay_index]
    if retention == 1.0:
        pooled_normalizer = None
        target_normalizers = None
        sensitivity = np.ones_like(eta, dtype=float)
    else:
        if n_positive == 0:
            raise ValueError("Cannot normalize censoring: fold training has no measured reference positives")
        pooled_eta = eta[train][train_positives]

        def solve(values: np.ndarray) -> float:
            lower = -float(np.max(values)) - 50.0
            upper = -float(np.min(values)) + 50.0
            return float(brentq(
                lambda intercept: expit(intercept + values).mean() - retention,
                lower, upper, xtol=1e-12,
            ))

        # The pooled value is used only for targets without an authorized
        # inner-training positive. It uses no validation/test outcomes and is
        # disclosed explicitly in the result diagnostics.
        pooled_normalizer = solve(pooled_eta)
        target_normalizers = np.empty(len(design.target_ids), dtype=float)
        for target in range(len(design.target_ids)):
            target_positive = train_positives[:, target]
            target_normalizers[target] = (
                solve(eta[train, target][target_positive])
                if np.any(target_positive) else pooled_normalizer
            )
        sensitivity = expit(target_normalizers[None, :] + eta)
    observed = positives & (design.uniforms < sensitivity)
    retained = int(observed[train].sum())
    expected = (1.0 if n_positive == 0
                else float(sensitivity[train][train_positives].mean()))
    realized = float(retained / n_positive) if n_positive else np.nan
    return CensoringResult(
        observed=observed, sensitivity=sensitivity, mechanism="heterogeneous",
        requested_retention=retention, expected_train_retention=expected,
        realized_train_retention=realized, normalizer=pooled_normalizer,
        train_positive_count=n_positive, train_retained_count=retained,
        target_normalizers=target_normalizers,
        target_train_positive_counts=target_counts,
        targets_without_train_positives=missing_targets,
    )


def apply_matched_uniform_censoring(
    reference,
    measured,
    row_assays: Sequence[object],
    train_rows: Sequence[int],
    retention: float,
    design: CensoringDesign,
    *,
    matched_train_retained_count: int | None = None,
) -> CensoringResult:
    """Apply SCAR, optionally matching a heterogeneous arm's realized count.

    When ``matched_train_retained_count`` is supplied, a fold-training quantile
    of the same fixed uniforms is used.  This exactly matches the number of
    retained training positives while leaving validation/test outcomes out of
    threshold selection.
    """
    retention = _retention(retention)
    _, _, train, _, positives, train_positives = _censoring_inputs(
        reference, measured, row_assays, train_rows, design)
    n_positive = int(train_positives.sum())
    target_counts = train_positives.sum(axis=0).astype(int)
    missing_targets = tuple(
        target for target, count in zip(design.target_ids, target_counts) if count == 0
    )
    if n_positive == 0 and retention < 1:
        raise ValueError("Cannot normalize censoring: fold training has no measured reference positives")
    if matched_train_retained_count is None:
        threshold = retention
        matched_count = None
    else:
        if (isinstance(matched_train_retained_count, (bool, np.bool_))
                or not isinstance(matched_train_retained_count, (int, np.integer))):
            raise TypeError("matched_train_retained_count must be an integer")
        matched_count = int(matched_train_retained_count)
        if not 0 <= matched_count <= n_positive:
            raise ValueError("matched_train_retained_count is outside the training-positive range")
        train_uniforms = np.sort(design.uniforms[train][train_positives])
        if matched_count == 0:
            threshold = 0.0
        elif matched_count == n_positive:
            threshold = 1.0
        else:
            below, above = train_uniforms[matched_count - 1:matched_count + 1]
            if below == above:
                raise ValueError("Tied training uniforms prevent an exact matched-SCAR threshold")
            threshold = float((below + above) / 2.0)
    sensitivity = np.full(positives.shape, threshold, dtype=float)
    observed = positives & (design.uniforms < threshold)
    retained = int(observed[train].sum())
    if matched_count is not None and retained != matched_count:
        raise RuntimeError("Matched-uniform censoring did not reproduce the requested training count")
    realized = float(retained / n_positive) if n_positive else np.nan
    normalizer = (None if threshold in (0.0, 1.0)
                  else float(np.log(threshold / (1.0 - threshold))))
    return CensoringResult(
        observed=observed, sensitivity=sensitivity, mechanism="matched_uniform",
        requested_retention=retention, expected_train_retention=float(threshold),
        realized_train_retention=realized, normalizer=normalizer,
        train_positive_count=n_positive, train_retained_count=retained,
        target_train_positive_counts=target_counts,
        targets_without_train_positives=missing_targets,
        matched_to_train_retained_count=matched_count,
    )


def apply_censoring_pair(
    reference,
    measured,
    row_assays: Sequence[object],
    train_rows: Sequence[int],
    retention: float,
    design: CensoringDesign,
) -> tuple[CensoringResult, CensoringResult]:
    """Return heterogeneous and exactly count-matched uniform censoring arms."""
    heterogeneous = apply_heterogeneous_censoring(
        reference, measured, row_assays, train_rows, retention, design)
    uniform = apply_matched_uniform_censoring(
        reference, measured, row_assays, train_rows, retention, design,
        matched_train_retained_count=heterogeneous.train_retained_count,
    )
    return heterogeneous, uniform


__all__ = [
    "CENSORING_VERSION",
    "DEFAULT_ASSAYS",
    "MEASUREMENT_DESIGN_VERSION",
    "CensoringDesign",
    "CensoringResult",
    "CyclicPanelDraw",
    "CyclicPanelFamily",
    "MeasurementSupportAudit",
    "MeasurementObservationPlan",
    "apply_censoring_pair",
    "apply_heterogeneous_censoring",
    "apply_matched_uniform_censoring",
    "apply_target_panel",
    "assign_virtual_assays",
    "audit_measurement_support",
    "draw_cyclic_panel_family",
    "make_censoring_design",
    "make_observation_plan",
    "panel_design_tables",
    "row_panel_mask",
]
