from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from siggeo.figures.utils import (
    REGIME_COLORS,
    annotate_bar_values,
    finalize_figure,
    format_small_metric_axis,
    prepare_axes,
)
from siggeo.io.atomic import write_csv_atomic


def _save_source(frame: pl.DataFrame, path: Path) -> None:
    write_csv_atomic(frame, path)


def _coerce_numeric_columns(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    casts = [pl.col(column).cast(pl.Float64, strict=False).alias(column) for column in columns if column in frame.columns]
    if not casts:
        return frame
    return frame.with_columns(casts)


def make_figure1_dataset(
    overlap: pl.DataFrame,
    gene_stats: pl.DataFrame,
    metrics: dict[str, Any],
    fig_path: Path,
    source_path: Path,
) -> None:
    source = pl.DataFrame(
        {
            "metric": [
                "overlap_genes",
                "keep_for_analysis_genes",
                "active_orf_count",
                "active_crispr_count",
                "active_both_count",
                "shared_feature_count",
            ],
            "value": [
                int(metrics.get("overlap_gene_count", overlap.height)),
                int(metrics.get("keep_for_analysis_count", overlap.filter(pl.col("keep_for_analysis")).height)),
                int(metrics.get("active_orf_count", 0)),
                int(metrics.get("active_crispr_count", 0)),
                int(metrics.get("active_both_count", 0)),
                int(metrics.get("shared_feature_count", 0)),
            ],
        }
    )
    _save_source(source, source_path)
    labels = [
        "Overlap genes",
        "Kept for analysis",
        "Active ORF",
        "Active CRISPR",
        "Active in both",
        "Shared features",
    ]
    values = source["value"].to_list()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    colors = ["#1F4E79", "#2D7F8D", "#C88A1B", "#B04A2F", "#3C7A3D", "#52606D"]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Dataset scale and analysis throughput", loc="left", fontweight="bold")
    annotate_bar_values(ax, fmt="{:.0f}", padding=max(values) * 0.01)
    prepare_axes(ax)
    finalize_figure(fig_path)


def make_figure2_regime_map(
    geometry: pl.DataFrame,
    fig_path: Path,
    source_path: Path,
    *,
    sign_z_threshold: float,
    asymmetry_threshold: float,
) -> None:
    source = geometry.select(["gene_id", "display_symbol", "sign_z", "delta_activity", "regime_label"])
    _save_source(source, source_path)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.axvspan(-sign_z_threshold, sign_z_threshold, color="#F4F1EA", alpha=0.7, zorder=0)
    ax.axhspan(-asymmetry_threshold, asymmetry_threshold, color="#F7F7F5", alpha=0.6, zorder=0)
    counts = source.group_by("regime_label").len()
    count_map = {row["regime_label"]: int(row["len"]) for row in counts.to_dicts()}
    for regime, subset in source.group_by("regime_label", maintain_order=True):
        label = regime[0] if isinstance(regime, tuple) else regime
        ax.scatter(
            subset["sign_z"].to_list(),
            subset["delta_activity"].to_list(),
            s=14,
            alpha=0.68,
            label=f"{label.replace('_', ' ')} (n={count_map.get(label, 0)})",
            color=REGIME_COLORS.get(label, "#4A5568"),
            edgecolors="none",
        )
    ax.axvline(sign_z_threshold, color="#243B53", linestyle="--", linewidth=1.0)
    ax.axvline(-sign_z_threshold, color="#243B53", linestyle="--", linewidth=1.0)
    ax.axhline(asymmetry_threshold, color="#243B53", linestyle=":", linewidth=1.0)
    ax.axhline(-asymmetry_threshold, color="#243B53", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Signed alignment z-score")
    ax.set_ylabel("Activity asymmetry (ORF - CRISPR)")
    ax.set_title("Structured yet predominantly asymmetric regime geometry", loc="left", fontweight="bold")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    prepare_axes(ax)
    finalize_figure(fig_path)


def make_figure3_stability(pipeline_agreement: pl.DataFrame, fig_path: Path, source_path: Path) -> None:
    _save_source(pipeline_agreement, source_path)
    metrics = [
        "overall_agreement_primary_only",
        "aligned_agreement",
        "inverse_agreement",
        "orthogonal_agreement",
        "asymmetric_orf_agreement",
        "asymmetric_crispr_agreement",
    ]
    display = (
        pipeline_agreement.filter(pl.col("has_real_comparison"))
        if "has_real_comparison" in pipeline_agreement.columns
        else pipeline_agreement
    )
    fig, ax = plt.subplots(figsize=(8.2, max(3.8, 1.6 + 0.85 * max(display.height, 1))))
    if display.height == 0:
        ax.text(0.5, 0.5, "Comparison runs not available yet", ha="center", va="center")
        ax.axis("off")
        ax.set_title("Robustness profile across locked comparison runs", loc="left", fontweight="bold")
        finalize_figure(fig_path)
        return

    matrix = np.array(display.select(metrics).to_numpy(), dtype=float)
    image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="YlGnBu")
    fig.colorbar(image, ax=ax, label="Agreement")
    metric_labels = [
        "Primary only",
        "Aligned",
        "Inverse",
        "Orthogonal",
        "Asym ORF",
        "Asym CRISPR",
    ]
    ax.set_xticks(range(len(metrics)), metric_labels, rotation=25, ha="right")
    ax.set_yticks(range(display.height), display["comparison_name"].to_list())
    for row_idx in range(display.height):
        for col_idx in range(len(metrics)):
            value = float(matrix[row_idx, col_idx])
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="#102A43", fontsize=8)
    ax.set_title("Robustness profile across locked comparison runs", loc="left", fontweight="bold")
    prepare_axes(ax)
    finalize_figure(fig_path)


def make_figure4_retrieval(summary: pl.DataFrame, fig_path: Path, source_path: Path) -> None:
    _save_source(summary, source_path)
    ordered = summary.with_columns(
        pl.when(pl.col("direction") == "orf_to_crispr").then(pl.lit("ORF to CRISPR")).otherwise(pl.lit("CRISPR to ORF")).alias("direction_label")
    )
    score_order = [
        "signed_cosine",
        "abs_cosine",
        "sign_z",
        "regime_score_lambda_0.25",
        "regime_score_lambda_0.5",
        "regime_score_lambda_1",
    ]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(score_order))
    width = 0.36
    colors = {"ORF to CRISPR": "#1F4E79", "CRISPR to ORF": "#C88A1B"}
    for idx, label in enumerate(["ORF to CRISPR", "CRISPR to ORF"]):
        subset = (
            ordered.filter(pl.col("direction_label") == label)
            .with_columns(pl.col("score_name").cast(pl.Categorical))
            .sort("score_name")
        )
        score_map = {row["score_name"]: float(row["mrr"]) for row in subset.to_dicts()}
        values = [score_map.get(score, 0.0) for score in score_order]
        ax.bar(x + (idx - 0.5) * width, values, width=width, color=colors[label], label=label)
    random_baseline = float(sum(1.0 / i for i in range(1, int(summary["n_queries"][0]) + 1)) / int(summary["n_queries"][0]))
    ax.axhline(random_baseline, color="#7B8794", linestyle="--", linewidth=1.2, label="Random baseline")
    ax.set_xticks(x, [name.replace("_", " ").replace("lambda ", "lambda=") for name in score_order], rotation=25, ha="right")
    ax.set_ylabel("MRR × 10$^{-3}$")
    ax.set_title("Retrieval remains near random despite regime-aware scoring", loc="left", fontweight="bold")
    format_small_metric_axis(ax, axis="y")
    prepare_axes(ax)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    finalize_figure(fig_path)


def make_figure5_enrichment(reactome: pl.DataFrame, corum: pl.DataFrame, fig_path: Path, source_path: Path) -> None:
    combined = pl.concat(
        [
            reactome.with_columns(pl.lit("reactome").alias("resource_name")) if reactome.height else reactome,
            corum.with_columns(pl.lit("corum").alias("resource_name")) if corum.height else corum,
        ],
        how="diagonal_relaxed",
    )
    combined = _coerce_numeric_columns(combined, ["p_value", "q_value"])
    if combined.height:
        source = combined.sort(["q_value", "p_value"]).head(12)
    else:
        source = pl.DataFrame({"term_name": [], "q_value": [], "resource_name": [], "regime": []})
    _save_source(source, source_path)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    if source.height:
        labels = [f"{term} ({regime})" for term, regime in zip(source["term_name"], source["regime"], strict=False)]
        values = [-np.log10(max(float(value), 1e-300)) for value in source["q_value"].fill_null(1.0).to_list()]
        colors = [REGIME_COLORS.get(regime, "#4A5568") for regime in source["regime"].to_list()]
        ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(r"$-\log_{10}(q\mathrm{-value})$")
    else:
        ax.text(0.5, 0.5, "No enrichment rows available", ha="center", va="center")
    ax.set_title("Biological signal concentrates in asymmetric regimes", loc="left", fontweight="bold")
    prepare_axes(ax)
    finalize_figure(fig_path)


def make_figure6_confounds(confounds: pl.DataFrame, fig_path: Path, source_path: Path) -> None:
    confounds = _coerce_numeric_columns(confounds, ["statistic_value"])
    _save_source(confounds, source_path)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    labels = [label.replace("_", " ") for label in confounds["confound_name"].to_list()]
    values = confounds["statistic_value"].fill_null(0.0).to_list()
    colors = ["#2F855A" if ok else "#C53030" for ok in confounds["supports_positive_story"].to_list()]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Statistic value")
    ax.set_title("Confound checks support fragility over a clean positive story", loc="left", fontweight="bold")
    annotate_bar_values(ax, fmt="{:.3f}", padding=max(values) * 0.015 if values else 0.01)
    prepare_axes(ax)
    finalize_figure(fig_path)
