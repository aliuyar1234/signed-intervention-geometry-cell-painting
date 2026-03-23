from __future__ import annotations

from pathlib import Path

import polars as pl

from siggeo.figures.generate import make_figure5_enrichment, make_figure6_confounds


def test_make_figure5_enrichment_coerces_string_q_values(tmp_path: Path) -> None:
    reactome = pl.DataFrame(
        {
            "term_name": ["Term A", "Term B"],
            "regime": ["aligned", "inverse"],
            "p_value": ["1e-6", "2e-4"],
            "q_value": ["5e-4", "1e-3"],
        }
    )
    corum = pl.DataFrame(
        {
            "term_name": ["Term C"],
            "regime": ["orthogonal"],
            "p_value": ["3e-4"],
            "q_value": ["2e-3"],
        }
    )

    fig_path = tmp_path / "figure5.png"
    source_path = tmp_path / "figure5.csv"

    make_figure5_enrichment(reactome, corum, fig_path, source_path)

    assert fig_path.exists()
    assert source_path.exists()
    source = pl.read_csv(source_path)
    assert source["q_value"].cast(pl.Float64, strict=False).null_count() == 0


def test_make_figure6_confounds_coerces_string_statistic_values(tmp_path: Path) -> None:
    confounds = pl.DataFrame(
        {
            "confound_name": ["artifact_sensitivity", "plate_holdout"],
            "statistic_value": ["0.42", "0.91"],
            "supports_positive_story": [False, True],
        }
    )

    fig_path = tmp_path / "figure6.png"
    source_path = tmp_path / "figure6.csv"

    make_figure6_confounds(confounds, fig_path, source_path)

    assert fig_path.exists()
    assert source_path.exists()
    source = pl.read_csv(source_path)
    assert source["statistic_value"].cast(pl.Float64, strict=False).null_count() == 0
