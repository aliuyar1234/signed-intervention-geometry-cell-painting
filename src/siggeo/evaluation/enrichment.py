from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from siggeo.io.atomic import write_csv_atomic


PRIMARY_REGIMES = {"aligned", "inverse", "orthogonal", "asymmetric_ORF", "asymmetric_CRISPR"}


@dataclass(frozen=True)
class EnrichmentConfig:
    reactome_min_set_size: int
    corum_min_set_size: int
    primary_min_regime_size: int
    fdr_method: str
    allow_legacy_display_symbol_fallback: bool = False


def _normalize_fdr_method(method: str) -> str:
    normalized = method.strip().casefold().replace("-", "_")
    aliases = {
        "bh": "fdr_bh",
        "benjamini_hochberg": "fdr_bh",
        "by": "fdr_by",
        "benjamini_yekutieli": "fdr_by",
    }
    return aliases.get(normalized, normalized)


def _parse_reactome_gmt(path: Path) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) < 3:
            continue
        term_name = parts[0]
        term_id = parts[1]
        genes = sorted(set(parts[2:]))
        terms.append({"term_id": term_id, "term_name": term_name, "genes": genes})
    return terms


def _parse_corum_tsv(path: Path) -> list[dict[str, object]]:
    frame = pl.read_csv(path, separator="\t", infer_schema_length=1000)
    column_lookup = {column.casefold(): column for column in frame.columns}
    term_id_col = column_lookup.get("complexid") or column_lookup.get("complex id") or frame.columns[0]
    term_name_col = column_lookup.get("complexname") or column_lookup.get("complex name") or frame.columns[1]
    gene_col = (
        column_lookup.get("subunits (gene name)")
        or column_lookup.get("subunits(gene name)")
        or column_lookup.get("subunits_gene_name")
        or column_lookup.get("gene members")
        or column_lookup.get("genes")
    )
    if gene_col is None:
        raise KeyError("Could not identify CORUM gene-member column.")
    terms: list[dict[str, object]] = []
    for row in frame.select([term_id_col, term_name_col, gene_col]).to_dicts():
        raw_members = str(row[gene_col]).replace(";", ",")
        genes = sorted({member.strip() for member in raw_members.split(",") if member.strip()})
        if genes:
            terms.append(
                {
                    "term_id": str(row[term_id_col]),
                    "term_name": str(row[term_name_col]),
                    "genes": genes,
                }
            )
    return terms


def _run_resource_enrichment(
    geometry: pl.DataFrame,
    terms: list[dict[str, object]],
    *,
    resource: str,
    min_set_size: int,
    primary_min_regime_size: int,
    fdr_method: str,
    allow_legacy_display_symbol_fallback: bool,
) -> pl.DataFrame:
    if "canonical_symbol" in geometry.columns:
        symbol_column = "canonical_symbol"
    elif allow_legacy_display_symbol_fallback and "display_symbol" in geometry.columns:
        symbol_column = "display_symbol"
    else:
        raise KeyError(
            "Geometry table is missing canonical_symbol. "
            "Publication enrichment now requires canonical symbols unless legacy fallback is explicitly enabled."
        )
    analyzed = geometry.filter(pl.col("regime_label").is_in(PRIMARY_REGIMES)).select(
        ["gene_id", symbol_column, "regime_label"]
    ).rename({symbol_column: "enrichment_symbol"})
    analyzed_symbols = {row["enrichment_symbol"] for row in analyzed.to_dicts() if row["enrichment_symbol"]}
    resource_universe = analyzed_symbols.intersection({gene for term in terms for gene in term["genes"]})
    rows: list[dict[str, object]] = []
    for regime in sorted(PRIMARY_REGIMES):
        regime_genes = set(
            analyzed.filter(pl.col("regime_label") == regime)["enrichment_symbol"].drop_nulls().to_list()
        ).intersection(resource_universe)
        if len(regime_genes) < primary_min_regime_size:
            continue
        regime_set = regime_genes
        for term in terms:
            term_genes = set(term["genes"])
            if len(term_genes) < min_set_size:
                continue
            background = resource_universe
            term_background = background.intersection(term_genes)
            if len(term_background) < min_set_size:
                continue
            overlap = regime_set.intersection(term_genes)
            a = len(overlap)
            b = len(regime_set - term_background)
            c = len(term_background - regime_set)
            d = len(background - regime_set - term_background)
            if min(a + b, a + c) == 0:
                continue
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            rows.append(
                {
                    "resource": resource,
                    "background_mode": "primary_confident_noninactive",
                    "regime": regime,
                    "term_id": term["term_id"],
                    "term_name": term["term_name"],
                    "term_size": len(term_background),
                    "overlap_size": a,
                    "background_size": len(background),
                    "odds_ratio": float(odds_ratio),
                    "p_value": float(p_value),
                    "q_value": None,
                    "overlap_genes": "|".join(sorted(overlap)),
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "resource": pl.String,
                "background_mode": pl.String,
                "regime": pl.String,
                "term_id": pl.String,
                "term_name": pl.String,
                "term_size": pl.Int64,
                "overlap_size": pl.Int64,
                "background_size": pl.Int64,
                "odds_ratio": pl.Float64,
                "p_value": pl.Float64,
                "q_value": pl.Float64,
                "overlap_genes": pl.String,
            }
        )
    frame = pl.DataFrame(rows)
    corrected = multipletests(frame["p_value"].to_list(), method=_normalize_fdr_method(fdr_method))[1]
    return frame.with_columns(pl.Series("q_value", corrected.tolist())).sort(["q_value", "term_id"])


def run_enrichment(
    geometry: pl.DataFrame,
    reactome_path: Path,
    corum_path: Path | None,
    *,
    config: EnrichmentConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    reactome = _run_resource_enrichment(
        geometry,
        _parse_reactome_gmt(reactome_path),
        resource="reactome",
        min_set_size=config.reactome_min_set_size,
        primary_min_regime_size=config.primary_min_regime_size,
        fdr_method=config.fdr_method,
        allow_legacy_display_symbol_fallback=config.allow_legacy_display_symbol_fallback,
    )
    if corum_path is not None and corum_path.exists():
        corum = _run_resource_enrichment(
            geometry,
            _parse_corum_tsv(corum_path),
            resource="corum",
            min_set_size=config.corum_min_set_size,
            primary_min_regime_size=config.primary_min_regime_size,
            fdr_method=config.fdr_method,
            allow_legacy_display_symbol_fallback=config.allow_legacy_display_symbol_fallback,
        )
    else:
        corum = pl.DataFrame(schema=reactome.schema)
    return reactome, corum


def write_enrichment_outputs(
    reactome: pl.DataFrame,
    corum: pl.DataFrame,
    reactome_path: Path,
    corum_path: Path,
) -> None:
    write_csv_atomic(reactome, reactome_path)
    write_csv_atomic(corum, corum_path)
