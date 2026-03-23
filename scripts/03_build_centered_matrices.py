from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from siggeo.config import load_analysis_config, resolve_repo_path
from siggeo.io.atomic import write_json_atomic, write_text_atomic
from siggeo.preprocess.feature_space import exact_feature_intersection
from siggeo.preprocess.matrix_builder import build_centered_matrix, write_centered_outputs
from siggeo.runs import tracked_phase, update_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact-intersection centered well matrices.")
    parser.add_argument("--config", required=True, type=Path, help="Path to an analysis config YAML.")
    args = parser.parse_args()

    with tracked_phase(args.config, "03_build_centered_matrices") as context:
        config = load_analysis_config(args.config)
        data_cfg = config.raw["data"]
        preprocess_cfg = config.raw.get("preprocess", {})
        db_path = resolve_repo_path(data_cfg["metadata_db"])
        orf_path = resolve_repo_path(data_cfg["orf_profile"])
        crispr_path = resolve_repo_path(data_cfg["crispr_profile"])

        if not bool(preprocess_cfg.get("use_exact_feature_intersection", True)):
            raise ValueError(
                "preprocess.use_exact_feature_intersection=false is unsupported in the locked pipeline. "
                "This run requires a shared exact feature space."
            )
        feature_list = exact_feature_intersection(orf_path, crispr_path)
        write_text_atomic(
            context.run_dir / "tables" / "feature_intersection.txt",
            "\n".join(feature_list) + ("\n" if feature_list else ""),
        )

        outputs: dict[str, object] = {"feature_count": len(feature_list)}
        centering_metrics: dict[str, object] = {}
        def _build_for_modality(modality: str, profile_path: Path):
            centered_df, qc = build_centered_matrix(
                profile_path=profile_path,
                db_path=db_path,
                modality=modality,
                feature_list=feature_list,
                preprocess_config=preprocess_cfg,
                mode=config.mode,
            )
            return modality, centered_df, qc

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_build_for_modality, modality, profile_path)
                for modality, profile_path in (("orf", orf_path), ("crispr", crispr_path))
            ]
            results = [future.result() for future in futures]

        for modality, centered_df, qc in sorted(results, key=lambda item: item[0]):
            matrix_path = context.run_dir / "tables" / f"centered_wells_{config.pipeline}_{modality}.parquet"
            qc_path = context.run_dir / "logs" / f"03_build_centered_matrices__{modality}.json"
            write_centered_outputs(centered_df, matrix_path, qc_path, qc)
            outputs[f"{modality}_wells"] = centered_df.height
            outputs[f"{modality}_cell_count_available"] = qc["cell_count_available"]
            centering_metrics[f"{modality}_plates_direct"] = int(qc.get("plates_direct", 0))
            centering_metrics[f"{modality}_plates_batch_fallback"] = int(qc.get("plates_batch_fallback", 0))
            centering_metrics[f"{modality}_plates_global_fallback"] = int(qc.get("plates_global_fallback", 0))
            centering_metrics[f"{modality}_artifact_removed_count"] = int(qc.get("artifact_removed_count", 0))

        update_metrics(
            context.run_dir,
            {
                "shared_feature_count": len(feature_list),
                "cell_count_available": bool(outputs.get("orf_cell_count_available") or outputs.get("crispr_cell_count_available")),
                "centering_fallback_counts": {
                    "orf": {
                        "plate": int(centering_metrics.get("orf_plates_direct", 0)),
                        "batch": int(centering_metrics.get("orf_plates_batch_fallback", 0)),
                        "global": int(centering_metrics.get("orf_plates_global_fallback", 0)),
                    },
                    "crispr": {
                        "plate": int(centering_metrics.get("crispr_plates_direct", 0)),
                        "batch": int(centering_metrics.get("crispr_plates_batch_fallback", 0)),
                        "global": int(centering_metrics.get("crispr_plates_global_fallback", 0)),
                    },
                },
                "artifact_removed_count": int(centering_metrics.get("orf_artifact_removed_count", 0))
                + int(centering_metrics.get("crispr_artifact_removed_count", 0)),
            },
        )
        write_json_atomic(context.run_dir / "logs" / "03_build_centered_matrices.json", {"status": "completed", "run_name": context.config.run_name, **outputs})
        print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
