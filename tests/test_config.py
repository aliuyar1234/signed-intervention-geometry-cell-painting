from siggeo.config import load_analysis_config, load_paths_config, repo_root


def test_load_analysis_config() -> None:
    config = load_analysis_config("configs/analysis_smoke.yaml")
    assert config.run_name.startswith("smoke__")
    assert config.result_dir == repo_root() / "results" / "runs"


def test_load_paths_config() -> None:
    config = load_paths_config("configs/paths.example.yaml")
    assert config.paths["raw_root"] == repo_root() / "data" / "raw"
