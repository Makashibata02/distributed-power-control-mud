"""Unified reproducibility command for the published research package."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from .parts.part1_two_user.pipeline import run as run_part1_two_user
from .workflows.part2_baselines import run_experiment as run_part2
from .workflows.part3_monte_carlo_baselines import run_experiment as run_part3_baselines
from .workflows.part3_pso_optimization import run_experiment as run_part3_pso
from .workflows.part3_method_comparison import run_experiment as run_part3_comparison
from .workflows.part3_sensitivity import run_experiment as run_part3_sensitivity


def _repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "configs").is_dir():
        return candidate
    cwd = Path.cwd().resolve()
    if (cwd / "configs").is_dir():
        return cwd
    raise FileNotFoundError("Run this command from the repository or install it in editable mode.")


def _result_dir(paths: dict[str, Path], key: str) -> Path:
    return Path(paths[key]).resolve().parent


def reproduce(*, quick: bool, output_dir: Path) -> Path:
    repo_root = _repo_root()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["DPC_MUD_REPO_ROOT"] = str(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")

    part1_dir = run_part1_two_user(output_dir / "part1_two_user")

    os.environ["DPC_MUD_OUTPUT_ROOT"] = str(output_dir / "part2_grouped_access")
    part2_paths = run_part2(
        quick=quick,
        config_path=repo_root / "configs" / "part2_grouped_access.yaml",
    )
    part2_dir = _result_dir(part2_paths, "summary_csv")

    os.environ["DPC_MUD_OUTPUT_ROOT"] = str(output_dir / "part3_stochastic_optimization")
    baseline_paths = run_part3_baselines(
        ch3_dir=part2_dir,
        quick=quick,
        config_path=repo_root / "configs" / "part3_stochastic_optimization.yaml",
    )
    baseline_dir = _result_dir(baseline_paths, "summary_csv")
    pso_paths = run_part3_pso(
        ch3_dir=part2_dir,
        quick=quick,
        config_path=repo_root / "configs" / "part3_stochastic_optimization.yaml",
    )
    pso_dir = _result_dir(pso_paths, "summary_csv")
    compare_paths = run_part3_comparison(baseline_mc_dir=baseline_dir, pso_dir=pso_dir)
    sensitivity_paths = run_part3_sensitivity(
        ch3_dir=part2_dir,
        quick=quick,
        config_path=repo_root / "configs" / "part3_stochastic_optimization.yaml",
    )

    def relative(path: Path) -> str:
        return path.resolve().relative_to(repo_root).as_posix()

    manifest = {
        "schema_version": 1,
        "mode": "quick" if quick else "full",
        "python": platform.python_version(),
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
        "configs": ["configs/part2_grouped_access.yaml", "configs/part3_stochastic_optimization.yaml"],
        "outputs": {
            "part1_two_user": relative(part1_dir),
            "part2_grouped_access": relative(part2_dir),
            "part3_baselines": relative(baseline_dir),
            "part3_pso": relative(pso_dir),
            "part3_comparison": relative(_result_dir(compare_paths, "compare_csv")),
            "part3_sensitivity": relative(_result_dir(sensitivity_paths, "summary_csv")),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce all published experiments.")
    parser.add_argument("command", choices=["reproduce"], help="Workflow to execute.")
    parser.add_argument("--quick", action="store_true", help="Use reduced search and Monte Carlo settings.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/generated"),
        help="Repository-relative output directory (default: results/generated).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = reproduce(quick=bool(args.quick), output_dir=args.output_dir)
    print(f"Reproduction manifest: {manifest}")


if __name__ == "__main__":
    main(sys.argv[1:])
