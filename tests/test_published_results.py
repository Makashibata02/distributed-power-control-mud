"""Regression checks for the compact published snapshot."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    published = ROOT / "results" / "published"
    png_files = list(published.rglob("*.png"))
    svg_files = list(published.rglob("*.svg"))
    assert len(png_files) == 23, len(png_files)
    assert len(svg_files) == 23, len(svg_files)

    part1_two_user = read_rows(published / "part1_two_user" / "data" / "table_2_3_discrete_gain_over_single.csv")
    gains = {row["接收机"]: float(row["绝对增益"]) for row in part1_two_user}
    assert abs(gains["SUD"] - 0.53312) < 1e-10
    assert abs(gains["SIC"] - 0.8) < 1e-10

    comparison = read_rows(published / "part3_stochastic_optimization" / "data" / "ch4_baseline_pso_compare.csv")
    values = {row["method"]: float(row["mc_mean_T_over_G"]) for row in comparison}
    assert abs(values["PSO_MC"] - 0.6508095) < 1e-10
    assert values["PSO_MC"] > values["local_search"] > values["structured_uniform_exp"]

    print("test_published_results: OK")


if __name__ == "__main__":
    main()
