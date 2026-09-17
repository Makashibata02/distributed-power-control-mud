"""Sanitize copied result artifacts and extract a compact Part II trace."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def portable_string(value: str) -> str:
    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    if lowered.startswith("results/ch3_") or lowered.startswith("results_quick/ch3_"):
        return "results/published/part2_grouped_access"
    if lowered.startswith("results/ch4_") or lowered.startswith("results_quick/ch4_"):
        return "results/published/part3_stochastic_optimization"
    if not WINDOWS_ABSOLUTE.match(value):
        return normalized
    if "part2_grouped_access.yaml" in lowered:
        return "configs/part2_grouped_access.yaml"
    if "part3_stochastic_optimization.yaml" in lowered:
        return "configs/part3_stochastic_optimization.yaml"
    if "ch3_exp10" in lowered:
        return "results/published/part2_grouped_access"
    if "ch4_exp" in lowered:
        return "results/published/part3_stochastic_optimization"
    return Path(normalized).name


def portable_data(value: Any) -> Any:
    if isinstance(value, str):
        return portable_string(value)
    if isinstance(value, list):
        return [portable_data(item) for item in value]
    if isinstance(value, dict):
        return {key: portable_data(item) for key, item in value.items()}
    return value


def sanitize_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(portable_data(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: portable_string(value) for key, value in row.items()} for row in rows])


def sanitize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\s*<metadata>.*?</metadata>", "", text, count=1, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")


def sanitize_png(path: Path) -> None:
    with Image.open(path) as image:
        pixels = image.copy()
    pixels.save(path, format="PNG", optimize=True)


def extract_compact_trace(source: Path, destination: Path) -> None:
    traces = json.loads(source.read_text(encoding="utf-8"))
    selected: dict[str, tuple[tuple[float, float, float], dict[str, Any]]] = {}
    for key, payload in traces.items():
        receiver = key.split("_run_")[0]
        trace = payload.get("search_trace", [])
        if not trace:
            continue
        score = (
            round(float(payload.get("best_result", {}).get("T_packets", 0.0)), 9),
            round(float(payload.get("init_result", {}).get("T_packets", 0.0)), 9),
            -round(float(payload.get("local_gain", 0.0)), 9),
        )
        if receiver not in selected or score > selected[receiver][0]:
            selected[receiver] = (score, payload)
    rows = []
    for receiver, (_, payload) in sorted(selected.items()):
        for item in payload.get("search_trace", []):
            rows.append(
                {
                    "receiver": receiver,
                    "step": item["step"],
                    "best_T_packets": item["best_T_packets"],
                }
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["receiver", "step", "best_T_packets"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--trace-source", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    extract_compact_trace(
        args.trace_source.resolve(),
        candidate / "results" / "published" / "part2_grouped_access" / "data" / "representative_search_trace.csv",
    )
    for path in candidate.rglob("*.json"):
        sanitize_json(path)
    for path in candidate.rglob("*.csv"):
        sanitize_csv(path)
    for path in candidate.rglob("*.svg"):
        sanitize_svg(path)
    for path in candidate.rglob("*.png"):
        sanitize_png(path)


if __name__ == "__main__":
    main()
