"""I/O helpers for Part III experiments."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, GroupedPolicy
from dpc_mud.parts.part2_grouped_access.experiment_utils import json_default, portable_value


def created_at() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(portable_value(data), f, ensure_ascii=False, indent=2, default=json_default)


def make_result_dir(project_root: Path, root_name: str, prefix: str) -> Path:
    override = os.environ.get("DPC_MUD_OUTPUT_ROOT")
    root = Path(override) if override else project_root / root_name
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{prefix}_{stamp}"
    for idx in range(1000):
        path = base if idx == 0 else root / f"{base.name}_{idx:02d}"
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(base)


def config_from_mapping(data: Dict[str, Any]) -> GroupedConfig:
    system = data.get("system", data)
    return GroupedConfig(
        N=int(system.get("N", 8)),
        S=int(system.get("S", 8)),
        R=float(system.get("R", 0.25)),
        G=float(system.get("G", 200.0)),
        Pbar=float(system.get("Pbar", 1.0)),
        sigma2=float(system.get("sigma2", 1.0)),
        stop_on_failure=bool(system.get("stop_on_failure", True)),
        use_full_power=bool(system.get("use_full_power", True)),
        q_tx_min=float(system.get("q_tx_min", 0.05)),
        q_tx_max=float(system.get("q_tx_max", 1.0)),
        Pmax=system.get("Pmax", None),
    )


def load_ch3_best_policies(ch3_dir: Path) -> tuple[Dict[str, Any], Dict[str, Dict[str, GroupedPolicy]]]:
    data = json.loads((ch3_dir / "ch3_best_policies.json").read_text(encoding="utf-8"))
    best = data["best_policies"]
    policies: Dict[str, Dict[str, GroupedPolicy]] = {}
    for receiver, receiver_payload in best.items():
        policies[receiver] = {}
        for baseline_type, payload in receiver_payload.items():
            policy_dict = payload["policy"]
            policies[receiver][baseline_type] = GroupedPolicy(
                name=str(policy_dict.get("name", baseline_type)),
                pi_full=np.asarray(policy_dict["pi_full"], dtype=float),
                P=np.asarray(policy_dict["P"], dtype=float),
                extra=dict(policy_dict.get("extra", {})),
            )
    return data.get("metadata", {}), policies


def flatten_ch3_best_payloads(ch3_dir: Path) -> list[dict[str, Any]]:
    data = json.loads((ch3_dir / "ch3_best_policies.json").read_text(encoding="utf-8"))
    rows = []
    for receiver, receiver_payload in data["best_policies"].items():
        for baseline_type, payload in receiver_payload.items():
            rows.append({"receiver": receiver, "baseline_type": baseline_type, **payload})
    return rows
