"""Experiment orchestration for the Part I two-user framework."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import EPS, energy_levels, gamma_from_rate
from .policies import (
    exponential_layers_covering_two_user_thresholds,
    power_support_by_level_count,
)
from .search import grid_search_fixed_power_policy, grid_search_four_point_policy, grid_search_policy, probability_grid


RECEIVERS = ("OD", "SUD", "SIC")


def _success_count_arrays(receiver: str, e1: np.ndarray, e2: np.ndarray, rate: float, sigma2: float) -> np.ndarray:
    """Vectorized two-user success count for candidate-specific powers."""
    gamma = gamma_from_rate(rate)
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    active1 = e1 > 0.0
    active2 = e2 > 0.0
    out = np.zeros_like(e1, dtype=float)

    only1 = active1 & (~active2)
    only2 = active2 & (~active1)
    both = active1 & active2
    out[only1] = (e1[only1] / sigma2 >= gamma).astype(float)
    out[only2] = (e2[only2] / sigma2 >= gamma).astype(float)

    if not np.any(both):
        return out

    a = e1[both]
    b = e2[both]

    if receiver == "OD":
        e_min, _, e_corner = energy_levels(rate, sigma2)
        e_sum = e_min + e_corner
        ok = (
            (a >= e_min - EPS)
            & (b >= e_min - EPS)
            & (a + b >= e_sum - EPS)
        )
        one_ok = (~ok) & ((a >= gamma * (sigma2 + b) - EPS) | (b >= gamma * (sigma2 + a) - EPS))
        out[both] = 2.0 * ok.astype(float) + one_ok.astype(float)
        return out

    if receiver == "SUD":
        ok1 = a >= gamma * (sigma2 + b) - EPS
        ok2 = b >= gamma * (sigma2 + a) - EPS
        out[both] = ok1.astype(float) + ok2.astype(float)
        return out

    if receiver == "SIC":
        first_if_1 = a >= gamma * (sigma2 + b) - EPS
        second_after_1 = b >= gamma * sigma2 - EPS
        count_if_1 = first_if_1.astype(float) * (1.0 + second_after_1.astype(float))

        first_if_2 = b >= gamma * (sigma2 + a) - EPS
        second_after_2 = a >= gamma * sigma2 - EPS
        count_if_2 = first_if_2.astype(float) * (1.0 + second_after_2.astype(float))
        out[both] = np.where(a >= b, count_if_1, count_if_2)
        return out

    raise ValueError(f"Unsupported receiver: {receiver}")


def _best_scaled_exponential_policy(
    *,
    receiver: str,
    q_arrival: float,
    pbar: float,
    rate: float,
    sigma2: float,
    layers: int,
    r_grid: np.ndarray,
    step: float,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """Search r and conditional probabilities for scaled exponential powers."""
    candidates_nonzero = probability_grid(layers, step)
    cond_pi = np.column_stack((1.0 - np.sum(candidates_nonzero, axis=1), candidates_nonzero))
    q_arrival = float(np.clip(q_arrival, 0.0, 1.0))

    best_score = -1.0
    best_r = 0.0
    best_support = np.zeros(layers, dtype=float)
    best_cond_pi = np.concatenate(([1.0], np.zeros(layers, dtype=float)))
    best_avg_power = 0.0

    effective_probs = q_arrival * cond_pi
    effective_probs[:, 0] += 1.0 - q_arrival

    for r in r_grid:
        weights = float(r) ** np.arange(layers, dtype=float)
        denom = candidates_nonzero @ weights
        scale = np.zeros_like(denom)
        valid = denom > 1e-12
        scale[valid] = pbar / denom[valid]

        powers = np.zeros((cond_pi.shape[0], layers + 1), dtype=float)
        powers[:, 1:] = scale[:, None] * weights[None, :]

        scores = np.zeros(cond_pi.shape[0], dtype=float)
        for i in range(layers + 1):
            for j in range(layers + 1):
                succ = _success_count_arrays(receiver, powers[:, i], powers[:, j], rate, sigma2)
                scores += effective_probs[:, i] * effective_probs[:, j] * succ
        scores *= rate

        idx = int(np.argmax(scores))
        if float(scores[idx]) > best_score:
            best_score = float(scores[idx])
            best_r = float(r)
            best_support = powers[idx, 1:].copy()
            best_cond_pi = cond_pi[idx].copy()
            best_avg_power = float(np.dot(cond_pi[idx], powers[idx]))

    return best_score, best_r, best_support, best_cond_pi, best_avg_power


@dataclass
class Ch2Config:
    rate: float = 0.8
    sigma2: float = 1.0
    pbar: float = 2.0
    grid_step: float = 0.02
    sensitivity_step: float = 0.04
    level_step: float = 0.05
    exp_compare_step: float = 0.05
    exp_compare_layer_counts: tuple[int, ...] = (2, 3, 4)
    load_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 2.0, 25))
    rate_grid: np.ndarray = field(default_factory=lambda: np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=float))
    pbar_grid: np.ndarray = field(default_factory=lambda: np.array([0.5, 1.0, 2.0, 4.0, 8.0], dtype=float))
    exp_r_grid: np.ndarray = field(
        default_factory=lambda: np.array(
            [1.02, 1.05, 1.08, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.70, 2.00],
            dtype=float,
        )
    )
    pbar_sensitivity_load: float = 1.4
    level_ablation_load: float = 1.4
    level_counts: tuple[int, ...] = (1, 2, 3, 4, 5)
    bridge_layer_counts: tuple[int, ...] = (2, 3, 4)


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write dictionaries to CSV if rows are available."""
    path.parent.mkdir(exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def experiment_params_table(cfg: Ch2Config) -> list[dict[str, float | str]]:
    return [
        {"parameter": "用户数", "symbol": "K", "value": 2, "description": "两用户随机接入"},
        {"parameter": "接收机类型", "symbol": "-", "value": "OD/SUD/SIC", "description": "最优检测、单用户检测、串行干扰消除"},
        {"parameter": "目标速率", "symbol": "R", "value": cfg.rate, "description": "bit/symbol"},
        {"parameter": "噪声功率", "symbol": "sigma^2", "value": cfg.sigma2, "description": "归一化噪声功率"},
        {"parameter": "平均功率约束", "symbol": "Pbar", "value": cfg.pbar, "description": "单用户平均功率上限"},
        {"parameter": "负载范围", "symbol": "G", "value": "[0, 2]", "description": "G=2q，q为外部消息到达概率"},
        {"parameter": "搜索步长", "symbol": "Delta pi", "value": cfg.grid_step, "description": "有消息用户的条件概率网格搜索步长"},
    ]


def experiment_load_scan(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """Compare single power with strict discrete distributed power control over load."""
    rows: list[dict[str, float | str]] = []
    e1, e2, e12 = energy_levels(cfg.rate, cfg.sigma2)

    for receiver in RECEIVERS:
        for load in cfg.load_grid:
            q_arrival = float(load / 2.0)
            fixed_result = grid_search_fixed_power_policy(
                receiver=receiver,
                q_arrival=q_arrival,
                pbar=cfg.pbar,
                rate=cfg.rate,
                sigma2=cfg.sigma2,
                step=cfg.grid_step,
            )
            fixed = fixed_result.policy
            t_fixed = fixed_result.throughput

            result = grid_search_four_point_policy(
                receiver=receiver,
                q_arrival=q_arrival,
                pbar=cfg.pbar,
                rate=cfg.rate,
                sigma2=cfg.sigma2,
                step=cfg.grid_step,
            )

            cond_pi = result.policy.cond_probs
            gain_abs = result.throughput - t_fixed
            rows.append(
                {
                    "receiver": receiver,
                    "G": float(load),
                    "q_arrival": q_arrival,
                    "R": cfg.rate,
                    "sigma2": cfg.sigma2,
                    "Pbar": cfg.pbar,
                    "T_fixed": t_fixed,
                    "T_random": result.throughput,
                    "T_single": t_fixed,
                    "T_discrete": result.throughput,
                    "gain_abs": gain_abs,
                    "gain_pct": 100.0 * gain_abs / t_fixed if t_fixed > 1e-12 else 0.0,
                    "pi0": float(cond_pi[0]),
                    "pi1": float(cond_pi[1]),
                    "pi2": float(cond_pi[2]),
                    "pi12": float(cond_pi[3]),
                    "fixed_pi0": float(fixed.cond_probs[0]),
                    "fixed_pi1": float(fixed.cond_probs[1]),
                    "fixed_power": float(fixed.powers[1]) if fixed.powers.size > 1 else 0.0,
                    "q_tx_random": result.policy.q_tx,
                    "q_tx_fixed": fixed.q_tx,
                    "avg_power_random": result.policy.avg_power,
                    "avg_power_fixed": fixed.avg_power,
                    "E1": e1,
                    "E2": e2,
                    "E12": e12,
                }
            )

    return rows


def experiment_discrete_gain_summary(
    load_rows: list[dict[str, float | str]],
    target_load: float = 1.0,
) -> list[dict[str, float | str]]:
    """Summarize discrete distributed power-control gain at the target load."""
    rows: list[dict[str, float | str]] = []

    for receiver in RECEIVERS:
        part = [row for row in load_rows if row["receiver"] == receiver]
        if not part:
            continue

        row = min(part, key=lambda item: abs(float(item["G"]) - target_load))
        t_single = float(row.get("T_single", row["T_fixed"]))
        t_discrete = float(row.get("T_discrete", row["T_random"]))
        gain_abs = t_discrete - t_single
        if abs(gain_abs) < 1e-12:
            gain_abs = 0.0
        gain_rel = "--" if t_single <= 1e-12 else f"{100.0 * gain_abs / t_single:.2f}%"

        rows.append(
            {
                "接收机": receiver,
                "负载 G": float(row["G"]),
                "单一功率控制吞吐量": t_single,
                "离散分布式功率控制吞吐量": t_discrete,
                "绝对增益": gain_abs,
                "相对增益": gain_rel,
            }
        )

    return rows


def experiment_rate_sensitivity(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """Scan R and record maximum random-power gain over the load grid."""
    rows: list[dict[str, float | str]] = []

    for rate in cfg.rate_grid:
        e1, e2, e12 = energy_levels(float(rate), cfg.sigma2)
        for receiver in RECEIVERS:
            best_gain = -1e9
            best_load = 0.0
            best_fixed = 0.0
            best_random = 0.0

            for load in cfg.load_grid:
                q_arrival = float(load / 2.0)
                fixed_result = grid_search_fixed_power_policy(
                    receiver=receiver,
                    q_arrival=q_arrival,
                    pbar=cfg.pbar,
                    rate=float(rate),
                    sigma2=cfg.sigma2,
                    step=cfg.sensitivity_step,
                )
                t_fixed = fixed_result.throughput
                result = grid_search_four_point_policy(
                    receiver=receiver,
                    q_arrival=q_arrival,
                    pbar=cfg.pbar,
                    rate=float(rate),
                    sigma2=cfg.sigma2,
                    step=cfg.sensitivity_step,
                )
                gain = result.throughput - t_fixed
                if gain > best_gain:
                    best_gain = gain
                    best_load = float(load)
                    best_fixed = t_fixed
                    best_random = result.throughput

            rows.append(
                {
                    "receiver": receiver,
                    "R": float(rate),
                    "Pbar": cfg.pbar,
                    "best_G": best_load,
                    "T_fixed_at_best_gain": best_fixed,
                    "T_random_at_best_gain": best_random,
                    "max_gain_abs": best_gain,
                    "max_gain_pct": 100.0 * best_gain / best_fixed if best_fixed > 1e-12 else 0.0,
                    "E1": e1,
                    "E2": e2,
                    "E12": e12,
                }
            )

    return rows


def experiment_pbar_sensitivity(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """Scan average power at a representative load."""
    rows: list[dict[str, float | str]] = []
    q_arrival = cfg.pbar_sensitivity_load / 2.0

    for pbar in cfg.pbar_grid:
        for receiver in RECEIVERS:
            fixed_result = grid_search_fixed_power_policy(
                receiver=receiver,
                q_arrival=q_arrival,
                pbar=float(pbar),
                rate=cfg.rate,
                sigma2=cfg.sigma2,
                step=cfg.sensitivity_step,
            )
            fixed = fixed_result.policy
            t_fixed = fixed_result.throughput
            result = grid_search_four_point_policy(
                receiver=receiver,
                q_arrival=q_arrival,
                pbar=float(pbar),
                rate=cfg.rate,
                sigma2=cfg.sigma2,
                step=cfg.sensitivity_step,
            )
            gain = result.throughput - t_fixed
            rows.append(
                {
                    "receiver": receiver,
                    "G": cfg.pbar_sensitivity_load,
                    "R": cfg.rate,
                    "Pbar": float(pbar),
                    "T_fixed": t_fixed,
                    "T_random": result.throughput,
                    "gain_abs": gain,
                    "gain_pct": 100.0 * gain / t_fixed if t_fixed > 1e-12 else 0.0,
                    "pi0": float(result.policy.cond_probs[0]),
                    "pi1": float(result.policy.cond_probs[1]),
                    "pi2": float(result.policy.cond_probs[2]),
                    "pi12": float(result.policy.cond_probs[3]),
                    "fixed_pi0": float(fixed.cond_probs[0]),
                    "fixed_power": float(fixed.powers[1]) if fixed.powers.size > 1 else 0.0,
                }
            )

    return rows


def experiment_level_ablation(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """Compare one fixed level with increasingly rich discrete supports."""
    rows: list[dict[str, float | str]] = []
    q_arrival = cfg.level_ablation_load / 2.0

    for receiver in RECEIVERS:
        for levels in cfg.level_counts:
            support = power_support_by_level_count(cfg.rate, cfg.sigma2, levels, cfg.pbar, q_arrival)

            if levels == 1:
                result = grid_search_fixed_power_policy(
                    receiver=receiver,
                    q_arrival=q_arrival,
                    pbar=cfg.pbar,
                    rate=cfg.rate,
                    sigma2=cfg.sigma2,
                    step=cfg.level_step,
                )
                policy = result.policy
                throughput = result.throughput
                avg_power = policy.avg_power
                pi0 = policy.cond_probs[0]
                support = policy.powers[1:]
            else:
                result = grid_search_policy(
                    receiver=receiver,
                    powers_nonzero=support,
                    q_arrival=q_arrival,
                    pbar=cfg.pbar,
                    rate=cfg.rate,
                    sigma2=cfg.sigma2,
                    step=cfg.level_step,
                    name=f"level_{levels}",
                )
                throughput = result.throughput
                avg_power = result.policy.avg_power
                pi0 = result.policy.pi0

            rows.append(
                {
                    "receiver": receiver,
                    "G": cfg.level_ablation_load,
                    "R": cfg.rate,
                    "Pbar": cfg.pbar,
                    "num_nonzero_levels": levels,
                    "throughput": throughput,
                    "pi0": float(pi0),
                    "avg_power": float(avg_power),
                    "power_support": " ".join(f"{x:.6g}" for x in support),
                }
            )

    return rows


def experiment_bridge_to_ch3(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """Create power-level data connecting {E1,E2,E12} to exponential layers."""
    rows: list[dict[str, float | str]] = []
    e1, e2, e12 = energy_levels(cfg.rate, cfg.sigma2)

    for label, value, idx in (("E1", e1, 1), ("E2", e2, 2), ("E12", e12, 3)):
        rows.append({"series": "two_user_threshold", "layer_count": 3, "layer_index": idx, "label": label, "power": value})

    for layers in cfg.bridge_layer_counts:
        powers = exponential_layers_covering_two_user_thresholds(cfg.rate, cfg.sigma2, layers)
        for idx, power in enumerate(powers, start=1):
            rows.append(
                {
                    "series": "exponential_layers",
                    "layer_count": layers,
                    "layer_index": idx,
                    "label": f"S={layers}",
                    "power": float(power),
                }
            )

    rows.append(
        {
            "series": "equivalent_ratio",
            "layer_count": 3,
            "layer_index": 0,
            "label": "E12/E1",
            "power": float(e12 / e1),
        }
    )
    return rows


def experiment_exponential_layer_compare(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """
    Compare the existing critical powers [E1,E2,E12] with exponential
    constructions using several nonzero layer counts.

    For each r and conditional probability candidate, P1 is scaled so that the
    exponential policy uses the average power budget:

        sum_s pi_s P1 r^(s-1) = Pbar.
    """
    rows: list[dict[str, float | str]] = []
    e1, e2, e12 = energy_levels(cfg.rate, cfg.sigma2)

    for receiver in RECEIVERS:
        for load in cfg.load_grid:
            q_arrival = float(load / 2.0)

            four_result = grid_search_four_point_policy(
                receiver=receiver,
                q_arrival=q_arrival,
                pbar=cfg.pbar,
                rate=cfg.rate,
                sigma2=cfg.sigma2,
                step=cfg.grid_step,
            )

            four_pi = four_result.policy.cond_probs

            for layers in cfg.exp_compare_layer_counts:
                best_exp_throughput, best_r, best_support, exp_pi, avg_power_exp = _best_scaled_exponential_policy(
                    receiver=receiver,
                    q_arrival=q_arrival,
                    pbar=cfg.pbar,
                    rate=cfg.rate,
                    sigma2=cfg.sigma2,
                    layers=layers,
                    r_grid=cfg.exp_r_grid,
                    step=cfg.exp_compare_step,
                )

                row: dict[str, float | str] = {
                    "receiver": receiver,
                    "G": float(load),
                    "q_arrival": q_arrival,
                    "R": cfg.rate,
                    "sigma2": cfg.sigma2,
                    "Pbar": cfg.pbar,
                    "exp_layers": layers,
                    "T_four_point": four_result.throughput,
                    "T_exp_layer": best_exp_throughput,
                    "exp_minus_four": best_exp_throughput - four_result.throughput,
                    "best_r": best_r,
                    "four_pi0": float(four_pi[0]),
                    "four_pi1": float(four_pi[1]),
                    "four_pi2": float(four_pi[2]),
                    "four_pi12": float(four_pi[3]),
                    "four_P1": e1,
                    "four_P2": e2,
                    "four_P3": e12,
                    "avg_power_four": four_result.policy.avg_power,
                    "avg_power_exp": avg_power_exp,
                    "exp_power_support": " ".join(f"{x:.10g}" for x in best_support),
                    "exp_probability": " ".join(f"{x:.10g}" for x in exp_pi),
                }

                for idx in range(4):
                    row[f"exp_pi{idx}"] = float(exp_pi[idx]) if idx < exp_pi.size else ""
                for idx in range(1, 5):
                    row[f"exp_P{idx}"] = float(best_support[idx - 1]) if idx <= best_support.size else ""

                rows.append(row)

    return rows


def experiment_exponential_three_layer_compare(cfg: Ch2Config) -> list[dict[str, float | str]]:
    """Backward-compatible alias for the multi-layer exponential comparison."""
    return experiment_exponential_layer_compare(cfg)
