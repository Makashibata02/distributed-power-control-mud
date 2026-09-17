"""
Grouped receiver references for Part II.

All receivers process only the positive-power layers P_1,...,P_S. The zero
power layer pi_0 contributes K0 = G*pi_0 but does not create interference and
is not counted in throughput.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .model import GroupedConfig, GroupedEvalResult, GroupedPolicy, validate_grouped_policy


SUPPORTED_GROUPED_RECEIVERS = {"global_od", "group_od", "group_mf", "group_lmmse"}


def _policy_loads(config: GroupedConfig, policy: GroupedPolicy) -> tuple[float, np.ndarray, float]:
    K0 = float(config.G * policy.pi0)
    K_groups = np.asarray(config.G * policy.pi_pos, dtype=float)
    G_tx = float(np.sum(K_groups))
    return K0, K_groups, G_tx


def _build_result(
    *,
    config: GroupedConfig,
    policy: GroupedPolicy,
    receiver: str,
    group_success_users: np.ndarray,
    group_success_flags: np.ndarray,
    failed_group: Optional[int],
    extra: Dict[str, Any],
    K_groups_override: Optional[np.ndarray] = None,
    K0_override: Optional[float] = None,
) -> GroupedEvalResult:
    K0, K_groups, G_tx = _policy_loads(config, policy)
    if K_groups_override is not None:
        K_groups = np.asarray(K_groups_override, dtype=float)
        G_tx = float(np.sum(K_groups))
    if K0_override is not None:
        K0 = float(K0_override)

    T_packets = float(np.sum(group_success_users))
    T_over_G = float(T_packets / config.G) if config.G > 0 else 0.0
    T_over_Gtx = float(T_packets / G_tx) if G_tx > 0 else 0.0
    return GroupedEvalResult(
        method=policy.name,
        receiver=receiver,
        T_packets=T_packets,
        T_rate=float(config.R * T_packets),
        T_over_G=T_over_G,
        T_over_Gtx=T_over_Gtx,
        success_ratio=T_over_Gtx,
        G=config.G,
        G_tx=G_tx,
        K0=K0,
        K_groups=K_groups.copy(),
        pi_full=policy.pi_full.copy(),
        pi0=policy.pi0,
        pi_pos=policy.pi_pos.copy(),
        q_tx=policy.q_tx,
        beta=policy.beta.copy(),
        P=np.asarray(policy.P, dtype=float).copy(),
        avg_power=policy.avg_power,
        group_success_users=np.asarray(group_success_users, dtype=float),
        group_success_flags=np.asarray(group_success_flags, dtype=bool),
        failed_group=failed_group,
        N=config.N,
        S=config.S,
        R=config.R,
        Pbar=config.Pbar,
        sigma2=config.sigma2,
        extra=extra,
    )


def evaluate_global_od_upper_bound(config: GroupedConfig, policy: GroupedPolicy) -> GroupedEvalResult:
    """Global OD sum-rate upper bound over all transmitting users."""

    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    P = np.asarray(policy.P, dtype=float)
    _, K, G_tx = _policy_loads(config, policy)
    P_sum = float(np.sum(K * P))
    d = float(min(config.N, G_tx))

    if G_tx <= 0 or d <= 0:
        C_global = 0.0
        T_packets = 0.0
        full_success = True
    else:
        C_global = float(d * np.log2(1.0 + P_sum / config.sigma2))
        full_success = bool(G_tx * config.R <= C_global + 1e-12)
        T_packets = G_tx if full_success else min(G_tx, C_global / config.R)

    users = K * (T_packets / G_tx) if G_tx > 0 else np.zeros(config.S)
    flags = users >= K - 1e-9
    return _build_result(
        config=config,
        policy=policy,
        receiver="global_od",
        group_success_users=users,
        group_success_flags=flags,
        failed_group=None if full_success else -1,
        extra={
            "interpretation": "global OD upper bound based on sum-rate constraint",
            "pi0_layer": "zero-power non-transmission layer; not decoded",
            "P_sum": P_sum,
            "spatial_dof": d,
            "C_global": C_global,
            "full_success": full_success,
        },
    )


def evaluate_group_od_sic_upper_bound(config: GroupedConfig, policy: GroupedPolicy) -> GroupedEvalResult:
    """Grouped OD upper bound with high-to-low inter-group SIC."""

    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    P = np.asarray(policy.P, dtype=float)
    _, K, _ = _policy_loads(config, policy)
    users = np.zeros(config.S, dtype=float)
    flags = np.zeros(config.S, dtype=bool)
    failed_group: Optional[int] = None
    records = []

    for s in reversed(range(config.S)):
        K_s = float(K[s])
        if K_s <= 1e-12:
            flags[s] = True
            records.append({"group": s + 1, "K_s": K_s, "success": True, "reason": "empty"})
            continue
        I_s = float(np.sum(K[:s] * P[:s]))
        d_s = float(min(config.N, K_s))
        lhs = float(K_s * config.R)
        rhs = float(d_s * np.log2(1.0 + K_s * P[s] / (config.sigma2 + I_s)))
        success = bool(lhs <= rhs + 1e-12)
        flags[s] = success
        users[s] = K_s if success else 0.0
        records.append(
            {
                "group": s + 1,
                "K_s": K_s,
                "P_s": float(P[s]),
                "I_lower": I_s,
                "spatial_dof": d_s,
                "required_rate": lhs,
                "group_od_rate": rhs,
                "success": success,
            }
        )
        if not success:
            failed_group = s + 1
            if config.stop_on_failure:
                break

    return _build_result(
        config=config,
        policy=policy,
        receiver="group_od",
        group_success_users=users,
        group_success_flags=flags,
        failed_group=failed_group,
        extra={"interpretation": "grouped OD upper bound with inter-group SIC", "records": records},
    )


def evaluate_group_mf_sic(config: GroupedConfig, policy: GroupedPolicy) -> GroupedEvalResult:
    """Grouped MF/MRC performance reference with high-to-low inter-group SIC."""

    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    P = np.asarray(policy.P, dtype=float)
    _, K, _ = _policy_loads(config, policy)
    users = np.zeros(config.S, dtype=float)
    flags = np.zeros(config.S, dtype=bool)
    failed_group: Optional[int] = None
    records = []

    for s in reversed(range(config.S)):
        K_s = float(K[s])
        if K_s <= 1e-12:
            flags[s] = True
            records.append({"group": s + 1, "K_s": K_s, "success": True, "reason": "empty"})
            continue
        I_s = float(np.sum(K[:s] * P[:s]))
        intra = float(max(K_s - 1.0, 0.0) * P[s])
        sinr_s = float(config.N * P[s] / (config.sigma2 + intra + I_s))
        success = bool(sinr_s >= config.gamma - 1e-12)
        flags[s] = success
        users[s] = K_s if success else 0.0
        records.append(
            {
                "group": s + 1,
                "K_s": K_s,
                "P_s": float(P[s]),
                "I_lower": I_s,
                "intra": intra,
                "sinr_s": sinr_s,
                "gamma": float(config.gamma),
                "success": success,
            }
        )
        if not success:
            failed_group = s + 1
            if config.stop_on_failure:
                break

    return _build_result(
        config=config,
        policy=policy,
        receiver="group_mf",
        group_success_users=users,
        group_success_flags=flags,
        failed_group=failed_group,
        extra={"interpretation": "grouped MF/MRC mean-load reference with inter-group SIC", "records": records},
    )


def evaluate_group_lmmse_sic(config: GroupedConfig, policy: GroupedPolicy) -> GroupedEvalResult:
    """Grouped LMMSE mean-load approximation reference.

    This is not explicit Rayleigh-channel LMMSE simulation.
    """

    validate_grouped_policy(policy, config, require_full_power=config.use_full_power)
    P = np.asarray(policy.P, dtype=float)
    _, K, _ = _policy_loads(config, policy)
    users = np.zeros(config.S, dtype=float)
    flags = np.zeros(config.S, dtype=bool)
    failed_group: Optional[int] = None
    records = []

    for s in reversed(range(config.S)):
        K_s = float(K[s])
        if K_s <= 1e-12:
            flags[s] = True
            records.append({"group": s + 1, "K_s": K_s, "success": True, "reason": "empty"})
            continue
        I_s = float(np.sum(K[:s] * P[:s]))
        intra = float(max(K_s - 1.0, 0.0) * P[s])
        K_remain = float(np.sum(K[: s + 1]))
        eta_s = float(1.0 / (1.0 + config.N / max(K_remain, 1e-12)))
        effective_interference = float(eta_s * (intra + I_s))
        sinr_s = float(config.N * P[s] / (config.sigma2 + effective_interference))
        success = bool(sinr_s >= config.gamma - 1e-12)
        flags[s] = success
        users[s] = K_s if success else 0.0
        records.append(
            {
                "group": s + 1,
                "K_s": K_s,
                "P_s": float(P[s]),
                "I_lower": I_s,
                "intra": intra,
                "K_remain": K_remain,
                "eta_s": eta_s,
                "effective_interference": effective_interference,
                "sinr_s": sinr_s,
                "gamma": float(config.gamma),
                "success": success,
            }
        )
        if not success:
            failed_group = s + 1
            if config.stop_on_failure:
                break

    return _build_result(
        config=config,
        policy=policy,
        receiver="group_lmmse",
        group_success_users=users,
        group_success_flags=flags,
        failed_group=failed_group,
        extra={
            "interpretation": "grouped LMMSE mean-load reference with inter-group SIC",
            "lmmse_model": "mean-load approximation, not explicit Rayleigh simulation",
            "records": records,
        },
    )


def evaluate_grouped_policy(config: GroupedConfig, policy: GroupedPolicy, receiver: str) -> GroupedEvalResult:
    """Unified grouped-load evaluation entry point."""

    if receiver == "global_od":
        return evaluate_global_od_upper_bound(config, policy)
    if receiver == "group_od":
        return evaluate_group_od_sic_upper_bound(config, policy)
    if receiver == "group_mf":
        return evaluate_group_mf_sic(config, policy)
    if receiver == "group_lmmse":
        return evaluate_group_lmmse_sic(config, policy)
    raise ValueError(
        f"Unsupported grouped receiver {receiver!r}; "
        f"supported receivers are {sorted(SUPPORTED_GROUPED_RECEIVERS)}."
    )
