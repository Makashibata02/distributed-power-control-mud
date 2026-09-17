"""
Consistency checks for the Part II grouped-load main framework.

Run from the repository root after installation:
    python tests/test_grouped_consistency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dpc_mud.parts.part2_grouped_access.initializers import make_initial_policy  # noqa: E402
from dpc_mud.parts.part2_grouped_access.local_search import local_search_grouped_policy  # noqa: E402
from dpc_mud.parts.part2_grouped_access.model import GroupedConfig, validate_grouped_policy  # noqa: E402
from dpc_mud.parts.part2_grouped_access.receivers import evaluate_grouped_policy  # noqa: E402


def main() -> None:
    cfg = GroupedConfig(
        N=8,
        S=4,
        R=0.25,
        G=20.0,
        Pbar=1.0,
        sigma2=1.0,
        stop_on_failure=True,
        use_full_power=True,
    )
    cfg.validate()

    rng = np.random.default_rng(123)
    policy = make_initial_policy(
        cfg,
        alpha_mode="exp_decreasing",
        power_mode="exponential",
        rng=rng,
        a=1.5,
        r=2.0,
    )
    validate_grouped_policy(policy, cfg, require_full_power=True)
    assert np.isclose(np.sum(policy.pi_full), 1.0)
    assert cfg.q_tx_min <= policy.q_tx <= cfg.q_tx_max
    assert np.all(np.diff(policy.P) > 0.0)
    assert np.isclose(policy.avg_power, cfg.Pbar)

    results = {
        receiver: evaluate_grouped_policy(cfg, policy, receiver)
        for receiver in ["global_od", "group_od", "group_mf", "group_lmmse"]
    }
    for result in results.values():
        assert np.isfinite(result.T_packets)
        assert result.K_groups.shape == (cfg.S,)
        assert result.group_success_users.shape == (cfg.S,)
        assert result.group_success_flags.shape == (cfg.S,)

    assert results["global_od"].T_packets + 1e-9 >= results["group_od"].T_packets
    assert results["group_od"].T_packets + 1e-9 >= results["group_mf"].T_packets

    run, _ = local_search_grouped_policy(
        config=cfg,
        init_policy=policy,
        receiver="group_lmmse",
        num_steps=5,
        rng=np.random.default_rng(456),
        beta_scale=0.1,
        power_scale=0.1,
        q_scale=0.2,
    )
    assert run.best_result.T_packets + 1e-9 >= run.init_result.T_packets

    print("check_model_consistency: OK")


if __name__ == "__main__":
    main()
