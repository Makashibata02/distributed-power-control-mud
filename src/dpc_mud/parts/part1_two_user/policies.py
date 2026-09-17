"""Power-probability policy helpers for Part I."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import energy_levels


@dataclass(frozen=True)
class Ch2Policy:
    """A two-user power selection policy."""

    name: str
    q_arrival: float
    cond_probs: np.ndarray
    probs: np.ndarray
    powers: np.ndarray

    @property
    def pi0(self) -> float:
        """Conditional zero-power probability for a user with a packet."""
        return float(self.cond_probs[0])

    @property
    def q_tx(self) -> float:
        """Unconditional non-zero transmission probability."""
        return float(np.sum(self.probs[1:]))

    @property
    def avg_power(self) -> float:
        """Average power conditioned on the user having a packet."""
        return float(np.dot(self.cond_probs, self.powers))

    @property
    def slot_avg_power(self) -> float:
        """Unconditional per-slot average power."""
        return float(np.dot(self.probs, self.powers))


def validate_policy(policy: Ch2Policy, *, pbar: float | None = None, atol: float = 1e-9) -> None:
    probs = np.asarray(policy.probs, dtype=float)
    cond_probs = np.asarray(policy.cond_probs, dtype=float)
    powers = np.asarray(policy.powers, dtype=float)

    if probs.ndim != 1 or cond_probs.ndim != 1 or powers.ndim != 1:
        raise ValueError("probs, cond_probs, and powers must be one-dimensional.")
    if cond_probs.size != powers.size:
        raise ValueError("cond_probs and powers must have the same length.")
    if probs.size != powers.size:
        raise ValueError("probs and powers must have the same length.")
    if powers[0] != 0.0:
        raise ValueError("the first power level must be zero.")
    if not (0.0 <= policy.q_arrival <= 1.0):
        raise ValueError("q_arrival must lie in [0, 1].")
    if np.any(cond_probs < -atol):
        raise ValueError(f"negative conditional probabilities found: {cond_probs}")
    if not np.isclose(np.sum(cond_probs), 1.0, atol=atol):
        raise ValueError(f"conditional probabilities must sum to one, got {np.sum(cond_probs)}.")
    expected_probs = cond_probs * policy.q_arrival
    expected_probs[0] += 1.0 - policy.q_arrival
    if not np.allclose(probs, expected_probs, atol=atol):
        raise ValueError("effective probabilities do not match q_arrival and cond_probs.")
    if np.any(probs < -atol):
        raise ValueError(f"negative probabilities found: {probs}")
    if np.any(powers < -atol):
        raise ValueError(f"negative powers found: {powers}")
    if not np.isclose(np.sum(probs), 1.0, atol=atol):
        raise ValueError(f"probabilities must sum to one, got {np.sum(probs)}.")
    if pbar is not None and policy.avg_power > pbar + atol:
        raise ValueError(f"average power {policy.avg_power} exceeds pbar {pbar}.")


def _effective_probs(q_arrival: float, cond_probs: np.ndarray) -> np.ndarray:
    q_arrival = float(np.clip(q_arrival, 0.0, 1.0))
    cond_probs = np.asarray(cond_probs, dtype=float)
    probs = q_arrival * cond_probs
    probs[0] += 1.0 - q_arrival
    return probs


def fixed_power_policy(pbar: float, q_arrival: float, pi0: float = 0.0) -> Ch2Policy:
    """
    Single-nonzero-level power-control strategy.

    A user with a packet transmits at the fixed non-zero power Pbar.  The
    strategy itself does not optimize zero-power probability, transmission
    probability, or rescale power.
    """
    if pbar < 0:
        raise ValueError("pbar must be non-negative.")
    q_arrival = float(np.clip(q_arrival, 0.0, 1.0))
    pi0 = 0.0
    tx_cond_prob = 1.0

    cond_probs = np.array([pi0, tx_cond_prob], dtype=float)
    probs = _effective_probs(q_arrival, cond_probs)

    policy = Ch2Policy(
        name="fixed",
        q_arrival=q_arrival,
        cond_probs=cond_probs,
        probs=probs,
        powers=np.array([0.0, pbar], dtype=float),
    )
    validate_policy(policy, pbar=pbar)
    return policy


def four_point_policy(rate: float, sigma2: float, q_arrival: float, cond_pi: np.ndarray) -> Ch2Policy:
    """Conditional policy on the theoretical support {0, E1, E2, E12}."""
    e1, e2, e12 = energy_levels(rate, sigma2)
    cond_pi = np.asarray(cond_pi, dtype=float)
    policy = Ch2Policy(
        name="four_point",
        q_arrival=float(np.clip(q_arrival, 0.0, 1.0)),
        cond_probs=cond_pi,
        probs=_effective_probs(q_arrival, cond_pi),
        powers=np.array([0.0, e1, e2, e12], dtype=float),
    )
    validate_policy(policy)
    return policy


def interpolated_power_policy(
    q_arrival: float,
    cond_pi: np.ndarray,
    powers_nonzero: np.ndarray,
    name: str = "interpolated",
) -> Ch2Policy:
    """Build a conditional policy from arbitrary non-zero power support."""
    powers_nonzero = np.asarray(powers_nonzero, dtype=float)
    cond_pi = np.asarray(cond_pi, dtype=float)
    policy = Ch2Policy(
        name=name,
        q_arrival=float(np.clip(q_arrival, 0.0, 1.0)),
        cond_probs=cond_pi,
        probs=_effective_probs(q_arrival, cond_pi),
        powers=np.concatenate(([0.0], powers_nonzero)),
    )
    validate_policy(policy)
    return policy


def power_support_by_level_count(rate: float, sigma2: float, levels: int, pbar: float, q_tx: float) -> np.ndarray:
    """
    Return non-zero support for the power-level ablation.

    levels=1 is the fixed-power baseline. Higher levels interpolate over the
    two-user critical interval [E1, E12], with levels=3 exactly giving
    [E1, E2, E12].
    """
    if levels <= 0:
        raise ValueError("levels must be positive.")
    if levels == 1:
        return np.array([pbar], dtype=float)

    e1, e2, e12 = energy_levels(rate, sigma2)
    if levels == 2:
        return np.array([e1, e12], dtype=float)
    if levels == 3:
        return np.array([e1, e2, e12], dtype=float)
    return np.linspace(e1, e12, levels)


def exponential_layers_covering_two_user_thresholds(rate: float, sigma2: float, layers: int) -> np.ndarray:
    """Return S exponential layers whose endpoints match E1 and E12."""
    if layers < 2:
        raise ValueError("layers must be at least 2.")
    e1, _, e12 = energy_levels(rate, sigma2)
    ratio = (e12 / e1) ** (1.0 / (layers - 1))
    return e1 * ratio ** np.arange(layers, dtype=float)
