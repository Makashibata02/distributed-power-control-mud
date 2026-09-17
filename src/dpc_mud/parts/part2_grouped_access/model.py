"""
Grouped power access model for Part II.

This framework is used by Part II, "Distributed Power Control for Massive
Users Based on a Grouped Model". The probability notation is pi, consistent
with Part I:

    pi_full = [pi_0, pi_1, ..., pi_S],  P_0 = 0

pi_0 is the zero-power/non-transmission probability. The total offered load is
G, and the actual transmitting load is derived from the policy:

    G_tx = G * q_tx,  q_tx = sum_{s=1}^S pi_s = 1 - pi_0

The average power constraint is

    sum_{s=0}^S pi_s P_s = sum_{s=1}^S pi_s P_s <= Pbar

The main experiments use full average power. With fixed S, pi_full contributes
S degrees of freedom and P contributes S-1 under the full-power equality, so
the search dimension is 2S-1. For S=8 this is 15.

No explicit Rayleigh fading or channel matrices are part of the Part II main
model. Poisson load Monte Carlo is validation only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import json
import warnings

import numpy as np


@dataclass(frozen=True, init=False)
class GroupedConfig:
    """System configuration for deterministic grouped-load evaluation."""

    N: int
    S: int
    R: float
    G: float
    Pbar: float
    sigma2: float
    stop_on_failure: bool
    use_full_power: bool
    q_tx_min: float
    q_tx_max: float
    Pmax: Optional[float]

    def __init__(
        self,
        *,
        N: int,
        S: int,
        R: float,
        Pbar: float,
        G: Optional[float] = None,
        G_tx: Optional[float] = None,
        sigma2: float = 1.0,
        stop_on_failure: bool = True,
        use_full_power: bool = True,
        q_tx_min: float = 0.05,
        q_tx_max: float = 1.0,
        Pmax: Optional[float] = None,
    ) -> None:
        if G is None:
            if G_tx is None:
                raise TypeError("GroupedConfig requires G. Legacy G_tx is only a fallback.")
            warnings.warn(
                "GroupedConfig received legacy G_tx without G; treating G=G_tx. "
                "New Part II code should use total offered load G.",
                RuntimeWarning,
                stacklevel=2,
            )
            G = G_tx

        object.__setattr__(self, "N", int(N))
        object.__setattr__(self, "S", int(S))
        object.__setattr__(self, "R", float(R))
        object.__setattr__(self, "G", float(G))
        object.__setattr__(self, "Pbar", float(Pbar))
        object.__setattr__(self, "sigma2", float(sigma2))
        object.__setattr__(self, "stop_on_failure", bool(stop_on_failure))
        object.__setattr__(self, "use_full_power", bool(use_full_power))
        object.__setattr__(self, "q_tx_min", float(q_tx_min))
        object.__setattr__(self, "q_tx_max", float(q_tx_max))
        object.__setattr__(self, "Pmax", None if Pmax is None else float(Pmax))
        self.validate()

    @property
    def gamma(self) -> float:
        return 2.0 ** self.R - 1.0

    @property
    def search_dimension_full_power(self) -> int:
        return 2 * self.S - 1

    def G_tx(self, policy: "GroupedPolicy") -> float:
        return float(self.G * policy.q_tx)

    def validate(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be positive, got {self.N}.")
        if self.S <= 0:
            raise ValueError(f"S must be positive, got {self.S}.")
        if self.R <= 0:
            raise ValueError(f"R must be positive, got {self.R}.")
        if self.G < 0:
            raise ValueError(f"G must be non-negative, got {self.G}.")
        if self.Pbar <= 0:
            raise ValueError(f"Pbar must be positive, got {self.Pbar}.")
        if self.sigma2 <= 0:
            raise ValueError(f"sigma2 must be positive, got {self.sigma2}.")
        if not (0.0 <= self.q_tx_min <= self.q_tx_max <= 1.0):
            raise ValueError(
                f"q_tx bounds must satisfy 0 <= min <= max <= 1, got "
                f"{self.q_tx_min}, {self.q_tx_max}."
            )
        if self.Pmax is not None and self.Pmax <= 0:
            raise ValueError(f"Pmax must be positive when set, got {self.Pmax}.")

    def with_overrides(self, **kwargs: Any) -> "GroupedConfig":
        data = {
            "N": self.N,
            "S": self.S,
            "R": self.R,
            "G": self.G,
            "Pbar": self.Pbar,
            "sigma2": self.sigma2,
            "stop_on_failure": self.stop_on_failure,
            "use_full_power": self.use_full_power,
            "q_tx_min": self.q_tx_min,
            "q_tx_max": self.q_tx_max,
            "Pmax": self.Pmax,
        }
        if "G_tx" in kwargs and "G" not in kwargs:
            kwargs["G"] = kwargs.pop("G_tx")
        data.update(kwargs)
        return GroupedConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "S": self.S,
            "R": self.R,
            "gamma": self.gamma,
            "G": self.G,
            "G_tx": "derived from policy q_tx = 1 - pi0",
            "Pbar": self.Pbar,
            "sigma2": self.sigma2,
            "stop_on_failure": self.stop_on_failure,
            "use_full_power": self.use_full_power,
            "q_tx_min": self.q_tx_min,
            "q_tx_max": self.q_tx_max,
            "Pmax": self.Pmax,
            "search_dimension_full_power": self.search_dimension_full_power,
        }


@dataclass(frozen=True, init=False)
class GroupedPolicy:
    """Grouped access policy using pi_full and positive power vector P."""

    name: str
    pi_full: np.ndarray
    P: np.ndarray
    extra: Dict[str, Any]

    def __init__(
        self,
        name: str,
        pi_full: Optional[np.ndarray] = None,
        P: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
        *,
        alpha_full: Optional[np.ndarray] = None,
        alpha: Optional[np.ndarray] = None,
    ) -> None:
        if P is None:
            raise TypeError("GroupedPolicy requires P.")
        if pi_full is None:
            if alpha_full is not None:
                pi_arr = np.asarray(alpha_full, dtype=float)
            elif alpha is not None:
                pi_arr = np.concatenate([[0.0], np.asarray(alpha, dtype=float)])
            else:
                raise TypeError("GroupedPolicy requires pi_full.")
        else:
            pi_arr = np.asarray(pi_full, dtype=float)

        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "pi_full", pi_arr)
        object.__setattr__(self, "P", np.asarray(P, dtype=float))
        object.__setattr__(self, "extra", dict(extra or {}))

    @property
    def S(self) -> int:
        return int(np.asarray(self.P).size)

    @property
    def pi0(self) -> float:
        return float(np.asarray(self.pi_full, dtype=float)[0])

    @property
    def pi_pos(self) -> np.ndarray:
        return np.asarray(self.pi_full, dtype=float)[1:].copy()

    @property
    def q_tx(self) -> float:
        return float(np.sum(self.pi_pos))

    @property
    def beta(self) -> np.ndarray:
        if self.q_tx <= 1e-15:
            return np.full(self.S, 1.0 / self.S, dtype=float)
        return self.pi_pos / self.q_tx

    @property
    def avg_power(self) -> float:
        return float(np.sum(self.pi_pos * np.asarray(self.P, dtype=float)))

    # Compatibility aliases for old experiment readers.
    @property
    def alpha_full(self) -> np.ndarray:
        return self.pi_full.copy()

    @property
    def alpha0(self) -> float:
        return self.pi0

    @property
    def alpha_pos(self) -> np.ndarray:
        return self.pi_pos

    @property
    def alpha(self) -> np.ndarray:
        return self.pi_pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pi_full": np.asarray(self.pi_full, dtype=float).tolist(),
            "pi0": self.pi0,
            "pi_pos": self.pi_pos.tolist(),
            "q_tx": self.q_tx,
            "beta": self.beta.tolist(),
            "P": np.asarray(self.P, dtype=float).tolist(),
            "avg_power": self.avg_power,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class GroupedEvalResult:
    """Result of one deterministic grouped-load evaluation."""

    method: str
    receiver: str
    T_packets: float
    T_rate: float
    T_over_G: float
    T_over_Gtx: float
    success_ratio: float
    G: float
    G_tx: float
    K0: float
    K_groups: np.ndarray
    pi_full: np.ndarray
    pi0: float
    pi_pos: np.ndarray
    q_tx: float
    beta: np.ndarray
    P: np.ndarray
    avg_power: float
    group_success_users: np.ndarray
    group_success_flags: np.ndarray
    failed_group: Optional[int]
    N: int
    S: int
    R: float
    Pbar: float
    sigma2: float
    extra: Dict[str, Any] = field(default_factory=dict)

    # Compatibility aliases.
    @property
    def alpha_full(self) -> np.ndarray:
        return self.pi_full.copy()

    @property
    def alpha0(self) -> float:
        return self.pi0

    @property
    def alpha_pos(self) -> np.ndarray:
        return self.pi_pos

    @property
    def alpha(self) -> np.ndarray:
        return self.pi_pos

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "receiver": self.receiver,
            "T_packets": float(self.T_packets),
            "T_rate": float(self.T_rate),
            "T_over_G": float(self.T_over_G),
            "T_over_Gtx": float(self.T_over_Gtx),
            "success_ratio": float(self.success_ratio),
            "G": float(self.G),
            "G_tx": float(self.G_tx),
            "K0": float(self.K0),
            "K_groups": np.asarray(self.K_groups, dtype=float).tolist(),
            "pi_full": np.asarray(self.pi_full, dtype=float).tolist(),
            "pi0": float(self.pi0),
            "pi_pos": np.asarray(self.pi_pos, dtype=float).tolist(),
            "q_tx": float(self.q_tx),
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "P": np.asarray(self.P, dtype=float).tolist(),
            "avg_power": float(self.avg_power),
            "group_success_users": np.asarray(self.group_success_users, dtype=float).tolist(),
            "group_success_flags": np.asarray(self.group_success_flags, dtype=bool).tolist(),
            "failed_group": None if self.failed_group is None else int(self.failed_group),
            "N": int(self.N),
            "S": int(self.S),
            "R": float(self.R),
            "Pbar": float(self.Pbar),
            "sigma2": float(self.sigma2),
            "extra": self.extra,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "receiver": self.receiver,
            "T_packets": float(self.T_packets),
            "T_rate": float(self.T_rate),
            "T_over_G": float(self.T_over_G),
            "T_over_Gtx": float(self.T_over_Gtx),
            "success_ratio": float(self.success_ratio),
            "G": float(self.G),
            "G_tx": float(self.G_tx),
            "K0": float(self.K0),
            "pi0": float(self.pi0),
            "q_tx": float(self.q_tx),
            "avg_power": float(self.avg_power),
            "failed_group": "" if self.failed_group is None else int(self.failed_group),
            "N": int(self.N),
            "S": int(self.S),
            "R": float(self.R),
            "Pbar": float(self.Pbar),
            "sigma2": float(self.sigma2),
            "pi_full": json.dumps(np.asarray(self.pi_full, dtype=float).tolist()),
            "pi_pos": json.dumps(np.asarray(self.pi_pos, dtype=float).tolist()),
            "beta": json.dumps(np.asarray(self.beta, dtype=float).tolist()),
            "P": json.dumps(np.asarray(self.P, dtype=float).tolist()),
            "K_groups": json.dumps(np.asarray(self.K_groups, dtype=float).tolist()),
            "group_success_users": json.dumps(np.asarray(self.group_success_users, dtype=float).tolist()),
            "group_success_flags": json.dumps(np.asarray(self.group_success_flags, dtype=bool).tolist()),
            "extra": json.dumps(self.extra, ensure_ascii=False),
        }


def validate_grouped_policy(
    policy: GroupedPolicy,
    config: GroupedConfig,
    *,
    require_full_power: Optional[bool] = None,
    atol: float = 1e-8,
) -> None:
    """Validate grouped policy constraints under the pi model."""

    config.validate()
    require_full_power = config.use_full_power if require_full_power is None else require_full_power
    pi_full = np.asarray(policy.pi_full, dtype=float)
    pi_pos = pi_full[1:]
    P = np.asarray(policy.P, dtype=float)
    q_tx = float(np.sum(pi_pos))

    if pi_full.shape != (config.S + 1,):
        raise ValueError(f"pi_full must have shape ({config.S + 1},), got {pi_full.shape}.")
    if P.shape != (config.S,):
        raise ValueError(f"P must have shape ({config.S},), got {P.shape}.")
    if not np.all(np.isfinite(pi_full)):
        raise ValueError("pi_full must contain finite values.")
    if not np.all(np.isfinite(P)):
        raise ValueError("P must contain finite values.")
    if np.any(pi_full < -atol):
        raise ValueError(f"pi_full must be non-negative, got {pi_full}.")
    if not np.isclose(float(np.sum(pi_full)), 1.0, atol=atol):
        raise ValueError(f"sum(pi_full) must be 1, got {np.sum(pi_full)}.")
    if q_tx < config.q_tx_min - atol or q_tx > config.q_tx_max + atol:
        raise ValueError(f"q_tx={q_tx} outside [{config.q_tx_min}, {config.q_tx_max}].")
    if np.any(P <= 0):
        raise ValueError(f"P must be strictly positive, got {P}.")
    if config.S > 1 and not np.all(np.diff(P) > atol):
        raise ValueError(f"P must be strictly increasing, got {P}.")
    if config.Pmax is not None and np.any(P > config.Pmax + atol):
        raise ValueError(f"P exceeds Pmax={config.Pmax}: got {P}.")

    avg_power = float(np.sum(pi_pos * P))
    if avg_power > config.Pbar + atol:
        raise ValueError(f"Average power exceeds Pbar: avg_power={avg_power}, Pbar={config.Pbar}.")
    if require_full_power and not np.isclose(avg_power, config.Pbar, atol=atol):
        raise ValueError(f"Full-power policy required: avg_power={avg_power}, Pbar={config.Pbar}.")
