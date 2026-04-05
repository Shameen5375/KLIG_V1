"""
Path parameterizations in (mu, logvar) space.

A path defines how the distribution parameters evolve from the prior
(mu=0, logvar=0, i.e. N(0,1)) to the final distribution at t=1.

Available paths
---------------
LinearPath          -- linear ramp in (mu, logvar); the baseline
PowerPath(gamma)    -- both parameters scale as t^gamma
                       gamma>1: slow start, attribution concentrated late
                       gamma<1: fast start, similar to IG character
DecoupledPath(a,b)  -- mu scales as t^alpha, logvar as t^beta independently
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch


class DistributionPath(ABC):
    """Abstract base for paths through (mu, logvar) space."""

    @abstractmethod
    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu_t, logvar_t) at position t in [0, 1]."""
        ...

    @abstractmethod
    def steps(self, n: int) -> torch.Tensor:
        """Return the t values to evaluate for numerical integration (length n)."""
        ...

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (d_mu_dt, d_logvar_dt) -- the instantaneous rate of change of
        the path parameters at time t.
        """
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Transform model-input x into the space where (mu, logvar) live."""
        return x

    def decode_sample(self, raw_sample: torch.Tensor) -> torch.Tensor:
        """Transform a batch of reparameterized samples back to model input space."""
        return raw_sample

    def compute_endpoint_kl(
        self,
        x: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> torch.Tensor | None:
        """Per-element KL divergence of the endpoint distribution from the reference."""
        return None

    def decode_attribution(self, attr_encoded: torch.Tensor) -> torch.Tensor:
        """Transform attribution from encoded space back to reporting space."""
        return attr_encoded


class LinearPath(DistributionPath):
    """
    Linear interpolation in (mu, logvar) space.

    mu(t)     = t * mu_final
    logvar(t) = t * logvar_final
    """

    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return t * mu_final, t * logvar_final

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mu_final, logvar_final

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)


class PowerPath(DistributionPath):
    """
    Non-linear path where both mu and logvar scale as t^gamma.

    gamma > 1: slow start, fast finish -- attribution concentrated late.
    gamma < 1: fast start, slow finish.
    gamma = 1: identical to LinearPath.
    """

    def __init__(self, gamma: float = 2.0) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma

    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = t ** self.gamma
        return scale * mu_final, scale * logvar_final

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rate = self.gamma * (t ** (self.gamma - 1))
        return rate * mu_final, rate * logvar_final

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)


class DecoupledPath(DistributionPath):
    """
    Path where mu and logvar evolve at independent rates.

    mu(t)     = t^alpha * mu_final
    logvar(t) = t^beta  * logvar_final
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError(f"alpha and beta must be >= 0, got alpha={alpha}, beta={beta}")
        self.alpha = alpha
        self.beta = beta

    def at(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_t = (t ** self.alpha) * mu_final
        logvar_t = (t ** self.beta) * logvar_final
        return mu_t, logvar_t

    def derivatives(
        self,
        t: float,
        mu_final: torch.Tensor,
        logvar_final: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dmu = (self.alpha * t ** (self.alpha - 1)) * mu_final if self.alpha > 0 else torch.zeros_like(mu_final)
        dlogvar = (self.beta * t ** (self.beta - 1)) * logvar_final if self.beta > 0 else torch.zeros_like(logvar_final)
        return dmu, dlogvar

    def steps(self, n: int) -> torch.Tensor:
        return torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)
