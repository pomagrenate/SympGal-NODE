"""
Long-horizon escape time regressor and survival probability curve head.
Supports discrete-time hazard formulation for right-censored stellar evaporation data.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from sympgal_node.config import ModelConfig


class SurvivalHead(nn.Module):
    """
    Survival Analysis Head for predicting stellar evaporation:
    1. Discrete-time hazard probabilities h_k across time bins [0, t_1, ..., t_K].
    2. Survival probability curve S(t) = Prod_{j=1}^k (1 - h_j).
    3. Expected escape time E[T_escape].
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        in_features: int = 128,
        t_max: float = 20.0,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.in_features = in_features
        self.n_bins = self.config.hazard_bins
        self.t_max = t_max

        # Bin cutoffs
        self.bin_edges = torch.linspace(0.0, t_max, self.n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Neural head
        self.hazard_mlp = nn.Sequential(
            nn.Linear(in_features, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.n_bins),
        )

        # Direct escape time regression branch
        self.time_regressor = nn.Sequential(
            nn.Linear(in_features, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Softplus(),  # Guarantees positive escape time
        )

    def forward(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: (N, in_features)
        Returns:
            hazards: (N, n_bins) in (0, 1)
            survival_curves: (N, n_bins)
            expected_times: (N, 1)
        """
        logits = self.hazard_mlp(features)
        hazards = torch.sigmoid(logits)  # (N, n_bins)

        # S(t_k) = prod_{j=1}^k (1 - h_j) = exp(sum_{j=1}^k log(1 - h_j + eps))
        log_surv_factors = torch.log(torch.clamp(1.0 - hazards, min=1e-7))
        cum_log_surv = torch.cumsum(log_surv_factors, dim=-1)
        survival_curves = torch.exp(cum_log_surv)  # (N, n_bins)

        # S(t_0) = 1.0 implicitly
        # Event density: f_k = h_k * S_{k-1}
        s_prev = torch.cat(
            [torch.ones((features.shape[0], 1), device=features.device), survival_curves[:, :-1]],
            dim=-1,
        )
        pdf = hazards * s_prev  # (N, n_bins)

        # Expected time
        centers = self.bin_centers.to(device=features.device, dtype=features.dtype)
        expected_times = torch.sum(pdf * centers.unsqueeze(0), dim=-1, keepdim=True)

        return hazards, survival_curves, expected_times

    def compute_loss(
        self,
        features: torch.Tensor,
        event_times: torch.Tensor,
        is_ejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative Log-Likelihood loss for right-censored survival data.
        features: (N, in_features)
        event_times: (N,) true escape times (or t_max if censored)
        is_ejected: (N,) boolean mask (True if ejected, False if censored/bound)
        """
        hazards, survival_curves, _ = self.forward(features)
        device = features.device

        # Determine target bin index for each sample
        bin_edges = self.bin_edges.to(device=device)
        # bucketize: 1 to n_bins
        bin_idx = torch.bucketize(event_times, bin_edges) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)

        # Gather hazard at event bin
        h_at_event = torch.gather(hazards, 1, bin_idx.unsqueeze(1)).squeeze(1)  # (N,)
        h_at_event = torch.clamp(h_at_event, min=1e-7, max=1.0 - 1e-7)

        # Gather survival prior to event bin
        # S_{k-1}
        log_surv_factors = torch.log(torch.clamp(1.0 - hazards, min=1e-7))
        cum_log_surv = torch.cumsum(log_surv_factors, dim=-1)
        cum_log_surv_padded = torch.cat(
            [torch.zeros((features.shape[0], 1), device=device), cum_log_surv[:, :-1]],
            dim=-1,
        )
        log_s_prev = torch.gather(cum_log_surv_padded, 1, bin_idx.unsqueeze(1)).squeeze(1)

        # For ejected: log(f_k) = log(h_k) + log(S_{k-1})
        log_likelihood_ejected = torch.log(h_at_event) + log_s_prev

        # For censored: log(S_k) = cum_log_surv at bin_idx
        log_s_at_event = torch.gather(cum_log_surv, 1, bin_idx.unsqueeze(1)).squeeze(1)
        log_likelihood_censored = log_s_at_event

        log_likelihood = torch.where(is_ejected, log_likelihood_ejected, log_likelihood_censored)
        nll_loss = -torch.mean(log_likelihood)

        return nll_loss
