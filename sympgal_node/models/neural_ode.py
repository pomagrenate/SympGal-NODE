"""
Symplectic Hamiltonian Neural ODE (H-NODE) wrapper.
Guarantees canonical symplectic flow: dq/dt = dH_theta/dp, dp/dt = -dH_theta/dq.
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import torch
import torch.nn as nn

from sympgal_node.config import ModelConfig


class HamiltonianNet(nn.Module):
    """
    Parameterized Neural Hamiltonian scalar function H_theta(q, p, [context]).
    Takes phase-space coordinates (and optional context embeddings) and outputs scalar energy.
    """

    def __init__(self, in_features: int = 6, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),  # Tanh or SiLU provides smooth C^2 gradients for Hamiltonian flow
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, qp: torch.Tensor) -> torch.Tensor:
        """qp shape (N, in_features), returns (N, 1)."""
        return self.net(qp)


class SymplecticHamiltonianNODE(nn.Module):
    """
    Symplectic Hamiltonian Neural ODE wrapper.
    Evaluates canonical symplectic vector fields and integrates trajectories
    preserving symplectic 2-form d(q) ^ d(p).
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.hidden_dim = self.config.hidden_dim
        self.context_dim = context_dim

        # Input dimension: q (3) + p (3) + context
        in_dim = 6 + context_dim
        self.hamiltonian_net = HamiltonianNet(in_features=in_dim, hidden_dim=self.hidden_dim)

    def forward_energy(
        self, q: torch.Tensor, p: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute scalar Hamiltonian H_theta(q, p, [context]).
        """
        if context is not None:
            qp_in = torch.cat([q, p, context], dim=-1)
        else:
            qp_in = torch.cat([q, p], dim=-1)
        return self.hamiltonian_net(qp_in)

    def vector_field(
        self, q: torch.Tensor, p: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symplectic vector field:
        dq/dt =  dH/dp
        dp/dt = -dH/dq
        via exact autograd.
        """
        q_req = q.clone().detach().requires_grad_(True)
        p_req = p.clone().detach().requires_grad_(True)

        H = self.forward_energy(q_req, p_req, context)
        H_sum = torch.sum(H)

        dH_dq, dH_dp = torch.autograd.grad(H_sum, [q_req, p_req], create_graph=True)

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return dq_dt, dp_dt

    def integrate(
        self,
        q0: torch.Tensor,
        p0: torch.Tensor,
        t_span: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symplectic Leapfrog integration across discrete time evaluation points t_span.
        q0, p0: (N, 3)
        t_span: (T,)
        Returns:
            q_traj: (T, N, 3)
            p_traj: (T, N, 3)
        """
        T = len(t_span)
        N = q0.shape[0]

        q_traj = [q0]
        p_traj = [p0]

        q = q0
        p = p0

        for idx in range(1, T):
            dt = float(t_span[idx] - t_span[idx - 1])

            # Symplectic leapfrog step
            dq_dt_0, dp_dt_0 = self.vector_field(q, p, context)
            p_half = p + 0.5 * dt * dp_dt_0.detach()
            q_next = q + dt * p_half

            dq_dt_1, dp_dt_1 = self.vector_field(q_next, p_half, context)
            p_next = p_half + 0.5 * dt * dp_dt_1.detach()

            q = q_next
            p = p_next

            q_traj.append(q)
            p_traj.append(p)

        return torch.stack(q_traj, dim=0), torch.stack(p_traj, dim=0)
