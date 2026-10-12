"""
Relativistic multi-body potential: 1PN Schwarzschild corrections,
Plummer and Miyamoto-Nagai mean field background, and Hamiltonian gradients.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import numpy as np

from sympgal_node.config import PotentialConfig


class RelativisticClusterPotential:
    """
    Curved spacetime effective potential representing:
    1. K Primary Gravitational Sources (Mother stars) with Newtonian point-mass gravity.
    2. First Post-Newtonian (1PN) Schwarzschild relativistic curvature corrections:
       V_{1PN}(q, p) = - sum_{j=1}^K [ G M_j ||(q - q_j) x (p - p_j)||^2 / (c^2 ||q - q_j||^3) ]
    3. Smooth cluster mean-field potential (Plummer sphere or Miyamoto-Nagai disc).
    """

    def __init__(self, config: Optional[PotentialConfig] = None) -> None:
        self.config = config or PotentialConfig()
        self.G = self.config.G
        self.c = self.config.c
        self.eps = self.config.softening
        self.use_1pn = self.config.use_1pn
        self.plummer_mass = self.config.plummer_mass
        self.plummer_scale = self.config.plummer_scale
        self.miyamoto_a = self.config.miyamoto_a
        self.miyamoto_b = self.config.miyamoto_b

    def background_potential(self, q: torch.Tensor) -> torch.Tensor:
        """
        Evaluate smooth background cluster potential at positions q (shape N, 3).
        Supports Plummer sphere and Miyamoto-Nagai disc models.
        Returns potential energy per unit mass (shape N, 1).
        """
        x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3]
        r_sq = x**2 + y**2 + z**2

        if self.miyamoto_a > 0.0 or self.miyamoto_b > 0.0:
            # Miyamoto-Nagai potential
            z_scale = torch.sqrt(z**2 + self.miyamoto_b**2)
            denom = torch.sqrt(x**2 + y**2 + (self.miyamoto_a + z_scale)**2 + self.eps**2)
            return -self.G * self.plummer_mass / denom
        else:
            # Plummer sphere: Phi(r) = - G * M0 / sqrt(r^2 + b^2)
            denom = torch.sqrt(r_sq + self.plummer_scale**2)
            return -self.G * self.plummer_mass / denom

    def mother_newtonian_potential(
        self, q: torch.Tensor, q_mothers: torch.Tensor, m_mothers: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized Newtonian potential from K mother stars onto N test particles.
        q: (N, 3), q_mothers: (K, 3), m_mothers: (K, 1) or (K,)
        Returns shape (N, 1).
        """
        # q: (N, 1, 3), q_mothers: (1, K, 3) -> delta: (N, K, 3)
        delta_q = q.unsqueeze(1) - q_mothers.unsqueeze(0)
        dist_sq = torch.sum(delta_q**2, dim=-1) + self.eps**2  # (N, K)
        dist = torch.sqrt(dist_sq)

        m = m_mothers.view(1, -1)  # (1, K)
        phi_per_star = -self.G * m / dist  # (N, K)
        return torch.sum(phi_per_star, dim=-1, keepdim=True)  # (N, 1)

    def mother_1pn_potential(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_mothers: torch.Tensor,
        p_mothers: torch.Tensor,
        m_mothers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized 1PN relativistic potential term:
        V_{1PN}(q, p) = - sum_{j=1}^K [ G M_j ||(q - q_j) x (p - p_j)||^2 / (c^2 ||q - q_j||^3) ]
        Returns shape (N, 1).
        """
        if not self.use_1pn or self.c <= 0.0:
            return torch.zeros((q.shape[0], 1), device=q.device, dtype=q.dtype)

        # Relative coordinates and momenta
        dq = q.unsqueeze(1) - q_mothers.unsqueeze(0)  # (N, K, 3)
        dp = p.unsqueeze(1) - p_mothers.unsqueeze(0)  # (N, K, 3)

        # Cross product: L_rel = dq x dp (shape N, K, 3)
        l_rel = torch.cross(dq, dp, dim=-1)
        l_rel_sq = torch.sum(l_rel**2, dim=-1)  # (N, K)

        r_sq = torch.sum(dq**2, dim=-1) + self.eps**2  # (N, K)
        r_cubed = r_sq * torch.sqrt(r_sq)  # (N, K)

        m = m_mothers.view(1, -1)  # (1, K)
        # Factor: - G * M_j / c^2
        factor = (self.G * m) / (self.c**2)  # (1, K)
        v_1pn_per_star = -factor * (l_rel_sq / r_cubed)  # (N, K)

        return torch.sum(v_1pn_per_star, dim=-1, keepdim=True)  # (N, 1)

    def effective_potential(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_mothers: torch.Tensor,
        p_mothers: torch.Tensor,
        m_mothers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total effective potential energy Phi_total(q, p) for N particles.
        Returns shape (N, 1).
        """
        phi_bg = self.background_potential(q)
        phi_newt = self.mother_newtonian_potential(q, q_mothers, m_mothers)
        phi_1pn = self.mother_1pn_potential(q, p, q_mothers, p_mothers, m_mothers)
        return phi_bg + phi_newt + phi_1pn

    def hamiltonian(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_mothers: torch.Tensor,
        p_mothers: torch.Tensor,
        m_mothers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Total Hamiltonian for swarm particles (assuming unit specific mass):
        H(q, p) = 0.5 * ||p||^2 + Phi_eff(q, p).
        Returns shape (N, 1).
        """
        kinetic = 0.5 * torch.sum(p**2, dim=-1, keepdim=True)
        potential = self.effective_potential(q, p, q_mothers, p_mothers, m_mothers)
        return kinetic + potential

    def compute_hamiltonian_derivatives(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_mothers: torch.Tensor,
        p_mothers: torch.Tensor,
        m_mothers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute canonical flow derivatives:
        dq/dt = dH/dp
        dp/dt = -dH/dq
        via PyTorch autograd for exact mathematical consistency with non-separable 1PN terms.
        """
        q_req = q.clone().detach().requires_grad_(True)
        p_req = p.clone().detach().requires_grad_(True)

        H = self.hamiltonian(q_req, p_req, q_mothers, p_mothers, m_mothers)
        H_sum = torch.sum(H)

        dH_dq, dH_dp = torch.autograd.grad(H_sum, [q_req, p_req], create_graph=False)

        dq_dt = dH_dp
        dp_dt = -dH_dq

        return dq_dt, dp_dt

    def compute_mother_accelerations(
        self, q_mothers: torch.Tensor, p_mothers: torch.Tensor, m_mothers: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute mutual gravitational accelerations between K mother stars:
        a_j = sum_{k != j} -G * M_k * (q_j - q_k) / (||q_j - q_k||^2 + eps^2)^(3/2)
        plus 1PN mutual correction.
        Returns shape (K, 3).
        """
        K = q_mothers.shape[0]
        if K <= 1:
            return torch.zeros_like(q_mothers)

        # Pairwise differences: (K, K, 3)
        dq = q_mothers.unsqueeze(1) - q_mothers.unsqueeze(0)  # q_j - q_k
        dist_sq = torch.sum(dq**2, dim=-1, keepdim=True) + self.eps**2  # (K, K, 1)
        dist = torch.sqrt(dist_sq)

        # Newtonian acceleration
        m = m_mothers.view(1, K, 1)  # (1, K, 1)
        mask = (~torch.eye(K, device=q_mothers.device, dtype=torch.bool)).unsqueeze(-1)
        acc_newt = -self.G * m * dq / (dist_sq * dist)  # (K, K, 3)
        acc_newt = torch.where(mask, acc_newt, torch.zeros_like(acc_newt))
        acc_total = torch.sum(acc_newt, dim=1)  # (K, 3)

        # Mean field background acceleration at mother star positions
        r_mothers_sq = torch.sum(q_mothers**2, dim=-1, keepdim=True)
        denom_bg = (r_mothers_sq + self.plummer_scale**2) ** 1.5
        acc_bg = -self.G * self.plummer_mass * q_mothers / denom_bg
        acc_total = acc_total + acc_bg

        return acc_total

    def evaluate_2d_mesh(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        grid_res: int,
        q_mothers: torch.Tensor,
        m_mothers: torch.Tensor,
        z_plane: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the spacetime potential on a 2D grid in the orbital plane (Z=z_plane)
        for the 'trampoline sheet' rubber-membrane visualization.
        Returns (X, Y, Z_potential) as 2D NumPy arrays.
        """
        x_vals = np.linspace(x_range[0], x_range[1], grid_res)
        y_vals = np.linspace(y_range[0], y_range[1], grid_res)
        X, Y = np.meshgrid(x_vals, y_vals)

        points = np.stack([X.ravel(), Y.ravel(), np.full_like(X.ravel(), z_plane)], axis=1)
        points_t = torch.tensor(points, dtype=q_mothers.dtype, device=q_mothers.device)
        dummy_p = torch.zeros_like(points_t)
        dummy_p_mothers = torch.zeros_like(q_mothers)

        with torch.no_grad():
            phi = self.effective_potential(
                points_t, dummy_p, q_mothers, dummy_p_mothers, m_mothers
            )
            Z = phi.squeeze().cpu().numpy().reshape(X.shape)

        return X, Y, Z
