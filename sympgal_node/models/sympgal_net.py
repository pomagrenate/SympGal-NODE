"""
Proprietary SympGal-Net Architecture:
1. Relativistic Geometric Invariant Encoder
2. Latent Symplectic Poincaré Jump Operator (Area/Volume-Preserving Canonical Map, det(J) == 1)
3. Multi-Risk Continuous-Time Survival Head (Bound, Ejection, Core Collision)
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sympgal_node.config import ModelConfig


class RelativisticGeometricInvariantEncoder(nn.Module):
    """
    Computes rigorous SO(3)-invariant geometric, relativistic, and tidal invariants
    from relative coordinates and momenta, projecting into latent representation space.
    """

    def __init__(self, in_features: int = 9, hidden_dim: int = 128, eps: float = 1e-4) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.eps = eps

        # Deep invariant edge transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Invariant attention aggregation weights across K mother stars
        self.attn_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def compute_invariants(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        m_s: Optional[torch.Tensor] = None,
        G: float = 1.0,
        c: float = 50.0,
    ) -> torch.Tensor:
        """
        Compute SO(3)-invariant scalar features between N test particles and K mother stars:
        1. ||d_ij|| = Euclidean distance
        2. ||v_ij|| = Relative velocity magnitude
        3. (d_ij . v_ij) / (||d_ij|| + eps) = Normalized radial velocity
        4. ||d_ij x v_ij|| = Specific angular momentum magnitude
        5. Phi_{1PN, ij} = Local 1PN potential depth
        6. T_{tidal, ij} = Local quadrupole tidal gravity gradient tensor magnitude
        7. M_j = Mother star mass
        8. ||q_i|| = Barycentric radius of test particle
        9. ||p_i|| = Barycentric speed of test particle
        10. m_i = Individual stellar particle mass (Kroupa IMF)
        11. M_j / m_i = Mass discrepancy ratio between primary and swarm star
        Returns shape (N, K, 11) or (N, K, 9) if in_features == 9.
        """
        N = q.shape[0]
        K = q_m.shape[0]

        # Relative separation and velocity: (N, K, 3)
        d_ij = q.unsqueeze(1) - q_m.unsqueeze(0)
        v_m = p_m / (m_m.view(-1, 1) + 1e-12)
        v_ij = p.unsqueeze(1) - v_m.unsqueeze(0)

        # 1. Distance ||d_ij||: (N, K, 1)
        d_norm = torch.norm(d_ij, dim=-1, keepdim=True)
        d_sq = d_norm**2 + self.eps**2

        # 2. Relative speed ||v_ij||: (N, K, 1)
        v_norm = torch.norm(v_ij, dim=-1, keepdim=True)

        # 3. Normalized radial velocity (d . v) / ||d||: (N, K, 1)
        d_dot_v = torch.sum(d_ij * v_ij, dim=-1, keepdim=True)
        radial_v = d_dot_v / (d_norm + self.eps)

        # 4. Angular momentum magnitude ||d x v||: (N, K, 1)
        ang_mom = torch.cross(d_ij, v_ij, dim=-1)
        ang_mom_mag = torch.norm(ang_mom, dim=-1, keepdim=True)

        # 5. Local 1PN potential depth: (N, K, 1)
        m_j = m_m.view(1, K, 1).expand(N, -1, -1)
        d_cubed = d_sq * torch.sqrt(d_sq)
        phi_1pn = -(G * m_j * (ang_mom_mag**2)) / ((c**2 + 1e-6) * d_cubed)

        # 6. Local quadrupole tidal strain: T_ab = G * M_j / d^3
        tidal_strain = (np.sqrt(6.0) * G * m_j) / d_cubed

        # 7. Mass: (N, K, 1) already m_j

        # 8 & 9. Test particle barycentric coordinates: (N, K, 1)
        r_bary = torch.norm(q, dim=-1, keepdim=True).unsqueeze(1).expand(-1, K, -1)
        v_bary = torch.norm(p, dim=-1, keepdim=True).unsqueeze(1).expand(-1, K, -1)

        inv_list = [d_norm, v_norm, radial_v, ang_mom_mag, phi_1pn, tidal_strain, m_j, r_bary, v_bary]

        # 10 & 11. Multi-scale mass discrepancy: m_i and (M_j / m_i)
        if self.in_features >= 11:
            if m_s is None:
                m_s_val = torch.ones((N, 1, 1), device=q.device, dtype=q.dtype).expand(-1, K, -1) * (1.0 / N)
            else:
                m_s_val = m_s.view(N, 1, 1).expand(-1, K, -1)
            mass_ratio = m_j / (m_s_val + 1e-12)
            inv_list.extend([m_s_val, torch.log10(torch.clamp(mass_ratio, min=1.0))])

        invariants = torch.cat(inv_list, dim=-1)
        return invariants

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        m_s: Optional[torch.Tensor] = None,
        G: float = 1.0,
        c: float = 50.0,
    ) -> torch.Tensor:
        """
        Encode raw phase-space states into SO(3)-invariant particle latent vectors.
        Returns shape (N, hidden_dim).
        """
        # (N, K, in_features)
        raw_inv = self.compute_invariants(q, p, q_m, p_m, m_m, m_s=m_s, G=G, c=c)
        edge_repr = self.edge_mlp(raw_inv)  # (N, K, hidden_dim)

        # Invariant attention weights over K mother stars
        attn_scores = self.attn_weights(edge_repr)  # (N, K, 1)
        attn_probs = F.softmax(attn_scores, dim=1)  # (N, K, 1)

        # Attention pooling over mother stars
        pooled = torch.sum(attn_probs * edge_repr, dim=1)  # (N, hidden_dim)
        out = self.node_proj(pooled)
        return out


class LatentSymplecticPoincareJump(nn.Module):
    """
    Latent Symplectic Poincaré Jump Operator.
    Implements a structure-preserving canonical symplectic map z_{k+1} = Psi_theta(z_k)
    by composing unit-triangular symplectic shear steps on partitioned canonical coordinates:
        z = [u, v] in R^{d/2} x R^{d/2}
        u1 = u + grad_v g_theta(v)
        v1 = v - grad_u f_theta(u1)
        u2 = u1 + grad_v h_theta(v1)
    Because each shear transformation has a unit-triangular Jacobian matrix with det = 1,
    the composite Jacobian has determinant IDENTICALLY EQUAL TO 1:
        det(J_Psi) == 1.0 (Exact Phase-Space Volume Preserving).
    """

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        if latent_dim % 2 != 0:
            raise ValueError(f"latent_dim must be even for symplectic partition, got {latent_dim}")

        self.latent_dim = latent_dim
        self.half_dim = latent_dim // 2

        # Scalar generating potentials g(v), f(u), h(v)
        self.potential_g = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim),
            nn.Tanh(),
            nn.Linear(self.half_dim, 1),
        )
        self.potential_f = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim),
            nn.Tanh(),
            nn.Linear(self.half_dim, 1),
        )
        self.potential_h = nn.Sequential(
            nn.Linear(self.half_dim, self.half_dim),
            nn.Tanh(),
            nn.Linear(self.half_dim, 1),
        )

    def _scalar_grad(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Compute exact gradient of scalar function net(x) with respect to x."""
        with torch.enable_grad():
            x_in = x if x.requires_grad else x.clone().requires_grad_(True)
            scalar = torch.sum(net(x_in))
            grad = torch.autograd.grad(scalar, x_in, create_graph=True)[0]
        return grad

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Symplectic Poincaré map: z -> Psi_theta(z)
        z: (N, latent_dim)
        Returns: (N, latent_dim) with det(dPsi/dz) == 1.0.
        """
        u = z[:, : self.half_dim]
        v = z[:, self.half_dim :]

        # Symplectic Shear Step 1: u1 = u + grad_v g(v)
        grad_g = self._scalar_grad(self.potential_g, v)
        u1 = u + grad_g

        # Symplectic Shear Step 2: v1 = v - grad_u f(u1)
        grad_f = self._scalar_grad(self.potential_f, u1)
        v1 = v - grad_f

        # Symplectic Shear Step 3: u2 = u1 + grad_v h(v1)
        grad_h = self._scalar_grad(self.potential_h, v1)
        u2 = u1 + grad_h

        return torch.cat([u2, v1], dim=-1)

    def compute_jacobian_determinant(self, z_samples: torch.Tensor) -> torch.Tensor:
        """
        Numerically evaluate det(J) via automatic differentiation.
        z_samples: (N, latent_dim) or (1, latent_dim).
        Returns: (N,) tensor of Jacobian determinants.
        """
        if z_samples.ndim == 1:
            z_samples = z_samples.unsqueeze(0)
        
        N, D = z_samples.shape
        det_list = []
        for i in range(N):
            z_i = z_samples[i:i+1].clone().detach().requires_grad_(True)
            z_out = self.forward(z_i)
            
            # Construct D x D Jacobian
            J_rows = []
            for d in range(D):
                grad_d = torch.autograd.grad(
                    outputs=z_out[0, d],
                    inputs=z_i,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                J_rows.append(grad_d[0])
            J = torch.stack(J_rows, dim=0)
            det_val = torch.linalg.det(J)
            det_list.append(det_val)
            
        return torch.stack(det_list, dim=0)


class MultiRiskSurvivalHead(nn.Module):
    """
    Multi-Risk Continuous-Time Survival Analysis Head.
    Predicts long-horizon right-censored competing terminal risks:
        Class 0: Bound / Trapped (Survival)
        Class 1: Hyperbolic Ejection
        Class 2: Core Tidal Collision
    Outputs discrete hazard rates h_k^(c) across time bins, cumulative survival S(t),
    and terminal risk classification probabilities.
    """

    def __init__(
        self,
        in_features: int = 128,
        n_bins: int = 20,
        t_max: float = 20.0,
        n_competing_risks: int = 2,  # 1: Ejection, 2: Collision
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_bins = n_bins
        self.t_max = t_max
        self.n_competing_risks = n_competing_risks

        self.bin_edges = torch.linspace(0.0, t_max, n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Multi-task hazard projection: (N, n_competing_risks * n_bins)
        self.hazard_mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features, n_competing_risks * n_bins),
        )

        # Terminal classification head: 3 classes (0: Bound, 1: Ejected, 2: Collided)
        self.terminal_classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(),
            nn.Linear(in_features // 2, 3),
        )

        # Expected escape time regressor
        self.escape_time_regressor = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.SiLU(),
            nn.Linear(in_features // 2, 1),
            nn.Softplus(),
        )

    def forward(
        self, features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        features: (N, in_features)
        Returns dictionary containing:
            - hazards: (N, n_risks, n_bins) in (0, 1)
            - total_survival: (N, n_bins)
            - expected_escape_times: (N, 1)
            - risk_logits: (N, 3)
            - risk_probs: (N, 3)
        """
        N = features.shape[0]

        # 1. Competing risk hazards: (N, n_risks, n_bins)
        raw_hazards = self.hazard_mlp(features).view(N, self.n_competing_risks, self.n_bins)
        hazards = torch.sigmoid(raw_hazards) * 0.5  # Ensure sum across risks < 1

        # Total event hazard at each bin: h_total = sum_c h^(c)
        total_hazard = torch.sum(hazards, dim=1)  # (N, n_bins)
        total_hazard = torch.clamp(total_hazard, min=1e-7, max=1.0 - 1e-7)

        # Cumulative survival function: S(t_k) = prod_{j=1}^k (1 - h_j^tot)
        log_surv_factors = torch.log(1.0 - total_hazard)
        cum_log_surv = torch.cumsum(log_surv_factors, dim=-1)
        total_survival = torch.exp(cum_log_surv)  # (N, n_bins)

        # 2. Expected escape time (from cause-specific ejection hazard)
        h_eject = hazards[:, 0, :]  # (N, n_bins) - Risk 1 is ejection
        s_prev = torch.cat([torch.ones((N, 1), device=features.device), total_survival[:, :-1]], dim=-1)
        f_eject = h_eject * s_prev  # Incidence density

        centers = self.bin_centers.to(device=features.device, dtype=features.dtype)
        expected_escape_times = torch.sum(f_eject * centers.unsqueeze(0), dim=-1, keepdim=True)

        # Auxiliary direct regressor
        direct_escape_time = self.escape_time_regressor(features)
        pred_times = 0.5 * (expected_escape_times + direct_escape_time)

        # 3. Terminal risk classification (Bound vs Ejected vs Collided)
        risk_logits = self.terminal_classifier(features)
        risk_probs = F.softmax(risk_logits, dim=-1)

        return {
            "hazards": hazards,
            "total_survival": total_survival,
            "expected_escape_times": pred_times,
            "risk_logits": risk_logits,
            "risk_probs": risk_probs,
        }

    def compute_loss(
        self,
        features: torch.Tensor,
        event_times: torch.Tensor,
        terminal_status: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task Loss:
        1. Competing-risk right-censored negative log-likelihood (NLL)
        2. Terminal outcome classification Cross-Entropy (Bound, Ejected, Collided)
        3. Escape time regression loss on observed ejections
        """
        preds = self.forward(features)
        hazards = preds["hazards"]  # (N, n_risks, n_bins)
        total_survival = preds["total_survival"]  # (N, n_bins)
        risk_logits = preds["risk_logits"]  # (N, 3)
        expected_times = preds["expected_escape_times"]  # (N, 1)

        device = features.device
        N = features.shape[0]

        # 1. Classification Loss (Cross Entropy)
        # terminal_status: 0 = Bound, 1 = Ejected, 2 = Collided
        clf_loss = F.cross_entropy(risk_logits, terminal_status)

        # 2. Right-censored survival NLL
        bin_edges = self.bin_edges.to(device=device)
        bin_idx = torch.bucketize(event_times, bin_edges) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins - 1)

        log_surv_factors = torch.log(torch.clamp(1.0 - torch.sum(hazards, dim=1), min=1e-7))
        cum_log_surv = torch.cumsum(log_surv_factors, dim=-1)
        cum_log_surv_padded = torch.cat([torch.zeros((N, 1), device=device), cum_log_surv[:, :-1]], dim=-1)

        log_s_prev = torch.gather(cum_log_surv_padded, 1, bin_idx.unsqueeze(1)).squeeze(1)
        log_s_at_event = torch.gather(cum_log_surv, 1, bin_idx.unsqueeze(1)).squeeze(1)

        # For ejected (status == 1):
        h_eject = torch.gather(hazards[:, 0, :], 1, bin_idx.unsqueeze(1)).squeeze(1)
        ll_ejected = torch.log(torch.clamp(h_eject, min=1e-7)) + log_s_prev

        # For collided (status == 2):
        h_collide = torch.gather(hazards[:, 1, :], 1, bin_idx.unsqueeze(1)).squeeze(1)
        ll_collided = torch.log(torch.clamp(h_collide, min=1e-7)) + log_s_prev

        # For bound/censored (status == 0):
        ll_bound = log_s_at_event

        ll = torch.where(
            terminal_status == 1,
            ll_ejected,
            torch.where(terminal_status == 2, ll_collided, ll_bound),
        )
        surv_nll = -torch.mean(ll)

        # 3. Escape time regression loss on observed ejections
        eject_mask = terminal_status == 1
        if eject_mask.any():
            reg_loss = F.mse_loss(expected_times[eject_mask].squeeze(-1), event_times[eject_mask])
        else:
            reg_loss = torch.tensor(0.0, device=device)

        total_loss = surv_nll + 0.8 * clf_loss + 0.05 * reg_loss

        return {
            "loss": total_loss,
            "surv_nll": surv_nll,
            "clf_loss": clf_loss,
            "reg_loss": reg_loss,
        }


class SympGalNet(nn.Module):
    """
    Novel SympGal-Net Architecture:
    Integrates the Relativistic Geometric Invariant Encoder,
    Latent Symplectic Poincaré Jump Operator (det(J) == 1),
    and Multi-Risk Continuous-Time Survival Head.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.hidden_dim = self.config.hidden_dim

        # 1. Invariant Encoder
        in_feat = getattr(self.config, "in_features", 11)
        self.encoder = RelativisticGeometricInvariantEncoder(
            in_features=in_feat, hidden_dim=self.hidden_dim
        )

        # 2. Canonical Symplectic Poincaré Jump Operator
        self.poincare_jump = LatentSymplecticPoincareJump(latent_dim=self.hidden_dim)

        # 3. Multi-Risk Survival Head
        self.survival_head = MultiRiskSurvivalHead(
            in_features=self.hidden_dim,
            n_bins=self.config.hazard_bins,
            t_max=20.0,
        )

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        m_s: Optional[torch.Tensor] = None,
        G: float = 1.0,
        c: float = 50.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SympGal-Net.
        Returns prediction dictionary and latent representation.
        """
        # Step 1: Geometric Relativistic Invariant Encoding
        z_enc = self.encoder(q, p, q_m, p_m, m_m, m_s=m_s, G=G, c=c)

        # Step 2: Canonical Symplectic Poincaré Jump
        z_symp = self.poincare_jump(z_enc)

        # Step 3: Multi-Risk Continuous-Time Survival Analysis
        out = self.survival_head(z_symp)
        out["latent_z"] = z_symp
        # Convenience aliases for scientific benchmark compatibility
        out["expected_escape_time"] = out["expected_escape_times"].squeeze(-1)
        out["survival_curve"] = out["total_survival"]
        out["risk_probabilities"] = out["risk_probs"]
        return out
