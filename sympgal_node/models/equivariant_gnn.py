"""
SE(3)/SO(3) invariant graph embeddings for curvature and tidal strain in stellar swarms.
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from sympgal_node.config import ModelConfig


class EquivariantSwarmGNN(nn.Module):
    """
    Equivariant Graph Neural Network for relativistic stellar swarms.
    Constructs SO(3)/SE(3) invariant geometric and tidal representations
    from relative positions, velocities, and gravitational tidal strain.
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()

        # Invariant edge feature dimension:
        # For each mother star j and particle i:
        # [||r_ij||, ||v_ij||, <r_ij, v_ij>, ||r_ij x v_ij||, M_j, M_j / ||r_ij||^3, ||q_i||, ||p_i||] -> 8 dims
        self.raw_feature_dim = 8
        self.node_dim = self.config.node_dim
        self.hidden_dim = self.config.hidden_dim

        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.raw_feature_dim, self.node_dim),
            nn.SiLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim),
        )

        # Message passing layers
        self.interaction_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.node_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.node_dim),
                    nn.LayerNorm(self.node_dim),
                )
                for _ in range(self.config.num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def extract_invariant_features(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute SO(3)-invariant geometric and tidal invariants between N particles and K mother stars.
        q, p: (N, 3)
        q_m, p_m: (K, 3), m_m: (K, 1)
        Returns: (N, K, 8) tensor of SO(3)-invariant scalar features.
        """
        N = q.shape[0]
        K = q_m.shape[0]

        dq = q.unsqueeze(1) - q_m.unsqueeze(0)  # (N, K, 3)
        dp = p.unsqueeze(1) - p_m.unsqueeze(0)  # (N, K, 3)

        # Invariant 1: Distance ||r_ij||
        r_ij = torch.norm(dq, dim=-1, keepdim=True)  # (N, K, 1)

        # Invariant 2: Relative speed ||v_ij||
        v_ij = torch.norm(dp, dim=-1, keepdim=True)  # (N, K, 1)

        # Invariant 3: Radial dot product <r_ij, v_ij>
        rv_dot = torch.sum(dq * dp, dim=-1, keepdim=True)  # (N, K, 1)

        # Invariant 4: Relative angular momentum magnitude ||r_ij x v_ij||
        ang_mom = torch.cross(dq, dp, dim=-1)
        ang_mom_mag = torch.norm(ang_mom, dim=-1, keepdim=True)  # (N, K, 1)

        # Invariant 5: Mother star mass M_j
        mass_j = m_m.view(1, K, 1).expand(N, -1, -1)  # (N, K, 1)

        # Invariant 6: Tidal strain proxy T_tidal = M_j / (||r_ij||^3 + eps)
        tidal_strain = mass_j / (r_ij**3 + eps)  # (N, K, 1)

        # Invariant 7: Barycentric radius ||q_i||
        r_bary = torch.norm(q, dim=-1, keepdim=True).unsqueeze(1).expand(-1, K, -1)  # (N, K, 1)

        # Invariant 8: Barycentric speed ||p_i||
        v_bary = torch.norm(p, dim=-1, keepdim=True).unsqueeze(1).expand(-1, K, -1)  # (N, K, 1)

        # Concatenate into (N, K, 8)
        features = torch.cat(
            [r_ij, v_ij, rv_dot, ang_mom_mag, mass_j, tidal_strain, r_bary, v_bary],
            dim=-1,
        )
        return features

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass producing SO(3)-invariant embeddings z_i in R^{hidden_dim}
        for each swarm particle.
        Returns tensor of shape (N, hidden_dim).
        """
        # 1. Compute invariant edge features: (N, K, 8)
        raw_feats = self.extract_invariant_features(q, p, q_m, p_m, m_m)

        # 2. Encode edges: (N, K, node_dim)
        edge_emb = self.edge_encoder(raw_feats)

        # 3. Aggregate across mother stars (mean pooling over K)
        node_emb = torch.mean(edge_emb, dim=1)  # (N, node_dim)

        # 4. Message passing refinement
        for layer in self.interaction_layers:
            node_emb = node_emb + layer(node_emb)

        # 5. Output projection
        out = self.output_proj(node_emb)  # (N, hidden_dim)
        return out
