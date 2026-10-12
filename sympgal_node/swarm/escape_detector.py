"""
Escape horizon monitor: Hyperbolic ejection (e >= 1.0, E > 0),
tidal evaporation beyond R_tidal, and mother-star collision detection.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from enum import IntEnum
import torch
import numpy as np

from sympgal_node.config import SwarmConfig
from sympgal_node.core.coordinates import PhaseSpaceState, CelestialFrame
from sympgal_node.core.potential import RelativisticClusterPotential


class ParticleStatus(IntEnum):
    BOUND = 0
    EJECTED = 1
    COLLIDED = 2


class EscapeDetector:
    """
    Monitors swarm test particles across time steps to detect:
    1. Ejection / Tidal Evaporation: Specific orbital energy E_i > 0 and r_i > R_tidal.
    2. Collision / Tidal capture: Particle distance to mother star < R_collision.
    Records escape timestamps, velocities, and celestial exit angles.
    """

    def __init__(
        self,
        potential: RelativisticClusterPotential,
        config: Optional[SwarmConfig] = None,
        n_particles: int = 1000,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.potential = potential
        self.config = config or SwarmConfig()
        self.n_particles = n_particles
        self.device = torch.device(device)
        self.dtype = dtype

        self.r_tidal = self.config.r_tidal
        self.collision_radius = self.config.collision_radius

        # Status tracking tensors
        self.status = torch.full(
            (n_particles,), ParticleStatus.BOUND, dtype=torch.long, device=self.device
        )
        self.escape_time = torch.full(
            (n_particles,), float("inf"), dtype=self.dtype, device=self.device
        )
        self.escape_position = torch.zeros(
            (n_particles, 3), dtype=self.dtype, device=self.device
        )
        self.escape_velocity = torch.zeros(
            (n_particles, 3), dtype=self.dtype, device=self.device
        )
        self.escape_celestial = torch.zeros(
            (n_particles, 2), dtype=self.dtype, device=self.device
        )  # (lambda, beta)

    def update(
        self,
        t: float,
        swarm: PhaseSpaceState,
        mothers: PhaseSpaceState,
    ) -> Tuple[int, int]:
        """
        Check all active bound particles at time t.
        Returns (num_newly_ejected, num_newly_collided).
        """
        bound_mask = self.status == ParticleStatus.BOUND
        if not bound_mask.any():
            return 0, 0

        q = swarm.q
        p = swarm.p

        # 1. Collision check with mother stars
        # Distance between particles and each mother star: (N, K)
        dq_mothers = q.unsqueeze(1) - mothers.q.unsqueeze(0)  # (N, K, 3)
        dist_mothers = torch.norm(dq_mothers, dim=-1)  # (N, K)
        min_dist_to_mother, _ = torch.min(dist_mothers, dim=-1)  # (N,)

        new_collided = bound_mask & (min_dist_to_mother < self.collision_radius)

        # 2. Escape / Tidal Evaporation check
        r_bary = torch.norm(q, dim=-1)  # (N,)
        with torch.no_grad():
            phi_eff = self.potential.effective_potential(
                q, p, mothers.q, mothers.p, mothers.masses
            ).squeeze(-1)  # (N,)
            v_sq = torch.sum(p**2, dim=-1)  # (N,)
            spec_energy = 0.5 * v_sq + phi_eff  # Specific orbital energy E = 0.5*v^2 + Phi

        # Ejected if beyond tidal radius and has positive specific energy
        new_ejected = bound_mask & (~new_collided) & (r_bary > self.r_tidal) & (spec_energy > 0.0)

        # Apply updates for collided
        if new_collided.any():
            self.status[new_collided] = ParticleStatus.COLLIDED
            self.escape_time[new_collided] = t
            self.escape_position[new_collided] = q[new_collided]
            self.escape_velocity[new_collided] = p[new_collided]

        # Apply updates for ejected
        if new_ejected.any():
            self.status[new_ejected] = ParticleStatus.EJECTED
            self.escape_time[new_ejected] = t
            self.escape_position[new_ejected] = q[new_ejected]
            self.escape_velocity[new_ejected] = p[new_ejected]

            # Compute celestial angles for ejected particles
            _, lon, lat = CelestialFrame.cartesian_to_spherical(q[new_ejected])
            self.escape_celestial[new_ejected, 0] = torch.tensor(lon, device=self.device, dtype=self.dtype)
            self.escape_celestial[new_ejected, 1] = torch.tensor(lat, device=self.device, dtype=self.dtype)

        return int(new_ejected.sum().item()), int(new_collided.sum().item())

    def get_summary(self) -> Dict[str, Any]:
        """Summary metrics of swarm survival."""
        total = self.n_particles
        ejected = int((self.status == ParticleStatus.EJECTED).sum().item())
        collided = int((self.status == ParticleStatus.COLLIDED).sum().item())
        bound = int((self.status == ParticleStatus.BOUND).sum().item())

        esc_times = self.escape_time[self.status == ParticleStatus.EJECTED].cpu().numpy()
        median_esc_time = float(np.median(esc_times)) if len(esc_times) > 0 else None

        return {
            "total_particles": total,
            "bound_count": bound,
            "ejected_count": ejected,
            "collided_count": collided,
            "bound_fraction": bound / total,
            "escape_fraction": ejected / total,
            "collision_fraction": collided / total,
            "median_escape_time": median_esc_time,
        }

    def export_dataset(self) -> Dict[str, np.ndarray]:
        """
        Export array dictionary suitable for training survival analysis & neural models.
        """
        return {
            "status": self.status.cpu().numpy(),
            "escape_time": self.escape_time.cpu().numpy(),
            "escape_position": self.escape_position.cpu().numpy(),
            "escape_velocity": self.escape_velocity.cpu().numpy(),
            "escape_celestial": self.escape_celestial.cpu().numpy(),
        }
