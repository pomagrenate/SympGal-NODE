"""
Stellar cluster generator: Multiple mother stars in barycentric balance
and N child stars / test particles in a relativistic curved background.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np

from sympgal_node.config import SympGalConfig, MotherStarConfig, SwarmConfig
from sympgal_node.core.coordinates import PhaseSpaceState


class StellarCluster:
    """
    Generator and manager for stellar swarm clusters.
    Enforces barycentric momentum balance across primary gravitational sources
    and populates test particles sampled from astrophysical density profiles.
    """

    def __init__(self, config: Optional[SympGalConfig] = None) -> None:
        self.config = config or SympGalConfig()
        self.device = self.config.simulation.device
        self.dtype = getattr(torch, self.config.simulation.dtype)

        # Initialize Mother Stars and Child Swarm
        self.mothers_state = self._init_mother_stars()
        self.swarm_state = self._init_swarm_particles()

    def _init_mother_stars(self) -> PhaseSpaceState:
        """
        Build PhaseSpaceState for K mother stars and enforce barycenter alignment:
        sum(m_i * q_i) = 0 and sum(m_i * p_i) = 0.
        """
        mothers_cfg = self.config.mother_stars
        k = len(mothers_cfg)

        q_list = [m.position for m in mothers_cfg]
        p_list = [m.velocity for m in mothers_cfg]
        mass_list = [m.mass for m in mothers_cfg]
        names = [m.name for m in mothers_cfg]

        q = torch.tensor(q_list, dtype=self.dtype, device=self.device)
        p = torch.tensor(p_list, dtype=self.dtype, device=self.device)
        masses = torch.tensor(mass_list, dtype=self.dtype, device=self.device).unsqueeze(1)

        state = PhaseSpaceState(
            q=q,
            p=p,
            masses=masses,
            names=names,
            t=0.0,
            device=self.device,
            dtype=self.dtype,
        )
        state.shift_to_barycentric_frame()
        return state

    def _sample_plummer_positions(self, n: int, a: float, rng: np.random.Generator) -> np.ndarray:
        """
        Sample positions from a Plummer sphere density profile:
        rho(r) = (3 * M / 4*pi*a^3) * (1 + r^2/a^2)^(-5/2).
        Using standard inversion method: r = a / sqrt(X^(-2/3) - 1).
        """
        X = rng.uniform(0.001, 0.999, size=n)
        r = a / np.sqrt(X ** (-2.0 / 3.0) - 1.0)

        # Isotropic angles
        cos_theta = rng.uniform(-1.0, 1.0, size=n)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * cos_theta
        return np.stack([x, y, z], axis=-1)

    def _sample_disk_positions(
        self, n: int, r_in: float, r_out: float, z_scale: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample positions from a circumbinary / circumstellar disc:
        Surface density Sigma(R) ~ 1/R, vertical Gaussian scale height z_scale.
        """
        u = rng.uniform(0.0, 1.0, size=n)
        R = r_in + u * (r_out - r_in)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

        x = R * np.cos(phi)
        y = R * np.sin(phi)
        z = rng.normal(0.0, z_scale, size=n)
        return np.stack([x, y, z], axis=-1)

    def _sample_velocities(
        self,
        positions: np.ndarray,
        v_disp: float,
        rng: np.random.Generator,
        is_disk: bool = False,
    ) -> np.ndarray:
        """
        Sample velocity vectors: Keplerian rotation + isotropic velocity dispersion.
        """
        n = len(positions)
        v_random = rng.normal(0.0, v_disp, size=(n, 3))

        if is_disk:
            # Add circular azimuthal velocity v_phi ~ sqrt(G * M_tot / R)
            x, y = positions[:, 0], positions[:, 1]
            R = np.sqrt(x**2 + y**2) + 1e-6
            m_mothers_tot = sum(m.mass for m in self.config.mother_stars)
            v_circ = np.sqrt(self.config.potential.G * m_mothers_tot / R)

            v_rot_x = -v_circ * (y / R)
            v_rot_y = v_circ * (x / R)
            v_rot_z = np.zeros_like(v_rot_x)
            v_rot = np.stack([v_rot_x, v_rot_y, v_rot_z], axis=-1)
            return v_rot + v_random

        return v_random

    def _init_swarm_particles(self) -> PhaseSpaceState:
        """
        Build PhaseSpaceState for N child swarm particles.
        """
        swarm_cfg = self.config.swarm
        rng = np.random.default_rng(swarm_cfg.seed)
        n = swarm_cfg.n_particles

        if swarm_cfg.distribution == "disk":
            q_np = self._sample_disk_positions(
                n,
                swarm_cfg.disk_inner_radius,
                swarm_cfg.disk_outer_radius,
                z_scale=0.05 * swarm_cfg.cluster_radius,
                rng=rng,
            )
            p_np = self._sample_velocities(
                q_np, swarm_cfg.velocity_dispersion, rng=rng, is_disk=True
            )
        else:
            q_np = self._sample_plummer_positions(
                n, a=swarm_cfg.cluster_radius, rng=rng
            )
            p_np = self._sample_velocities(
                q_np, swarm_cfg.velocity_dispersion, rng=rng, is_disk=False
            )

        q = torch.tensor(q_np, dtype=self.dtype, device=self.device)
        p = torch.tensor(p_np, dtype=self.dtype, device=self.device)
        masses = torch.ones((n, 1), dtype=self.dtype, device=self.device) * (1.0 / n)
        names = [f"child_{i:06d}" for i in range(n)]

        return PhaseSpaceState(
            q=q,
            p=p,
            masses=masses,
            names=names,
            t=0.0,
            device=self.device,
            dtype=self.dtype,
        )

    def get_states(self) -> Tuple[PhaseSpaceState, PhaseSpaceState]:
        """Return (swarm_state, mothers_state)."""
        return self.swarm_state, self.mothers_state
