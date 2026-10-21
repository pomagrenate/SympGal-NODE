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

        # If exactly 2 mother stars, compute exact Keplerian orbital elements at periastron (nu = 0)
        e = self.config.swarm.binary_eccentricity
        a = self.config.swarm.binary_semi_major_axis
        if k == 2 and (e > 0.0 or a != 1.0 or not hasattr(mothers_cfg[0], "custom_orbit")):
            m1 = mothers_cfg[0].mass
            m2 = mothers_cfg[1].mass
            m_tot = m1 + m2
            G = self.config.potential.G

            # Periastron separation: r_peri = a * (1 - e)
            r_peri = a * (1.0 - e)

            # Vis-viva relative speed at periastron: v_peri = sqrt(G * M_tot * (1 + e) / (a * (1 - e)))
            v_rel = np.sqrt(G * m_tot * (1.0 + e) / max(r_peri, 1e-6))

            # Barycentric position and velocity
            q_1 = [- (m2 / m_tot) * r_peri, 0.0, 0.0]
            q_2 = [  (m1 / m_tot) * r_peri, 0.0, 0.0]

            v_1 = [0.0, - (m2 / m_tot) * v_rel, 0.0]
            v_2 = [0.0,   (m1 / m_tot) * v_rel, 0.0]

            # Canonical momentum p = m * v
            p_1 = [m1 * v_1[0], m1 * v_1[1], m1 * v_1[2]]
            p_2 = [m2 * v_2[0], m2 * v_2[1], m2 * v_2[2]]

            q_list = [q_1, q_2]
            p_list = [p_1, p_2]
        else:
            q_list = [m.position for m in mothers_cfg]
            p_list = [[m.mass * v for v in m.velocity] for m in mothers_cfg]

        mass_list = [m.mass for m in mothers_cfg]
        spin_list = [getattr(m, "spin", [0.0, 0.0, 0.1]) for m in mothers_cfg]
        names = [m.name for m in mothers_cfg]

        q = torch.tensor(q_list, dtype=self.dtype, device=self.device)
        p = torch.tensor(p_list, dtype=self.dtype, device=self.device)
        masses = torch.tensor(mass_list, dtype=self.dtype, device=self.device).unsqueeze(1)
        spins = torch.tensor(spin_list, dtype=self.dtype, device=self.device)

        state = PhaseSpaceState(
            q=q,
            p=p,
            masses=masses,
            spins=spins,
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

    def _sample_imf_masses(
        self, n: int, imf_type: str, m_min: float, m_max: float, total_mass: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample stellar masses from an astrophysical Initial Mass Function (IMF):
        1. Kroupa (2001) broken power-law:
           xi(m) ~ m^(-1.3) for 0.08 <= m < 0.5 M_sun
           xi(m) ~ m^(-2.3) for 0.5 <= m <= m_max
        2. Salpeter (1955) power-law:
           xi(m) ~ m^(-2.35)
        3. Uniform / Equal mass:
           m_i = total_mass / n
        """
        if imf_type.lower() == "salpeter":
            alpha = 2.35
            u = rng.uniform(0.0, 1.0, size=n)
            # Inverse CDF of m^(-alpha)
            exp = 1.0 - alpha
            masses = (m_min**exp + u * (m_max**exp - m_min**exp)) ** (1.0 / exp)
        elif imf_type.lower() == "kroupa":
            # Probability in low-mass vs high-mass regimes
            # f1(m) = c1 * m^(-1.3), 0.08 <= m < 0.5
            # f2(m) = c2 * m^(-2.3), 0.5 <= m <= m_max
            # Continuity at 0.5: c1 * 0.5^(-1.3) = c2 * 0.5^(-2.3) => c2 = c1 * 0.5
            c1 = 1.0
            c2 = c1 * 0.5
            int1 = (c1 / (1.0 - 1.3)) * (0.5**(1.0 - 1.3) - m_min**(1.0 - 1.3))
            int2 = (c2 / (1.0 - 2.3)) * (m_max**(1.0 - 2.3) - 0.5**(1.0 - 2.3))
            p1 = int1 / (int1 + int2)

            u = rng.uniform(0.0, 1.0, size=n)
            masses = np.zeros(n)
            mask1 = u < p1
            mask2 = ~mask1

            # Regime 1: 0.08 <= m < 0.5
            u1 = u[mask1] / p1
            exp1 = 1.0 - 1.3
            masses[mask1] = (m_min**exp1 + u1 * (0.5**exp1 - m_min**exp1)) ** (1.0 / exp1)

            # Regime 2: 0.5 <= m <= m_max
            u2 = (u[mask2] - p1) / (1.0 - p1)
            exp2 = 1.0 - 2.3
            masses[mask2] = (0.5**exp2 + u2 * (m_max**exp2 - 0.5**exp2)) ** (1.0 / exp2)
        else:
            masses = np.ones(n)

        # Normalize total swarm mass
        masses = masses / np.sum(masses) * total_mass
        return masses

    def _sample_velocities(
        self,
        positions: np.ndarray,
        v_disp: float,
        masses: np.ndarray,
        rng: np.random.Generator,
        is_disk: bool = False,
        equipartition: bool = True,
    ) -> np.ndarray:
        """
        Sample velocity vectors with energy equipartition: sigma_v_i ~ (m_mean / m_i)^0.5.
        Massive stars move slower and sink (mass segregation), light stars move faster.
        """
        n = len(positions)
        if equipartition:
            m_mean = np.mean(masses)
            scale = np.sqrt(m_mean / np.maximum(masses, 1e-7))
            scale = np.clip(scale, 0.2, 5.0).reshape(n, 1)
        else:
            scale = 1.0

        v_random = rng.normal(0.0, v_disp, size=(n, 3)) * scale

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
        Build PhaseSpaceState for N child swarm particles with realistic mass spectrum.
        """
        swarm_cfg = self.config.swarm
        rng = np.random.default_rng(swarm_cfg.seed)
        n = swarm_cfg.n_particles

        # 1. Sample realistic stellar masses (Kroupa IMF by default)
        masses_np = self._sample_imf_masses(
            n=n,
            imf_type=getattr(swarm_cfg, "imf_type", "kroupa"),
            m_min=getattr(swarm_cfg, "imf_m_min", 0.08),
            m_max=getattr(swarm_cfg, "imf_m_max", 50.0),
            total_mass=getattr(swarm_cfg, "total_swarm_mass", 0.5),
            rng=rng,
        )

        # 2. Sample positions
        if swarm_cfg.distribution == "disk":
            q_np = self._sample_disk_positions(
                n,
                swarm_cfg.disk_inner_radius,
                swarm_cfg.disk_outer_radius,
                z_scale=0.05 * swarm_cfg.cluster_radius,
                rng=rng,
            )
            p_np = self._sample_velocities(
                q_np,
                swarm_cfg.velocity_dispersion,
                masses=masses_np,
                rng=rng,
                is_disk=True,
                equipartition=getattr(swarm_cfg, "equipartition", True),
            )
        else:
            q_np = self._sample_plummer_positions(
                n, a=swarm_cfg.cluster_radius, rng=rng
            )
            p_np = self._sample_velocities(
                q_np,
                swarm_cfg.velocity_dispersion,
                masses=masses_np,
                rng=rng,
                is_disk=False,
                equipartition=getattr(swarm_cfg, "equipartition", True),
            )

        q = torch.tensor(q_np, dtype=self.dtype, device=self.device)
        # Canonical momentum p = m * v
        p_canonical = p_np * masses_np.reshape(-1, 1)
        p = torch.tensor(p_canonical, dtype=self.dtype, device=self.device)
        masses = torch.tensor(masses_np, dtype=self.dtype, device=self.device).reshape(n, 1)
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
