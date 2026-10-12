"""
Barycentric reference frame, phase-space state vectors (q, p), and celestial coordinates.
"""

from __future__ import annotations
from typing import Optional, Tuple, Union, Dict, Any
import torch
import numpy as np


class PhaseSpaceState:
    """
    Canonical phase-space state representation for an N-body or swarm system.
    Holds position q and canonical momentum/velocity p in R^{N x 3}.
    """

    def __init__(
        self,
        q: Union[torch.Tensor, np.ndarray],
        p: Union[torch.Tensor, np.ndarray],
        masses: Optional[Union[torch.Tensor, np.ndarray]] = None,
        spins: Optional[Union[torch.Tensor, np.ndarray]] = None,
        names: Optional[list[str]] = None,
        t: float = 0.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.t = float(t)

        # Normalize q
        if isinstance(q, np.ndarray):
            self.q = torch.from_numpy(q).to(device=self.device, dtype=self.dtype)
        else:
            self.q = q.to(device=self.device, dtype=self.dtype)

        # Normalize p
        if isinstance(p, np.ndarray):
            self.p = torch.from_numpy(p).to(device=self.device, dtype=self.dtype)
        else:
            self.p = p.to(device=self.device, dtype=self.dtype)

        if self.q.shape != self.p.shape:
            raise ValueError(f"Shape mismatch: q {self.q.shape} vs p {self.p.shape}")
        if self.q.ndim != 2 or self.q.shape[1] != 3:
            raise ValueError(f"Expected shape (N, 3), got {self.q.shape}")

        self.n_bodies = self.q.shape[0]

        # Normalize masses
        if masses is not None:
            if isinstance(masses, np.ndarray):
                self.masses = torch.from_numpy(masses).to(device=self.device, dtype=self.dtype)
            else:
                self.masses = masses.to(device=self.device, dtype=self.dtype)
            if self.masses.ndim == 1:
                self.masses = self.masses.unsqueeze(1)
        else:
            self.masses = torch.ones((self.n_bodies, 1), device=self.device, dtype=self.dtype)

        # Normalize spins
        if spins is not None:
            if isinstance(spins, np.ndarray):
                self.spins = torch.from_numpy(spins).to(device=self.device, dtype=self.dtype)
            else:
                self.spins = spins.to(device=self.device, dtype=self.dtype)
        else:
            self.spins = torch.zeros((self.n_bodies, 3), device=self.device, dtype=self.dtype)

        self.names = names or [f"particle_{i}" for i in range(self.n_bodies)]

    def clone(self) -> PhaseSpaceState:
        """Create a deep copy of the state."""
        return PhaseSpaceState(
            q=self.q.clone(),
            p=self.p.clone(),
            masses=self.masses.clone(),
            spins=self.spins.clone(),
            names=list(self.names),
            t=self.t,
            device=str(self.device),
            dtype=self.dtype,
        )

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> PhaseSpaceState:
        """Transfer state to another device or dtype."""
        dev = torch.device(device)
        dt = dtype or self.dtype
        return PhaseSpaceState(
            q=self.q.to(device=dev, dtype=dt),
            p=self.p.to(device=dev, dtype=dt),
            masses=self.masses.to(device=dev, dtype=dt),
            spins=self.spins.to(device=dev, dtype=dt),
            names=list(self.names),
            t=self.t,
            device=str(dev),
            dtype=dt,
        )

    def numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return position and momentum as NumPy arrays."""
        return self.q.detach().cpu().numpy(), self.p.detach().cpu().numpy()

    @property
    def r(self) -> torch.Tensor:
        """Heliocentric/Barycentric radial distances: ||q|| (shape N, 1)."""
        return torch.norm(self.q, dim=-1, keepdim=True)

    @property
    def v_sq(self) -> torch.Tensor:
        """Squared velocity ||p||^2 (shape N, 1)."""
        return torch.sum(self.p**2, dim=-1, keepdim=True)

    @property
    def kinetic_energy(self) -> torch.Tensor:
        """Kinetic energy T = 0.5 * sum(m * ||p||^2)."""
        return 0.5 * torch.sum(self.masses * torch.sum(self.p**2, dim=-1, keepdim=True))

    @property
    def angular_momentum(self) -> torch.Tensor:
        """Total or per-particle angular momentum L = q x p (shape N, 3)."""
        return torch.cross(self.q, self.p, dim=-1)

    def compute_barycenter(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Center of Mass (CoM) position and velocity:
        q_com = sum(m_i * q_i) / M_total
        p_com = sum(m_i * p_i) / M_total
        """
        total_mass = torch.sum(self.masses)
        q_com = torch.sum(self.masses * self.q, dim=0, keepdim=True) / total_mass
        p_com = torch.sum(self.masses * self.p, dim=0, keepdim=True) / total_mass
        return q_com, p_com

    def shift_to_barycentric_frame(self) -> PhaseSpaceState:
        """
        In-place shift phase-space state to ensure barycentric momentum balance:
        sum(m_i * q_i) = 0 and sum(m_i * p_i) = 0.
        """
        q_com, p_com = self.compute_barycenter()
        self.q = self.q - q_com
        self.p = self.p - p_com
        return self


class CelestialFrame:
    """
    Transforms Cartesian phase-space coordinates into Celestial Sphere coordinates:
    Celestial longitude (lambda in [-pi, pi]) and latitude (beta in [-pi/2, pi/2]),
    as well as 2D Mollweide projection coordinates (x_moll, y_moll).
    """

    @staticmethod
    def cartesian_to_spherical(
        q: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert Cartesian (x, y, z) to (r, longitude lambda, latitude beta).
        lambda in [-pi, pi], beta in [-pi/2, pi/2].
        """
        if isinstance(q, torch.Tensor):
            q_np = q.detach().cpu().numpy()
        else:
            q_np = np.asarray(q)

        x, y, z = q_np[:, 0], q_np[:, 1], q_np[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-12

        # Celestial Longitude lambda = atan2(y, x) in [-pi, pi]
        lon = np.arctan2(y, x)

        # Celestial Latitude beta = arcsin(z / r) in [-pi/2, pi/2]
        lat = np.arcsin(np.clip(z / r, -1.0, 1.0))

        return r, lon, lat

    @staticmethod
    def mollweide_projection(
        lon: np.ndarray, lat: np.ndarray, max_iter: int = 10, tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute standard Mollweide projection:
        2*theta + sin(2*theta) = pi * sin(beta)
        x = (2 * sqrt(2) / pi) * lambda * cos(theta)
        y = sqrt(2) * sin(theta)
        """
        # Newton-Raphson iteration to solve for auxiliary angle theta
        theta = lat.copy()
        pi_sin_lat = np.pi * np.sin(lat)

        for _ in range(max_iter):
            delta = -(2.0 * theta + np.sin(2.0 * theta) - pi_sin_lat) / (
                2.0 + 2.0 * np.cos(2.0 * theta) + 1e-12
            )
            theta += delta
            if np.max(np.abs(delta)) < tol:
                break

        x = (2.0 * np.sqrt(2.0) / np.pi) * lon * np.cos(theta)
        y = np.sqrt(2.0) * np.sin(theta)
        return x, y
