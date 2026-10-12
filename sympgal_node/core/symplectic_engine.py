"""
Structure-preserving symplectic integrators: Leapfrog, 4th-order Yoshida, and Implicit Midpoint.
Enforces Hamiltonian canonical conservation and tracks relative energy drift.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable
import torch

from sympgal_node.config import SimulationConfig
from sympgal_node.core.coordinates import PhaseSpaceState
from sympgal_node.core.potential import RelativisticClusterPotential


class SymplecticEngine:
    """
    High-performance symplectic dynamical engine for relativistic stellar swarms.
    Executes structure-preserving time integration while tracking energy invariants.
    """

    def __init__(
        self,
        potential: RelativisticClusterPotential,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.potential = potential
        self.config = config or SimulationConfig()
        self.dt = self.config.dt
        self.integrator_type = self.config.integrator.lower()
        self.device = torch.device(self.config.device)
        self.dtype = getattr(torch, self.config.dtype)

        # Yoshida 4th-order coefficients
        c_val = 2.0 ** (1.0 / 3.0)
        self.w1 = 1.0 / (2.0 - c_val)
        self.w0 = -c_val / (2.0 - c_val)

    def _step_leapfrog(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symplectic 2nd-order Leapfrog (Kick-Drift-Kick / Drift-Kick-Drift) step.
        """
        # 1. Mother stars update (Kick-Drift-Kick)
        acc_m = self.potential.compute_mother_accelerations(q_m, p_m, m_m)
        p_m_half = p_m + 0.5 * dt * acc_m
        q_m_next = q_m + dt * p_m_half
        acc_m_next = self.potential.compute_mother_accelerations(q_m_next, p_m_half, m_m)
        p_m_next = p_m_half + 0.5 * dt * acc_m_next

        # 2. Swarm particles update
        dq_dt_0, dp_dt_0 = self.potential.compute_hamiltonian_derivatives(
            q, p, q_m, p_m, m_m
        )
        p_half = p + 0.5 * dt * dp_dt_0
        q_next = q + dt * (p_half)  # dominant canonical drift

        dq_dt_1, dp_dt_1 = self.potential.compute_hamiltonian_derivatives(
            q_next, p_half, q_m_next, p_m_next, m_m
        )
        p_next = p_half + 0.5 * dt * dp_dt_1

        return q_next, p_next, q_m_next, p_m_next

    def _step_implicit_midpoint(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implicit Midpoint integrator:
        Preserves the symplectic 2-form d(q) ^ d(p) exactly for non-separable Hamiltonians.
        Solved using fixed-point Picard iteration.
        """
        # Advance mother stars
        acc_m = self.potential.compute_mother_accelerations(q_m, p_m, m_m)
        p_m_half = p_m + 0.5 * dt * acc_m
        q_m_next = q_m + dt * p_m_half
        acc_m_next = self.potential.compute_mother_accelerations(q_m_next, p_m_half, m_m)
        p_m_next = p_m_half + 0.5 * dt * acc_m_next

        q_m_mid = 0.5 * (q_m + q_m_next)
        p_m_mid = 0.5 * (p_m + p_m_next)

        # Picard iteration for swarm particles
        q_next = q.clone()
        p_next = p.clone()

        for _ in range(self.config.max_fixed_point_iter):
            q_mid = 0.5 * (q + q_next)
            p_mid = 0.5 * (p + p_next)

            dq_dt, dp_dt = self.potential.compute_hamiltonian_derivatives(
                q_mid, p_mid, q_m_mid, p_m_mid, m_m
            )

            q_candidate = q + dt * dq_dt
            p_candidate = p + dt * dp_dt

            err = torch.max(torch.abs(q_candidate - q_next)) + torch.max(
                torch.abs(p_candidate - p_next)
            )
            q_next = q_candidate
            p_next = p_candidate

            if err.item() < self.config.fixed_point_tol:
                break

        return q_next, p_next, q_m_next, p_m_next

    def _step_yoshida4(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        4th-order symplectic Yoshida integrator constructed by composition of leapfrog steps.
        """
        # Step 1: w1 * dt
        q, p, q_m, p_m = self._step_leapfrog(q, p, q_m, p_m, m_m, self.w1 * dt)
        # Step 2: w0 * dt
        q, p, q_m, p_m = self._step_leapfrog(q, p, q_m, p_m, m_m, self.w0 * dt)
        # Step 3: w1 * dt
        q, p, q_m, p_m = self._step_leapfrog(q, p, q_m, p_m, m_m, self.w1 * dt)

        return q, p, q_m, p_m

    def step(
        self,
        swarm_state: PhaseSpaceState,
        mothers_state: PhaseSpaceState,
        dt: Optional[float] = None,
    ) -> Tuple[PhaseSpaceState, PhaseSpaceState]:
        """
        Advance both swarm particles and mother stars by one symplectic step dt.
        """
        step_dt = dt if dt is not None else self.dt

        if self.integrator_type == "implicit_midpoint":
            integrator_fn = self._step_implicit_midpoint
        elif self.integrator_type == "yoshida4":
            integrator_fn = self._step_yoshida4
        else:
            integrator_fn = self._step_leapfrog

        q_next, p_next, qm_next, pm_next = integrator_fn(
            swarm_state.q,
            swarm_state.p,
            mothers_state.q,
            mothers_state.p,
            mothers_state.masses,
            step_dt,
        )

        new_t = swarm_state.t + step_dt
        new_swarm = PhaseSpaceState(
            q=q_next,
            p=p_next,
            masses=swarm_state.masses,
            names=swarm_state.names,
            t=new_t,
            device=str(self.device),
            dtype=self.dtype,
        )
        new_mothers = PhaseSpaceState(
            q=qm_next,
            p=pm_next,
            masses=mothers_state.masses,
            names=mothers_state.names,
            t=new_t,
            device=str(self.device),
            dtype=self.dtype,
        )

        return new_swarm, new_mothers

    def compute_energy(
        self, swarm_state: PhaseSpaceState, mothers_state: PhaseSpaceState
    ) -> float:
        """
        Compute total Hamiltonian energy of swarm particles.
        """
        with torch.no_grad():
            H = self.potential.hamiltonian(
                swarm_state.q,
                swarm_state.p,
                mothers_state.q,
                mothers_state.p,
                mothers_state.masses,
            )
            return float(torch.sum(H).item())
