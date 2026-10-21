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
        self.total_dissipated_work: float = 0.0

        # Yoshida 4th-order coefficients
        c_val = 2.0 ** (1.0 / 3.0)
        self.w1 = 1.0 / (2.0 - c_val)
        self.w0 = -c_val / (2.0 - c_val)

    def _apply_dynamical_friction_half_step(
        self,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, float]:
        """
        Half-step dissipative kick on mother stars:
        p_m <- p_m + 0.5 * dt * M_m * a_{df}
        Returns (new_p_m, work_done).
        """
        if not self.potential.enable_dynamical_friction:
            return p_m, 0.0

        a_df = self.potential.compute_dynamical_friction(q_m, p_m, m_m)
        delta_p = 0.5 * dt * (m_m.view(-1, 1) * a_df)
        v_m = p_m / (m_m.view(-1, 1) + 1e-12)
        work_done = float(torch.sum(delta_p * v_m).item())

        p_m_new = p_m + delta_p
        return p_m_new, work_done

    def _step_leapfrog(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        spins_m: Optional[torch.Tensor],
        dt: float,
        m_swarm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Symplectic 2nd-order Leapfrog step with conservative 1PN + 1.5PN Lense-Thirring terms
        and mutual swarm self-gravity + mother back-reaction.
        """
        # 1. Mother stars update (Kick-Drift-Kick under mutual gravity and swarm pull)
        acc_m = self.potential.compute_mother_accelerations(q_m, p_m, m_m, q_swarm=q, m_swarm=m_swarm)
        p_m_half = p_m + 0.5 * dt * acc_m
        q_m_next = q_m + dt * (p_m_half / (m_m.view(-1, 1) + 1e-12))
        acc_m_next = self.potential.compute_mother_accelerations(q_m_next, p_m_half, m_m, q_swarm=q, m_swarm=m_swarm)
        p_m_next = p_m_half + 0.5 * dt * acc_m_next

        # 2. Swarm particles update
        dq_dt_0, dp_dt_0 = self.potential.compute_hamiltonian_derivatives(
            q, p, q_m, p_m, m_m, m_swarm=m_swarm
        )
        a_lt_0 = self.potential.compute_lense_thirring_acceleration(
            q, p, q_m, p_m, m_m, spins_m
        )
        p_half = p + 0.5 * dt * (dp_dt_0 + a_lt_0)
        q_next = q + dt * (p_half)

        dq_dt_1, dp_dt_1 = self.potential.compute_hamiltonian_derivatives(
            q_next, p_half, q_m_next, p_m_next, m_m, m_swarm=m_swarm
        )
        a_lt_1 = self.potential.compute_lense_thirring_acceleration(
            q_next, p_half, q_m_next, p_m_next, m_m, spins_m
        )
        p_next = p_half + 0.5 * dt * (dp_dt_1 + a_lt_1)

        return q_next, p_next, q_m_next, p_m_next

    def _step_implicit_midpoint(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        spins_m: Optional[torch.Tensor],
        dt: float,
        m_swarm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Implicit Midpoint integrator:
        Preserves the symplectic 2-form d(q) ^ d(p) exactly for non-separable Hamiltonians
        with mutual self-gravity and back-reaction.
        """
        # Advance mother stars
        acc_m = self.potential.compute_mother_accelerations(q_m, p_m, m_m, q_swarm=q, m_swarm=m_swarm)
        p_m_half = p_m + 0.5 * dt * acc_m
        q_m_next = q_m + dt * (p_m_half / (m_m.view(-1, 1) + 1e-12))
        acc_m_next = self.potential.compute_mother_accelerations(q_m_next, p_m_half, m_m, q_swarm=q, m_swarm=m_swarm)
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
                q_mid, p_mid, q_m_mid, p_m_mid, m_m, m_swarm=m_swarm
            )
            a_lt_mid = self.potential.compute_lense_thirring_acceleration(
                q_mid, p_mid, q_m_mid, p_m_mid, m_m, spins_m
            )

            q_candidate = q + dt * dq_dt
            p_candidate = p + dt * (dp_dt + a_lt_mid)

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
        spins_m: Optional[torch.Tensor],
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        4th-order symplectic Yoshida integrator constructed by composition of leapfrog steps.
        """
        # Step 1: w1 * dt
        q, p, q_m, p_m = self._step_leapfrog(q, p, q_m, p_m, m_m, spins_m, self.w1 * dt)
        # Step 2: w0 * dt
        q, p, q_m, p_m = self._step_leapfrog(q, p, q_m, p_m, m_m, spins_m, self.w0 * dt)
        # Step 3: w1 * dt
        q, p, q_m, p_m = self._step_leapfrog(q, p, q_m, p_m, m_m, spins_m, self.w1 * dt)

        return q, p, q_m, p_m

    def step(
        self,
        swarm_state: PhaseSpaceState,
        mothers_state: PhaseSpaceState,
        dt: Optional[float] = None,
    ) -> Tuple[PhaseSpaceState, PhaseSpaceState]:
        """
        Advance both swarm particles and mother stars by one time step dt
        using Strang Operator Splitting for dissipative friction + symplectic flow.
        """
        step_dt = dt if dt is not None else self.dt

        if self.integrator_type == "implicit_midpoint":
            integrator_fn = self._step_implicit_midpoint
        elif self.integrator_type == "yoshida4":
            integrator_fn = self._step_yoshida4
        else:
            integrator_fn = self._step_leapfrog

        # 1. Strang Splitting Half-Step: Dissipative Dynamical Friction Kick
        p_m_half, w1 = self._apply_dynamical_friction_half_step(
            mothers_state.q, mothers_state.p, mothers_state.masses, step_dt
        )

        # 2. Strang Splitting Full-Step: Conservative Relativistic Symplectic Step
        q_next, p_next, qm_next, pm_mid = integrator_fn(
            swarm_state.q,
            swarm_state.p,
            mothers_state.q,
            p_m_half,
            mothers_state.masses,
            mothers_state.spins,
            step_dt,
            m_swarm=swarm_state.masses,
        )

        # 3. Strang Splitting Half-Step: Dissipative Dynamical Friction Kick
        pm_next, w2 = self._apply_dynamical_friction_half_step(
            qm_next, pm_mid, mothers_state.masses, step_dt
        )

        self.total_dissipated_work += (w1 + w2)

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
            spins=mothers_state.spins,
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
                m_swarm=swarm_state.masses,
            )
            return float(torch.sum(H).item())

    def compute_total_energy(
        self, swarm_state: PhaseSpaceState, mothers_state: PhaseSpaceState
    ) -> float:
        """
        Compute total energy accounting for work dissipated by dynamical friction:
        E_total = E_canonical - W_df (where W_df <= 0, so -W_df >= 0 restores conserved total).
        """
        e_can = self.compute_energy(swarm_state, mothers_state)
        return e_can - self.total_dissipated_work
