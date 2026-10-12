"""
Unit and integration tests for SympGal-NODE framework.
"""

import torch

import numpy as np

from sympgal_node.config import SympGalConfig, PotentialConfig, SwarmConfig, SimulationConfig
from sympgal_node.core.coordinates import PhaseSpaceState, CelestialFrame
from sympgal_node.core.potential import RelativisticClusterPotential
from sympgal_node.core.symplectic_engine import SymplecticEngine
from sympgal_node.swarm.cluster import StellarCluster
from sympgal_node.swarm.escape_detector import EscapeDetector, ParticleStatus
from sympgal_node.models.equivariant_gnn import EquivariantSwarmGNN
from sympgal_node.models.neural_ode import SymplecticHamiltonianNODE
from sympgal_node.models.survival_head import SurvivalHead


def test_barycentric_frame():
    """Verify center of mass and barycentric frame momentum balance."""
    q = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=torch.float32)
    m = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

    state = PhaseSpaceState(q, p, masses=m)
    q_com, p_com = state.compute_barycenter()
    assert torch.allclose(q_com, torch.tensor([[2.0, 0.0, 0.0]]))
    assert torch.allclose(p_com, torch.tensor([[0.0, 0.0, 0.0]]))

    state.shift_to_barycentric_frame()
    q_com_shifted, p_com_shifted = state.compute_barycenter()
    assert torch.allclose(q_com_shifted, torch.zeros_like(q_com_shifted), atol=1e-6)
    assert torch.allclose(p_com_shifted, torch.zeros_like(p_com_shifted), atol=1e-6)


def test_celestial_coordinates():
    """Verify Cartesian to Spherical and Mollweide projection."""
    # Point on X axis -> lon=0, lat=0
    q = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    r, lon, lat = CelestialFrame.cartesian_to_spherical(q)

    assert np.isclose(lon[0], 0.0)
    assert np.isclose(lat[0], 0.0)
    assert np.isclose(lon[1], np.pi / 2)
    assert np.isclose(lat[2], np.pi / 2)

    x_moll, y_moll = CelestialFrame.mollweide_projection(lon, lat)
    assert len(x_moll) == 3
    assert len(y_moll) == 3


def test_relativistic_potential_and_1pn():
    """Verify effective potential computation and 1PN correction term."""
    pot_cfg = PotentialConfig(G=1.0, c=50.0, use_1pn=True)
    pot = RelativisticClusterPotential(pot_cfg)

    # Test particles
    q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32)

    # Mother stars
    qm = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    pm = torch.tensor([[0.0, -0.7, 0.0], [0.0, 0.7, 0.0]], dtype=torch.float32)
    mm = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

    phi = pot.effective_potential(q, p, qm, pm, mm)
    assert phi.shape == (2, 1)
    assert torch.all(phi < 0.0)  # Gravitational potential is attractive / negative

    # Canonical flow derivatives
    dq_dt, dp_dt = pot.compute_hamiltonian_derivatives(q, p, qm, pm, mm)
    assert dq_dt.shape == (2, 3)
    assert dp_dt.shape == (2, 3)


def test_symplectic_integrator():
    """Verify symplectic integration step and energy tracking."""
    cfg = SympGalConfig()
    cfg.simulation.dt = 0.01
    cfg.simulation.integrator = "leapfrog"
    cfg.swarm.n_particles = 100

    cluster = StellarCluster(cfg)
    swarm_state, mothers_state = cluster.get_states()
    pot = RelativisticClusterPotential(cfg.potential)
    engine = SymplecticEngine(pot, cfg.simulation)

    e0 = engine.compute_energy(swarm_state, mothers_state)

    for _ in range(10):
        swarm_state, mothers_state = engine.step(swarm_state, mothers_state)

    e1 = engine.compute_energy(swarm_state, mothers_state)
    rel_error = abs((e1 - e0) / e0)
    assert rel_error < 0.05  # Energy conservation to high precision


def test_equivariant_gnn():
    """Verify SE(3)/SO(3) invariance of the GNN embedding."""
    gnn = EquivariantSwarmGNN()

    N = 5
    q = torch.randn(N, 3)
    p = torch.randn(N, 3)
    qm = torch.tensor([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    pm = torch.tensor([[0.0, -0.7, 0.0], [0.0, 0.7, 0.0]])
    mm = torch.tensor([[1.0], [1.0]])

    z_original = gnn(q, p, qm, pm, mm)

    # Apply 3D orthogonal rotation matrix R (SO(3))
    theta = 0.785398  # 45 degrees
    c, s = np.cos(theta), np.sin(theta)
    R = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    q_rot = torch.matmul(q, R.T)
    p_rot = torch.matmul(p, R.T)
    qm_rot = torch.matmul(qm, R.T)
    pm_rot = torch.matmul(pm, R.T)

    z_rotated = gnn(q_rot, p_rot, qm_rot, pm_rot, mm)

    # Node embeddings should be invariant to global SO(3) rotations
    assert torch.allclose(z_original, z_rotated, atol=1e-4)


def test_survival_head():
    """Verify SurvivalHead hazard and curve outputs and NLL loss computation."""
    head = SurvivalHead(in_features=64, t_max=20.0)
    x = torch.randn(20, 64)

    hazards, surv_curve, exp_times = head(x)
    assert hazards.shape == (20, head.n_bins)
    assert surv_curve.shape == (20, head.n_bins)
    assert exp_times.shape == (20, 1)

    # Monotonicity check of survival function S(t_k) <= S(t_{k-1})
    diffs = surv_curve[:, 1:] - surv_curve[:, :-1]
    assert torch.all(diffs <= 1e-6)

    # Test NLL loss
    times = torch.rand(20) * 20.0
    is_ejected = torch.randint(0, 2, (20,)).bool()
    loss = head.compute_loss(x, times, is_ejected)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_hamiltonian_node():
    """Verify Symplectic Hamiltonian NODE vector field and trajectory integration."""
    node = SymplecticHamiltonianNODE()
    q0 = torch.randn(10, 3)
    p0 = torch.randn(10, 3)

    dq_dt, dp_dt = node.vector_field(q0, p0)
    assert dq_dt.shape == (10, 3)
    assert dp_dt.shape == (10, 3)

    t_span = torch.linspace(0.0, 1.0, 10)
    q_traj, p_traj = node.integrate(q0, p0, t_span)
    assert q_traj.shape == (10, 10, 3)
    assert p_traj.shape == (10, 10, 3)
