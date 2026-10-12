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


def test_eccentric_binary_periastron():
    """Verify that eccentric binary initial separation matches exact periastron r_peri = a * (1 - e)."""
    cfg = SympGalConfig()
    a_val = 1.5
    e_val = 0.6
    cfg.swarm.binary_semi_major_axis = a_val
    cfg.swarm.binary_eccentricity = e_val

    cluster = StellarCluster(cfg)
    _, mothers = cluster.get_states()

    q_m, _ = mothers.numpy()
    # Separation between Star 1 and Star 2
    actual_sep = np.linalg.norm(q_m[1] - q_m[0])
    expected_sep = a_val * (1.0 - e_val)  # 1.5 * 0.4 = 0.60

    assert np.isclose(actual_sep, expected_sep, atol=1e-5), (
        f"Expected periastron separation {expected_sep}, got {actual_sep}"
    )


def test_dynamical_friction_deceleration_and_work():
    """Verify that Chandrasekhar dynamical friction opposes velocity and produces negative work."""
    cfg = PotentialConfig(enable_dynamical_friction=True, coulomb_log=4.0, df_sigma_v=0.5)
    pot = RelativisticClusterPotential(cfg)

    # Mother star moving at velocity [0.5, 0.0, 0.0] through the Plummer cluster
    q_m = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32)
    v_m = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    m_m = torch.tensor([[1.0]], dtype=torch.float32)
    p_m = m_m * v_m

    a_df = pot.compute_dynamical_friction(q_m, p_m, m_m)

    # Deceleration must oppose the velocity vector: a_df . v < 0
    dot_product = torch.sum(a_df * v_m).item()
    assert dot_product < 0.0, f"Dynamical friction must oppose velocity, got dot product {dot_product}"
    assert a_df[0, 0].item() < 0.0, "Deceleration must be in the negative X direction"

    # Test work done in SymplecticEngine
    engine = SymplecticEngine(pot)
    _, dW = engine._apply_dynamical_friction_half_step(q_m, p_m, m_m, dt=0.01)
    assert dW < 0.0, f"Work done by dynamical friction must be negative (dissipative), got {dW}"


def test_lense_thirring_inclined_orbit_torque():
    """Verify that 1.5PN Lense-Thirring gravitomagnetic acceleration induces out-of-plane deflection on inclined orbits."""
    cfg = PotentialConfig(G=1.0, c=20.0, use_1_5pn=True)
    pot = RelativisticClusterPotential(cfg)

    # Central spinning mother star with spin along Z axis: S = [0, 0, 1.0]
    q_m = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    p_m = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    m_m = torch.tensor([[2.0]], dtype=torch.float32)
    spins_m = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    # Inclined test particle with position [1.0, 0.0, 0.5] and velocity [0.0, 1.0, 0.2]
    q_part = torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float32)
    p_part = torch.tensor([[0.0, 1.0, 0.2]], dtype=torch.float32)

    a_lt = pot.compute_lense_thirring_acceleration(q_part, p_part, q_m, p_m, m_m, spins_m)

    # 1.5PN gravitomagnetic acceleration must be non-zero
    assert a_lt.shape == (1, 3)
    assert torch.norm(a_lt).item() > 0.0, "Lense-Thirring acceleration must be non-zero"

    # Must produce out-of-plane (Z-component) acceleration for inclined orbital velocity
    assert abs(a_lt[0, 2].item()) > 1e-6, "Lense-Thirring must induce non-zero out-of-plane Z acceleration"


def test_sympgal_net_architecture_and_forward():
    """Verify SympGalNet forward pass, latent shape, and multi-task loss computation."""
    from sympgal_node.models.sympgal_net import SympGalNet

    cfg_model = SympGalConfig().model
    model = SympGalNet(cfg_model)

    N, K = 32, 2
    q = torch.randn(N, 3)
    p = torch.randn(N, 3)
    qm = torch.randn(K, 3)
    pm = torch.randn(K, 3)
    mm = torch.tensor([[1.0], [1.0]])

    out = model(q, p, qm, pm, mm)

    assert "hazards" in out
    assert "total_survival" in out
    assert "expected_escape_times" in out
    assert "expected_escape_time" in out
    assert "risk_logits" in out
    assert "risk_probabilities" in out
    assert "latent_z" in out

    assert out["latent_z"].shape == (N, cfg_model.hidden_dim)
    assert out["risk_probabilities"].shape == (N, 3)
    assert out["survival_curve"].shape == (N, cfg_model.hazard_bins)

    # Verify probability distributions sum to 1.0
    risk_sum = torch.sum(out["risk_probabilities"], dim=-1)
    assert torch.allclose(risk_sum, torch.ones_like(risk_sum), atol=1e-5)

    # Test loss function
    event_times = torch.rand(N) * 20.0
    status = torch.randint(0, 3, (N,))
    loss_dict = model.survival_head.compute_loss(out["latent_z"], event_times, status)

    assert "loss" in loss_dict
    assert "surv_nll" in loss_dict
    assert "clf_loss" in loss_dict
    assert not torch.isnan(loss_dict["loss"])
    assert loss_dict["loss"].item() > 0.0


def test_symplectic_poincare_jump_determinant():
    """Verify that LatentSymplecticPoincareJump preserves volume with det(J) == 1.0."""
    from sympgal_node.models.sympgal_net import LatentSymplecticPoincareJump
    from sympgal_node.core.metrics import symplectic_volume_dissipation

    latent_dim = 64
    jump_op = LatentSymplecticPoincareJump(latent_dim=latent_dim)

    # Check via automatic differentiation
    max_dissipation = symplectic_volume_dissipation(jump_op, latent_dim=latent_dim, num_samples=10)
    assert max_dissipation < 1e-4, f"Symplectic jump must preserve volume |det(J) - 1.0| < 1e-4, got {max_dissipation}"


def test_scientific_metrics_zero_on_ground_truth_and_catches_degraded():
    """Verify metrics return ideal values on perfect dummy data and catch degraded predictions."""
    from sympgal_node.core.metrics import (
        concordance_index,
        integrated_brier_score,
        regression_metrics,
        population_distribution_metrics,
        classification_metrics,
    )

    N = 100
    K = 20
    # Ground truth
    true_times = torch.linspace(1.0, 20.0, N)
    is_event = torch.ones(N)

    # Perfect predictions: identical order
    perfect_c_index = concordance_index(true_times, true_times, is_event)
    assert np.isclose(perfect_c_index, 1.0), f"Expected perfect C-index 1.0, got {perfect_c_index}"

    # Degraded / reversed predictions: C-index should collapse near 0.0
    reversed_times = torch.flip(true_times, dims=[0])
    degraded_c_index = concordance_index(true_times, reversed_times, is_event)
    assert degraded_c_index < 0.1, f"Expected collapsed C-index < 0.1, got {degraded_c_index}"

    # Regression MAE / RMSE
    perf_reg = regression_metrics(true_times, true_times, is_event.bool())
    assert np.isclose(perf_reg["mae_log10"], 0.0, atol=1e-6)
    assert np.isclose(perf_reg["rmse_log10"], 0.0, atol=1e-6)

    bad_reg = regression_metrics(true_times, true_times * 10.0, is_event.bool())
    assert bad_reg["mae_log10"] > 0.5, "MAE on log10 must catch systematic degradation"

    # Population macroscopic metrics
    time_bins = torch.linspace(0.0, 20.0, K)
    curve_a = torch.linspace(1.0, 0.0, K)
    perf_pop = population_distribution_metrics(curve_a, curve_a, time_bins)
    assert np.isclose(perf_pop["wasserstein_distance"], 0.0, atol=1e-6)
    assert np.isclose(perf_pop["ks_statistic"], 0.0, atol=1e-6)

    curve_b = torch.linspace(0.0, 1.0, K)
    bad_pop = population_distribution_metrics(curve_a, curve_b, time_bins)
    assert bad_pop["ks_statistic"] > 0.5, "KS test must catch opposing survival curves"

    # Classification metrics
    labels = torch.randint(0, 3, (N,))
    one_hot = torch.nn.functional.one_hot(labels, num_classes=3).float()
    perf_cls = classification_metrics(labels, one_hot, num_classes=3)
    assert np.isclose(perf_cls["macro_f1"], 1.0), f"Perfect classification must have macro F1 = 1.0, got {perf_cls['macro_f1']}"

    # Degraded classification (random or flipped)
    flipped_labels = (labels + 1) % 3
    bad_cls = classification_metrics(labels, torch.nn.functional.one_hot(flipped_labels, num_classes=3).float(), num_classes=3)
    assert np.isclose(bad_cls["macro_f1"], 0.0), f"Completely misclassified labels must have macro F1 = 0.0, got {bad_cls['macro_f1']}"

