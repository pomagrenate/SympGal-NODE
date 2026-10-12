"""
SympGal-NODE CLI dispatcher.
Provides commands for swarm initialization, symplectic simulation,
neural model training, survival prediction, and publication-ready visualization.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import numpy as np

from sympgal_node.config import SympGalConfig
from sympgal_node.core.coordinates import PhaseSpaceState
from sympgal_node.core.potential import RelativisticClusterPotential
from sympgal_node.core.symplectic_engine import SymplecticEngine
from sympgal_node.swarm.cluster import StellarCluster
from sympgal_node.swarm.escape_detector import EscapeDetector, ParticleStatus
from sympgal_node.visualization.sky_maps import SkyMapVisualizer
from sympgal_node.visualization.potential_mesh import TrampolineMeshVisualizer
from sympgal_node.models.equivariant_gnn import EquivariantSwarmGNN
from sympgal_node.models.survival_head import SurvivalHead
from sympgal_node.models.neural_ode import SymplecticHamiltonianNODE
from sympgal_node.models.sympgal_net import SympGalNet
from sympgal_node.core.metrics import ScientificEvaluationEngine
import json


BANNER = r"""
================================================================================
   ____                      ____       _         _   _  ___  ____  _____ 
  / ___| _   _ _ __ ___  _ __/ ___| __ _| |       | \ | |/ _ \|  _ \| ____|
  \___ \| | | | '_ ` _ \| '_ \ |  _ / _` | | _____|  \| | | | | | | |  _|  
   ___) | |_| | | | | | | |_) | |_| | (_| | |_____| |\  | |_| | |_| | |___ 
  |____/ \__, |_| |_| |_| .__/ \____|\__,_|_|     |_| \_|\___/|____/|_____|
         |___/          |_|                                                 
  Relativistic Stellar Swarm Dynamics, 1PN Curvature & Survival Prediction
================================================================================
"""


def cmd_init_swarm(args: argparse.Namespace) -> None:
    """Initialize cluster with mother stars and child swarm."""
    print(BANNER)
    print(f"[*] Initializing stellar swarm with {args.particles} particles...")

    cfg = SympGalConfig()
    cfg.swarm.n_particles = args.particles
    cfg.swarm.distribution = args.distribution
    cfg.swarm.velocity_dispersion = args.v_disp
    cfg.swarm.seed = args.seed
    cfg.swarm.binary_eccentricity = getattr(args, "eccentricity", 0.0)
    cfg.swarm.binary_semi_major_axis = getattr(args, "semi_major_axis", 1.0)

    # Set spins on mother stars
    spin_z = getattr(args, "spin_z", 0.1)
    for m in cfg.mother_stars:
        m.spin = [0.0, 0.0, spin_z]

    cluster = StellarCluster(cfg)
    swarm_state, mothers_state = cluster.get_states()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "swarm_init.pt"

    torch.save(
        {
            "config": cfg.to_dict(),
            "swarm_q": swarm_state.q.cpu(),
            "swarm_p": swarm_state.p.cpu(),
            "swarm_masses": swarm_state.masses.cpu(),
            "swarm_names": swarm_state.names,
            "mothers_q": mothers_state.q.cpu(),
            "mothers_p": mothers_state.p.cpu(),
            "mothers_masses": mothers_state.masses.cpu(),
            "mothers_spins": mothers_state.spins.cpu(),
            "mothers_names": mothers_state.names,
        },
        out_path,
    )

    cfg.save_json(out_dir / "config.json")
    print(f"[+] Successfully generated swarm and saved checkpoint to: {out_path}")
    print(f"[+] Mother stars: {len(mothers_state.names)} ({', '.join(mothers_state.names)})")
    print(f"[+] Central Binary: a={cfg.swarm.binary_semi_major_axis}, e={cfg.swarm.binary_eccentricity:.2f}")
    print(f"[+] Swarm distribution: {cfg.swarm.distribution.upper()}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Run symplectic simulation with 1PN, 1.5PN Lense-Thirring, and Dynamical Friction."""
    print(BANNER)
    print(f"[*] Starting symplectic integration...")

    input_path = Path(args.init_state)
    if not input_path.exists():
        print(f"[-] Init state not found at {input_path}. Generating on the fly...")
        cfg = SympGalConfig()
        cfg.swarm.n_particles = args.particles
        cluster = StellarCluster(cfg)
        swarm_state, mothers_state = cluster.get_states()
    else:
        data = torch.load(input_path, map_location=args.device, weights_only=False)
        cfg = SympGalConfig.from_dict(data["config"])
        swarm_state = PhaseSpaceState(
            q=data["swarm_q"],
            p=data["swarm_p"],
            masses=data["swarm_masses"],
            names=data["swarm_names"],
            device=args.device,
        )
        mothers_state = PhaseSpaceState(
            q=data["mothers_q"],
            p=data["mothers_p"],
            masses=data["mothers_masses"],
            spins=data.get("mothers_spins", None),
            names=data["mothers_names"],
            device=args.device,
        )

    # Overrides
    if args.t_end is not None:
        cfg.simulation.t_end = args.t_end
    if args.dt is not None:
        cfg.simulation.dt = args.dt
    if args.integrator is not None:
        cfg.simulation.integrator = args.integrator
    cfg.simulation.device = args.device
    cfg.potential.use_1pn = not args.no_1pn
    cfg.potential.use_1_5pn = not getattr(args, "no_1_5pn", False)
    cfg.potential.enable_dynamical_friction = not getattr(args, "no_df", False)
    if hasattr(args, "coulomb_log") and args.coulomb_log is not None:
        cfg.potential.coulomb_log = args.coulomb_log
    if hasattr(args, "df_sigma_v") and args.df_sigma_v is not None:
        cfg.potential.df_sigma_v = args.df_sigma_v

    potential = RelativisticClusterPotential(cfg.potential)
    engine = SymplecticEngine(potential, cfg.simulation)
    detector = EscapeDetector(
        potential,
        cfg.swarm,
        n_particles=swarm_state.n_bodies,
        device=args.device,
    )

    t_start = cfg.simulation.t_start
    t_end = cfg.simulation.t_end
    dt = cfg.simulation.dt
    n_steps = int((t_end - t_start) / dt)

    initial_energy = engine.compute_energy(swarm_state, mothers_state)
    print(f"[*] Simulation Horizon: T=[{t_start:.1f}, {t_end:.1f}], dt={dt}, Total Steps={n_steps}")
    print(f"[*] Integrator: {cfg.simulation.integrator} (Strang Operator Splitting for Dissipation)")
    print(f"[*] Physics: 1PN={cfg.potential.use_1pn} | 1.5PN Lense-Thirring={cfg.potential.use_1_5pn} | DynFriction={cfg.potential.enable_dynamical_friction}")
    print(f"[*] Initial Total Hamiltonian Energy: {initial_energy:.6e}")

    # Track mother trajectory for orbital visualization
    mother_traj_list = [mothers_state.q.detach().cpu().numpy()]
    sample_child_indices = np.random.choice(swarm_state.n_bodies, min(20, swarm_state.n_bodies), replace=False)
    sample_child_traj = [swarm_state.q[sample_child_indices].detach().cpu().numpy()]

    start_wall_time = time.time()
    last_print = start_wall_time

    for step_idx in range(1, n_steps + 1):
        swarm_state, mothers_state = engine.step(swarm_state, mothers_state, dt=dt)
        curr_t = swarm_state.t

        # Escape detection check
        new_esc, new_col = detector.update(curr_t, swarm_state, mothers_state)

        if step_idx % cfg.simulation.record_interval == 0:
            mother_traj_list.append(mothers_state.q.detach().cpu().numpy())
            sample_child_traj.append(swarm_state.q[sample_child_indices].detach().cpu().numpy())

        # Progress reporting
        now = time.time()
        if now - last_print > 1.5 or step_idx == n_steps:
            progress = (step_idx / n_steps) * 100.0
            cur_tot_e = engine.compute_total_energy(swarm_state, mothers_state)
            rel_err = abs((cur_tot_e - initial_energy) / (abs(initial_energy) + 1e-12))
            summary = detector.get_summary()
            w_df = engine.total_dissipated_work
            print(
                f"  [{progress:5.1f}%] t={curr_t:6.2f} | dE/E0={rel_err:.2e} | W_df={w_df:.3e} | "
                f"Bound={summary['bound_count']} | Escaped={summary['ejected_count']} | Collided={summary['collided_count']}"
            )
            last_print = now

    elapsed = time.time() - start_wall_time
    final_tot_energy = engine.compute_total_energy(swarm_state, mothers_state)
    total_dE = abs((final_tot_energy - initial_energy) / (abs(initial_energy) + 1e-12))
    summary = detector.get_summary()

    print(f"[+] Simulation Completed in {elapsed:.2f}s ({n_steps / max(elapsed, 1e-3):.1f} steps/s)")
    print(f"[+] Final Energy Conservation Error (Strang Bookkeeping): |dE/E0| = {total_dE:.4e}")
    print(f"[+] Total Dynamical Friction Work Dissipated: W_df = {engine.total_dissipated_work:.4e}")
    print(f"[+] Final Swarm Survival: Bound {summary['bound_fraction']*100:.1f}%, Escaped {summary['escape_fraction']*100:.1f}%, Collided {summary['collision_fraction']*100:.1f}%")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "simulation_results.pt"

    torch.save(
        {
            "config": cfg.to_dict(),
            "swarm_final_q": swarm_state.q.cpu(),
            "swarm_final_p": swarm_state.p.cpu(),
            "mothers_final_q": mothers_state.q.cpu(),
            "mothers_final_p": mothers_state.p.cpu(),
            "mothers_spins": mothers_state.spins.cpu(),
            "mothers_trajectory": np.stack(mother_traj_list, axis=0),
            "sample_child_trajectory": np.stack(sample_child_traj, axis=0),
            "escape_dataset": detector.export_dataset(),
            "summary": summary,
            "total_dissipated_work": engine.total_dissipated_work,
        },
        out_path,
    )
    print(f"[+] Simulation results saved to: {out_path}")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Generate top-down orbital plane map, celestial Mollweide map, and 3D trampoline surface."""
    print(BANNER)
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[-] Results file {results_path} not found. Run 'simulate' first.")
        sys.exit(1)

    print(f"[*] Loading simulation results from {results_path}...")
    data = torch.load(results_path, map_location="cpu", weights_only=False)
    cfg = SympGalConfig.from_dict(data["config"])

    swarm_final = PhaseSpaceState(q=data["swarm_final_q"], p=data["swarm_final_p"])
    mothers_final = PhaseSpaceState(
        q=data["mothers_final_q"],
        p=data["mothers_final_p"],
        names=[m.name for m in cfg.mother_stars],
    )
    escape_dataset = data["escape_dataset"]
    mothers_traj = data.get("mothers_trajectory")
    child_traj = data.get("sample_child_trajectory")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sky_vis = SkyMapVisualizer()
    potential = RelativisticClusterPotential(cfg.potential)
    trampoline_vis = TrampolineMeshVisualizer(potential)

    # 1. Top-Down Ecliptic / Orbital Plane Map
    if args.type in ["all", "orbital"]:
        print("[*] Generating Top-Down Orbital Plane Map...")
        fig1_path = out_dir / "topdown_orbital_plane.png"
        sky_vis.plot_topdown_orbital_plane(
            swarm_final=swarm_final,
            mothers_final=mothers_final,
            escape_dataset=escape_dataset,
            mothers_trajectory=mothers_traj,
            r_tidal=cfg.swarm.r_tidal,
            output_path=fig1_path,
            dpi=args.dpi,
        )
        print(f"[+] Saved: {fig1_path}")

    # 2. Celestial Mollweide Map
    if args.type in ["all", "mollweide"]:
        print("[*] Generating Full-Sky Celestial Mollweide Map...")
        fig2_path = out_dir / "mollweide_sky_map.png"
        sky_vis.plot_mollweide_sky_map(
            swarm_final=swarm_final,
            escape_dataset=escape_dataset,
            output_path=fig2_path,
            dpi=args.dpi,
        )
        print(f"[+] Saved: {fig2_path}")

    # 3. 3D Trampoline Spacetime Curvature Surface
    if args.type in ["all", "trampoline"]:
        print("[*] Rendering 3D Spacetime Curvature Trampoline Mesh...")
        fig3_path = out_dir / "trampoline_surface.png"
        trampoline_vis.plot_trampoline_surface(
            mothers_state=mothers_final,
            grid_range=(-cfg.swarm.r_tidal * 0.9, cfg.swarm.r_tidal * 0.9),
            particle_trajectories=child_traj,
            output_path=fig3_path,
            dpi=args.dpi,
        )
        print(f"[+] Saved: {fig3_path}")

    print("[+] All requested visualizations completed successfully!")


def cmd_train_model(args: argparse.Namespace) -> None:
    """Train Equivariant GNN + Survival Head on simulated escape data."""
    print(BANNER)
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[-] Results file {results_path} not found. Please simulate first.")
        sys.exit(1)

    print(f"[*] Loading training dataset from {results_path}...")
    data = torch.load(results_path, map_location=args.device, weights_only=False)
    cfg = SympGalConfig.from_dict(data["config"])

    # Extract initial or final swarm state
    init_path = Path("data/swarm_init.pt")
    if init_path.exists():
        init_data = torch.load(init_path, map_location=args.device, weights_only=False)
        q0 = init_data["swarm_q"].to(args.device)
        p0 = init_data["swarm_p"].to(args.device)
        qm = init_data["mothers_q"].to(args.device)
        pm = init_data["mothers_p"].to(args.device)
        mm = init_data["mothers_masses"].to(args.device)
    else:
        q0 = data["swarm_final_q"].to(args.device)
        p0 = data["swarm_final_p"].to(args.device)
        qm = data["mothers_final_q"].to(args.device)
        pm = data["mothers_final_p"].to(args.device)
        mm = torch.tensor([[m.mass] for m in cfg.mother_stars], dtype=q0.dtype, device=args.device)

    esc_data = data["escape_dataset"]
    esc_times = torch.tensor(esc_data["escape_time"], dtype=torch.float32, device=args.device)
    status = torch.tensor(esc_data["status"], dtype=torch.long, device=args.device)
    is_ejected = status == int(ParticleStatus.EJECTED)

    model_type = getattr(args, "model_type", "sympgal_net")
    if model_type == "sympgal_net":
        print(f"[*] Training Proprietary SympGal-Net (Encoder + Symplectic Jump + Multi-Risk Head)...")
        model = SympGalNet(cfg.model).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=cfg.model.weight_decay,
        )

        N = q0.shape[0]
        for epoch in range(1, args.epochs + 1):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, N, args.batch_size):
                b_idx = perm[b_start : b_start + args.batch_size]
                q_b = q0[b_idx]
                p_b = p0[b_idx]
                t_b = esc_times[b_idx]
                status_b = status[b_idx]

                z_enc = model.encoder(q_b, p_b, qm, pm, mm)
                z_symp = model.poincare_jump(z_enc)
                loss_dict = model.survival_head.compute_loss(z_symp, t_b, status_b)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  [Epoch {epoch:3d}/{args.epochs}] SympGal-Net Multi-Task Loss: {avg_loss:.4f}")

        # Save model checkpoint
        model_dir = Path(args.model_output).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "sympgal_net",
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
            },
            args.model_output,
        )
        print(f"[+] SympGal-Net checkpoint saved to: {args.model_output}")
    else:
        # Instantiate Equivariant GNN + Survival Head
        gnn = EquivariantSwarmGNN(cfg.model).to(args.device)
        survival_head = SurvivalHead(
            cfg.model, in_features=cfg.model.hidden_dim, t_max=cfg.simulation.t_end
        ).to(args.device)

        optimizer = torch.optim.AdamW(
            list(gnn.parameters()) + list(survival_head.parameters()),
            lr=args.lr,
            weight_decay=cfg.model.weight_decay,
        )

        print(f"[*] Training Equivariant GNN + Survival Head for {args.epochs} epochs (Batch Size={args.batch_size})...")
        N = q0.shape[0]

        for epoch in range(1, args.epochs + 1):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, N, args.batch_size):
                b_idx = perm[b_start : b_start + args.batch_size]

                q_b = q0[b_idx]
                p_b = p0[b_idx]
                t_b = esc_times[b_idx]
                ejected_b = is_ejected[b_idx]

                # Invariant embeddings
                z_b = gnn(q_b, p_b, qm, pm, mm)
                loss = survival_head.compute_loss(z_b, t_b, ejected_b)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  [Epoch {epoch:3d}/{args.epochs}] Negative Log-Likelihood Loss: {avg_loss:.4f}")

        # Save model checkpoint
        model_dir = Path(args.model_output).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_type": "gnn",
                "gnn_state": gnn.state_dict(),
                "survival_head_state": survival_head.state_dict(),
                "config": cfg.to_dict(),
            },
            args.model_output,
        )
        print(f"[+] Model checkpoint saved to: {args.model_output}")


def cmd_predict_escape(args: argparse.Namespace) -> None:
    """Predict escape times and survival probability curve for a swarm."""
    print(BANNER)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[-] Model checkpoint {model_path} not found. Run 'train-model' first.")
        sys.exit(1)

    print(f"[*] Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = SympGalConfig.from_dict(ckpt["config"])
    # Check model architecture
    model_type = ckpt.get("model_type", "gnn" if "gnn_state" in ckpt else "sympgal_net")
    if model_type == "sympgal_net":
        model = SympGalNet(cfg.model)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        is_sympgal = True
    else:
        gnn = EquivariantSwarmGNN(cfg.model)
        gnn.load_state_dict(ckpt["gnn_state"])
        gnn.eval()

        survival_head = SurvivalHead(
            cfg.model, in_features=cfg.model.hidden_dim, t_max=cfg.simulation.t_end
        )
        survival_head.load_state_dict(ckpt["survival_head_state"])
        survival_head.eval()
        is_sympgal = False

    # Load initial swarm or generate sample
    init_path = Path(args.init_state)
    if init_path.exists():
        data = torch.load(init_path, map_location="cpu", weights_only=False)
        q = data["swarm_q"]
        p = data["swarm_p"]
        qm = data["mothers_q"]
        pm = data["mothers_p"]
        mm = data["mothers_masses"]
    else:
        print("[*] Generating new test swarm...")
        cluster = StellarCluster(cfg)
        sw, mo = cluster.get_states()
        q, p, qm, pm, mm = sw.q, sw.p, mo.q, mo.p, mo.masses

    with torch.no_grad():
        if is_sympgal:
            preds = model(q, p, qm, pm, mm)
            survival_curves = preds["survival_curve"]
            expected_times = preds["expected_escape_time"]
            bin_centers = model.survival_head.bin_centers
        else:
            z = gnn(q, p, qm, pm, mm)
            hazards, survival_curves, expected_times = survival_head(z)
            bin_centers = survival_head.bin_centers

    surv_np = survival_curves.cpu().numpy()
    exp_times_np = expected_times.squeeze().cpu().numpy()

    # Mean survival curve across swarm
    mean_survival = np.mean(surv_np, axis=0)
    median_pred_time = float(np.median(exp_times_np))

    print(f"[+] Predicted Expected Escape Time (Median): {median_pred_time:.2f} [N-body units]")
    print(f"[+] Collective Swarm Survival Probability Curve:")
    time_bins = bin_centers.cpu().numpy()
    for t_val, s_val in zip(time_bins[::3], mean_survival[::3]):
        bar = "#" * int(s_val * 30)
        print(f"    t = {t_val:5.1f} | S(t) = {s_val:5.3f} | [{bar:<30}]")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        time_bins=time_bins,
        mean_survival=mean_survival,
        individual_survival=surv_np,
        predicted_escape_times=exp_times_np,
    )
    print(f"[+] Full predictions saved to: {out_path}")


def cmd_evaluate_model(args: argparse.Namespace) -> None:
    """Run comprehensive scientific evaluation benchmark on SympGal-Net."""
    print(BANNER)
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[-] Model checkpoint {model_path} not found. Run 'train-model' first.")
        sys.exit(1)

    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        print(f"[-] Test dataset {test_data_path} not found. Run 'simulate' first.")
        sys.exit(1)

    print(f"[*] Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location=args.device, weights_only=False)
    cfg = SympGalConfig.from_dict(ckpt["config"])

    model_type = ckpt.get("model_type", "sympgal_net")
    if model_type == "sympgal_net":
        model = SympGalNet(cfg.model).to(args.device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("[-] Evaluation engine currently requires SympGal-Net architecture.")
        sys.exit(1)

    model.eval()

    print(f"[*] Loading evaluation test data from {test_data_path}...")
    sim_data = torch.load(test_data_path, map_location=args.device, weights_only=False)

    init_path = Path("data/swarm_init.pt")
    if init_path.exists():
        init_data = torch.load(init_path, map_location=args.device, weights_only=False)
        q = init_data["swarm_q"].to(args.device)
        p = init_data["swarm_p"].to(args.device)
        qm = init_data["mothers_q"].to(args.device)
        pm = init_data["mothers_p"].to(args.device)
        mm = init_data["mothers_masses"].to(args.device)
    else:
        q = sim_data["swarm_final_q"].to(args.device)
        p = sim_data["swarm_final_p"].to(args.device)
        qm = sim_data["mothers_final_q"].to(args.device)
        pm = sim_data["mothers_final_p"].to(args.device)
        mm = torch.tensor([[m.mass] for m in cfg.mother_stars], dtype=q.dtype, device=args.device)

    esc_data = sim_data["escape_dataset"]
    true_times = torch.tensor(esc_data["escape_time"], dtype=torch.float32, device=args.device)
    true_status = torch.tensor(esc_data["status"], dtype=torch.long, device=args.device)

    time_bins = model.survival_head.bin_centers.to(args.device)

    print("[*] Executing Scientific Evaluation Engine across test population...")
    eval_engine = ScientificEvaluationEngine(model)
    report = eval_engine.evaluate(
        q=q,
        p=p,
        q_m=qm,
        p_m=pm,
        m_m=mm,
        true_event_times=true_times,
        true_terminal_status=true_status,
        time_bins=time_bins,
        n_rotations=args.n_rotations,
    )

    # Print Executive Summary Table
    surv = report["survival_metrics"]
    phys = report["physics_stability"]
    pop = report["population_macroscopic"]
    cls_rep = report["classification_metrics"]

    print("\n" + "=" * 78)
    print("                SYMPGAL-NET PUBLICATION BENCHMARK REPORT")
    print("=" * 78)
    print(f" {'Metric Category':<30} | {'Metric Name':<30} | {'Value':<12}")
    c_idx_label = "Harrell's C-Index (>=0.5)"
    print(f" {'Survival Discrimination':<30} | {c_idx_label:<30} | {surv['c_index']:<12.4f}")
    print(f" {'Survival Calibration':<30} | {'Integrated Brier Score (IBS)':<30} | {surv['integrated_brier_score']:<12.4f}")
    print(f" {'Regression Accuracy':<30} | {'MAE log10(T_esc)':<30} | {surv['mae_log10_escape_time']:<12.4f}")
    print(f" {'Regression Accuracy':<30} | {'RMSE log10(T_esc)':<30} | {surv['rmse_log10_escape_time']:<12.4f}")
    print(f" {'Symmetry Preservation':<30} | {'SO(3) Invariance Error':<30} | {phys['so3_rotational_invariance_error']:<12.2e}")
    print(f" {'Symplectic Structure':<30} | {'|det(J) - 1.0| Dissipation':<30} | {phys['symplectic_volume_dissipation']:<12.2e}")
    print(f" {'Macroscopic Alignment':<30} | {'Wasserstein Distance W_1':<30} | {pop['wasserstein_distance']:<12.4f}")
    print(f" {'Macroscopic Alignment':<30} | {'Kolmogorov-Smirnov D_KS':<30} | {pop['kolmogorov_smirnov_statistic']:<12.4f}")
    print(f" {'Terminal Fate Prediction':<30} | {'Macro F1-Score':<30} | {cls_rep['macro_f1']:<12.4f}")
    print("=" * 78)

    print("\n[+] Confusion Matrix (Rows=True, Cols=Pred: 0=Bound, 1=Ejection, 2=Collision):")
    cm = cls_rep["confusion_matrix"]
    for row in cm:
        print("    " + "  ".join(f"{val:6d}" for val in row))

    out_report_path = Path(args.output_report)
    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Full publication report saved to: {out_report_path}")


def main() -> None:
    """CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="sympgal",
        description="SympGal-NODE: Symplectic Relativistic Swarm Dynamics & Survival Analysis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # init-swarm
    p_init = subparsers.add_parser("init-swarm", help="Initialize stellar swarm and mother stars")
    p_init.add_argument("--particles", "-n", type=int, default=2000, help="Number of child stars")
    p_init.add_argument("--distribution", "-d", type=str, default="plummer", choices=["plummer", "disk"], help="Spatial distribution")
    p_init.add_argument("--v-disp", type=float, default=0.25, help="Velocity dispersion")
    p_init.add_argument("--eccentricity", "-e", type=float, default=0.0, help="Central binary orbital eccentricity e in [0.0, 0.85]")
    p_init.add_argument("--semi-major-axis", "-a", type=float, default=1.0, help="Central binary semi-major axis a")
    p_init.add_argument("--spin-z", type=float, default=0.1, help="Intrinsic spin S_z for mother stars")
    p_init.add_argument("--seed", type=int, default=42, help="RNG seed")
    p_init.add_argument("--output-dir", "-o", type=str, default="data", help="Output directory")

    # simulate
    p_sim = subparsers.add_parser("simulate", help="Run symplectic simulation with 1PN, 1.5PN, and Dynamical Friction")
    p_sim.add_argument("--init-state", "-i", type=str, default="data/swarm_init.pt", help="Path to initial state")
    p_sim.add_argument("--particles", "-n", type=int, default=2000, help="Particle count if generating on fly")
    p_sim.add_argument("--t-end", "-t", type=float, default=20.0, help="End time")
    p_sim.add_argument("--dt", type=float, default=0.005, help="Integration time step")
    p_sim.add_argument("--integrator", type=str, default="leapfrog", choices=["leapfrog", "implicit_midpoint", "yoshida4"], help="Symplectic integrator")
    p_sim.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda')")
    p_sim.add_argument("--no-1pn", action="store_true", help="Disable 1PN relativistic corrections")
    p_sim.add_argument("--no-1-5pn", action="store_true", help="Disable 1.5PN Lense-Thirring gravitomagnetic spin-orbit acceleration")
    p_sim.add_argument("--no-df", action="store_true", help="Disable Chandrasekhar dynamical friction on mother stars")
    p_sim.add_argument("--coulomb-log", type=float, default=3.5, help="Coulomb logarithm ln(Lambda) for dynamical friction")
    p_sim.add_argument("--df-sigma-v", type=float, default=0.5, help="Background velocity dispersion sigma_v for dynamical friction")
    p_sim.add_argument("--output-dir", "-o", type=str, default="data", help="Output directory")

    # visualize-map
    p_vis = subparsers.add_parser("visualize-map", help="Generate publication-ready celestial maps and 3D trampoline mesh")
    p_vis.add_argument("--results", "-r", type=str, default="data/simulation_results.pt", help="Simulation results path")
    p_vis.add_argument("--output-dir", "-o", type=str, default="figures", help="Figures output directory")
    p_vis.add_argument("--type", type=str, default="all", choices=["all", "orbital", "mollweide", "trampoline"], help="Visualization type")
    p_vis.add_argument("--dpi", type=int, default=300, help="Figure DPI")

    # train-model
    p_train = subparsers.add_parser("train-model", help="Train Equivariant GNN or SympGal-Net")
    p_train.add_argument("--results", "-r", type=str, default="data/simulation_results.pt", help="Results file path")
    p_train.add_argument("--model-type", type=str, default="sympgal_net", choices=["sympgal_net", "gnn"], help="Architecture to train")
    p_train.add_argument("--epochs", "-e", type=int, default=20, help="Training epochs")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p_train.add_argument("--batch-size", "-b", type=int, default=256, help="Batch size")
    p_train.add_argument("--device", type=str, default="cpu", help="Compute device")
    p_train.add_argument("--model-output", "-m", type=str, default="data/sympgal_model.pt", help="Model checkpoint path")

    # predict-escape
    p_pred = subparsers.add_parser("predict-escape", help="Predict swarm survival and escape times")
    p_pred.add_argument("--model", "-m", type=str, default="data/sympgal_model.pt", help="Trained model path")
    p_pred.add_argument("--init-state", "-i", type=str, default="data/swarm_init.pt", help="Swarm state path")
    p_pred.add_argument("--output", "-o", type=str, default="data/predictions.npz", help="Output predictions path")

    # evaluate-model
    p_eval = subparsers.add_parser("evaluate-model", help="Run comprehensive scientific evaluation benchmark")
    p_eval.add_argument("--model", "-m", type=str, default="data/sympgal_model.pt", help="Path to trained model")
    p_eval.add_argument("--test-data", "-t", type=str, default="data/simulation_results.pt", help="Path to test simulation results")
    p_eval.add_argument("--output-report", "-o", type=str, default="results/metrics_report.json", help="Path to output JSON report")
    p_eval.add_argument("--n-rotations", type=int, default=5, help="Number of random SO(3) rotations")
    p_eval.add_argument("--device", type=str, default="cpu", help="Compute device")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "init-swarm":
        cmd_init_swarm(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "visualize-map":
        cmd_visualize(args)
    elif args.command == "train-model":
        cmd_train_model(args)
    elif args.command == "predict-escape":
        cmd_predict_escape(args)
    elif args.command == "evaluate-model":
        cmd_evaluate_model(args)


if __name__ == "__main__":
    main()
