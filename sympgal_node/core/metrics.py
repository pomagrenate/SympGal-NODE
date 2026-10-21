"""
Rigorous Scientific Evaluation Suite for SympGal-NODE & Chaotic Astrodynamics:

Scientific Rationale & Q1/Q2 Literature Alignment:
1. Chaos & Lyapunov Predictability Horizon:
   In gravitational multi-body and three-body problems, positive maximal Lyapunov exponents
   lambda > 0 induce exponential trajectory divergence: delta x(t) ~ delta x(0) * exp(lambda * t)
   (Morbidelli 2002; Valtonen & Karttunen 2006; Breen et al. 2020, MNRAS).
   Evaluating pointwise trajectory Euclidean error ||q_pred(t) - q_true(t)|| at long horizons
   t >> T_Lyapunov is physically ill-posed because all models inevitably saturate the cluster diameter.
   Therefore, Q1/Q2 standards mandate evaluating:
   (a) Geometric Phase-Space Invariants (Symplectic 2-form, Liouville volume preservation).
   (b) Individual Event Horizons (Escape / Ejection epoch T_escape).
   (c) Macroscopic Population Statistical Distributions (Stone & Leigh 2019, Nature; Heggie 1975, MNRAS).

2. Right-Censoring & Competing-Risk Survival Metrics:
   In finite simulation or observation campaigns [0, T_max], bound particles have unknown true escape times.
   Naive regression assigning T_escape = T_max introduces severe survivorship bias.
   - Harrell's Concordance Index (C-index, Harrell et al. 1982) ranks pairs robustly under censoring.
   - Integrated Brier Score (IBS, Graf et al. 1999) applies Inverse Probability of Censoring
     Weighting (IPCW) via Kaplan-Meier censoring survival distribution G(t).
   - Log-Scale Error: Mean Absolute Error and RMSE on log10(T_escape) prevent late-escape outliers
     from dominating power-law distributed ejection timescales.

3. Macroscopic Ensemble Alignment:
   - Kolmogorov-Smirnov (KS) statistic: max_t |S_true(t) - S_pred(t)| tests identical cumulative decay.
   - 1D Wasserstein distance (W_1) measures optimal transport work between empirical survival curves,
     verifying the dynamical cluster half-life t_{1/2}.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List, Union
import math
import numpy as np
import torch
import torch.nn as nn


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integration compatible across NumPy 1.x and 2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def concordance_index(
    event_times: torch.Tensor,
    predicted_times: torch.Tensor,
    is_event: torch.Tensor,
) -> float:
    """
    Computes Harrell's Concordance Index (C-index) for right-censored data.
    
    A pair (i, j) is comparable if:
      - Both experienced an event and event_times[i] != event_times[j]
      - i experienced an event at t_i and j was censored/event at t_j > t_i
    
    Concordant if:
      - t_i < t_j implies pred_i < pred_j
      - tied predictions receive 0.5 score
      
    Args:
        event_times: (N,) Observed event or censoring times.
        predicted_times: (N,) Predicted expected event times (or risk scores where higher means shorter time).
                         Here predicted_times are expected times (higher means survives longer).
        is_event: (N,) Binary indicator (1 if event observed, 0 if right-censored).
        
    Returns:
        float: C-index in [0.0, 1.0], where 1.0 is perfect concordance.
    """
    t = event_times.detach().cpu().numpy().flatten()
    pred = predicted_times.detach().cpu().numpy().flatten()
    e = is_event.detach().cpu().numpy().flatten().astype(bool)

    n = len(t)
    if n < 2 or not np.any(e):
        return 0.5  # Undefined/trivial baseline

    concordant = 0.0
    discordant = 0.0
    tied = 0.0
    pairs = 0

    # Vectorized / pairwise comparison
    for i in range(n):
        if not e[i]:
            continue  # Censored subject cannot be the reference earlier event
        t_i = t[i]
        pred_i = pred[i]
        
        # Compare with all subjects j where t_j > t_i
        mask_j = (t > t_i)
        if not np.any(mask_j):
            continue
            
        pred_js = pred[mask_j]
        # Since pred is expected survival time, t_i < t_j implies pred_i < pred_j
        num_conc = np.sum(pred_i < pred_js)
        num_disc = np.sum(pred_i > pred_js)
        num_tie = np.sum(pred_i == pred_js)
        
        concordant += num_conc
        discordant += num_disc
        tied += num_tie
        pairs += (num_conc + num_disc + num_tie)

    if pairs == 0:
        return 0.5

    return float((concordant + 0.5 * tied) / pairs)


def integrated_brier_score(
    event_times: torch.Tensor,
    is_event: torch.Tensor,
    predicted_survival_curves: torch.Tensor,
    time_bins: torch.Tensor,
) -> float:
    """
    Computes Integrated Brier Score (IBS) across continuous mission timeline.
    
    Brier Score at time t:
      BS(t) = 1/N sum_i ( I(T_i > t) - S_hat(t|x_i) )^2  (for uncensored/fully observed)
    With right-censoring, weights by inverse probability of censoring weighting (IPCW).
    
    Args:
        event_times: (N,) Observed event or censoring times.
        is_event: (N,) Binary indicator (1 if event observed, 0 if right-censored).
        predicted_survival_curves: (N, K) Predicted survival probabilities S(t_k | x).
        time_bins: (K,) Discrete time evaluation points.
        
    Returns:
        float: Integrated Brier Score >= 0.0 (lower is better, 0.0 is perfect).
    """
    t = event_times.detach().cpu().numpy().flatten()
    e = is_event.detach().cpu().numpy().flatten().astype(bool)
    S_pred = predicted_survival_curves.detach().cpu().numpy()  # (N, K)
    eval_times = time_bins.detach().cpu().numpy().flatten()    # (K,)

    N = len(t)
    K = len(eval_times)
    if N == 0 or K == 0:
        return 0.0

    # Estimate censoring survival distribution G(t) via Kaplan-Meier on censoring indicator (1 - e)
    # For robustness and self-contained computation without external libraries:
    order = np.argsort(t)
    t_sorted = t[order]
    c_sorted = (~e)[order]  # Censoring indicator
    
    # KM estimator for censoring
    unique_t, counts = np.unique(t_sorted, return_counts=True)
    n_at_risk = N - np.cumsum(counts) + counts
    c_events = np.array([np.sum(c_sorted[t_sorted == ut]) for ut in unique_t])
    
    hazard_c = np.where(n_at_risk > 0, c_events / np.maximum(n_at_risk, 1), 0.0)
    G_steps = np.cumprod(np.maximum(1.0 - hazard_c, 1e-4))
    
    def get_G(query_times: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(unique_t, query_times, side="right") - 1
        G_vals = np.ones_like(query_times, dtype=np.float64)
        valid = idx >= 0
        G_vals[valid] = G_steps[np.clip(idx[valid], 0, len(G_steps) - 1)]
        return np.maximum(G_vals, 1e-3)

    G_t_obs = get_G(t)  # G(T_i)

    brier_scores = []
    for k in range(K):
        tk = eval_times[k]
        G_tk = get_G(np.array([tk]))[0]
        
        # Subject states at time tk:
        # Case 1: T_i > tk -> still alive (uncensored before tk) -> weight = 1 / G(tk)
        alive_mask = t > tk
        # Case 2: T_i <= tk and e_i == 1 -> experienced event before tk -> weight = 1 / G(T_i)
        event_before_mask = (t <= tk) & e
        
        weights = np.zeros(N, dtype=np.float64)
        actual = np.zeros(N, dtype=np.float64)
        
        weights[alive_mask] = 1.0 / G_tk
        actual[alive_mask] = 1.0
        
        weights[event_before_mask] = 1.0 / G_t_obs[event_before_mask]
        actual[event_before_mask] = 0.0
        
        # Valid evaluation subset (censored before tk without event are removed via IPCW)
        valid_eval = alive_mask | event_before_mask
        if np.sum(valid_eval) == 0:
            brier_scores.append(0.0)
            continue
            
        sq_diff = (actual[valid_eval] - S_pred[valid_eval, k]) ** 2
        bs_k = np.mean(weights[valid_eval] * sq_diff)
        brier_scores.append(float(bs_k))

    brier_scores = np.array(brier_scores)
    # Trapezoidal integration across timeline divided by duration
    if eval_times[-1] > eval_times[0]:
        ibs = _trapz(brier_scores, eval_times) / (eval_times[-1] - eval_times[0])
    else:
        ibs = np.mean(brier_scores)

    return float(np.clip(ibs, 0.0, 1.0))


def regression_metrics(
    true_times: torch.Tensor,
    pred_times: torch.Tensor,
    event_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Computes MAE and RMSE on log10(T_escape) for observed events.
    
    Args:
        true_times: (N,) Ground truth escape times.
        pred_times: (N,) Predicted escape times.
        event_mask: Optional (N,) boolean/binary mask for observed events.
        
    Returns:
        Dict containing:
            "mae_log10": Mean Absolute Error in log10 space
            "rmse_log10": Root Mean Squared Error in log10 space
            "mae_linear": Mean Absolute Error in linear space
    """
    t_true = true_times.detach().cpu().numpy().flatten()
    t_pred = pred_times.detach().cpu().numpy().flatten()
    
    if event_mask is not None:
        mask = event_mask.detach().cpu().numpy().flatten().astype(bool)
        t_true = t_true[mask]
        t_pred = t_pred[mask]
        
    if len(t_true) == 0:
        return {"mae_log10": 0.0, "rmse_log10": 0.0, "mae_linear": 0.0}

    log_true = np.log10(np.maximum(t_true, 1e-4))
    log_pred = np.log10(np.maximum(t_pred, 1e-4))
    
    diff_log = log_pred - log_true
    mae_log = float(np.mean(np.abs(diff_log)))
    rmse_log = float(np.sqrt(np.mean(diff_log ** 2)))
    mae_lin = float(np.mean(np.abs(t_pred - t_true)))

    return {
        "mae_log10": mae_log,
        "rmse_log10": rmse_log,
        "mae_linear": mae_lin,
    }


def random_rotation_matrix(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Generates a uniform random 3D SO(3) rotation matrix using QR decomposition of random Gaussian.
    """
    random_mat = torch.randn(3, 3, device=device)
    q, r = torch.linalg.qr(random_mat)
    # Ensure determinant is +1 (SO(3), not O(3))
    d = torch.diag(torch.sign(torch.diag(r)))
    q = torch.matmul(q, d)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def so3_invariance_error(
    model: nn.Module,
    q: torch.Tensor,
    p: torch.Tensor,
    q_m: torch.Tensor,
    p_m: torch.Tensor,
    m_m: torch.Tensor,
    n_rotations: int = 5,
) -> float:
    """
    Evaluates SO(3) rotational invariance error of SympGal-Net:
      Err_SO3 = mean_rot || T_pred(R * q, R * p) - T_pred(q, p) || / || T_pred(q, p) ||
      
    Args:
        model: SympGalNet instance.
        q, p: Child particle phase-space vectors (N, 3).
        q_m, p_m, m_m: Mother star coordinates (K, 3), momenta (K, 3), masses (K,).
        n_rotations: Number of random 3D rotation tests.
        
    Returns:
        float: Relative L1 invariance error across random Euler rotations.
    """
    model.eval()
    device = q.device
    with torch.no_grad():
        base_out = model(q, p, q_m, p_m, m_m)
        base_t = base_out["expected_escape_time"]  # (N,)

        rel_errors = []
        for _ in range(n_rotations):
            R = random_rotation_matrix(device=device)  # (3, 3)
            # Rotate all positions and momenta
            q_rot = torch.matmul(q, R.T)
            p_rot = torch.matmul(p, R.T)
            q_m_rot = torch.matmul(q_m, R.T)
            p_m_rot = torch.matmul(p_m, R.T)

            rot_out = model(q_rot, p_rot, q_m_rot, p_m_rot, m_m)
            rot_t = rot_out["expected_escape_time"]

            err = torch.mean(torch.abs(rot_t - base_t)) / (torch.mean(torch.abs(base_t)) + 1e-6)
            rel_errors.append(err.item())

    return float(np.mean(rel_errors))


def symplectic_volume_dissipation(
    jump_operator: nn.Module,
    latent_dim: int = 128,
    num_samples: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Verifies that the Poincaré jump operator preserves phase-space volume:
      Dissipation = max_samples | det(J) - 1.0 |
      
    Evaluates exact Jacobian determinant via PyTorch automatic differentiation.
    
    Args:
        jump_operator: LatentSymplecticPoincareJump instance.
        latent_dim: Latent vector dimension (must be even).
        num_samples: Number of sample latent points z ~ N(0, 1).
        device: Device to execute autograd on.
        
    Returns:
        float: Maximum absolute deviation from unity |det(J) - 1.0|.
    """
    jump_operator.eval()
    z_samples = torch.randn(num_samples, latent_dim, device=device)
    
    if hasattr(jump_operator, "compute_jacobian_determinant"):
        det_J = jump_operator.compute_jacobian_determinant(z_samples)
        deviations = torch.abs(det_J - 1.0)
        return float(torch.max(deviations).item())
        
    # Fallback to direct autograd Jacobian
    max_dev = 0.0
    for i in range(min(num_samples, 5)):
        z_single = z_samples[i:i+1].clone().detach().requires_grad_(True)
        z_out = jump_operator(z_single)
        
        # Compute D x D jacobian
        D = latent_dim
        J = torch.zeros(D, D, device=device)
        for d in range(D):
            grad_out = torch.zeros_like(z_out)
            grad_out[0, d] = 1.0
            grads = torch.autograd.grad(
                outputs=z_out,
                inputs=z_single,
                grad_outputs=grad_out,
                retain_graph=True,
                create_graph=False,
            )[0]
            J[d, :] = grads[0]
            
        det_val = torch.linalg.det(J).item()
        dev = abs(det_val - 1.0)
        if dev > max_dev:
            max_dev = dev

    return float(max_dev)


def population_distribution_metrics(
    true_survival_curve: torch.Tensor,
    pred_survival_curve: torch.Tensor,
    time_bins: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes population macroscopic distribution alignment:
      - 1D Wasserstein distance between macroscopic survival curves
      - Kolmogorov-Smirnov (KS) test statistic: max_t | S_true(t) - S_pred(t) |
      
    Args:
        true_survival_curve: (K,) Ground truth population empirical survival curve S(t).
        pred_survival_curve: (K,) Predicted population average survival curve S(t).
        time_bins: (K,) Discrete time evaluation points.
        
    Returns:
        Dict containing "wasserstein_distance" and "ks_statistic".
    """
    s_true = true_survival_curve.detach().cpu().numpy().flatten()
    s_pred = pred_survival_curve.detach().cpu().numpy().flatten()
    t_vals = time_bins.detach().cpu().numpy().flatten()

    # KS statistic is maximal absolute discrepancy along the curve
    ks_stat = float(np.max(np.abs(s_true - s_pred)))

    # 1D Wasserstein distance between survival distributions:
    # W_1(F, G) = \int_0^T | F(t) - G(t) | dt = \int_0^T | S_true(t) - S_pred(t) | dt
    if len(t_vals) > 1 and t_vals[-1] > t_vals[0]:
        wass_dist = float(_trapz(np.abs(s_true - s_pred), t_vals))
    else:
        wass_dist = float(np.mean(np.abs(s_true - s_pred)))

    return {
        "wasserstein_distance": wass_dist,
        "ks_statistic": ks_stat,
    }


def classification_metrics(
    true_status: torch.Tensor,
    pred_probs: torch.Tensor,
    num_classes: int = 3,
) -> Dict[str, Any]:
    """
    Computes multi-class terminal fate classification metrics:
      - 3x3 Confusion Matrix
      - Per-class precision, recall, F1-score
      - Macro-averaged F1-score
      
    Classes:
      0: Bound / Trapped
      1: Hyperbolic Ejection
      2: Core Tidal Collision
      
    Args:
        true_status: (N,) Ground truth class labels {0, 1, 2}.
        pred_probs: (N, num_classes) Predicted probability distribution or logits.
        num_classes: Number of risk classes (default 3).
        
    Returns:
        Dict containing confusion matrix, per-class F1, and macro F1.
    """
    y_true = true_status.detach().cpu().numpy().flatten().astype(int)
    probs = pred_probs.detach().cpu().numpy()
    if probs.ndim == 2:
        y_pred = np.argmax(probs, axis=1)
    else:
        y_pred = probs.flatten().astype(int)

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t_val, p_val in zip(y_true, y_pred):
        if 0 <= t_val < num_classes and 0 <= p_val < num_classes:
            cm[t_val, p_val] += 1

    f1_scores = []
    class_names = ["Bound", "Ejection", "Collision"]
    per_class = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        f1_scores.append(f1)
        name = class_names[c] if c < len(class_names) else f"Class_{c}"
        per_class[name] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "support": int(np.sum(cm[c, :])),
        }

    macro_f1 = float(np.mean(f1_scores))

    return {
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "macro_f1": macro_f1,
    }


class ScientificEvaluationEngine:
    """
    Comprehensive Scientific Benchmark & Evaluation Engine for SympGal-NODE.
    Executes all metrics:
      - Harrell's C-index, Integrated Brier Score (IBS), Log-MAE, Log-RMSE
      - SO(3) Rotational Invariance Error, Symplectic Volume Dissipation
      - Population Wasserstein Distance & KS Statistic
      - Multi-class Confusion Matrix & Macro F1-Score
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        q_m: torch.Tensor,
        p_m: torch.Tensor,
        m_m: torch.Tensor,
        true_event_times: torch.Tensor,
        true_terminal_status: torch.Tensor,
        time_bins: torch.Tensor,
        m_s: Optional[torch.Tensor] = None,
        n_rotations: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation pipeline on test swarm batch.
        
        Args:
            q, p: Child star coordinates and momenta (N, 3).
            q_m, p_m, m_m: Mother stars coordinates (K, 3), momenta (K, 3), masses (K,).
            true_event_times: (N,) True observed/censored event times.
            true_terminal_status: (N,) True status: 0=Bound, 1=Ejection, 2=Collision.
            time_bins: (K_bins,) Discrete evaluation time grid.
            n_rotations: Number of SO(3) rotation tests.
            
        Returns:
            Dict[str, Any] containing all structured evaluation metrics.
        """
        self.model.eval()
        device = q.device

        with torch.no_grad():
            outputs = self.model(q, p, q_m, p_m, m_m, m_s=m_s)

        pred_times = outputs["expected_escape_time"]  # (N,)
        pred_survival = outputs["survival_curve"]     # (N, K_bins)
        pred_risk_probs = outputs["risk_probabilities"] # (N, 3)

        # Event mask for ejections/collisions (any non-bound particle)
        is_event = (true_terminal_status > 0).float()

        # 1. Survival Analysis Metrics
        c_index = concordance_index(true_event_times, pred_times, is_event)
        ibs = integrated_brier_score(true_event_times, is_event, pred_survival, time_bins)
        reg_metrics = regression_metrics(true_event_times, pred_times, is_event.bool())

        # 2. Physics & Stability Metrics
        so3_err = so3_invariance_error(
            self.model, q, p, q_m, p_m, m_m, n_rotations=n_rotations
        )
        
        symp_diss = 0.0
        if hasattr(self.model, "poincare_jump"):
            symp_diss = symplectic_volume_dissipation(
                self.model.poincare_jump,
                latent_dim=self.model.hidden_dim,
                num_samples=10,
                device=device,
            )

        # 3. Population Macroscopic Metrics
        # True empirical survival curve: fraction of particles still bound at time t
        t_np = true_event_times.detach().cpu().numpy()
        tb_np = time_bins.detach().cpu().numpy()
        true_pop_surv = np.array([np.mean(t_np > tk) for tk in tb_np])
        pred_pop_surv = torch.mean(pred_survival, dim=0).detach().cpu().numpy()

        pop_metrics = population_distribution_metrics(
            torch.from_numpy(true_pop_surv),
            torch.from_numpy(pred_pop_surv),
            time_bins,
        )

        # Classification Metrics
        cls_metrics = classification_metrics(true_terminal_status, pred_risk_probs, num_classes=3)

        report = {
            "survival_metrics": {
                "c_index": c_index,
                "integrated_brier_score": ibs,
                "mae_log10_escape_time": reg_metrics["mae_log10"],
                "rmse_log10_escape_time": reg_metrics["rmse_log10"],
                "mae_linear_escape_time": reg_metrics["mae_linear"],
            },
            "physics_stability": {
                "so3_rotational_invariance_error": so3_err,
                "symplectic_volume_dissipation": symp_diss,
            },
            "population_macroscopic": {
                "wasserstein_distance": pop_metrics["wasserstein_distance"],
                "kolmogorov_smirnov_statistic": pop_metrics["ks_statistic"],
            },
            "classification_metrics": cls_metrics,
        }

        return report
