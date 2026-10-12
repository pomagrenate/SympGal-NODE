# SympGal-NODE: Symplectic Relativistic Stellar Swarms & Neural Dynamics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SympGal-NODE** is an end-to-end, high-performance, object-oriented computational astrophysics framework designed to simulate, visualize, and predict the long-horizon collective orbital escape and tidal evaporation of massive relativistic stellar swarms ($N \sim 10^3 - 10^6$) moving in an effective curved spacetime potential ("trampoline sheet" rubber-membrane analogy with 1PN corrections).

---

## 1. Key Features

- **Relativistic Curved Spacetime Potential**:
  - $K$ Primary Gravitational Sources (binary/hierarchical Mother Stars) satisfying barycentric momentum balance: $\sum_{i} m_i q_i = 0$ and $\sum_{i} m_i p_i = 0$.
  - First Post-Newtonian (1PN) Schwarzschild relativistic curvature corrections:
    $$V_{1\text{PN}}(q, p) = -\sum_{j=1}^{K} \frac{G M_j \Vert(q - q_j) \times (p - p_j)\Vert^2}{c^2 \Vert q - q_j \Vert^3}$$
  - Smooth cluster mean-field background (Plummer sphere / Miyamoto-Nagai flattened disc).
- **Structure-Preserving Symplectic Engines**:
  - 2nd-order Symplectic Leapfrog (Kick-Drift-Kick).
  - 4th-order Yoshida symplectic integrator.
  - Implicit Midpoint integrator (unconditionally preserves the symplectic 2-form $dq \wedge dp$ for non-separable Hamiltonians).
  - Energy conservation monitor tracking relative drift $|\Delta E / E_0|$.
- **Escape Horizon Detection**:
  - Hyperbolic ejection: Specific energy $\mathcal{E}_i = \frac{1}{2} \|v_i\|^2 + \Phi(q_i) > 0$ and barycentric radius $r_i > R_{\text{tidal}}$.
  - Physical collision / tidal capture radius monitoring with primary stars.
- **$\mathrm{SE}(3)$ / $\mathrm{SO}(3)$ Equivariant Graph Embeddings & Neural Dynamics**:
  - Invariant graph features: pairwise Euclidean separations, velocities, scalar products, angular momentum magnitudes, and tidal strain tensors.
  - Hamiltonian Neural ODE ($\mathcal{H}_{\theta}(q, p)$) preserving canonical symplectic phase-space volume.
  - Survival Analysis Head predicting hazard functions $h(t)$, survival probability curves $S(t)$, and expected escape times $\mathbb{E}[T_{\text{escape}}]$.
- **Publication-Ready Astrometric Visualizations**:
  - **Top-Down Ecliptic / Orbital Plane Map**: Dynamic mother orbits, swarm density, and escaped trajectory vectors color-coded by $\log_{10} T_{\text{escape}}$.
  - **3D Spacetime Curvature "Trampoline" Surface**: Deformed 3D rubber sheet where deep funnels represent mother stars and saddle points highlight Lagrange escape bottlenecks.
  - **Full-Sky Celestial Mollweide Map**: All-sky projection $(\lambda, \beta)$ mapping anisotropic escape corridors.

---

## 2. Package Architecture

```
sympgal_node/
├── __init__.py
├── cli.py                     # Entrypoint CLI command dispatcher
├── config.py                  # Dataclasses for simulation and model hyperparams
├── core/
│   ├── coordinates.py         # Barycentric reference frame, phase-space state vectors (q, p)
│   ├── potential.py           # Relativistic multi-body potential: 1PN Schwarzschild + Plummer/Miyamoto-Nagai
│   └── symplectic_engine.py   # Structure-preserving implicit/explicit symplectic integrators
├── swarm/
│   ├── cluster.py             # Stellar cluster generator (Mother stars + N Child stars / test particles)
│   └── escape_detector.py     # Escape horizon monitor (hyperbolic ejection e >= 1.0, tidal disruption, collision)
├── models/
│   ├── equivariant_gnn.py     # SE(3)/SO(3) invariant graph embeddings for curvature & tidal strain
│   ├── neural_ode.py          # Symplectic Hamiltonian Neural ODE wrapper
│   └── survival_head.py       # Long-horizon escape time regressor & survival probability curve
└── visualization/
    ├── sky_maps.py            # Celestial sphere (Mollweide projection) and top-down orbital plane projection
    └── potential_mesh.py      # 3D Spacetime curvature surface ("trampoline sheet" gravity-well depth map)
```

---

## 3. Installation

```bash
# Clone the repository
cd SympGal-NODE

# Install dependencies and editable package
pip install -e .
```

---

## 4. CLI Workflow & Sample Commands

### 1. Initialize a Stellar Swarm
Generate a binary core with 2,000 child swarm stars sampled from a Plummer profile:
```bash
python -m sympgal_node.cli init-swarm --particles 2000 --distribution plummer --v-disp 0.25 --output-dir data
```

### 2. Run Relativistic Symplectic Simulation
Simulate with 1PN corrections and symplectic leapfrog integration:
```bash
python -m sympgal_node.cli simulate --init-state data/swarm_init.pt --t-end 20.0 --dt 0.005 --integrator leapfrog --output-dir data
```

### 3. Generate Publication-Quality Visualizations
Produce Top-Down orbital maps, 3D trampoline spacetime curvature surface, and Full-Sky Mollweide map:
```bash
python -m sympgal_node.cli visualize-map --results data/simulation_results.pt --output-dir figures --type all
```

Output figures:
- `figures/topdown_orbital_plane.png`
- `figures/mollweide_sky_map.png`
- `figures/trampoline_surface.png`

### 4. Train the Equivariant GNN + Survival Head
Train the neural model on the simulated escape records using discrete-time hazard NLL loss:
```bash
python -m sympgal_node.cli train-model --results data/simulation_results.pt --epochs 25 --lr 0.001 --batch-size 256
```

### 5. Predict Swarm Survival & Escape Horizon
Predict the long-horizon collective survival probability curve $S(t)$ and expected escape times:
```bash
python -m sympgal_node.cli predict-escape --model data/survival_model.pt --init-state data/swarm_init.pt --output data/predictions.npz
```

---

## 5. Mathematical Formulations

### Hamiltonian Flow
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

### Relativistic 1PN Potential
$$V_{1\text{PN}}(q_i, p_i) = -\sum_{j=1}^{K} \frac{G M_j \Vert(q_i - q_j) \times (p_i - p_j)\Vert^2}{c^2 (\Vert q_i - q_j \Vert^2 + \epsilon^2)^{3/2}}$$

### Right-Censored Survival Hazard
$$S(t_k) = \prod_{j=1}^{k} (1 - h_j), \quad f(t_k) = h_k S(t_{k-1})$$
$$\mathcal{L}_{\text{NLL}} = -\sum_{i \in \text{ejected}} \log f(t_{k_i}) - \sum_{i \in \text{censored}} \log S(T_{\max})$$
