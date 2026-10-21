"""
Configuration dataclasses for SympGal-NODE simulation, potentials, swarms, and neural models.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class MotherStarConfig:
    """Configuration for a primary gravitational source (Mother Star)."""
    name: str = "Mother_1"
    mass: float = 1.0
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    spin: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.1])  # Intrinsic spin S_j in R^3
    radius: float = 0.02


@dataclass
class PotentialConfig:
    """Configuration for the effective curved spacetime potential and dissipative forces."""
    G: float = 1.0                     # Gravitational constant in N-body units
    c: float = 50.0                    # Speed of light in N-body units (for 1PN / 1.5PN)
    softening: float = 0.05            # Gravitational softening length epsilon
    use_1pn: bool = True               # Enable first Post-Newtonian corrections
    use_1_5pn: bool = True             # Enable 1.5PN Lense-Thirring gravitomagnetic spin-orbit acceleration
    enable_dynamical_friction: bool = True  # Enable Chandrasekhar dynamical friction on mother stars
    coulomb_log: float = 3.5           # Coulomb logarithm ln(Lambda)
    df_sigma_v: float = 0.5            # Background velocity dispersion sigma_v for dynamical friction
    plummer_mass: float = 0.5          # Background cluster mass (Plummer sphere)
    plummer_scale: float = 2.5         # Plummer scale radius b
    miyamoto_a: float = 0.0            # Miyamoto-Nagai radial scale a (0 = Plummer)
    miyamoto_b: float = 0.0            # Miyamoto-Nagai vertical scale b
    enable_self_gravity: bool = True   # Enable mutual gravitational interaction between swarm members
    self_gravity_chunk_size: int = 4096 # Vectorized tile chunk size to scale to N >= 10^5 without OOM


@dataclass
class SwarmConfig:
    """Configuration for child stars / test particles and central binary elements."""
    n_particles: int = 2000            # Total number of test particles in swarm
    cluster_radius: float = 1.2        # Initial spatial dispersion radius
    velocity_dispersion: float = 0.25  # Maxwell-Boltzmann velocity dispersion
    distribution: str = "plummer"      # 'plummer', 'disk', or 'uniform'
    imf_type: str = "kroupa"           # Stellar Initial Mass Function: 'kroupa', 'salpeter', or 'uniform'
    imf_m_min: float = 0.08            # Minimum stellar mass in solar units M_sun
    imf_m_max: float = 50.0            # Maximum stellar mass in solar units M_sun
    total_swarm_mass: float = 0.5      # Total normalized mass of swarm relative to central mother stars
    equipartition: bool = True         # Energy equipartition: sigma_v ~ m_i^(-0.5)
    disk_inner_radius: float = 0.4     # For circumbinary disk distribution
    disk_outer_radius: float = 2.0
    r_tidal: float = 5.0               # Tidal escape boundary radius
    collision_radius: float = 0.03     # Physical collision capture radius with mother stars
    seed: int = 42                     # RNG seed for reproducibility
    binary_semi_major_axis: float = 1.0  # Semi-major axis a of central binary
    binary_eccentricity: float = 0.0    # Orbital eccentricity e in [0.0, 0.85]
    binary_mass_ratio: float = 1.0      # Mass ratio q = M2 / M1


@dataclass
class SimulationConfig:
    """Configuration for symplectic numerical integration."""
    t_start: float = 0.0
    t_end: float = 20.0
    dt: float = 0.005
    integrator: str = "leapfrog"       # 'leapfrog', 'yoshida4', 'implicit_midpoint'
    record_interval: int = 4           # Save state every N steps
    device: str = "cpu"                # 'cpu' or 'cuda'
    dtype: str = "float32"             # 'float32' or 'float64'
    max_fixed_point_iter: int = 15     # For implicit midpoint
    fixed_point_tol: float = 1e-7      # Fixed point convergence tolerance


@dataclass
class ModelConfig:
    """Configuration for Equivariant GNN, Neural ODE, and Survival Head."""
    node_dim: int = 64
    edge_dim: int = 32
    in_features: int = 11              # 9 original invariants + m_i + mass ratio (M_j / m_i)
    hidden_dim: int = 128
    num_layers: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    epochs: int = 50
    time_horizons: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0])
    hazard_bins: int = 20


@dataclass
class SympGalConfig:
    """Master configuration binding all sub-configs."""
    potential: PotentialConfig = field(default_factory=PotentialConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mother_stars: List[MotherStarConfig] = field(default_factory=list)

    def __post_init__(self):
        # Default binary core if none provided
        if not self.mother_stars:
            # Equal mass binary in circular orbit around barycenter at separation a=1.0
            # M1 = M2 = 1.0, r1 = [-0.5, 0, 0], r2 = [0.5, 0, 0]
            # v_circ = sqrt(G * M_tot / (4 * a)) = sqrt(1.0 * 2.0 / 4) = sqrt(0.5) ~ 0.7071
            self.mother_stars = [
                MotherStarConfig(
                    name="Mother_A",
                    mass=1.0,
                    position=[-0.5, 0.0, 0.0],
                    velocity=[0.0, -0.7071, 0.0],
                    radius=0.03
                ),
                MotherStarConfig(
                    name="Mother_B",
                    mass=1.0,
                    position=[0.5, 0.0, 0.0],
                    velocity=[0.0, 0.7071, 0.0],
                    radius=0.03
                )
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        """Save configuration to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SympGalConfig:
        """Construct configuration from dictionary."""
        mother_stars_data = data.pop("mother_stars", [])
        mother_stars = [MotherStarConfig(**m) for m in mother_stars_data]
        potential = PotentialConfig(**data.get("potential", {}))
        swarm = SwarmConfig(**data.get("swarm", {}))
        simulation = SimulationConfig(**data.get("simulation", {}))
        model = ModelConfig(**data.get("model", {}))
        return cls(
            potential=potential,
            swarm=swarm,
            simulation=simulation,
            model=model,
            mother_stars=mother_stars,
        )

    @classmethod
    def from_json(cls, path: Path | str) -> SympGalConfig:
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
