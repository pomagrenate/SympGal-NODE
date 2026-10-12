"""
SympGal-NODE: Symplectic Hamiltonian Neural ODE Framework
for Relativistic Stellar Swarm Dynamics, Tidal Evaporation, and Orbital Escape.
"""

__version__ = "0.1.0"
__author__ = "Principal Computational Astrophysicist & Senior Systems Architect"

from sympgal_node.config import (
    SimulationConfig,
    PotentialConfig,
    SwarmConfig,
    MotherStarConfig,
    ModelConfig,
)

__all__ = [
    "SimulationConfig",
    "PotentialConfig",
    "SwarmConfig",
    "MotherStarConfig",
    "ModelConfig",
]
