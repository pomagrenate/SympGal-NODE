"""
Baselines module for SympGal-NODE framework.
"""

from sympgal_node.baselines.models import (
    VanillaNODE,
    SoftLossGatedPINN,
    HNN_RK4,
    EnhancedSympNet,
    ContinuousEGNODE,
    SymplecticFourierOperator,
    PhysicsGuidedBiLSTM,
    CubicSplineBaseline,
)

__all__ = [
    "VanillaNODE",
    "SoftLossGatedPINN",
    "HNN_RK4",
    "EnhancedSympNet",
    "ContinuousEGNODE",
    "SymplecticFourierOperator",
    "PhysicsGuidedBiLSTM",
    "CubicSplineBaseline",
]
