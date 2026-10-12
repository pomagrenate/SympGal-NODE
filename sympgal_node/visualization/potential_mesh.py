"""
3D Spacetime Curvature Surface ('Trampoline Sheet' rubber-membrane gravity-well depth map).
Visualizes relativistic potential funnels, Lagrange saddle points, and escaping flow corridors.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

from sympgal_node.core.coordinates import PhaseSpaceState
from sympgal_node.core.potential import RelativisticClusterPotential


class TrampolineMeshVisualizer:
    """
    Renders the effective spacetime potential as a 3D deformed rubber-sheet membrane.
    Deep gravity funnels mark the mother stars, while saddle corridors represent
    chaotic escape bottlenecks.
    """

    def __init__(self, potential: RelativisticClusterPotential, style: str = "dark_background") -> None:
        self.potential = potential
        self.style = style

    def plot_trampoline_surface(
        self,
        mothers_state: PhaseSpaceState,
        grid_range: Tuple[float, float] = (-4.0, 4.0),
        grid_res: int = 120,
        z_clip_min: float = -4.0,
        particle_trajectories: Optional[np.ndarray] = None,
        output_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Render 3D Spacetime Curvature Surface.
        particle_trajectories: Optional array of shape (T, N_sample, 3) to overlay flowing paths.
        """
        plt.style.use(self.style)
        fig = plt.figure(figsize=(12, 9), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # 1. Compute 2D grid potential
        qm = mothers_state.q
        mm = mothers_state.masses
        X, Y, Z_pot = self.potential.evaluate_2d_mesh(
            grid_range, grid_range, grid_res, qm, mm, z_plane=0.0
        )

        # Clip deep singularity cusps for clean visual aesthetic
        Z_clipped = np.clip(Z_pot, z_clip_min, 0.0)

        # 2. Render 3D Surface
        surf = ax.plot_surface(
            X,
            Y,
            Z_clipped,
            cmap="viridis",
            rstride=2,
            cstride=2,
            linewidth=0.1,
            edgecolors="cyan",
            alpha=0.88,
            antialiased=True,
        )

        # 3. Equipotential contour projection on bottom floor
        z_floor = z_clip_min * 1.15
        ax.contour(
            X,
            Y,
            Z_clipped,
            zdir="z",
            offset=z_floor,
            levels=25,
            cmap="plasma",
            linewidths=1.2,
            alpha=0.6,
        )

        # 4. Mark mother stars at the funnel depths
        qm_np = qm.detach().cpu().numpy()
        for k in range(qm_np.shape[0]):
            x_k, y_k = qm_np[k, 0], qm_np[k, 1]
            z_k = z_clip_min * 0.95
            
            # Bottom well marker
            ax.scatter(
                [x_k],
                [y_k],
                [z_k],
                color="#ffb703",
                s=200,
                edgecolors="white",
                linewidths=1.5,
                label=f"{mothers_state.names[k]}",
                zorder=10,
            )
            # Surface anchor marker
            ax.scatter(
                [x_k],
                [y_k],
                [0.0],
                color="#ffb703",
                s=80,
                alpha=0.8,
                edgecolors="white",
                zorder=10,
            )
            # Vertical drop line from z=0 down to the well
            ax.plot(
                [x_k, x_k],
                [y_k, y_k],
                [0.0, z_k],
                color="#ffd166",
                linestyle="--",
                linewidth=2.0,
                alpha=0.9,
            )

        # 5. Overlay sample escaping trajectories if provided
        if particle_trajectories is not None:
            # particle_trajectories: (T, N_sample, 3)
            T, N_sample, _ = particle_trajectories.shape
            for i in range(min(N_sample, 15)):
                traj_x = particle_trajectories[:, i, 0]
                traj_y = particle_trajectories[:, i, 1]
                
                # Mask to grid range
                mask = (
                    (traj_x >= grid_range[0])
                    & (traj_x <= grid_range[1])
                    & (traj_y >= grid_range[0])
                    & (traj_y <= grid_range[1])
                )
                if not np.any(mask):
                    continue
                
                tx = traj_x[mask]
                ty = traj_y[mask]
                
                # Map z to slightly above the local potential sheet
                r_traj = np.sqrt(tx**2 + ty**2) + 0.1
                traj_z = -1.0 / (r_traj) + 0.05
                traj_z = np.clip(traj_z, z_clip_min * 0.85, 0.0)

                ax.plot(
                    tx,
                    ty,
                    traj_z,
                    color="#f72585",
                    alpha=0.75,
                    linewidth=1.2,
                    label="Escape Flow Path" if i == 0 else None,
                )

        # View and labels
        ax.view_init(elev=38, azim=-55)
        ax.set_xlim(grid_range[0], grid_range[1])
        ax.set_ylim(grid_range[0], grid_range[1])
        ax.set_zlim(z_floor, 0.2)
        ax.set_xlabel(r"$X$ [Orbital Plane]", fontsize=11, labelpad=8)
        ax.set_ylabel(r"$Y$ [Orbital Plane]", fontsize=11, labelpad=8)
        ax.set_zlabel(r"Potential Depth $\Phi_{\mathrm{eff}}(X, Y)$", fontsize=11, labelpad=8)
        ax.set_title(
            r"Spacetime Curvature Trampoline Mesh: Gravity Funnels & Escape Saddles",
            fontsize=13,
            pad=15,
        )

        cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.1)
        cbar.set_label(r"Potential $\Phi_{\mathrm{eff}}$", fontsize=11)
        ax.legend(loc="upper right", framealpha=0.6, fontsize=9)

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight", dpi=dpi)

        return fig
