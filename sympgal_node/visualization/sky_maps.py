"""
Publication-ready celestial mapping:
1. Top-Down Ecliptic/Orbital Plane Map with color-coded escape timestamps.
2. Full-Sky Celestial Mollweide Map showing escape directions and anisotropy corridors.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from sympgal_node.core.coordinates import PhaseSpaceState, CelestialFrame
from sympgal_node.swarm.escape_detector import ParticleStatus


class SkyMapVisualizer:
    """
    Astrophysical visualizer for stellar swarms:
    Generates high-resolution top-down orbital plane projections
    and celestial sphere Mollweide maps.
    """

    def __init__(self, style: str = "dark_background") -> None:
        self.style = style

    def plot_topdown_orbital_plane(
        self,
        swarm_final: PhaseSpaceState,
        mothers_final: PhaseSpaceState,
        escape_dataset: Dict[str, np.ndarray],
        mothers_trajectory: Optional[np.ndarray] = None,
        r_tidal: float = 5.0,
        output_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        1. Top-Down Ecliptic/Orbital Plane Map:
        Shows the dynamic mother stars, their mutual orbit, the surrounding child swarm density,
        and highlights escaped trajectory vectors with color-coded escape timestamps (log10 T_escape).
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

        q_final, _ = swarm_final.numpy()
        status = escape_dataset["status"]
        esc_times = escape_dataset["escape_time"]
        esc_pos = escape_dataset["escape_position"]
        esc_vel = escape_dataset["escape_velocity"]

        # 1. Plot bound swarm particles
        bound_mask = status == int(ParticleStatus.BOUND)
        if np.any(bound_mask):
            ax.scatter(
                q_final[bound_mask, 0],
                q_final[bound_mask, 1],
                s=4,
                c="#4cc9f0",
                alpha=0.35,
                label=f"Bound Swarm (N={np.sum(bound_mask)})",
                edgecolors="none",
                zorder=2,
            )

        # 2. Plot escaped particles with log10(T_escape) colormap
        esc_mask = status == int(ParticleStatus.EJECTED)
        if np.any(esc_mask):
            valid_times = esc_times[esc_mask]
            # Avoid log of zero
            min_pos_time = np.min(valid_times[valid_times > 0]) if np.any(valid_times > 0) else 1e-3
            valid_times_clipped = np.maximum(valid_times, min_pos_time)
            log_times = np.log10(valid_times_clipped)

            norm = mcolors.Normalize(vmin=np.min(log_times), vmax=np.max(log_times))
            cmap = plt.cm.plasma

            sc = ax.scatter(
                esc_pos[esc_mask, 0],
                esc_pos[esc_mask, 1],
                c=log_times,
                cmap=cmap,
                norm=norm,
                s=20,
                alpha=0.85,
                label=f"Escaped Swarm (N={np.sum(esc_mask)})",
                edgecolors="white",
                linewidths=0.3,
                zorder=4,
            )

            # Quiver velocity vectors for a subset of escaped particles
            stride = max(1, np.sum(esc_mask) // 40)
            esc_indices = np.where(esc_mask)[0][::stride]
            ax.quiver(
                esc_pos[esc_indices, 0],
                esc_pos[esc_indices, 1],
                esc_vel[esc_indices, 0],
                esc_vel[esc_indices, 1],
                color="#f72585",
                alpha=0.7,
                width=0.003,
                scale=15.0,
                zorder=5,
            )

            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(r"$\log_{10}(T_{\mathrm{escape}})$", fontsize=12)

        # 3. Plot Mother Stars and Trajectories
        qm_final, _ = mothers_final.numpy()
        if mothers_trajectory is not None and mothers_trajectory.ndim == 3:
            # mothers_trajectory shape: (T, K, 3)
            for k in range(mothers_trajectory.shape[1]):
                ax.plot(
                    mothers_trajectory[:, k, 0],
                    mothers_trajectory[:, k, 1],
                    color="#ffd166",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.8,
                    label="Mother Orbit" if k == 0 else None,
                    zorder=6,
                )

        # Mother star markers
        for k in range(qm_final.shape[0]):
            ax.scatter(
                qm_final[k, 0],
                qm_final[k, 1],
                s=160,
                c="#ffb703",
                edgecolors="#ffffff",
                linewidths=1.5,
                zorder=7,
                label=f"{mothers_final.names[k]}" if k < 2 else None,
            )

        # 4. Tidal escape boundary circle
        tidal_circle = Circle(
            (0, 0),
            r_tidal,
            color="#e63946",
            fill=False,
            linestyle=":",
            linewidth=1.8,
            alpha=0.75,
            label=f"Tidal Horizon $R_{{tidal}}={r_tidal}$",
            zorder=3,
        )
        ax.add_patch(tidal_circle)

        # Formatting - Focus on stellar cluster and tidal evaporation horizon
        view_limit = min(r_tidal * 1.8, 12.0)
        ax.set_xlim(-view_limit, view_limit)
        ax.set_ylim(-view_limit, view_limit)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Orbital Plane $X$ [N-body units]", fontsize=13)
        ax.set_ylabel(r"Orbital Plane $Y$ [N-body units]", fontsize=13)
        ax.set_title(
            r"$\mathbf{SympGal-NODE}$: Relativistic Swarm Orbital Plane & Escape Vectors",
            fontsize=14,
            pad=12,
        )
        ax.legend(loc="upper right", framealpha=0.6, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight", dpi=dpi)

        return fig

    def plot_mollweide_sky_map(
        self,
        swarm_final: PhaseSpaceState,
        escape_dataset: Dict[str, np.ndarray],
        output_path: Optional[Path | str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        2. Full-Sky Celestial Mollweide Map:
        Projects the swarm density and escape directions onto celestial coordinates:
        longitude lambda in [-pi, pi], latitude beta in [-pi/2, pi/2].
        """
        plt.style.use(self.style)
        fig = plt.figure(figsize=(12, 6.5), dpi=dpi)
        ax = fig.add_subplot(111, projection="mollweide")

        q_final, _ = swarm_final.numpy()
        status = escape_dataset["status"]
        esc_cel = escape_dataset["escape_celestial"]
        esc_times = escape_dataset["escape_time"]

        # 1. Bound particles spherical projection
        bound_mask = status == int(ParticleStatus.BOUND)
        if np.any(bound_mask):
            _, lon_b, lat_b = CelestialFrame.cartesian_to_spherical(q_final[bound_mask])
            ax.scatter(
                lon_b,
                lat_b,
                s=6,
                c="#48cae4",
                alpha=0.3,
                edgecolors="none",
                label="Bound Stars",
            )

        # 2. Escaped particles celestial exit points
        esc_mask = status == int(ParticleStatus.EJECTED)
        if np.any(esc_mask):
            lon_esc = esc_cel[esc_mask, 0]
            lat_esc = esc_cel[esc_mask, 1]
            valid_times = esc_times[esc_mask]
            min_pos_time = np.min(valid_times[valid_times > 0]) if np.any(valid_times > 0) else 1e-3
            log_times = np.log10(np.maximum(valid_times, min_pos_time))

            norm = mcolors.Normalize(vmin=np.min(log_times), vmax=np.max(log_times))
            sc = ax.scatter(
                lon_esc,
                lat_esc,
                c=log_times,
                cmap="plasma",
                norm=norm,
                s=35,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.4,
                label="Ejection Directions",
            )
            cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08)
            cbar.set_label(r"Ejection Time $\log_{10}(T_{\mathrm{escape}})$", fontsize=11)

        ax.set_title(
            r"Full-Sky Celestial Mollweide Map: Swarm Escape Corridors $(\lambda, \beta)$",
            fontsize=13,
            pad=18,
        )
        ax.set_xlabel(r"Celestial Longitude $\lambda$", fontsize=11)
        ax.set_ylabel(r"Celestial Latitude $\beta$", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="lower left", framealpha=0.6, fontsize=9)

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, bbox_inches="tight", dpi=dpi)

        return fig
