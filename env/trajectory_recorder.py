from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


@dataclass
class TrajectoryFrame:
    step_no: int
    rendered_map: np.ndarray
    agent_position: tuple[int, int]
    npc_positions: list[tuple[int, int]]
    battery: int
    score: int
    terminated: bool
    truncated: bool


@dataclass
class TrajectoryRecorder:
    """Optional trajectory recorder for offline animation rendering."""

    frames: list[TrajectoryFrame] = field(default_factory=list)

    def clear(self) -> None:
        self.frames.clear()

    def record(
        self,
        *,
        step_no: int,
        rendered_map: np.ndarray,
        agent_position: tuple[int, int],
        npc_positions: list[tuple[int, int]],
        battery: int,
        score: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.frames.append(
            TrajectoryFrame(
                step_no=step_no,
                rendered_map=np.array(rendered_map, copy=True),
                agent_position=tuple(agent_position),
                npc_positions=[tuple(position) for position in npc_positions],
                battery=int(battery),
                score=int(score),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )

    def export_gif(self, output_path: str | Path = "artifacts/trajectory.gif", fps: int = 4) -> Path:
        if not self.frames:
            raise ValueError("No trajectory frames recorded.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#111827")
        fig.subplots_adjust(right=0.78)
        cmap = ListedColormap(["#6B7280", "#374151", "#FBBF24", "#FB7185", "#C084FC", "#60A5FA"])

        agent_path_x: list[int] = []
        agent_path_z: list[int] = []
        npc_path_x: list[list[int]] = [[] for _ in self.frames[0].npc_positions]
        npc_path_z: list[list[int]] = [[] for _ in self.frames[0].npc_positions]

        first = self.frames[0]
        image = ax.imshow(first.rendered_map, cmap=cmap, vmin=0, vmax=5, origin="upper")
        agent_scatter = ax.scatter([first.agent_position[0]], [first.agent_position[1]], c="#60A5FA", s=80, label="agent", edgecolors="#DBEAFE", linewidths=1.5)
        npc_scatter = ax.scatter(
            [position[0] for position in first.npc_positions],
            [position[1] for position in first.npc_positions],
            c="#FB7185",
            s=50,
            label="npc",
        )
        (agent_line,) = ax.plot([], [], color="#60A5FA", linewidth=1.5, alpha=0.8)
        npc_lines = [ax.plot([], [], "--", linewidth=1.0)[0] for _ in first.npc_positions]

        legend_handles = [
            Patch(facecolor="#6B7280", edgecolor="none", label="blocked"),
            Patch(facecolor="#374151", edgecolor="#525252", label="floor"),
            Patch(facecolor="#FBBF24", edgecolor="none", label="dirty"),
            Patch(facecolor="#C084FC", edgecolor="none", label="charging station"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#FB7185", markersize=8, label="npc"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#60A5FA", markersize=8, label="agent"),
            Line2D([0], [0], color="#60A5FA", linewidth=1.5, alpha=0.8, label="agent path"),
            Line2D([0], [0], color="#FB7185", linestyle="--", linewidth=1.0, label="npc path"),
        ]

        ax.set_title("Trajectory Replay", color="#E2E8F0", fontsize=14)
        ax.set_xlabel("x", color="#94A3B8")
        ax.set_ylabel("z", color="#94A3B8")
        ax.tick_params(colors="#94A3B8", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
                  facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")

        def update(frame_idx: int) -> list[Any]:
            frame = self.frames[frame_idx]
            image.set_data(frame.rendered_map)

            agent_path_x.append(frame.agent_position[0])
            agent_path_z.append(frame.agent_position[1])
            agent_scatter.set_offsets(np.array([[frame.agent_position[0], frame.agent_position[1]]]))
            agent_line.set_data(agent_path_x, agent_path_z)

            if frame.npc_positions:
                npc_scatter.set_offsets(np.array(frame.npc_positions))
            else:
                npc_scatter.set_offsets(np.empty((0, 2)))

            for idx, position in enumerate(frame.npc_positions):
                npc_path_x[idx].append(position[0])
                npc_path_z[idx].append(position[1])
                npc_lines[idx].set_data(npc_path_x[idx], npc_path_z[idx])

            ax.set_title(
                f"Step {frame.step_no} | Battery {frame.battery} | Score {frame.score}"
                + (" | done" if frame.terminated or frame.truncated else ""),
                color="#E2E8F0", fontsize=14,
            )
            return [image, agent_scatter, npc_scatter, agent_line, *npc_lines]

        ani = animation.FuncAnimation(fig, update, frames=len(self.frames), interval=max(1, int(1000 / fps)), blit=False)
        ani.save(output, writer=animation.PillowWriter(fps=fps))
        plt.close(fig)
        return output

    def export_log(self, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            f.write("step\tagent_x\tagent_z\tbattery\tscore\tterminated\ttruncated\tnpc_count\tnpc_positions\n")
            for frame in self.frames:
                npc_list = ";".join(f"{x},{z}" for x, z in frame.npc_positions) if frame.npc_positions else ""
                f.write(
                    f"{frame.step_no}\t"
                    f"{frame.agent_position[0]}\t{frame.agent_position[1]}\t"
                    f"{frame.battery}\t{frame.score}\t"
                    f"{int(frame.terminated)}\t{int(frame.truncated)}\t"
                    f"{len(frame.npc_positions)}\t{npc_list}\n"
                )
        return output
