from __future__ import annotations

import argparse
import math
import sys
import tomllib
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.map_loader import load_map_config

CMAP = mcolors.ListedColormap(["#1e1e1e", "#ffffff", "#cdaa3c"])
CMAP_OVERLAY = mcolors.ListedColormap(["#1e1e1e", "#ffffff", "#cdaa3c", "#dc3c3c", "#3c8cf0", "#50dc78"])

LABELS = {0: "Blocked", 1: "Clean", 2: "Dirty"}
FONTSIZE = 8


def _default_config_path() -> Path:
    return PROJECT_ROOT / "configs" / "train_config.toml"


def _parse_maps_from_config(config_path: Path) -> list[int]:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    env = raw.get("env", {})
    return env.get("default_map_list", [1])


def _plot_single_map(ax: plt.Axes, cfg: dict[str, Any], title: str) -> None:
    grid = cfg["custom_map"]
    h, w = grid.shape

    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=2, origin="upper", interpolation="nearest")

    station_pool = cfg.get("station_pool", [])
    for s in station_pool:
        sx, sz = s["x"], s["z"]
        dx, dz = s.get("dx", 3), s.get("dz", 3)
        rect = mpatches.Rectangle(
            (sx - 0.5, sz - 0.5), dx, dz,
            fill=True, facecolor="#3c8cf0", alpha=0.20, edgecolor="#3c8cf0",
            linewidth=1.5, zorder=5,
        )
        ax.add_patch(rect)

    agent_pool = cfg.get("agent_spawn_pool", [])
    if agent_pool:
        xs, zs = zip(*agent_pool)
        ax.scatter(xs, zs, c="#50dc78", s=50, marker="o", edgecolors="white",
                   linewidths=1.0, zorder=10, label="Agent Spawn")

    npc_pool = cfg.get("npc_spawn_pool", [])
    if npc_pool:
        xs, zs = zip(*npc_pool)
        ax.scatter(xs, zs, c="#dc3c3c", s=40, marker="o", edgecolors="white",
                   linewidths=1.0, zorder=10, label="NPC Spawn")

    ax.set_xlim(-2, w + 2)
    ax.set_ylim(h + 2, -2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=FONTSIZE)


def plot_maps(
    map_ids: list[int],
    config_path: Path | None = None,
    output: str | None = None,
    show: bool = True,
) -> plt.Figure:
    config_label = f" (config: {config_path.name})" if config_path else ""
    n = len(map_ids)
    cols = min(n, 3)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.5),
                             squeeze=False)
    fig.suptitle(f"Training Map Overview{config_label}", fontsize=14, fontweight="bold", y=0.98)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i][j]
            if idx < n:
                map_id = map_ids[idx]
                cfg = load_map_config(map_id)
                _plot_single_map(ax, cfg, f"Map {map_id}")
            else:
                ax.axis("off")

    legend_elements = [
        mpatches.Patch(facecolor="#1e1e1e", edgecolor="none", label="Blocked"),
        mpatches.Patch(facecolor="#cdaa3c", edgecolor="none", label="Dirty"),
        mpatches.Patch(facecolor="#3c8cf0", alpha=0.25, edgecolor="#3c8cf0", linewidth=1.5, label="Charging Station"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#50dc78",
                   markersize=8, markeredgecolor="white", label="Agent Spawn"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc3c3c",
                   markersize=8, markeredgecolor="white", label="NPC Spawn"),
    ]

    legend_ax = axes[-1][-1] if n < rows * cols else fig.add_subplot()
    if n < rows * cols:
        legend_ax.axis("off")
        legend_ax.legend(handles=legend_elements, loc="center", fontsize=9,
                         framealpha=1, ncol=1)
    else:
        fig.legend(handles=legend_elements, loc="lower center",
                   fontsize=9, ncol=len(legend_elements), framealpha=1)
        fig.subplots_adjust(bottom=0.06)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"[plot_maps] Saved: {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all maps from the training config")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training config TOML (default: configs/train_config.toml)")
    parser.add_argument("--maps", type=int, nargs="+", default=None,
                        help="Map IDs to plot (default: read from config)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save path for output image (e.g. maps_overview.png)")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display the window, only save")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else _default_config_path()

    if args.maps:
        map_ids = args.maps
    else:
        map_ids = _parse_maps_from_config(config_path)
        if not map_ids:
            print("[plot_maps] No default_map_list found in config")
            sys.exit(1)

    print(f"[plot_maps] Loading maps: {map_ids}")
    for mid in map_ids:
        try:
            cfg = load_map_config(mid)
            print(f"  Map {mid}: size={cfg['size']}  "
                  f"agent_spawns={len(cfg.get('agent_spawn_pool', []))}  "
                  f"npc_spawns={len(cfg.get('npc_spawn_pool', []))}  "
                  f"stations={len(cfg.get('station_pool', []))}")
        except ValueError as e:
            print(f"  [error] Map {mid}: {e}")
            sys.exit(1)

    plot_maps(map_ids, config_path=config_path,
              output=args.output, show=not args.no_show)


if __name__ == "__main__":
    from core import setup_logging
    setup_logging()
    main()
