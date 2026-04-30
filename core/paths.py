from __future__ import annotations

from pathlib import Path


def get_artifacts_root(artifacts_dir: str | Path) -> Path:
    return Path(artifacts_dir)


def get_checkpoints_root(artifacts_root: Path) -> Path:
    """Root directory for all training run checkpoints."""
    return artifacts_root / "multi_map" / "checkpoints"


def get_run_dir(checkpoints_root: Path, run_id: str) -> Path:
    """Directory for a specific training run (checkpoints live here directly)."""
    return checkpoints_root / run_id


def get_checkpoint_path(run_dir: Path, step: int) -> Path:
    """Path to a specific checkpoint file within a run directory."""
    return run_dir / f"checkpoint_{step}.pt"


def get_run_info_path(run_dir: Path) -> Path:
    """Path to run_info.json within a run directory."""
    return run_dir / "run_info.json"


def get_train_log_path(run_dir: Path) -> Path:
    return run_dir / "train.log"


def get_eval_dir(run_dir: Path, step: int, custom: Path | None = None) -> Path:
    """Directory for evaluation artifacts. Uses step-based naming by default."""
    return custom if custom is not None else run_dir / f"eval_{step}"


def find_run_dir(checkpoints_root: Path, run_id: str | None) -> Path | None:
    if not checkpoints_root.is_dir():
        return None
    if run_id:
        run_dir = checkpoints_root / run_id
        return run_dir if run_dir.is_dir() else None
    runs = sorted(d for d in checkpoints_root.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def find_checkpoint(run_dir: Path, step: int | None = None) -> Path | None:
    if step is not None:
        path = get_checkpoint_path(run_dir, step)
        return path if path.exists() else None
    checkpoints = sorted(
        run_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def find_nearest_checkpoint(directory: Path) -> Path | None:
    """Find the checkpoint with the highest step number in a directory."""
    if not directory.exists():
        return None
    ckpt_files = list(directory.glob("checkpoint_*.pt"))
    if not ckpt_files:
        return None

    def _step_from_name(p: Path) -> int:
        import re
        m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else 0

    ckpt_files.sort(key=_step_from_name, reverse=True)
    return ckpt_files[0]
