from __future__ import annotations

import logging
from pathlib import Path

from core.paths import find_checkpoint, find_run_dir, get_checkpoints_root

logger = logging.getLogger(__name__)


def find_latest_run(checkpoints_root: Path) -> Path | None:
    return find_run_dir(checkpoints_root, run_id=None)


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    return find_checkpoint(run_dir, step=None)


def find_checkpoint_by_step(run_dir: Path, step: int) -> Path | None:
    return find_checkpoint(run_dir, step=step)


def resolve_checkpoint(
    resume_from: Path | str | None,
    artifacts_root: Path,
    run_id: str | None = None,
) -> Path | None:
    if resume_from is None:
        return None
    resume_path = Path(resume_from)

    if resume_path.is_dir():
        found = find_latest_checkpoint(resume_path)
        if found is None:
            logger.warning("No checkpoint found in %s", resume_path)
            return None
        resume_path = found
    elif not resume_path.is_absolute():
        checkpoints_root = get_checkpoints_root(artifacts_root)
        run_dir = checkpoints_root / resume_from if run_id is None else checkpoints_root / run_id
        if run_dir.is_dir():
            found = find_latest_checkpoint(run_dir)
            if found is not None:
                resume_path = found

    if not resume_path.exists():
        logger.warning("Checkpoint not found: %s", resume_path)
        return None
    return resume_path


def resolve_auto_resume(artifacts_root: Path) -> Path | None:
    checkpoints_root = get_checkpoints_root(artifacts_root)
    if not checkpoints_root.exists():
        logger.warning("No checkpoints root: %s", checkpoints_root)
        return None
    run_dir = find_latest_run(checkpoints_root)
    if run_dir is None:
        logger.warning("No run directories found")
        return None
    found = find_latest_checkpoint(run_dir)
    if found is None:
        logger.warning("No checkpoint in %s", run_dir)
        return None
    logger.info("Auto-resume: found %s", found)
    return found


def validate_checkpoint_path(path: Path) -> bool:
    return path.exists() and path.suffix == ".pt"
