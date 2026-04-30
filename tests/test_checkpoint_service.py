"""Checkpoint 服务层测试：services/checkpoint_service.py"""

from __future__ import annotations

import unittest
from pathlib import Path

from services.checkpoint_service import (
    find_latest_run,
    find_latest_checkpoint,
    find_checkpoint_by_step,
    resolve_checkpoint,
    resolve_auto_resume,
    validate_checkpoint_path,
)


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestFindLatestRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_cs_run"
        self.ckpt_root = self.tmp / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_find_latest_run(self):
        for name in ["run_1", "run_2", "run_3"]:
            (self.ckpt_root / name).mkdir()
        result = find_latest_run(self.ckpt_root)
        self.assertEqual(result.name, "run_3")

    def test_empty_root(self):
        result = find_latest_run(self.ckpt_root)
        self.assertIsNone(result)

    def test_nonexistent_root(self):
        result = find_latest_run(Path("/nonexistent"))
        self.assertIsNone(result)


class TestFindLatestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_cs_latest"
        self.run_dir = self.tmp / "test_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_find_latest_checkpoint(self):
        _touch(self.run_dir / "checkpoint_100.pt")
        _touch(self.run_dir / "checkpoint_500.pt")
        _touch(self.run_dir / "checkpoint_1000.pt")
        result = find_latest_checkpoint(self.run_dir)
        self.assertEqual(result.name, "checkpoint_1000.pt")

    def test_no_checkpoints(self):
        result = find_latest_checkpoint(self.run_dir)
        self.assertIsNone(result)

    def test_non_existent_dir(self):
        result = find_latest_checkpoint(Path("/nonexistent"))
        self.assertIsNone(result)

    def test_ignores_non_checkpoint_files(self):
        _touch(self.run_dir / "some_file.txt")
        _touch(self.run_dir / "checkpoint_50.pt")
        result = find_latest_checkpoint(self.run_dir)
        self.assertEqual(result.name, "checkpoint_50.pt")


class TestFindCheckpointByStep(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_cs_step"
        self.run_dir = self.tmp / "test_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_find_by_exact_step(self):
        _touch(self.run_dir / "checkpoint_2000.pt")
        result = find_checkpoint_by_step(self.run_dir, 2000)
        self.assertEqual(result.name, "checkpoint_2000.pt")

    def test_step_not_found(self):
        result = find_checkpoint_by_step(self.run_dir, 999)
        self.assertIsNone(result)


class TestResolveCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_cs_resolve"
        self.artifacts_root = self.tmp / "artifacts"
        self.ckpt_root = self.artifacts_root / "multi_map" / "checkpoints"
        self.run_dir = self.ckpt_root / "test_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_resolve_none(self):
        result = resolve_checkpoint(None, self.artifacts_root)
        self.assertIsNone(result)

    def test_resolve_exact_file(self):
        ckpt = self.run_dir / "checkpoint_5000.pt"
        _touch(ckpt)
        result = resolve_checkpoint(ckpt, self.artifacts_root)
        self.assertEqual(result, ckpt)

    def test_resolve_directory_finds_latest(self):
        _touch(self.run_dir / "checkpoint_100.pt")
        _touch(self.run_dir / "checkpoint_200.pt")
        result = resolve_checkpoint(self.run_dir, self.artifacts_root)
        self.assertEqual(result.name, "checkpoint_200.pt")

    def test_resolve_directory_no_checkpoints(self):
        result = resolve_checkpoint(self.run_dir, self.artifacts_root)
        self.assertIsNone(result)

    def test_resolve_nonexistent_logs_warning(self):
        with self.assertLogs("services.checkpoint_service", level="WARNING") as log:
            result = resolve_checkpoint(Path("/nonexistent/ckpt.pt"), self.artifacts_root)
        self.assertIsNone(result)
        self.assertTrue(any("Checkpoint not found" in msg for msg in log.output))


class TestResolveAutoResume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_cs_auto"
        self.artifacts_root = self.tmp / "artifacts"
        self.ckpt_root = self.artifacts_root / "multi_map" / "checkpoints"
        self.run_dir = self.ckpt_root / "latest_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_auto_resume_ok(self):
        _touch(self.run_dir / "checkpoint_500.pt")
        result = resolve_auto_resume(self.artifacts_root)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "checkpoint_500.pt")

    def test_auto_resume_no_checkpoints_root(self):
        result = resolve_auto_resume(Path("/nonexistent"))
        self.assertIsNone(result)

    def test_auto_resume_no_runs(self):
        empty_root = self.tmp / "empty_artifacts"
        empty_root.mkdir(parents=True, exist_ok=True)
        result = resolve_auto_resume(empty_root)
        self.assertIsNone(result)

    def test_auto_resume_run_no_checkpoints(self):
        result = resolve_auto_resume(self.artifacts_root)
        self.assertIsNone(result)


class TestValidateCheckpointPath(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_cs_valid"

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_valid_pt_file(self):
        path = self.tmp / "checkpoint_100.pt"
        _touch(path)
        self.assertTrue(validate_checkpoint_path(path))

    def test_non_existent_file(self):
        self.assertFalse(validate_checkpoint_path(Path("/nonexistent.pt")))

    def test_wrong_extension(self):
        path = self.tmp / "checkpoint_100.txt"
        _touch(path)
        self.assertFalse(validate_checkpoint_path(path))


if __name__ == "__main__":
    unittest.main()
