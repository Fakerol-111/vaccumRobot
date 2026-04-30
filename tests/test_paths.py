"""路径层测试：core/paths.py"""

from __future__ import annotations

import unittest
from pathlib import Path

from core.paths import (
    get_artifacts_root,
    get_checkpoints_root,
    get_run_dir,
    get_checkpoint_path,
    get_run_info_path,
    get_train_log_path,
    get_eval_dir,
    find_run_dir,
    find_checkpoint,
    find_nearest_checkpoint,
)


class TestPathGetters(unittest.TestCase):
    def setUp(self):
        self.root = Path("/tmp/test_artifacts")
        self.checkpoints_root = self.root / "multi_map" / "checkpoints"
        self.run_dir = self.checkpoints_root / "run_001"

    def test_get_artifacts_root(self):
        self.assertEqual(get_artifacts_root("/tmp/test_artifacts"), Path("/tmp/test_artifacts"))
        self.assertEqual(get_artifacts_root("relative/path"), Path("relative/path"))

    def test_get_checkpoints_root(self):
        result = get_checkpoints_root(self.root)
        self.assertEqual(result, self.root / "multi_map" / "checkpoints")

    def test_get_run_dir(self):
        result = get_run_dir(self.checkpoints_root, "run_001")
        self.assertEqual(result, self.checkpoints_root / "run_001")

    def test_get_checkpoint_path(self):
        path = get_checkpoint_path(self.run_dir, 5000)
        self.assertEqual(path, self.run_dir / "checkpoint_5000.pt")

    def test_get_run_info_path(self):
        path = get_run_info_path(self.run_dir)
        self.assertEqual(path, self.run_dir / "run_info.json")

    def test_get_train_log_path(self):
        path = get_train_log_path(self.run_dir)
        self.assertEqual(path, self.run_dir / "train.log")

    def test_get_eval_dir_default(self):
        path = get_eval_dir(self.run_dir, 5000)
        self.assertEqual(path, self.run_dir / "eval_5000")

    def test_get_eval_dir_custom(self):
        custom = Path("/tmp/custom_eval")
        path = get_eval_dir(self.run_dir, 5000, custom)
        self.assertEqual(path, custom)

    def test_path_separator_independence(self):
        root = Path("artifacts")
        ckpt = get_checkpoints_root(root)
        self.assertEqual(ckpt, Path("artifacts/multi_map/checkpoints"))


class TestFindRunDir(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.ckpt_root = self.tmp / "checkpoints"
        self.ckpt_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def test_find_run_dir_by_id(self):
        run_dir = self.ckpt_root / "test_run"
        run_dir.mkdir()
        result = find_run_dir(self.ckpt_root, "test_run")
        self.assertEqual(result, run_dir)

    def test_find_run_dir_nonexistent(self):
        result = find_run_dir(self.ckpt_root, "nonexistent")
        self.assertIsNone(result)

    def test_find_run_dir_latest(self):
        for name in ["run_a", "run_b", "run_c"]:
            (self.ckpt_root / name).mkdir()
        result = find_run_dir(self.ckpt_root, None)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "run_c")

    def test_find_run_dir_empty(self):
        result = find_run_dir(self.ckpt_root, None)
        self.assertIsNone(result)

    def test_find_run_dir_nonexistent_root(self):
        result = find_run_dir(Path("/nonexistent/path"), "x")
        self.assertIsNone(result)


class TestFindCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_ckpt"
        self.run_dir = self.tmp / "test_run"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def _touch(self, name: str):
        (self.run_dir / name).touch()

    def test_find_checkpoint_by_step(self):
        self._touch("checkpoint_1000.pt")
        result = find_checkpoint(self.run_dir, 1000)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "checkpoint_1000.pt")

    def test_find_checkpoint_by_step_missing(self):
        result = find_checkpoint(self.run_dir, 999)
        self.assertIsNone(result)

    def test_find_latest_checkpoint(self):
        self._touch("checkpoint_100.pt")
        self._touch("checkpoint_500.pt")
        self._touch("checkpoint_1000.pt")
        result = find_checkpoint(self.run_dir, None)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "checkpoint_1000.pt")

    def test_find_checkpoint_no_files(self):
        result = find_checkpoint(self.run_dir, None)
        self.assertIsNone(result)


class TestFindNearestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).parent / "_test_tmp_near"
        self.ckpt_dir = self.tmp / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp)

    def _touch(self, name: str):
        (self.ckpt_dir / name).touch()

    def test_find_nearest_checkpoint(self):
        self._touch("checkpoint_100.pt")
        self._touch("checkpoint_50.pt")
        self._touch("checkpoint_200.pt")
        result = find_nearest_checkpoint(self.ckpt_dir)
        self.assertEqual(result.name, "checkpoint_200.pt")

    def test_find_nearest_no_files(self):
        result = find_nearest_checkpoint(self.ckpt_dir)
        self.assertIsNone(result)

    def test_find_nearest_non_existent_dir(self):
        result = find_nearest_checkpoint(self.tmp / "nonexistent")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
