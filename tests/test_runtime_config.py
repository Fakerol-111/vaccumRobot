"""配置层测试：configs/runtime_config.py"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from configs.runtime_config import (
    get_default_train_config_path,
    get_default_test_config_path,
    load_train_config_bundle,
    load_test_config_bundle,
    build_multi_env_configs,
    _parse_ppo,
    _parse_env,
    _parse_curriculum,
    _parse_dashboard,
    _parse_metrics,
    _parse_general,
    _parse_training,
    load_ppo_config,
    load_env_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDefaultPaths(unittest.TestCase):
    def test_default_train_config_path(self):
        path = get_default_train_config_path()
        self.assertTrue(path.exists(), f"train config not found: {path}")
        self.assertEqual(path.suffix, ".toml")
        self.assertIn("train_config", path.name)

    def test_default_test_config_path(self):
        path = get_default_test_config_path()
        self.assertTrue(path.exists(), f"test config not found: {path}")
        self.assertEqual(path.suffix, ".toml")
        self.assertIn("test_config", path.name)


class TestLoadTrainConfigBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_train_config_bundle()

    def test_bundle_is_namespace(self):
        self.assertIsInstance(self.bundle, SimpleNamespace)

    def test_ppo_section(self):
        ppo = self.bundle.ppo
        self.assertIsInstance(ppo, SimpleNamespace)
        for attr in ("learning_rate", "gamma", "gae_lambda", "clip_epsilon",
                     "batch_size", "mini_batch_size", "total_timesteps",
                     "save_interval", "log_interval"):
            self.assertTrue(hasattr(ppo, attr), f"ppo missing: {attr}")

    def test_ppo_values_are_typed(self):
        ppo = self.bundle.ppo
        self.assertIsInstance(ppo.learning_rate, float)
        self.assertIsInstance(ppo.gamma, float)
        self.assertIsInstance(ppo.batch_size, int)
        self.assertIsInstance(ppo.total_timesteps, int)

    def test_env_section(self):
        env = self.bundle.env
        self.assertIsInstance(env, dict)
        self.assertIn("default_map_list", env)
        self.assertIn("default_npc_count", env)
        self.assertIn("default_station_count", env)
        self.assertIn("map_strategy", env)

    def test_curriculum_section(self):
        cur = self.bundle.curriculum
        self.assertIsInstance(cur, dict)
        self.assertIn("enabled", cur)
        self.assertIn("stages", cur)
        self.assertIsInstance(cur["stages"], list)

    def test_dashboard_section(self):
        dash = self.bundle.dashboard
        self.assertIsInstance(dash, dict)
        self.assertIn("enabled", dash)
        self.assertIn("host", dash)
        self.assertIn("port", dash)

    def test_metrics_section(self):
        metrics = self.bundle.metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn("max_updates", metrics)
        self.assertIn("max_episodes", metrics)

    def test_general_section(self):
        general = self.bundle.general
        self.assertIsInstance(general, dict)
        self.assertIn("seed", general)

    def test_training_section(self):
        training = self.bundle.training
        self.assertIsInstance(training, dict)
        self.assertIn("artifacts_dir", training)

    def test_config_path_is_set(self):
        self.assertIsInstance(self.bundle.config_path, Path)
        self.assertTrue(str(self.bundle.config_path).endswith("train_config.toml"))

    def test_custom_config_path(self):
        custom = PROJECT_ROOT / "configs" / "train_config.toml"
        bundle = load_train_config_bundle(custom)
        self.assertEqual(bundle.config_path, custom)


class TestLoadTestConfigBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_test_config_bundle()

    def test_returns_dict(self):
        self.assertIsInstance(self.cfg, dict)

    def test_has_required_keys(self):
        for key in ("maps", "episodes", "npc_count", "station_count",
                    "run_id", "step", "gif_fps", "output_dir"):
            self.assertIn(key, self.cfg, f"missing key: {key}")

    def test_types(self):
        self.assertIsInstance(self.cfg["maps"], list)
        self.assertIsInstance(self.cfg["episodes"], int)
        self.assertIsInstance(self.cfg["gif_fps"], int)

    def test_custom_config_path(self):
        custom = PROJECT_ROOT / "configs" / "test_config_quick.toml"
        cfg = load_test_config_bundle(custom)
        self.assertIsInstance(cfg, dict)


class TestParseFunctions(unittest.TestCase):
    def test_parse_ppo_with_defaults(self):
        raw = {"ppo": {"learning_rate": 0.001}}
        ppo = _parse_ppo(raw)
        self.assertEqual(ppo.learning_rate, 0.001)
        self.assertEqual(ppo.gamma, 0.99)

    def test_parse_env(self):
        raw = {"env": {"default_npc_count": "2"}}
        env = _parse_env(raw)
        self.assertEqual(env["default_npc_count"], 2)

    def test_parse_curriculum_disabled(self):
        raw = {}
        cur = _parse_curriculum(raw)
        self.assertFalse(cur["enabled"])
        self.assertEqual(cur["stages"], [])

    def test_parse_dashboard_disabled(self):
        raw = {}
        dash = _parse_dashboard(raw)
        self.assertFalse(dash["enabled"])
        self.assertEqual(dash["host"], "0.0.0.0")
        self.assertEqual(dash["port"], 8088)

    def test_parse_metrics_defaults(self):
        raw = {}
        m = _parse_metrics(raw)
        self.assertEqual(m["max_updates"], 500)
        self.assertEqual(m["max_episodes"], 500)

    def test_parse_general_defaults(self):
        raw = {}
        g = _parse_general(raw)
        self.assertEqual(g["seed"], 42)

    def test_parse_training(self):
        raw = {"training": {"artifacts_dir": "test_artifacts"}}
        t = _parse_training(raw)
        self.assertEqual(t["artifacts_dir"], "test_artifacts")


class TestBuildMultiEnvConfigs(unittest.TestCase):
    def test_build_env_configs(self):
        configs = build_multi_env_configs([1, 2], npc_count=2, station_count=3)
        self.assertEqual(len(configs), 2)
        for cfg in configs:
            self.assertIsInstance(cfg, dict)
            self.assertIn("custom_map", cfg)
            self.assertIn("npc_count", cfg)
            self.assertEqual(cfg["npc_count"], 2)
            self.assertEqual(cfg["station_count"], 3)

    def test_single_map(self):
        configs = build_multi_env_configs([1], npc_count=0, station_count=1)
        self.assertEqual(len(configs), 1)


class TestCompatFunctions(unittest.TestCase):
    def test_load_ppo_config(self):
        ppo = load_ppo_config()
        self.assertIsInstance(ppo, SimpleNamespace)
        self.assertTrue(hasattr(ppo, "learning_rate"))

    def test_load_env_config(self):
        env = load_env_config()
        self.assertIsInstance(env, dict)
        self.assertIn("default_map_list", env)


if __name__ == "__main__":
    unittest.main()
