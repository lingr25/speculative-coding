"""Regression tests for strict speculative vs main provider pool isolation and config loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speculative_coding.config import ProviderEndpoint, SpecConfig
from speculative_coding.llm.router import ProviderRouter


class ProviderPoolIsolationTests(unittest.TestCase):
    def _config_from_raw(self, raw: dict) -> SpecConfig:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg_path = root / "config.json"
        cfg_path.write_text(json.dumps(raw), encoding="utf-8")
        return SpecConfig.load_from_file(cfg_path)

    def test_main_defaults_ignore_spec_only_config_and_fallbacks(self):
        cfg = self._config_from_raw(
            {
                "default_provider_id": "egg",
                "fallback_provider_ids": ["egg"],
                "providers": [
                    {
                        "provider_id": "egg",
                        "model": "free-model",
                        "api_base": "https://example.invalid/v1",
                        "keys": ["test-key"],
                        "pool": "speculative",
                    }
                ],
            }
        )
        self.assertEqual(cfg.default_provider_id, "")
        self.assertEqual(cfg.fallback_provider_ids, [])
        self.assertEqual([ep.provider_id for ep in cfg.providers_for_pool("main")], [])
        self.assertEqual([ep.provider_id for ep in cfg.providers_for_pool("speculative")], ["egg"])

    def test_pool_overrides_reject_cross_pool_providers(self):
        cfg = SpecConfig(
            providers={
                "main": ProviderEndpoint(
                    provider_id="main",
                    model="paid-model",
                    api_base="https://example.invalid/v1",
                    keys=["k1"],
                    pool="main",
                ),
                "egg": ProviderEndpoint(
                    provider_id="egg",
                    model="free-model",
                    api_base="https://example.invalid/v1",
                    keys=["k2"],
                    pool="speculative",
                ),
            },
            default_provider_id="main",
            speculative_provider_id="egg",
        )
        with self.assertRaisesRegex(ValueError, "main"):
            cfg.set_main_default("egg")
        with self.assertRaisesRegex(ValueError, "speculative"):
            cfg.set_speculative_default("main")

        main_router = ProviderRouter(cfg, pool="main")
        spec_router = ProviderRouter(cfg, pool="speculative")
        self.assertEqual([ep.provider_id for ep in main_router.candidate_chain()], ["main"])
        self.assertEqual([ep.provider_id for ep in spec_router.candidate_chain()], ["egg"])

        # Even if preferred_id points to the other pool, router strictly filters by its pool.
        self.assertEqual([ep.provider_id for ep in spec_router.candidate_chain(preferred_id="main")], ["egg"])
        self.assertEqual([ep.provider_id for ep in main_router.candidate_chain(preferred_id="egg")], ["main"])

    def test_speculative_pool_cooling_never_falls_back_to_main(self):
        cfg = SpecConfig(
            providers={
                "main": ProviderEndpoint(
                    provider_id="main",
                    model="paid-model",
                    api_base="https://example.invalid/v1",
                    keys=["k1"],
                    pool="main",
                ),
                "egg": ProviderEndpoint(
                    provider_id="egg",
                    model="free-model",
                    api_base="https://example.invalid/v1",
                    keys=["k2"],
                    pool="speculative",
                ),
            },
            default_provider_id="main",
            speculative_provider_id="egg",
            cooldown_fail_threshold=1,
            cooldown_seconds=60.0,
        )
        spec_router = ProviderRouter(cfg, pool="speculative")
        spec_router._mark_failure("egg")
        self.assertEqual(spec_router.candidate_chain(), [])

    def test_config_env_var_interpolation(self):
        with patch.dict("os.environ", {"TEST_SPEC_KEY": "interpolated-key", "TEST_SPEC_TOKEN": "t" * 32}, clear=False):
            cfg = self._config_from_raw(
                {
                    "worker_token": "${TEST_SPEC_TOKEN}",
                    "providers": [
                        {
                            "provider_id": "spec-1",
                            "model": "deepseek-chat",
                            "api_base": "https://example.invalid/v1",
                            "keys": ["${TEST_SPEC_KEY}"],
                            "pool": "speculative",
                        }
                    ],
                }
            )
        self.assertEqual(cfg.worker_token, "t" * 32)
        self.assertEqual(cfg.providers["spec-1"].keys, ["interpolated-key"])
        self.assertEqual(cfg.providers["spec-1"].context_window, 131_072)


if __name__ == "__main__":
    unittest.main()
