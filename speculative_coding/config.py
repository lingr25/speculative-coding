"""Standalone configuration for Speculative Coding with strict provider pool isolation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_CONTEXT_WINDOW_HINTS = (
    ("gemini", 1_048_576),
    ("deepseek", 131_072),
    ("glm", 131_072),
    ("kimi", 262_144),
    ("qwen", 131_072),
    ("claude", 200_000),
    ("gpt-5", 400_000),
)


def infer_context_window(model: str) -> int:
    """Conservative family fallback; an explicit provider context_window wins."""
    lowered = str(model or "").lower()
    for hint, window in _CONTEXT_WINDOW_HINTS:
        if hint in lowered:
            return window
    return 0


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_REF_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


@dataclass
class ProviderEndpoint:
    provider_id: str
    model: str
    api_base: str
    keys: list[str]
    source_id: str = ""
    timeout: int = 120
    modalities: list[str] = field(default_factory=lambda: ["text", "image", "tool_use"])
    custom_headers: dict[str, str] = field(default_factory=dict)
    api_type: str = "openai_chat_completion"
    context_window: int = 0
    max_output_tokens: int = 0
    omit_max_tokens: bool = False
    pool: str = "speculative"
    prefer_stream: bool = True

    def __post_init__(self) -> None:
        if self.pool not in {"main", "speculative"}:
            raise ValueError(f"invalid provider pool: {self.pool!r}")
        if not self.context_window:
            self.context_window = infer_context_window(self.model)

    @property
    def base_url(self) -> str:
        return self.api_base


@dataclass
class SpecConfig:
    """Self-contained runtime configuration for the Speculative Coding control plane."""

    data_dir: Path = field(default_factory=lambda: Path("./data"))
    db_path: Path | None = None
    sessions_db_path: Path | None = None
    providers: dict[str, ProviderEndpoint] = field(default_factory=dict)
    default_provider_id: str = ""
    speculative_provider_id: str = ""
    fallback_provider_ids: list[str] = field(default_factory=list)
    cooldown_enabled: bool = True
    cooldown_fail_threshold: int = 2
    cooldown_seconds: float = 180.0
    max_cooldown_seconds: float = 1800.0
    worker_token: str = ""
    api_token: str = ""
    default_owner_id: str = "local"
    default_timezone: str = "UTC"
    github_app_id: str = ""
    github_installation_id: str = ""
    github_private_key_path: str = ""

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.db_path is None:
            self.db_path = self.data_dir / "speculative.db"
        else:
            self.db_path = Path(self.db_path)
        if self.sessions_db_path is None:
            self.sessions_db_path = self.data_dir / "sessions.db"
        else:
            self.sessions_db_path = Path(self.sessions_db_path)
        self._sanitize_pool_defaults()

    def _sanitize_pool_defaults(self) -> None:
        """Guarantee strict bidirectional pool isolation between main and speculative pools."""
        if self.default_provider_id:
            ep = self.providers.get(self.default_provider_id)
            if not ep or ep.pool != "main":
                self.default_provider_id = ""
        if self.speculative_provider_id:
            ep = self.providers.get(self.speculative_provider_id)
            if not ep or ep.pool != "speculative":
                self.speculative_provider_id = ""
        self.fallback_provider_ids = [
            pid
            for pid in self.fallback_provider_ids
            if pid in self.providers and self.providers[pid].pool == "main"
        ]

    def set_main_default(self, provider_id: str) -> None:
        ep = self.providers.get(provider_id)
        if not ep or ep.pool != "main":
            raise ValueError(f"provider {provider_id!r} must belong to the 'main' pool")
        self.default_provider_id = provider_id

    def set_speculative_default(self, provider_id: str) -> None:
        ep = self.providers.get(provider_id)
        if not ep or ep.pool != "speculative":
            raise ValueError(f"provider {provider_id!r} must belong to the 'speculative' pool")
        self.speculative_provider_id = provider_id

    def providers_for_pool(self, pool: str) -> list[ProviderEndpoint]:
        if pool not in {"main", "speculative"}:
            raise ValueError(f"invalid provider pool: {pool!r}")
        return [ep for ep in self.providers.values() if ep.pool == pool]

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_dir: Path | None = None) -> SpecConfig:
        expanded = _expand_env(raw)
        root = Path(base_dir) if base_dir is not None else Path(".")
        data_dir_raw = expanded.get("data_dir") or "./data"
        data_dir = Path(data_dir_raw)
        if not data_dir.is_absolute():
            data_dir = (root / data_dir).resolve()

        providers: dict[str, ProviderEndpoint] = {}
        for item in expanded.get("providers") or []:
            keys = item.get("keys")
            if isinstance(keys, str):
                keys = [k for k in (keys,) if k]
            elif keys is None and item.get("api_key"):
                keys = [str(item["api_key"])]
            else:
                keys = [str(k) for k in (keys or []) if str(k).strip()]
            ep = ProviderEndpoint(
                provider_id=str(item["provider_id"]),
                source_id=str(item.get("source_id") or item["provider_id"]),
                model=str(item["model"]),
                api_base=str(item.get("api_base") or item.get("base_url") or ""),
                keys=keys,
                timeout=int(item.get("timeout", 120)),
                modalities=list(item.get("modalities") or ["text", "image", "tool_use"]),
                custom_headers=dict(item.get("custom_headers") or {}),
                api_type=str(item.get("api_type") or "openai_chat_completion"),
                context_window=int(item.get("context_window") or 0),
                max_output_tokens=int(item.get("max_output_tokens") or 0),
                omit_max_tokens=bool(item.get("omit_max_tokens", False)),
                pool=str(item.get("pool") or "speculative"),
                prefer_stream=bool(item.get("prefer_stream", True)),
            )
            providers[ep.provider_id] = ep

        cooldown = expanded.get("cooldown") or {}
        github = expanded.get("github_app") or {}
        return cls(
            data_dir=data_dir,
            db_path=Path(expanded["db_path"]) if expanded.get("db_path") else None,
            sessions_db_path=Path(expanded["sessions_db_path"]) if expanded.get("sessions_db_path") else None,
            providers=providers,
            default_provider_id=str(expanded.get("default_provider_id") or ""),
            speculative_provider_id=str(expanded.get("speculative_provider_id") or ""),
            fallback_provider_ids=list(expanded.get("fallback_provider_ids") or []),
            cooldown_enabled=bool(cooldown.get("enable", True)),
            cooldown_fail_threshold=max(1, int(cooldown.get("failures", 2))),
            cooldown_seconds=max(5.0, float(cooldown.get("cooldown_seconds", 180.0))),
            max_cooldown_seconds=max(5.0, float(cooldown.get("max_cooldown_seconds", 1800.0))),
            worker_token=str(expanded.get("worker_token") or os.environ.get("SPEC_WORKER_TOKEN", "")).strip(),
            api_token=str(expanded.get("api_token") or os.environ.get("SPEC_API_TOKEN", "")).strip(),
            default_owner_id=str(expanded.get("default_owner_id") or os.environ.get("SPEC_OWNER_ID", "local")).strip(),
            default_timezone=str(expanded.get("default_timezone") or os.environ.get("SPEC_TIMEZONE", "UTC")).strip(),
            github_app_id=str(github.get("app_id") or os.environ.get("SPEC_GITHUB_APP_ID", "")).strip(),
            github_installation_id=str(
                github.get("installation_id") or os.environ.get("SPEC_GITHUB_APP_INSTALLATION_ID", "")
            ).strip(),
            github_private_key_path=str(
                github.get("private_key_path") or os.environ.get("SPEC_GITHUB_APP_PRIVATE_KEY_PATH", "")
            ).strip(),
        )

    @classmethod
    def load_from_file(cls, path: str | Path) -> SpecConfig:
        file_path = Path(path).resolve()
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a JSON object")
        return cls.from_dict(raw, base_dir=file_path.parent)

    @classmethod
    def from_env(cls, *, data_dir: Path | None = None) -> SpecConfig:
        config_path = os.environ.get("SPEC_CONFIG_PATH", "").strip()
        if config_path and Path(config_path).is_file():
            return cls.load_from_file(config_path)

        providers: dict[str, ProviderEndpoint] = {}
        spec_default = ""
        api_key = os.environ.get("SPEC_OPENAI_API_KEY", "").strip()
        model = os.environ.get("SPEC_MODEL", "").strip()
        if api_key and model:
            ep = ProviderEndpoint(
                provider_id="spec-default",
                source_id="env-openai",
                model=model,
                api_base=os.environ.get("SPEC_OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
                keys=[api_key],
                api_type=os.environ.get("SPEC_API_TYPE", "openai_chat_completion").strip(),
                pool="speculative",
            )
            providers[ep.provider_id] = ep
            spec_default = ep.provider_id

        resolved_data = data_dir or Path(os.environ.get("SPEC_DATA_DIR", "./data"))
        return cls(
            data_dir=resolved_data,
            providers=providers,
            speculative_provider_id=spec_default,
            worker_token=os.environ.get("SPEC_WORKER_TOKEN", "").strip(),
            api_token=os.environ.get("SPEC_API_TOKEN", "").strip(),
            default_owner_id=os.environ.get("SPEC_OWNER_ID", "local").strip(),
            default_timezone=os.environ.get("SPEC_TIMEZONE", "UTC").strip(),
            github_app_id=os.environ.get("SPEC_GITHUB_APP_ID", "").strip(),
            github_installation_id=os.environ.get("SPEC_GITHUB_APP_INSTALLATION_ID", "").strip(),
            github_private_key_path=os.environ.get("SPEC_GITHUB_APP_PRIVATE_KEY_PATH", "").strip(),
        )
