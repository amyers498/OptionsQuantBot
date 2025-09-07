from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _resolve_env_refs(node: Any) -> Any:
    """Recursively resolve values like 'ENV:VAR_NAME' from environment variables."""
    if isinstance(node, dict):
        return {k: _resolve_env_refs(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_env_refs(v) for v in node]
    if isinstance(node, str) and node.startswith("ENV:"):
        env_key = node.split(":", 1)[1]
        return os.getenv(env_key, "")
    return node


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _resolve_env_refs(cfg)

