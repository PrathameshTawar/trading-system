from __future__ import annotations

from pathlib import Path

import yaml


class FeatureRegistry:
    """Loads declarative feature definitions from YAML and exposes their names."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path is not None else Path("configs/features.yaml")
        self.specs = self._load_specs()

    def _load_specs(self) -> list[dict]:
        if not self.config_path.exists():
            return []
        with self.config_path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        features = payload.get("features", [])
        if not isinstance(features, list):
            return []
        return features

    @property
    def feature_names(self) -> list[str]:
        return [spec.get("name") for spec in self.specs if isinstance(spec, dict) and spec.get("name")]

    def get_specs(self) -> list[dict]:
        return list(self.specs)

    def filter_names(self, available: list[str]) -> list[str]:
        return [name for name in self.feature_names if name in available]
