from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import dump, load


METADATA_NAME = "metadata.json"
MODEL_NAME = "model.joblib"
REFERENCE_NAME = "reference_features.csv"


@dataclass
class ProductionBundle:
    model: Any
    feature_cols: list[str]
    target_col: str
    timestamp_col: str
    train_end: str
    reference: pd.DataFrame
    long_only: bool = True
    max_names: int = 10
    max_weight: float = 0.05
    gross_exposure: float = 1.0
    cost_bps: float = 10.0
    max_drawdown_pause: float = 0.10
    execution: str = "next_open"
    model_version: str = "v2.0-demonstration"
    checksum: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "checksum": self.checksum,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "timestamp_col": self.timestamp_col,
            "train_end": self.train_end,
            "long_only": self.long_only,
            "max_names": self.max_names,
            "max_weight": self.max_weight,
            "gross_exposure": self.gross_exposure,
            "cost_bps": self.cost_bps,
            "max_drawdown_pause": self.max_drawdown_pause,
            "execution": self.execution,
        }


def save_bundle(bundle: ProductionBundle, model_dir: str | Path) -> Path:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    dump(bundle.model, path / MODEL_NAME)
    bundle.reference.to_csv(path / REFERENCE_NAME, index=False)
    with (path / METADATA_NAME).open("w", encoding="utf-8") as fh:
        json.dump(bundle.to_metadata(), fh, indent=2, default=str)
    return path


def load_bundle(model_dir: str | Path) -> ProductionBundle:
    path = Path(model_dir)
    meta_path = path / METADATA_NAME
    model_path = path / MODEL_NAME
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Frozen model not found in {path}. Run `python main.py train` first."
        )
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    model = load(model_path)
    ref_path = path / REFERENCE_NAME
    reference = pd.read_csv(ref_path) if ref_path.exists() else pd.DataFrame()
    return ProductionBundle(
        model=model,
        feature_cols=list(meta["feature_cols"]),
        target_col=meta.get("target_col", "future_return_5d"),
        timestamp_col=meta.get("timestamp_col", "timestamp"),
        train_end=str(meta.get("train_end", "")),
        reference=reference,
        long_only=bool(meta.get("long_only", True)),
        max_names=int(meta.get("max_names", 5)),
        max_weight=float(meta.get("max_weight", 0.30)),
        gross_exposure=float(meta.get("gross_exposure", 1.0)),
        cost_bps=float(meta.get("cost_bps", 10.0)),
        max_drawdown_pause=float(meta.get("max_drawdown_pause", 0.10)),
        execution=str(meta.get("execution", "next_open")),
    )
