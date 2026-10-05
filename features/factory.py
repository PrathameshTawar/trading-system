import pandas as pd

from signalforge.features.cross_asset import add_cross_asset_features
from signalforge.features.nlp import add_nlp_features
from signalforge.features.price import add_price_features
from signalforge.features.registry import FeatureRegistry
from signalforge.features.sectors import add_sector_tags, compute_sector_neutral_targets
from signalforge.modeling.sanitize import sanitize_frame


class FeatureFactory:
    """Automated Feature Factory generating price, volume, volatility, cross-asset, sector, and Indic NLP financial features."""

    def __init__(self, config_path: str | None = None):
        self.registry = FeatureRegistry(config_path=config_path or "configs/features.yaml")
        self.config = self.registry.config if hasattr(self.registry, "config") else {}

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate candidate feature matrix from raw input dataframe."""
        out = df.copy()

        # Add Sector tags & sector-neutralized target
        out = add_sector_tags(out)
        out = compute_sector_neutral_targets(out, target_col="future_return_5d")

        # Initialize news sentiment & Indic features if missing
        if "news_sentiment" not in out.columns:
            out["news_sentiment"] = 0.0

        out = add_price_features(out)
        out = add_cross_asset_features(out)
        out = add_nlp_features(out)

        out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        feature_cols = self.get_feature_names(out)
        return sanitize_frame(out, feature_cols)

    def get_feature_names(self, df: pd.DataFrame | None = None) -> list[str]:
        """Return list of scale-free stationary candidate feature column names.

        Excludes raw price level indicators that leak symbol identity to tree models.
        """
        raw_exact = {
            "timestamp", "symbol", "sector", "open", "high", "low", "close", "volume", "turnover",
            "adj_close", "nifty_close", "obv", "volume_price_trend", "atr_14",
            "future_return_5d", "future_return_1d", "future_return_5d_sn", "headline"
        }

        def is_candidate(col: str) -> bool:
            if col in raw_exact or col.startswith("future_"):
                return False
            # Retain scale-free normalized features
            if col.startswith("norm_") or col.startswith("indic_") or col.startswith("sentiment_"):
                return True
            # Exclude raw level indicators like sma_*, ema_*, std_*, atr_*
            if col.startswith("sma_") or col.startswith("ema_") or col.startswith("std_") or col.startswith("atr_"):
                return False
            return True

        if df is not None:
            return [col for col in df.columns if is_candidate(col)]
        return [f for f in self.registry.feature_names if is_candidate(f)]
