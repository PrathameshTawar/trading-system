"""SignalForge package."""

__all__ = [
    "FeatureFactory",
    "FeatureSelector",
    "FeatureRegistry",
    "LeakageValidator",
    "WalkForwardRegressor",
    "generate_synthetic_market_data",
    "load_market_csvs",
    "compute_ic",
    "compute_sharpe",
]

from signalforge.data import generate_synthetic_market_data
from signalforge.features.factory import FeatureFactory
from signalforge.features.registry import FeatureRegistry
from signalforge.ingestion.csv_loader import load_market_csvs
from signalforge.selection.selector import FeatureSelector
from signalforge.validation.leakage import LeakageValidator
from signalforge.modeling.walk_forward import WalkForwardRegressor
from signalforge.backtesting.metrics import compute_ic, compute_sharpe
