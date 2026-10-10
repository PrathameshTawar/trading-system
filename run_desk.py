#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from agents import chief
from signalforge.ingestion.csv_loader import load_market_csvs
from signalforge.ingestion.market_data import download_nse_daily, save_market_csv


def load_rules(path: str | Path = "configs/rules.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {
            "mode": "paper_only",
            "balance": 1_000_000.0,
            "watchlist": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"],
            "max_size_pct": 5.0,
            "max_loss_pct": 1.0,
            "daily_loss_limit_pct": 2.0,
            "max_cards_per_day": 3,
            "min_price": 50.0,
            "embargo_days": 1,
        }
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="SignalForge Indigenous Autonomous Quant Desk")
    parser.add_argument("--config", type=str, default="configs/rules.yaml", help="Path to rules YAML config")
    parser.add_argument("--data", type=str, default="data/eod.csv", help="Market data CSV path")
    parser.add_argument("--fetch", action="store_true", help="Download fresh EOD market data before running")
    parser.add_argument("--loss-pct", type=float, default=0.0, help="Today's portfolio loss percentage for Skeptic check")
    args = parser.parse_args()

    rules = load_rules(args.config)

    if args.fetch:
        symbols = rules.get("watchlist", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"])
        print(f"Downloading fresh EOD market data for {symbols}...")
        df_fetched = download_nse_daily(symbols=symbols, period="2y")
        save_market_csv(df_fetched, args.data)

    df = load_market_csvs(args.data)
    print(f"Loaded {len(df)} market rows from {args.data}")

    cards = chief.run(df, rules, todays_loss_pct=args.loss_pct)

    print("\n==================================================")
    print("      INDIGENOUS DESK GENERATED DECISION CARDS")
    print("==================================================\n")

    if not cards:
        print("No leads passed Skeptic hard risk gates today.")
    else:
        for card in cards:
            print(card)
            print("\n--------------------------------------------------\n")

    print(f"\nDecision cards written to desk/decision_cards.md")
    print(f"Paper ledger synchronized to desk/paper-ledger.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
