from __future__ import annotations

import argparse
import json
from pathlib import Path

from signalforge.data import generate_synthetic_market_data
from signalforge.ingestion.csv_loader import load_market_csvs
from signalforge.ingestion.market_data import DEFAULT_SYMBOLS, download_nse_daily, save_market_csv
from signalforge.live.artifacts import load_bundle
from signalforge.live.paper import load_ledger, paper_step, save_ledger
from signalforge.live.score import score_universe, write_signals
from signalforge.live.train import train_production_model
from signalforge.pipelines.daily_pipeline import DailyPipeline


def _summarize_result(result: dict) -> dict:
    return {
        "total_candidate_features": result.get("total_candidate_features"),
        "quality_filtered_features": result.get("quality_filtered_features"),
        "selected_features": result.get("selected_features", []),
        "leakage_issues": result.get("leakage_issues", []),
        "top_feature_rankings": result.get("evaluation_ranking", [])[:5],
        "walk_forward_metrics": result.get("walk_forward_metrics", {}),
        "baseline_vs_full_comparison": result.get("baseline_vs_full_comparison", []),
        "backtest_results": result.get("backtest_results", {}),
        "drift_reports": result.get("drift_reports", []),
    }


def cmd_generate_sample(args: argparse.Namespace) -> int:
    df = generate_synthetic_market_data(days=args.days, symbols=tuple(args.symbols.split(",")))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved synthetic sample data to {output_path}")
    else:
        print(df.head().to_markdown(index=False))
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    if args.input:
        df = load_market_csvs(args.input)
    else:
        df = generate_synthetic_market_data(days=args.days, symbols=tuple(args.symbols.split(",")))

    pipeline = DailyPipeline(
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        config_path=args.config,
        selected_limit=args.selected_limit,
    )
    result = pipeline.run(df, feature_lag_map={})

    summary = _summarize_result(result)
    print(json.dumps(summary, indent=2, default=str))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        print(f"Saved summary JSON to {output_path}")

    return 0


def cmd_ingest_csvs(args: argparse.Namespace) -> int:
    df = load_market_csvs(args.input)
    output_path = Path(args.output) if args.output else Path("data/standardized_market.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ingested {len(df)} rows and saved to {output_path}")
    return 0


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def cmd_fetch_eod(args: argparse.Namespace) -> int:
    symbols = _parse_symbols(args.symbols)
    df = download_nse_daily(symbols=symbols, period=args.period)
    path = save_market_csv(df, args.output)
    print(f"Saved {len(df)} real NSE daily rows for {symbols} to {path}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    df = load_market_csvs(args.input)
    summary, saved = train_production_model(
        df,
        args.model_dir,
        selected_limit=args.selected_limit,
        config_path=args.config,
        target_col=args.target_col,
        timestamp_col=args.timestamp_col,
        long_only=not args.allow_short,
        max_names=args.max_names,
        max_weight=args.max_weight,
        cost_bps=args.cost_bps,
    )
    slim = _summarize_result(summary)
    slim["frozen_model_dir"] = str(saved)
    slim["frozen_feature_cols"] = summary.get("frozen_feature_cols")
    slim["train_end"] = summary.get("train_end")
    print(json.dumps(slim, indent=2, default=str))
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "train_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=2, default=str)
    print(f"Frozen production model at {saved}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    df = load_market_csvs(args.input)
    bundle = load_bundle(args.model_dir)
    signals, drift_reports, high_drift = score_universe(df, bundle)
    if high_drift:
        signals["weight"] = 0.0
        print("HIGH feature drift detected; flattening weights to cash.")
    path = write_signals(signals, args.output_dir)
    print(json.dumps({"signal_file": str(path), "drift": drift_reports, "high_drift": high_drift}, indent=2, default=str))
    print(signals.to_string(index=False))
    return 0


def cmd_live_day(args: argparse.Namespace) -> int:
    if args.fetch:
        symbols = _parse_symbols(args.symbols)
        market = download_nse_daily(symbols=symbols, period=args.period)
        save_market_csv(market, args.input)
    else:
        market = load_market_csvs(args.input)

    bundle = load_bundle(args.model_dir)
    signals, drift_reports, high_drift = score_universe(market, bundle)
    if high_drift:
        signals["weight"] = 0.0
        print("HIGH feature drift detected; new orders flattened to cash.")

    signal_path = write_signals(signals, args.signals_dir)
    ledger = load_ledger(args.ledger, starting_cash=args.starting_cash, cost_bps=bundle.cost_bps)
    ledger = paper_step(
        market,
        ledger,
        signals,
        max_drawdown_pause=bundle.max_drawdown_pause,
    )
    save_ledger(ledger, args.ledger)
    print(f"Signals: {signal_path}")
    print(f"Paper equity: {ledger.get('last_equity'):.2f}  cash: {ledger.get('cash'):.2f}  paused: {ledger.get('paused')}")
    if drift_reports:
        print(json.dumps({"drift": drift_reports}, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SignalForge: Multi-Asset Feature Engineering & Selection Platform")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser("generate-sample", help="Generate a synthetic market dataset")
    sample_parser.add_argument("--days", type=int, default=120)
    sample_parser.add_argument("--symbols", type=str, default="RELIANCE,INFY,TCS,HDFCBANK")
    sample_parser.add_argument("--output", type=str, default=None)
    sample_parser.set_defaults(func=cmd_generate_sample)

    run_parser = subparsers.add_parser("run-pipeline", help="Run the feature engine, IC selection, walk-forward, and backtest")
    run_parser.add_argument("--input", type=str, default=None)
    run_parser.add_argument("--days", type=int, default=120)
    run_parser.add_argument("--symbols", type=str, default="RELIANCE,INFY,TCS,HDFCBANK")
    run_parser.add_argument("--config", type=str, default="configs/features.yaml")
    run_parser.add_argument("--target-col", type=str, default="future_return_5d")
    run_parser.add_argument("--timestamp-col", type=str, default="timestamp")
    run_parser.add_argument("--min-ic", type=float, default=0.01)
    run_parser.add_argument("--max-corr", type=float, default=0.90)
    run_parser.add_argument("--selected-limit", type=int, default=15)
    run_parser.add_argument("--output", type=str, default=None)
    run_parser.set_defaults(func=cmd_run_pipeline)

    ingest_parser = subparsers.add_parser("ingest-csvs", help="Normalize a raw exchange CSV directory into a single standardized DataFrame")
    ingest_parser.add_argument("--input", type=str, required=True, help="CSV file or directory of CSVs")
    ingest_parser.add_argument("--output", type=str, default="data/standardized_market.csv")
    ingest_parser.set_defaults(func=cmd_ingest_csvs)

    default_symbols = ",".join(DEFAULT_SYMBOLS)

    fetch_parser = subparsers.add_parser("fetch-eod", help="Download real NSE daily bars (no synthetic fallback)")
    fetch_parser.add_argument("--symbols", type=str, default=default_symbols)
    fetch_parser.add_argument("--period", type=str, default="2y")
    fetch_parser.add_argument("--output", type=str, default="data/eod.csv")
    fetch_parser.set_defaults(func=cmd_fetch_eod)

    train_parser = subparsers.add_parser("train", help="Select features, evaluate, and freeze a production model")
    train_parser.add_argument("--input", type=str, required=True)
    train_parser.add_argument("--model-dir", type=str, default="models")
    train_parser.add_argument("--config", type=str, default="configs/features.yaml")
    train_parser.add_argument("--target-col", type=str, default="future_return_5d")
    train_parser.add_argument("--timestamp-col", type=str, default="timestamp")
    train_parser.add_argument("--selected-limit", type=int, default=15)
    train_parser.add_argument("--max-names", type=int, default=5)
    train_parser.add_argument("--max-weight", type=float, default=0.30)
    train_parser.add_argument("--cost-bps", type=float, default=10.0)
    train_parser.add_argument("--allow-short", action="store_true", help="Allow short weights (not cash long-only)")
    train_parser.set_defaults(func=cmd_train)

    score_parser = subparsers.add_parser("score", help="Score latest bars with the frozen model; write signals/today.csv")
    score_parser.add_argument("--input", type=str, default="data/eod.csv")
    score_parser.add_argument("--model-dir", type=str, default="models")
    score_parser.add_argument("--output-dir", type=str, default="signals")
    score_parser.set_defaults(func=cmd_score)

    live_parser = subparsers.add_parser(
        "live-day",
        help="Daily next-open cash book: score frozen model, write signals, fill paper ledger at next open",
    )
    live_parser.add_argument("--input", type=str, default="data/eod.csv")
    live_parser.add_argument("--model-dir", type=str, default="models")
    live_parser.add_argument("--signals-dir", type=str, default="signals")
    live_parser.add_argument("--ledger", type=str, default="paper/ledger.json")
    live_parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    live_parser.add_argument("--fetch", action="store_true", help="Download fresh NSE EOD before scoring")
    live_parser.add_argument("--symbols", type=str, default=default_symbols)
    live_parser.add_argument("--period", type=str, default="2y")
    live_parser.set_defaults(func=cmd_live_day)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
