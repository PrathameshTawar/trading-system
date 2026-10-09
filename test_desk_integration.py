from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from signalforge.live.ledger_sync import export_ledger_to_csv, import_csv_to_ledger
from signalforge.mcp_server import handle_tool_call
from signalforge.data import generate_synthetic_market_data
from signalforge.live.train import train_production_model


def test_desk_rules_exist_and_contain_bot_roles(tmp_path):
    rules_file = Path("desk/my-rules.md")
    assert rules_file.exists(), "desk/my-rules.md should exist"
    content = rules_file.read_text(encoding="utf-8")
    assert "WATCHLIST =" in content
    assert "Quant" in content
    assert "Skeptic" in content
    assert "older than 1 trading day" in content


def test_ledger_sync_export_and_import(tmp_path):
    sample_ledger = {
        "starting_cash": 1_000_000.0,
        "cash": 800_000.0,
        "last_equity": 1_050_000.0,
        "peak_equity": 1_050_000.0,
        "paused": False,
        "pause_reason": "",
        "positions": {
            "RELIANCE": {"qty": 100, "avg_price": 2500.0}
        }
    }
    
    csv_file = tmp_path / "paper-ledger.csv"
    export_ledger_to_csv(sample_ledger, csv_file)
    assert csv_file.exists()
    
    df = pd.read_csv(csv_file)
    assert not df.empty
    assert "RELIANCE" in df["symbol"].values
    assert float(df.loc[df["symbol"] == "RELIANCE", "quantity"].iloc[0]) == 100
    
    json_out = tmp_path / "ledger.json"
    reimported = import_csv_to_ledger(csv_file, json_out)
    assert json_out.exists()
    assert reimported["cash"] == 800_000.0
    assert "RELIANCE" in reimported["positions"]
    assert reimported["positions"]["RELIANCE"]["qty"] == 100


def test_mcp_server_tool_handlers(tmp_path):
    status_res = handle_tool_call("get_paper_status", {"ledger_path": "nonexistent.json"})
    assert status_res["status"] == "success"
    assert status_res["cash"] == 1_000_000.0
    
    # Generate synthetic market data & train model
    market_df = generate_synthetic_market_data(days=120, symbols=("RELIANCE", "TCS"))
    market_csv = tmp_path / "market.csv"
    market_df.to_csv(market_csv, index=False)
    
    model_dir = tmp_path / "models"
    train_production_model(market_df, model_dir=str(model_dir), selected_limit=5)
    
    output_dir = tmp_path / "desk_out"
    score_res = handle_tool_call("score_universe", {
        "input_file": str(market_csv),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir)
    })
    assert score_res["status"] == "success"
    assert Path(score_res["signal_file"]).exists()
