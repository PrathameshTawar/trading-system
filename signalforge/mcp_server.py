from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from signalforge.ingestion.market_data import download_nse_daily, save_market_csv
from signalforge.live.artifacts import load_bundle
from signalforge.live.ledger_sync import export_ledger_to_csv, import_csv_to_ledger
from signalforge.live.paper import load_ledger, paper_step, save_ledger
from signalforge.live.score import score_universe, write_signals
from signalforge.live.train import train_production_model
from signalforge.pipelines.daily_pipeline import DailyPipeline


TOOLS_MANIFEST = [
    {
        "name": "fetch_eod",
        "description": "Download real daily market bars for specified symbols (e.g., RELIANCE, TCS, INFY)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "symbols": {"type": "string", "description": "Comma-separated stock ticker symbols"},
                "period": {"type": "string", "default": "2y", "description": "Data history period (e.g. 1y, 2y, 5y)"},
                "output": {"type": "string", "default": "data/eod.csv", "description": "Destination CSV path"}
            },
            "required": ["symbols"]
        }
    },
    {
        "name": "score_universe",
        "description": "Score latest daily bars against frozen model and output signals to desk/signals.csv",
        "inputSchema": {
            "type": "object",
            "properties": {
                "input_file": {"type": "string", "default": "data/eod.csv", "description": "Market data CSV"},
                "model_dir": {"type": "string", "default": "models", "description": "Directory of frozen model bundle"},
                "output_dir": {"type": "string", "default": "desk", "description": "Directory to write signals.csv"}
            }
        }
    },
    {
        "name": "get_paper_status",
        "description": "Retrieve current paper trading ledger status, cash, positions, equity, and pause state",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ledger_path": {"type": "string", "default": "paper/ledger.json"}
            }
        }
    },
    {
        "name": "sync_desk_ledger",
        "description": "Synchronize paper/ledger.json with desk/paper-ledger.csv",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ledger_path": {"type": "string", "default": "paper/ledger.json"},
                "csv_path": {"type": "string", "default": "desk/paper-ledger.csv"}
            }
        }
    }
]


def handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handler for MCP tool execution."""
    if name == "fetch_eod":
        raw_syms = arguments.get("symbols", "RELIANCE,TCS,INFY,HDFCBANK")
        symbols = [s.strip().upper() for s in raw_syms.split(",") if s.strip()]
        period = arguments.get("period", "2y")
        output = arguments.get("output", "data/eod.csv")
        
        df = download_nse_daily(symbols=symbols, period=period)
        path = save_market_csv(df, output)
        return {
            "status": "success",
            "symbols": symbols,
            "rows_downloaded": len(df),
            "output_path": str(path)
        }

    elif name == "score_universe":
        input_file = arguments.get("input_file", "data/eod.csv")
        model_dir = arguments.get("model_dir", "models")
        output_dir = arguments.get("output_dir", "desk")
        
        from signalforge.ingestion.csv_loader import load_market_csvs
        df = load_market_csvs(input_file)
        bundle = load_bundle(model_dir)
        signals, drift_reports, high_drift = score_universe(df, bundle)
        
        if high_drift:
            signals["weight"] = 0.0
            
        path = write_signals(signals, output_dir)
        # Also mirror to desk/signals.csv if output_dir isn't desk
        desk_path = Path("desk/signals.csv")
        desk_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_csv(desk_path, index=False)
        
        return {
            "status": "success",
            "signal_file": str(path),
            "desk_signal_file": str(desk_path),
            "high_drift_flag": high_drift,
            "signals": signals.to_dict("records") if not signals.empty else []
        }

    elif name == "get_paper_status":
        ledger_path = arguments.get("ledger_path", "paper/ledger.json")
        ledger = load_ledger(ledger_path)
        return {
            "status": "success",
            "cash": ledger.get("cash", 0.0),
            "last_equity": ledger.get("last_equity", ledger.get("cash", 0.0)),
            "peak_equity": ledger.get("peak_equity", ledger.get("cash", 0.0)),
            "paused": ledger.get("paused", False),
            "positions": ledger.get("positions", {})
        }

    elif name == "sync_desk_ledger":
        ledger_path = arguments.get("ledger_path", "paper/ledger.json")
        csv_path = arguments.get("csv_path", "desk/paper-ledger.csv")
        exported_path = export_ledger_to_csv(ledger_path, csv_path)
        return {
            "status": "success",
            "exported_csv": str(exported_path)
        }

    else:
        raise ValueError(f"Unknown tool name: {name}")


def run_stdio_mcp_server() -> None:
    """Stdio JSON-RPC MCP Server loop."""
    sys.stderr.write("SignalForge MCP Server starting on stdio...\n")
    sys.stderr.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as err:
            sys.stderr.write(f"Invalid JSON: {err}\n")
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "SignalForge-MCP-Server", "version": "1.0.0"}
                }
            }
        elif method == "tools/list":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS_MANIFEST}
            }
        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result_data = handle_tool_call(name, arguments)
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result_data, indent=2)}]
                    }
                }
            except Exception as e:
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }
        else:
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_mcp_server()
