from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd

from . import scout, charts, quant, news, sentiment, skeptic
from .base import ask
from signalforge.live.ledger_sync import export_ledger_to_csv, import_csv_to_ledger
from signalforge.live.paper import load_ledger, save_ledger


def run(
    df: pd.DataFrame,
    rules: dict[str, Any],
    headlines_by_symbol: dict[str, list[str]] | None = None,
    todays_loss_pct: float = 0.0,
    ledger_path: str | Path = "paper/ledger.json",
    csv_ledger_path: str | Path = "desk/paper-ledger.csv",
) -> list[str]:
    """Orchestrates indigenous multi-agent desk pipeline."""
    headlines_by_symbol = headlines_by_symbol or {}
    max_cards = int(rules.get("max_cards_per_day", 3))
    cards: list[str] = []
    skipped_log: list[dict[str, Any]] = []

    leads = scout.find_leads(df, rules)
    ledger = load_ledger(ledger_path)

    for i, lead in enumerate(leads, 1):
        lead_id = f"#L{i:03d}"
        sym = lead["symbol"]

        c = charts.levels(df, sym)
        q = quant.score(df, sym)
        n = news.check(sym, headlines_by_symbol.get(sym, []))
        s = sentiment.evaluate_sentiment(df, sym)

        verdict, why = skeptic.review(lead, c, q, rules, todays_loss_pct)

        if verdict == "REJECT":
            skipped_log.append({"lead_id": lead_id, "symbol": sym, "verdict": verdict, "reason": why})
            continue

        # Format Decision Card via Base/LLM Helper
        card_prompt = (
            f"LEAD ID: {lead_id}\n"
            f"SYMBOL: {sym} | Price: ₹{c['price']} | Move: {lead['price_move_pct']}%\n"
            f"LEVELS: Entry=₹{c['entry']}, Stop=₹{c['stop']}, Target=₹{c['target']}, ATR-14=₹{c['atr_14']}\n"
            f"QUANT MODEL: Pred Return={q['prediction']}, OOS IC={q['oos_ic']}, NW t-stat={q['nw_tstat']}, Drift={q['high_drift']}\n"
            f"NEWS & SENTIMENT: Score={s['composite_sentiment']} ({s['status']})\n"
            f"VERDICT: {verdict} ({why})"
        )

        card_text = ask(
            system="You are Chief, an executive quantitative desk coordinator. Write a clean 5-line decision summary card for beginners.",
            user=card_prompt
        )

        cards.append(card_text)

        # Log paper trading record
        if verdict in ("APPROVE", "REDUCE"):
            positions = ledger.get("positions", {})
            max_size_pct = float(rules.get("max_size_pct", 5.0))
            if verdict == "REDUCE":
                max_size_pct *= 0.5

            allocated_cash = float(ledger.get("cash", 1_000_000.0)) * (max_size_pct / 100.0)
            fill_price = c["entry"]
            qty = int(allocated_cash / fill_price) if fill_price > 0 else 0

            if qty > 0:
                positions[sym] = {"qty": qty, "avg_price": fill_price}
                ledger["positions"] = positions
                save_ledger(ledger, ledger_path)
                export_ledger_to_csv(ledger, csv_ledger_path)

        if len(cards) >= max_cards:
            break

    # Write cards to desk/decision_cards.md
    out_cards = Path("desk/decision_cards.md")
    out_cards.parent.mkdir(parents=True, exist_ok=True)
    with out_cards.open("w", encoding="utf-8") as f:
        f.write("# Executive Quant Desk Decision Cards\n\n")
        f.write("\n\n---\n\n".join(cards) if cards else "No leads approved by Skeptic hard risk gates today.")
        if skipped_log:
            f.write("\n\n## Skipped Leads Audit Log\n\n")
            for item in skipped_log:
                f.write(f"- **{item['lead_id']} ({item['symbol']})**: REJECTED - {item['reason']}\n")

    return cards
