from __future__ import annotations

import pandas as pd
import numpy as np


# Statutory NSE / Zerodha Equity Charges Schedule (2026)
STT_DELIVERY = 0.001          # 0.1% STT on buy and sell for delivery
NSE_TURNOVER_FEE = 0.0000297     # 0.00297% (2.97 bps per lakh)
SEBI_TURNOVER_FEE = 0.000001    # 0.0001% (0.1 bps per lakh)
STAMP_DUTY_BUY = 0.00015        # 0.015% (15 bps per lakh) on Buy orders only
GST_RATE = 0.18                # 18% GST on (Brokerage + Exchange Fee + SEBI Fee)
SLIPPAGE_BPS = 2.5             # Estimated execution market impact for Nifty 50 liquid names (2.5 bps per side)


def compute_indian_transaction_cost(
    notional: float,
    action: str = "buy",
    trade_type: str = "delivery",
) -> float:
    """Calculate exact statutory Indian cash equity transaction charges (Brokerage, STT, Exchange, GST, Stamp Duty)."""
    if notional <= 0:
        return 0.0

    if trade_type.lower() == "delivery":
        brokerage = 0.0  # Zerodha delivery brokerage is Rs 0
        stt = STT_DELIVERY * notional  # 0.1% STT on buy and sell
        stamp = STAMP_DUTY_BUY * notional if action.lower() == "buy" else 0.0
    else:
        brokerage = min(20.0, 0.0003 * notional)  # 0.03% or Rs 20
        stt = 0.00025 * notional if action.lower() == "sell" else 0.0
        stamp = 0.00003 * notional if action.lower() == "buy" else 0.0

    exchange_fee = NSE_TURNOVER_FEE * notional
    sebi_fee = SEBI_TURNOVER_FEE * notional
    gst = GST_RATE * (brokerage + exchange_fee + sebi_fee)
    slippage = (SLIPPAGE_BPS / 10000.0) * notional

    total_cost = brokerage + stt + exchange_fee + sebi_fee + gst + stamp + slippage
    return float(total_cost)


def get_effective_round_trip_cost_bps(is_delivery: bool = True) -> float:
    """Compute statutory round-trip transaction costs + market slippage in basis points.
    
    Breakdown for 100 INR buy + 100 INR sell:
    - STT (Buy + Sell): 20.0 bps
    - Stamp Duty (Buy): 1.5 bps
    - NSE Exchange Fee (2x): 0.594 bps
    - SEBI Fee (2x): 0.02 bps
    - GST (18% on exch+sebi): ~0.11 bps
    - Execution Slippage (2x 2.5 bps): 5.0 bps
    Total Round-Trip Cost: ~27.22 bps (~13.61 bps one-way per rebalance turn).
    """
    buy_cost = compute_indian_transaction_cost(100000.0, action="buy", trade_type="delivery" if is_delivery else "intraday")
    sell_cost = compute_indian_transaction_cost(100000.0, action="sell", trade_type="delivery" if is_delivery else "intraday")
    rt_bps = ((buy_cost + sell_cost) / 100000.0) * 10000.0
    return float(rt_bps)


def get_one_way_cost_bps(is_delivery: bool = True) -> float:
    """One-way transaction cost per rebalance turn in basis points (~13.61 bps)."""
    return get_effective_round_trip_cost_bps(is_delivery=is_delivery) / 2.0
