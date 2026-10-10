from __future__ import annotations

from typing import Any


def review(
    lead: dict[str, Any],
    charts: dict[str, Any],
    quant: dict[str, Any],
    rules: dict[str, Any],
    todays_loss_pct: float = 0.0,
) -> tuple[str, str]:
    """Pure code hard risk gate enforcement. Zero LLM reliance.
    
    Returns:
        (verdict, reasoning_string) where verdict is "APPROVE", "REJECT", or "REDUCE"
    """
    daily_limit = float(rules.get("daily_loss_limit_pct", 2.0))
    min_price = float(rules.get("min_price", 50.0))
    max_size_pct = float(rules.get("max_size_pct", 5.0))
    max_loss_pct = float(rules.get("max_loss_pct", 1.0))
    embargo_days = int(rules.get("embargo_days", 1))

    # 1. Daily Loss Limit Check
    if todays_loss_pct >= daily_limit:
        return "REJECT", f"Daily loss limit hit ({todays_loss_pct:.2f}% >= {daily_limit:.2f}%)"

    # 2. Minimum Price Check
    price = float(charts.get("price", 0.0))
    if price < min_price:
        return "REJECT", f"Stock price below minimum threshold (₹{price:.2f} < ₹{min_price:.2f})"

    # 3. Stale Data Check
    data_age = int(charts.get("data_age_days", 99))
    if data_age > embargo_days:
        return "REJECT", f"Stale market data ({data_age} days old > {embargo_days} day limit)"

    # 4. Feature Drift Check
    if quant.get("high_drift", False):
        return "REJECT", "HIGH feature drift detected by monitoring engine"

    # 5. Risk-to-Capital Sizing Check
    entry = float(charts.get("entry", price))
    stop = float(charts.get("stop", price * 0.95))
    if entry > 0:
        risk_pct = max(0.0, (entry - stop) / entry * 100.0)
        potential_loss_pct = risk_pct * (max_size_pct / 100.0)
        if potential_loss_pct > max_loss_pct:
            return "REDUCE", f"Stop distance too wide ({risk_pct:.2f}% risk exceeds max loss cap {max_loss_pct:.2f}%)"

    return "APPROVE", "Passes all quantitative risk and freshness guardrails"
