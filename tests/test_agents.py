from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from agents import scout, charts, quant, news, sentiment, skeptic, chief
from signalforge.data import generate_synthetic_market_data


@pytest.fixture
def sample_market_df():
    return generate_synthetic_market_data(days=120, symbols=("RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"))


@pytest.fixture
def sample_rules():
    return {
        "mode": "paper_only",
        "balance": 1_000_000.0,
        "watchlist": ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"],
        "max_size_pct": 5.0,
        "max_loss_pct": 1.0,
        "daily_loss_limit_pct": 2.0,
        "max_cards_per_day": 3,
        "min_price": 50.0,
        "embargo_days": 1,
    }


def test_scout_finds_leads(sample_market_df, sample_rules):
    leads = scout.find_leads(sample_market_df, sample_rules)
    assert isinstance(leads, list)
    if leads:
        assert "symbol" in leads[0]
        assert "vol_ratio" in leads[0]


def test_charts_levels_calculation(sample_market_df):
    c = charts.levels(sample_market_df, "RELIANCE")
    assert c["symbol"] == "RELIANCE"
    assert c["entry"] > 0
    assert c["stop"] < c["entry"]
    assert c["target"] > c["entry"]
    assert c["atr_14"] >= 0


def test_quant_scoring(sample_market_df):
    q = quant.score(sample_market_df, "RELIANCE")
    assert q["symbol"] == "RELIANCE"
    assert "prediction" in q
    assert q["oos_ic"] == 0.0042


def test_skeptic_hard_risk_gates(sample_rules):
    c_good = {"price": 2500.0, "entry": 2500.0, "stop": 2450.0, "data_age_days": 0}
    q_good = {"high_drift": False}
    lead = {"symbol": "RELIANCE"}

    # Approve Case
    verdict, why = skeptic.review(lead, c_good, q_good, sample_rules, todays_loss_pct=0.0)
    assert verdict == "APPROVE"

    # Daily Loss Limit Rejection
    verdict_loss, why_loss = skeptic.review(lead, c_good, q_good, sample_rules, todays_loss_pct=2.5)
    assert verdict_loss == "REJECT"
    assert "Daily loss limit" in why_loss

    # Min Price Rejection
    c_cheap = {"price": 10.0, "entry": 10.0, "stop": 9.0, "data_age_days": 0}
    verdict_cheap, why_cheap = skeptic.review(lead, c_cheap, q_good, sample_rules, todays_loss_pct=0.0)
    assert verdict_cheap == "REJECT"
    assert "minimum threshold" in why_cheap

    # Stale Data Rejection
    c_stale = {"price": 2500.0, "entry": 2500.0, "stop": 2450.0, "data_age_days": 5}
    verdict_stale, why_stale = skeptic.review(lead, c_stale, q_good, sample_rules, todays_loss_pct=0.0)
    assert verdict_stale == "REJECT"
    assert "Stale market data" in why_stale

    # High Drift Rejection
    q_drift = {"high_drift": True}
    verdict_drift, why_drift = skeptic.review(lead, c_good, q_drift, sample_rules, todays_loss_pct=0.0)
    assert verdict_drift == "REJECT"
    assert "drift" in why_drift.lower()


def test_chief_orchestration(sample_market_df, sample_rules, tmp_path):
    ledger_path = tmp_path / "ledger.json"
    csv_ledger = tmp_path / "paper-ledger.csv"
    
    cards = chief.run(
        sample_market_df,
        sample_rules,
        todays_loss_pct=0.0,
        ledger_path=ledger_path,
        csv_ledger_path=csv_ledger
    )
    assert isinstance(cards, list)
