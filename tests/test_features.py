import pandas as pd
from signalforge.features.indic_nlp import add_indic_nlp_features, compute_indic_financial_sentiment, detect_indic_language
from signalforge.features.price import add_price_features
from signalforge.features.cross_asset import add_cross_asset_features


def test_detect_indic_language():
    assert detect_indic_language("कंपनी का तिमाही मुनाफा बढ़ा") == "hi_mr"
    assert detect_indic_language("કંપનીનો નફો વધ્યો") == "gu"
    assert detect_indic_language("কোম্পানির মুনাফা বৃদ্ধি পেয়েছে") == "bn"
    assert detect_indic_language("Company quarterly profit increased") == "en"


def test_compute_indic_financial_sentiment():
    res = compute_indic_financial_sentiment("मुनाफा बढ़ल, रिकॉर्ड लाभ")
    assert res["sentiment"] > 0
    res_neg = compute_indic_financial_sentiment("गिरावट नुकसान मंदी जोखिम")
    assert res_neg["sentiment"] < 0
    assert res_neg["risk"] == 1.0


def test_add_indic_nlp_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        "symbol": ["RELIANCE"] * 5,
        "headline": [
            "कंपनी का तिमाही मुनाफा बढ़ा",
            "नुकसान और गिरावट का डर",
            "उत्कृष्ट परिणाम लाभ",
            "जोखिम और मंदी",
            "Quarterly result announced",
        ],
    })
    out = add_indic_nlp_features(df)
    assert "indic_sentiment" in out.columns
    assert "indic_risk_signal" in out.columns
    assert "indic_sentiment_7d" in out.columns
    assert len(out) == 5


def test_price_features_expanded():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
        "symbol": ["TCS"] * 30,
        "open": [100.0 + i for i in range(30)],
        "high": [102.0 + i for i in range(30)],
        "low": [99.0 + i for i in range(30)],
        "close": [101.0 + i for i in range(30)],
        "volume": [1000 + i * 10 for i in range(30)],
    })
    out = add_price_features(df)
    assert "rsi_14" in out.columns
    assert "macd" in out.columns
    assert "stoch_k" in out.columns
    assert "atr_14" in out.columns
    assert "obv" in out.columns
    assert "future_return_5d" in out.columns
