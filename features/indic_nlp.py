from __future__ import annotations

import re
import pandas as pd

# Indic financial lexicons for Hindi/Marathi/Gujarati/Bengali financial sentiment
INDIC_FINANCIAL_KEYWORDS = {
    "positive": [
        "मुनाफा", "बढ़त", "लाभ", "विकास", "तेजी", "उत्कृष्ट", "सकारात्मक", "रिकॉर्ड", "वृद्धि",
        "नफा", "वाढ", "उत्पन्न", "लाभांश", "फायदा", "ઉછાળો", "નફો", "વિકાસ", "સુધારો", "ઉત્પાદન",
        "লাভ", "বৃদ্ধি", "মুনাফা", "ইতিবাচক"
    ],
    "negative": [
        "घाटा", "नुकसान", "मंदी", "गिरावट", "जोखिम", "कमी", "नकारात्मक", "दबाव", "मामला",
        "तोटा", "घसरण", "मंदी", "धोका", "ખૂટ", "ઘટાડો", "મંદી", "જોખમ", "નુકસાન",
        "ক্ষতি", "পতন", "ঝুঁকি", "হ্রাস"
    ],
    "risk": ["जोखिम", "धोका", "જોખમ", "ঝুঁকি", "मुकदमा", "कर्ज", "ऋण", "कर्जबाजरी", "तोटा"],
    "revenue": ["राजस्व", "आय", "महसूल", "આવક", "বিক্রি", "উৎপাদন", "सेल", "कमाई"],
    "earnings": ["परिणाम", "नतीजे", "निकाल", "પરિણામો", "ফলাফল", "त्रैमासिक", "तिमाही"],
}


def detect_indic_language(text: str) -> str:
    """Detect if text contains Devanagari (Hindi/Marathi), Gujarati, Bengali, or English scripts."""
    if not isinstance(text, str) or not text.strip():
        return "en"

    # Script Unicode ranges
    devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
    gujarati_count = len(re.findall(r"[\u0A80-\u0AFF]", text))
    bengali_count = len(re.findall(r"[\u0980-\u09FF]", text))

    if devanagari_count > 2:
        return "hi_mr"
    elif gujarati_count > 2:
        return "gu"
    elif bengali_count > 2:
        return "bn"
    return "en"


def compute_indic_financial_sentiment(text: str) -> dict[str, float]:
    """Extract financial sentiment score and category flags from Indic text."""
    if not isinstance(text, str) or not text.strip():
        return {"sentiment": 0.0, "risk": 0.0, "revenue": 0.0, "earnings": 0.0}

    text_lower = text.lower()
    pos_score = sum(1 for kw in INDIC_FINANCIAL_KEYWORDS["positive"] if kw in text)
    neg_score = sum(1 for kw in INDIC_FINANCIAL_KEYWORDS["negative"] if kw in text)

    total = pos_score + neg_score
    sentiment = (pos_score - neg_score) / max(total, 1.0) if total > 0 else 0.0

    risk_flag = 1.0 if any(kw in text for kw in INDIC_FINANCIAL_KEYWORDS["risk"]) else 0.0
    rev_flag = 1.0 if any(kw in text for kw in INDIC_FINANCIAL_KEYWORDS["revenue"]) else 0.0
    earnings_flag = 1.0 if any(kw in text for kw in INDIC_FINANCIAL_KEYWORDS["earnings"]) else 0.0

    return {
        "sentiment": float(sentiment),
        "risk": float(risk_flag),
        "revenue": float(rev_flag),
        "earnings": float(earnings_flag),
    }


def add_indic_nlp_features(df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
    """Process Indic language news text and generate multilingual financial signals."""
    out = df.copy()
    if text_col not in out.columns:
        out["indic_sentiment"] = 0.0
        out["indic_risk_signal"] = 0.0
        out["indic_multilingual_flag"] = 0.0
        return out

    sentiments = []
    risks = []
    langs = []

    for text in out[text_col]:
        lang = detect_indic_language(text)
        res = compute_indic_financial_sentiment(text)
        sentiments.append(res["sentiment"])
        risks.append(res["risk"])
        langs.append(1.0 if lang != "en" else 0.0)

    out["indic_sentiment"] = sentiments
    out["indic_risk_signal"] = risks
    out["indic_multilingual_flag"] = langs

    grouped = out.groupby("symbol", group_keys=False)
    out["indic_sentiment_7d"] = grouped["indic_sentiment"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["indic_risk_mentions_7d"] = grouped["indic_risk_signal"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    return out
