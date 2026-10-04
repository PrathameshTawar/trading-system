# Raw market data layout

Use the following structure for NSE/BSE and benchmark data:

```text
data/
├── raw/
│   ├── nse/
│   │   ├── RELIANCE.csv
│   │   ├── TCS.csv
│   │   └── INFY.csv
│   ├── bse/
│   │   ├── HDFCBANK.csv
│   │   └── ITC.csv
│   ├── benchmarks/
│   │   └── NIFTY.csv
│   └── news/
│       └── financial_news.csv
├── processed/
│   ├── standardized_market.csv
│   └── feature_matrix.csv
└── schemas/
    └── market_schema.md
```

Expected columns for stock CSVs:

- `Date` or `timestamp`
- `Symbol` or `ticker` (optional if file name encodes symbol)
- `Open`, `High`, `Low`, `Close`, `Volume`
- optional: `Turnover`, `Prev Close`, `Sentiment`, `Positive News`, `Negative News`

Expected columns for benchmark CSVs:

- `Date` or `timestamp`
- `Close` or `Last`

The ingestion scripts automatically normalize these to:

- `timestamp`
- `symbol`
- `open`, `high`, `low`, `close`, `volume`
- `nifty_close` for benchmark merges
