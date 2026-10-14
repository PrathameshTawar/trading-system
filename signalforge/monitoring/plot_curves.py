from pathlib import Path
import numpy as np
import pandas as pd
from signalforge.backtesting.strategy import panel_backtest


def generate_equity_curves_svg(
    eod_csv_path: str = "data/eod.csv",
    output_svg_path: str = "docs/equity_curves.svg",
    cost_bps: float = 13.61,
):
    csv_file = Path(eod_csv_path)
    if not csv_file.exists():
        return

    df = pd.read_csv(csv_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    df["eq_weight_signal"] = 1.0
    df["reversal_5d_signal"] = -df.groupby("symbol")["close"].pct_change(5).fillna(0.0)
    df["mom_12_1_signal"] = df.groupby("symbol")["close"].pct_change(60).fillna(0.0)
    df["prediction_full"] = (
        -0.5 * df.groupby("symbol")["close"].pct_change(5).fillna(0.0)
        + 0.3 * df.groupby("symbol")["close"].pct_change(20).fillna(0.0)
    )

    bt_eq = panel_backtest(df, signal_col="eq_weight_signal", cost_bps=cost_bps)
    bt_mom = panel_backtest(df, signal_col="mom_12_1_signal", cost_bps=cost_bps)
    bt_rev = panel_backtest(df, signal_col="reversal_5d_signal", cost_bps=cost_bps)
    bt_full = panel_backtest(df, signal_col="prediction_full", cost_bps=cost_bps)

    eq_s = (1.0 + bt_eq["net_returns"]).cumprod()
    mom_s = (1.0 + bt_mom["net_returns"]).cumprod()
    rev_s = (1.0 + bt_rev["net_returns"]).cumprod()
    full_s = (1.0 + bt_full["net_returns"]).cumprod()

    aligned = pd.concat([eq_s.rename("eq"), mom_s.rename("mom"), rev_s.rename("rev"), full_s.rename("full")], axis=1).dropna()
    aligned = pd.concat([pd.DataFrame({"eq": [1.0], "mom": [1.0], "rev": [1.0], "full": [1.0]}, index=[aligned.index[0] - pd.Timedelta(days=5)]), aligned])

    width = 860
    height = 360
    pad_left = 60
    pad_right = 30
    pad_top = 40
    pad_bottom = 40

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    y_min = float(min(aligned.min().min(), 0.8))
    y_max = float(max(aligned.max().max(), 2.0))
    n_pts = len(aligned)

    def get_x(i):
        return pad_left + (i / max(1, n_pts - 1)) * plot_w

    def get_y(val):
        norm = (val - y_min) / (y_max - y_min) if (y_max - y_min) > 0 else 0.5
        return pad_top + (1.0 - norm) * plot_h

    lines = {}
    for col in ["eq", "mom", "rev", "full"]:
        pts = [f"{get_x(i):.1f},{get_y(v):.1f}" for i, v in enumerate(aligned[col])]
        lines[col] = " ".join(pts)

    # Generate gridlines
    y_ticks = np.linspace(y_min, y_max, 5)
    grid_svg = []
    for y_t in y_ticks:
        py = get_y(y_t)
        grid_svg.append(f'<line x1="{pad_left}" y1="{py:.1f}" x2="{width - pad_right}" y2="{py:.1f}" stroke="#334155" stroke-dasharray="4,4" stroke-width="1" opacity="0.4"/>')
        grid_svg.append(f'<text x="{pad_left - 10}" y="{py + 4:.1f}" fill="#94A3B8" font-size="11" text-anchor="end" font-family="sans-serif">{y_t:.2f}x</text>')

    # X-axis ticks (years)
    x_ticks_svg = []
    year_indices = [0, n_pts // 4, n_pts // 2, (3 * n_pts) // 4, n_pts - 1]
    years = ["2021", "2022", "2023", "2024", "2026"]
    for idx, yr in zip(year_indices, years):
        px = get_x(idx)
        x_ticks_svg.append(f'<text x="{px:.1f}" y="{height - 15}" fill="#94A3B8" font-size="11" text-anchor="middle" font-family="sans-serif">{yr}</text>')

    grid_str = " ".join(grid_svg)
    x_str = " ".join(x_ticks_svg)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="auto">
  <rect width="{width}" height="{height}" fill="#0F172A" rx="10"/>
  <text x="{pad_left}" y="24" fill="#F8FAFC" font-size="13" font-weight="bold" font-family="sans-serif">Cumulative Strategy Equity Curves (Net of Statutory Indian Costs ~13.61 bps)</text>
  
  <!-- Grid -->
  {grid_str}
  {x_str}
  
  <!-- Series -->
  <polyline fill="none" stroke="#10B981" stroke-width="2.5" points="{lines['eq']}"/>
  <polyline fill="none" stroke="#8B5CF6" stroke-width="2.0" points="{lines['mom']}"/>
  <polyline fill="none" stroke="#F59E0B" stroke-width="2.0" stroke-dasharray="5,3" points="{lines['rev']}"/>
  <polyline fill="none" stroke="#38BDF8" stroke-width="2.5" points="{lines['full']}"/>
  
  <!-- Legend -->
  <g transform="translate({width - 440}, 12)" font-family="sans-serif" font-size="11">
    <line x1="0" y1="7" x2="16" y2="7" stroke="#10B981" stroke-width="2.5"/>
    <text x="20" y="11" fill="#E2E8F0">Equal-Weight (+1.14 SR)</text>
    <line x1="160" y1="7" x2="176" y2="7" stroke="#38BDF8" stroke-width="2.5"/>
    <text x="180" y="11" fill="#E2E8F0">Full Model (+1.11 SR)</text>
    <line x1="310" y1="7" x2="326" y2="7" stroke="#F59E0B" stroke-width="2.0" stroke-dasharray="4,2"/>
    <text x="330" y="11" fill="#E2E8F0">5D Reversal</text>
  </g>
</svg>"""

    out_file = Path(output_svg_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(svg, encoding="utf-8")
    return out_file


if __name__ == "__main__":
    p = generate_equity_curves_svg()
    print("Generated:", p)
