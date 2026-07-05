#!/usr/bin/env python3
"""
S&P 500 cross-section: YoY earnings growth vs YoY stock price performance.

One point per S&P 500 constituent:
  x = YoY diluted-EPS growth (%)  -- SEC EDGAR XBRL frames, CY2026Q1 vs CY2025Q1
  y = YoY price return (%)        -- Databento DBEQ.BASIC ohlcv-1d, ~trailing 12m

Price is split-adjusted (Databento bars are raw/unadjusted; splits are detected
as clean overnight ratios and backed out). Loss-base points (year-ago EPS <= 0)
are dropped because YoY % growth is undefined/meaningless there.

Env: DATABENTO_API_KEY. SEC EDGAR is free (needs a descriptive User-Agent).
Caches raw pulls to CSV so re-runs don't re-hit the APIs.
"""
import os
import io
import sys
import numpy as np
import pandas as pd
import requests
import databento as db

UA = {"User-Agent": "personal-projects research mitchellcornellwtp@gmail.com"}
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = "DBEQ.BASIC"
START, END = "2025-07-01", "2026-07-04"
CUR_Q, PRV_Q = "CY2026Q1", "CY2025Q1"
SPLIT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 1.5,
                0.5, 1/3, 0.25, 0.2, 0.1, 2/3, 0.75]


def constituents():
    f = os.path.join(HERE, "spx_constituents.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers=UA, timeout=60).text
    df = pd.read_html(io.StringIO(html))[0]
    df = df.rename(columns={"GICS Sector": "sector"})
    df["dbsym"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    df["cik"] = df["CIK"].astype(int)
    out = df[["Symbol", "dbsym", "sector", "cik"]].copy()
    out.to_csv(f, index=False)
    return out


def pull_prices(symbols):
    f = os.path.join(HERE, "spx_ohlcv_raw.csv")
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["date"])
    c = db.Historical()
    store = c.timeseries.get_range(dataset=DATASET, symbols=symbols, schema="ohlcv-1d",
                                   start=START, end=END, stype_in="raw_symbol")
    df = store.to_df()  # prices as float dollars, symbol column mapped
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["ts_event"]).dt.tz_localize(None).dt.normalize()
    out = df[["date", "symbol", "open", "high", "low", "close", "volume"]].copy()
    out.to_csv(f, index=False)
    return out


def dedupe_primary(px):
    """DBEQ.BASIC returns one bar per listing venue; keep the primary
    (max-volume) bar per symbol/day so each symbol is a single clean series."""
    return (px.sort_values("volume")
            .drop_duplicates(["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"]))


FWD_SPLITS = [0.5, 1/3, 0.25, 0.2, 1/6, 0.1, 0.05, 2/3, 0.75, 0.8]


def clean_return(g, n_days):
    """
    Split-adjusted YoY return for one symbol (%), or NaN if the series looks
    contaminated. Rules:
      - dedupe already done upstream (one bar/day).
      - require >=95% coverage of trading days.
      - adjust clean *forward* split ratios (price drops to 1/N): multiply the
        pre-split prices by the ratio so the series is continuous.
      - a clean *reverse-split-like* jump (ratio >~1.4) is almost always a
        splice/spin-off artifact in raw_symbol data, not a real reverse split
        -> reject the symbol.
      - any residual >~2.4x overnight move after adjustment -> reject.
    Real earnings moves (e.g. -40%) are below these thresholds and kept.
    """
    g = g.sort_values("date")
    c = g["close"].to_numpy(dtype=float).copy()
    if len(c) < 0.95 * n_days:
        return np.nan

    # De-spike single-day bad prints (a jump that reverts the next day).
    for i in range(1, len(c) - 1):
        step, back = c[i] / c[i - 1], c[i + 1] / c[i]
        if abs(step * back - 1) < 0.12 and (step < 0.7 or step > 1.43):
            c[i] = np.sqrt(c[i - 1] * c[i + 1])   # replace with neighbor geomean

    ratio = c[1:] / c[:-1]
    factor = np.ones(len(c))                       # multiplies pre-split prices
    for i, rr in enumerate(ratio):
        if rr > 1.4 and any(abs(rr / s - 1) < 0.04 for s in SPLIT_RATIOS if s > 1):
            return np.nan                          # reverse-split-like -> splice
        for sr in FWD_SPLITS:
            if abs(rr / sr - 1) < 0.04:            # clean forward split
                factor[:i + 1] *= sr               # scale pre-split prices DOWN
                break
    adj = c * factor
    resid = adj[1:] / adj[:-1]
    if np.nanmax(np.abs(np.log(resid))) > np.log(2.4):
        return np.nan                              # unexplained residual jump
    return (adj[-1] / adj[0] - 1) * 100.0


def edgar_eps(period):
    url = (f"https://data.sec.gov/api/xbrl/frames/us-gaap/"
           f"EarningsPerShareDiluted/USD-per-shares/{period}.json")
    d = requests.get(url, headers=UA, timeout=60).json()["data"]
    # Keep one value per CIK (frames already dedupe by entity for the period).
    return {int(x["cik"]): float(x["val"]) for x in d}


def plot(full, corr, beta, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    XLO, XHI, YLO, YHI = -100, 200, -100, 200      # clip axes; outliers annotated
    sectors = sorted(full["sector"].unique())
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]
    cmap = {s: palette[i % len(palette)] for i, s in enumerate(sectors)}

    fig, ax = plt.subplots(figsize=(12, 9))
    fx = full["yoy_eps"].clip(XLO, XHI)
    fy = full["yoy_price"].clip(YLO, YHI)
    for s in sectors:
        m = full["sector"] == s
        ax.scatter(fx[m], fy[m], s=34, alpha=0.8, color=cmap[s],
                   edgecolor="white", linewidth=0.4, label=s)

    ax.axhline(0, color="#888", lw=0.8)
    ax.axvline(0, color="#888", lw=0.8)
    xs = np.array([XLO, XHI])
    ax.plot(xs, alpha + beta * xs, color="#111", lw=1.4, linestyle="--",
            label=f"fit: slope={beta:.2f}, r={corr:.2f}")

    # Label a few notable / outlier names (clipped to the frame).
    notable = pd.concat([full.nlargest(4, "yoy_price"), full.nsmallest(3, "yoy_price"),
                         full.nlargest(2, "yoy_eps")]).drop_duplicates("Symbol")
    for _, r in notable.iterrows():
        ax.annotate(r["Symbol"],
                    (np.clip(r["yoy_eps"], XLO, XHI), np.clip(r["yoy_price"], YLO, YHI)),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color="#333")

    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("YoY diluted-EPS growth (%)  -  latest quarter (CY2026Q1 vs CY2025Q1)",
                  fontsize=11)
    ax.set_ylabel("YoY price return (%)  -  trailing ~12 months", fontsize=11)
    ax.set_title("S&P 500: earnings growth vs price performance  (each dot = one company)",
                 fontsize=14, pad=12)
    ax.grid(True, color="#eee", lw=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.text(0.0, -0.09,
            "Sources: Databento DBEQ.BASIC (split-adjusted daily closes) + SEC EDGAR XBRL "
            "frames (diluted EPS). Loss-base names & data-contaminated tickers excluded; "
            "axes clipped to +/-200% (outliers labeled).",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666")

    fig.tight_layout()
    out = os.path.join(HERE, "spx_earnings_vs_price.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)


def main():
    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("Set DATABENTO_API_KEY.")
    con = constituents()
    print(f"Constituents: {len(con)}")

    prices = dedupe_primary(pull_prices(con["dbsym"].tolist()))
    n_days = prices["date"].nunique()
    print(f"Price: {prices['symbol'].nunique()} symbols, {n_days} trading days")
    rets = {sym: clean_return(g, n_days) for sym, g in prices.groupby("symbol")}
    rejected = [s for s, v in rets.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    con["yoy_price"] = con["dbsym"].map(rets)

    cur, prv = edgar_eps(CUR_Q), edgar_eps(PRV_Q)
    print(f"EDGAR EPS coverage: {CUR_Q}={len(cur)}, {PRV_Q}={len(prv)} filers")

    def eps_growth(cik):
        a, b = cur.get(cik), prv.get(cik)
        if a is None or b is None or b <= 0:    # need positive base for a % growth
            return np.nan
        return (a - b) / abs(b) * 100.0
    con["yoy_eps"] = con["cik"].map(eps_growth)

    full = con.dropna(subset=["yoy_price", "yoy_eps"]).copy()
    print(f"Price rejected (splice/low-coverage): {len(rejected)}  e.g. {rejected[:8]}")
    print(f"Usable points (both metrics, positive EPS base): {len(full)} / {len(con)}")

    # Robust fit (guard against remaining extreme leverage by using clipped x for slope).
    corr = full["yoy_eps"].corr(full["yoy_price"])
    x = full["yoy_eps"].clip(-100, 200).to_numpy()
    y = full["yoy_price"].clip(-100, 200).to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    print(f"Correlation r={corr:.3f}  |  fit slope={beta:.3f}, intercept={alpha:.1f}")

    full.to_csv(os.path.join(HERE, "spx_earnings_vs_price.csv"), index=False)
    print("\nTop price performers:")
    print(full[["Symbol", "sector", "yoy_eps", "yoy_price"]]
          .sort_values("yoy_price", ascending=False).head(6).to_string(index=False))
    plot(full, corr, beta, alpha)


if __name__ == "__main__":
    main()
