#!/usr/bin/env python3
"""
(1) Formal test of residual seasonality in core PCE via a trend-controlled
    month-dummy regression, and
(2) a residual-seasonally-adjusted version of the 3-year, 3m-annualized chart
    for the four inflation measures.

Regression (core PCE, ex-COVID 2013-2025 excl. 2020-21):
    annualized_mm_t = b0 + b1*trend + sum_m d_m * Month_m + e_t
    Month dummies relative to the calendar-month mean (sum-to-zero), so each
    coefficient is that month's premium/discount vs an average month after
    removing the linear trend. Reports coefficients, t-stats, p-values, and a
    joint F-test that all month effects are zero.

Residual-seasonal adjustment:
    For each series, estimate additive seasonal factors on log m/m changes via
    classic decomposition (deviation from a centered 12-term moving average),
    averaged by calendar month over 2013-2025 ex-COVID and centered to sum to
    zero. Subtract from log m/m changes, then recompute the trailing-3-month
    annualized rate.

Data: BEA NIPA Underlying Detail, monthly (2.4.4U prices / 2.4.5U nominal).
Requires env var BEA_API_KEY.
"""
import os
import sys
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

BASE = "https://apps.bea.gov/api/data"
HEAD, CORE, MKT, IPE = "1", "374", "402", "48"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fetch(table, key):
    r = requests.get(BASE, params={
        "UserID": key, "method": "GetData", "datasetname": "NIUnderlyingDetail",
        "TableName": table, "Frequency": "M", "Year": "ALL", "ResultFormat": "JSON",
    }, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["BEAAPI"]["Results"]["Data"])
    df["value"] = pd.to_numeric(df["DataValue"].str.replace(",", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["TimePeriod"], format="%YM%m")
    return df


def line(df, ln):
    return df[df["LineNumber"] == ln].set_index("date")["value"].sort_index()


def remove_component(p_agg, n_agg, p_ipe, n_ipe):
    d = pd.DataFrame({"p_agg": p_agg, "n_agg": n_agg,
                      "p_ipe": p_ipe, "n_ipe": n_ipe}).dropna()
    share = d["n_ipe"] / d["n_agg"]
    avg_share = 0.5 * (share + share.shift(1))
    dln = (np.log(d["p_agg"]).diff() - avg_share * np.log(d["p_ipe"]).diff()) / (1 - avg_share)
    return np.exp(dln.fillna(0).cumsum()) * 100.0


def seasonal_factors(p):
    """Additive seasonal factors on log m/m changes (classic decomposition)."""
    dln = np.log(p).diff().dropna()
    trend = dln.rolling(12, center=True).mean()
    detr = (dln - trend).dropna()
    est = detr[(detr.index.year >= 2013) & (detr.index.year <= 2025) &
               (~detr.index.year.isin([2020, 2021]))]
    fac = est.groupby(est.index.month).mean()
    fac = fac - fac.mean()                      # center: sum to zero
    return fac.reindex(range(1, 13)).fillna(0.0)


def ann3m_from_dln(dln_adj):
    """Trailing-3-month annualized rate from adjusted log m/m changes (%)."""
    return (np.exp(dln_adj.rolling(3).sum()) ** 4 - 1) * 100.0


def run_regression(p_core):
    mm = ((p_core / p_core.shift(1)) ** 12 - 1) * 100.0
    df = pd.DataFrame({"rate": mm}).dropna()
    df = df[(df.index.year >= 2013) & (df.index.year <= 2025) &
            (~df.index.year.isin([2020, 2021]))].copy()
    df["trend"] = np.arange(len(df))
    df["month"] = pd.Categorical(df.index.month)
    # Sum-to-zero coding so each month coef is vs the average month.
    model = smf.ols("rate ~ trend + C(month, Sum)", data=df).fit(
        cov_type="HAC", cov_kwds={"maxlags": 12})
    print("=" * 64)
    print("Month-dummy regression: core PCE annualized m/m (2013-2025 ex-COVID)")
    print("HAC (Newey-West, 12 lags) standard errors")
    print("=" * 64)
    params = model.params
    # Reconstruct each month's effect (Sum coding gives 11; Dec = -sum of rest).
    month_eff = {}
    for m in range(1, 12):
        key = f"C(month, Sum)[S.{m}]"
        if key in params:
            month_eff[m] = (params[key], model.bse[key], model.pvalues[key])
    dec = -sum(v[0] for v in month_eff.values())
    print(f"\n  trend coef: {params['trend']:.4f}  (p={model.pvalues['trend']:.3f})")
    print("\n  Month premium vs average month (pp), after de-trending:")
    print(f"  {'Month':5s} {'coef':>8s} {'se':>7s} {'p-value':>9s}")
    for m in range(1, 13):
        if m in month_eff:
            c, se, pv = month_eff[m]
            star = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
            print(f"  {MONTHS[m-1]:5s} {c:8.3f} {se:7.3f} {pv:9.3f} {star}")
        else:
            print(f"  {MONTHS[m-1]:5s} {dec:8.3f} {'(implied)':>17s}")
    # Joint F-test that all month effects = 0.
    terms = [f"C(month, Sum)[S.{m}]" for m in range(1, 12)]
    ftest = model.f_test(" = 0, ".join(terms) + " = 0")
    print(f"\n  Joint F-test (all months = 0): F={float(ftest.fvalue):.2f}, "
          f"p={float(ftest.pvalue):.4g}")
    print(f"  -> {'REJECT' if float(ftest.pvalue) < 0.05 else 'cannot reject'} "
          "no-seasonality at 5%.\n")


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    px, nom = fetch("U20404", key), fetch("U20405", key)
    p_head = line(px, HEAD)
    p_core, n_core = line(px, CORE), line(nom, CORE)
    p_mkt, n_mkt = line(px, MKT), line(nom, MKT)
    p_ipe, n_ipe = line(px, IPE), line(nom, IPE)

    run_regression(p_core)

    indices = {
        "Headline PCE": p_head,
        "Core PCE": p_core,
        "Core PCE less IPE": remove_component(p_core, n_core, p_ipe, n_ipe),
        "Mkt-based core PCE less IPE": remove_component(p_mkt, n_mkt, p_ipe, n_ipe),
    }

    end = min(v.index.max() for v in indices.values())
    start = end - pd.DateOffset(years=3)

    orig, adj = {}, {}
    for k, p in indices.items():
        dln = np.log(p).diff()
        fac = seasonal_factors(p)
        dln_adj = dln - dln.index.month.map(fac).to_numpy()
        o = ((p / p.shift(3)) ** 4 - 1) * 100.0
        a = ann3m_from_dln(pd.Series(dln_adj, index=dln.index))
        orig[k] = o[(o.index >= start) & (o.index <= end)].dropna()
        adj[k] = a[(a.index >= start) & (a.index <= end)].dropna()

    print("Latest 3m annualized: original vs residual-seasonally adjusted")
    for k in indices:
        print(f"  {k:30s} orig {orig[k].iloc[-1]:5.2f}%   adj {adj[k].iloc[-1]:5.2f}%")

    colors = {
        "Headline PCE": "#888888",
        "Core PCE": "#111111",
        "Core PCE less IPE": "#2e8b57",
        "Mkt-based core PCE less IPE": "#1a2a6c",
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for k in indices:
        ax.plot(orig[k].index, orig[k].values, color=colors[k], linewidth=1.0,
                alpha=0.35, linestyle="-")
        ax.plot(adj[k].index, adj[k].values, color=colors[k], linewidth=2.0,
                marker="o", markersize=3, label=k)
        ax.annotate(f"{adj[k].iloc[-1]:.2f}", xy=(adj[k].index[-1], adj[k].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=colors[k], fontweight="bold")

    ax.set_title("Residual-seasonally adjusted 3m-annualized inflation, last 3 years\n"
                 "(faint = as-reported, bold = adjusted)",
                 fontsize=13, color="#333333", pad=12)
    ax.set_ylim(0, 7)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.axhline(2.0, color="#999999", linewidth=0.9, linestyle="--", label="2% target")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    ax.margins(x=0.03)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.text(0.0, -0.22,
            "Source: BEA (NIPA Underlying Detail Tables 2.4.4U / 2.4.5U), author's "
            "calculations. Residual-seasonal factors from 2013-2025 ex-COVID.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_deseasonalized.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)
    pd.DataFrame({**{f"{k} (orig)": orig[k] for k in indices},
                  **{f"{k} (adj)": adj[k] for k in indices}}).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_deseasonalized.csv"))


if __name__ == "__main__":
    main()
