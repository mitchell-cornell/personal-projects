#!/usr/bin/env python3
"""
3-month annualized (SAAR) versions of the core PCE measures.

Instead of a 12-month change, this annualizes the trailing 3 months:
    rate = (P_t / P_{t-3})**4 - 1
which equals annualizing the average of the last 3 monthly changes
(BEA monthly PCE price indexes are seasonally adjusted, so SAAR is valid).

Series (BEA NIPA Underlying Detail, monthly; 2.4.4U prices / 2.4.5U nominal):
  - Core PCE (ex food & energy)                  line 374
  - Market-based core PCE                        line 402
  - Core PCE less info proc equip                374 less 48
  - Mkt-based core PCE less info proc equip       402 less 48  (Omair's series)

Requires env var BEA_API_KEY.
"""
import os
import sys
import numpy as np
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE = "https://apps.bea.gov/api/data"
CORE, MKT, IPE = "374", "402", "48"
START = "1996-05-01"


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


def ann3m(p):
    """Trailing-3-month annualized rate, in percent."""
    return ((p / p.shift(3)) ** 4 - 1) * 100.0


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    px, nom = fetch("U20404", key), fetch("U20405", key)
    p_core, n_core = line(px, CORE), line(nom, CORE)
    p_mkt, n_mkt = line(px, MKT), line(nom, MKT)
    p_ipe, n_ipe = line(px, IPE), line(nom, IPE)

    indices = {
        "Core PCE": p_core,
        "Market-based core PCE": p_mkt,
        "Core PCE less info proc equip": remove_component(p_core, n_core, p_ipe, n_ipe),
        "Mkt-based core PCE less info proc equip": remove_component(p_mkt, n_mkt, p_ipe, n_ipe),
    }
    series = {k: ann3m(v)[ann3m(v).index >= START].dropna() for k, v in indices.items()}

    headline = "Mkt-based core PCE less info proc equip"
    print("3-month annualized (SAAR), latest 3 months:")
    idx = series[headline].index[-3:]
    for k, v in series.items():
        vals = "  ".join(f"{v.loc[d]:5.2f}%" for d in idx)
        print(f"  {k:42s} {vals}   (latest {v.iloc[-1]:.2f}%)")
    print("\nMonthly % changes that feed the latest SAAR (annualized in parens):")
    p = indices[headline]
    for d in idx:
        m = (p.loc[d] / p.shift(1).loc[d] - 1) * 100
        print(f"  {d:%b-%Y}: {m:+.3f}% m/m  ->  {((1+m/100)**12-1)*100:+.2f}% annualized")

    colors = {
        "Core PCE": "#111111",
        "Market-based core PCE": "#c0392b",
        "Core PCE less info proc equip": "#2e8b57",
        "Mkt-based core PCE less info proc equip": "#1a2a6c",
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for k, v in series.items():
        ax.plot(v.index, v.values, label=k, color=colors[k],
                linewidth=1.8 if k == headline else 1.1,
                alpha=1.0 if k == headline else 0.7)
        ax.annotate(f"{v.iloc[-1]:.2f}", xy=(v.index[-1], v.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=colors[k])

    ax.set_title("Core PCE measures - 3-month annualized (SAAR, %)",
                 fontsize=14, color="#333333", pad=12)
    ax.set_ylim(-1, 11)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.axhline(2.0, color="#999999", linewidth=0.8, linestyle="--")  # Fed target
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ticks = pd.date_range(START, series["Core PCE"].index[-1], freq="24MS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.strftime("%b-%y") for d in ticks], rotation=90, fontsize=8)
    ax.margins(x=0.02)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.text(0.0, -0.20,
            "Source: BEA (NIPA Underlying Detail Tables 2.4.4U / 2.4.5U), author's calculations. "
            "Dashed line = 2% target.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_3m_annualized.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nSaved", out)
    pd.DataFrame(series).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_3m_annualized.csv"))


if __name__ == "__main__":
    main()
