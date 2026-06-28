#!/usr/bin/env python3
"""
3-month annualized (SAAR) inflation, last 3 years, for four measures:
  - Headline PCE                            BEA line 1
  - Core PCE (ex food & energy)             BEA line 374
  - Core PCE less info proc equip           374 less 48
  - Mkt-based core PCE less info proc equip 402 less 48

Rate = (P_t / P_t-3)^4 - 1 (BEA monthly PCE price indexes are seasonally adj).
Data: BEA NIPA Underlying Detail, monthly (2.4.4U prices / 2.4.5U nominal).
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
import matplotlib.dates as mdates

BASE = "https://apps.bea.gov/api/data"
HEAD, CORE, MKT, IPE = "1", "374", "402", "48"


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
    return ((p / p.shift(3)) ** 4 - 1) * 100.0


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    px, nom = fetch("U20404", key), fetch("U20405", key)
    p_head = line(px, HEAD)
    p_core, n_core = line(px, CORE), line(nom, CORE)
    p_mkt, n_mkt = line(px, MKT), line(nom, MKT)
    p_ipe, n_ipe = line(px, IPE), line(nom, IPE)

    indices = {
        "Headline PCE": p_head,
        "Core PCE": p_core,
        "Core PCE less IPE": remove_component(p_core, n_core, p_ipe, n_ipe),
        "Mkt-based core PCE less IPE": remove_component(p_mkt, n_mkt, p_ipe, n_ipe),
    }
    end = min(v.index.max() for v in indices.values())
    start = end - pd.DateOffset(years=3)
    series = {k: ann3m(v)[(ann3m(v).index >= start) & (ann3m(v).index <= end)].dropna()
              for k, v in indices.items()}

    print(f"3m annualized (SAAR), window {start:%b-%Y} to {end:%b-%Y}")
    print(f"Latest ({end:%b-%Y}):")
    for k, v in series.items():
        print(f"  {k:30s} {v.iloc[-1]:.2f}%")

    colors = {
        "Headline PCE": "#888888",
        "Core PCE": "#111111",
        "Core PCE less IPE": "#2e8b57",
        "Mkt-based core PCE less IPE": "#1a2a6c",
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for k, v in series.items():
        ax.plot(v.index, v.values, label=k, color=colors[k],
                linewidth=2.0 if "Mkt-based" in k else 1.6, marker="o", markersize=3)
        ax.annotate(f"{v.iloc[-1]:.2f}", xy=(v.index[-1], v.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=colors[k], fontweight="bold")

    ax.set_title("Inflation - 3-month annualized (SAAR), last 3 years",
                 fontsize=14, color="#333333", pad=12)
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
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(0.0, -0.22,
            "Source: BEA (NIPA Underlying Detail Tables 2.4.4U / 2.4.5U), author's "
            "calculations. Dashed line = 2% target.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_3m_last3y.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)
    pd.DataFrame(series).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_3m_last3y.csv"))


if __name__ == "__main__":
    main()
