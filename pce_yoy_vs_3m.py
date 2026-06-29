#!/usr/bin/env python3
"""
YoY vs 3-month annualized (SAAR) for market-based core PCE less info-proc equip.

Both views of the same Tornqvist-constructed index on one chart:
  - YoY:  P_t / P_t-12 - 1
  - 3m annualized:  (P_t / P_t-3)^4 - 1

Data: BEA NIPA Underlying Detail, monthly (2.4.4U prices / 2.4.5U nominal),
line 402 (market-based core PCE) less line 48 (info processing equipment).
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
MKT, IPE = "402", "48"
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


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    px, nom = fetch("U20404", key), fetch("U20405", key)
    p = remove_component(line(px, MKT), line(nom, MKT), line(px, IPE), line(nom, IPE))

    yoy = ((p / p.shift(12) - 1) * 100.0)
    ann3 = ((p / p.shift(3)) ** 4 - 1) * 100.0
    yoy = yoy[yoy.index >= START].dropna()
    ann3 = ann3[ann3.index >= START].dropna()

    print(f"Latest {yoy.index[-1]:%b-%Y}:  YoY = {yoy.iloc[-1]:.2f}%   "
          f"3m annualized = {ann3.iloc[-1]:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(ann3.index, ann3.values, color="#cc7a00", linewidth=1.0, alpha=0.65,
            label="3-month annualized (SAAR)")
    ax.plot(yoy.index, yoy.values, color="#1a2a6c", linewidth=1.9,
            label="Year-over-year")
    for s, c in [(yoy, "#1a2a6c"), (ann3, "#cc7a00")]:
        ax.annotate(f"{s.iloc[-1]:.2f}", xy=(s.index[-1], s.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=c, fontweight="bold")

    ax.set_title("Mkt-Based Core PCE less Info Proc Equip: YoY vs 3-month annualized",
                 fontsize=14, color="#333333", pad=12)
    ax.set_ylim(-1, 11)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.axhline(2.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ticks = pd.date_range(START, yoy.index[-1], freq="24MS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.strftime("%b-%y") for d in ticks], rotation=90, fontsize=8)
    ax.margins(x=0.02)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    ax.text(0.0, -0.20,
            "Source: BEA (NIPA Underlying Detail Tables 2.4.4U / 2.4.5U), author's "
            "calculations. Dashed line = 2% target.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_yoy_vs_3m.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)
    pd.DataFrame({"yoy": yoy, "ann_3m": ann3}).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_yoy_vs_3m.csv"))


if __name__ == "__main__":
    main()
