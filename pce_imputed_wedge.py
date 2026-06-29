#!/usr/bin/env python3
"""
The "imputed wedge": core PCE minus market-based core PCE (YoY%).

The market-based filter removes ~16% of core PCE -- imputed financial services
(FISIM, portfolio mgmt), most insurance, gambling, used-vehicle margins, etc.
This wedge = YoY(core) - YoY(market-based core). When positive, those
non-market / imputed components are adding to inflation; when negative, they
are pulling it down.

Top panel:    core PCE vs market-based core PCE (YoY%)
Bottom panel: the wedge (core - market-based core), shaded around zero

Data: BEA NIPA Underlying Detail, monthly (2.4.4U prices), lines 374 / 402.
Requires env var BEA_API_KEY.
"""
import os
import sys
import pandas as pd
import requests
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE = "https://apps.bea.gov/api/data"
CORE, MKT = "374", "402"
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


def yoy(s):
    return (s / s.shift(12) - 1) * 100.0


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    px = fetch("U20404", key)
    core = yoy(line(px, CORE))
    mkt = yoy(line(px, MKT))
    df = pd.DataFrame({"core": core, "mkt": mkt}).dropna()
    df = df[df.index >= START]
    df["wedge"] = df["core"] - df["mkt"]

    print(f"Latest {df.index[-1]:%b-%Y}:  core {df['core'].iloc[-1]:.2f}%  "
          f"mkt-based {df['mkt'].iloc[-1]:.2f}%  wedge {df['wedge'].iloc[-1]:+.2f}pp")
    print(f"Wedge stats since 1996:  mean {df['wedge'].mean():+.2f}pp  "
          f"min {df['wedge'].min():+.2f}  max {df['wedge'].max():+.2f}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})

    ax1.plot(df.index, df["core"], color="#111111", linewidth=1.4, label="Core PCE")
    ax1.plot(df.index, df["mkt"], color="#c0392b", linewidth=1.4,
             label="Market-based core PCE")
    ax1.set_title("Core PCE vs market-based core PCE, and the imputed wedge (YoY%)",
                  fontsize=14, color="#333333", pad=12)
    ax1.set_ylim(0, 7)
    ax1.yaxis.set_major_locator(MultipleLocator(1.0))
    ax1.legend(loc="upper left", fontsize=10, frameon=False)

    ax2.fill_between(df.index, df["wedge"], 0, where=df["wedge"] >= 0,
                     color="#c0392b", alpha=0.55, interpolate=True,
                     label="imputed adding to inflation")
    ax2.fill_between(df.index, df["wedge"], 0, where=df["wedge"] < 0,
                     color="#2e7bb6", alpha=0.55, interpolate=True,
                     label="imputed subtracting")
    ax2.axhline(0, color="#444444", linewidth=0.8)
    ax2.set_ylabel("wedge (pp)", fontsize=9)
    ax2.set_ylim(-1.0, 1.0)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax2.annotate(f"{df['wedge'].iloc[-1]:+.2f}",
                 xy=(df.index[-1], df["wedge"].iloc[-1]),
                 xytext=(6, 0), textcoords="offset points", va="center",
                 fontsize=10, color="#c0392b", fontweight="bold")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
    ax1.spines["left"].set_visible(False)

    ticks = pd.date_range(START, df.index[-1], freq="24MS")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([d.strftime("%b-%y") for d in ticks], rotation=90, fontsize=8)
    ax2.margins(x=0.02)
    ax2.text(0.0, -0.32,
             "Source: BEA (NIPA Underlying Detail Table 2.4.4U), author's calculations. "
             "Wedge = core minus market-based core.",
             transform=ax2.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_imputed_wedge.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pce_imputed_wedge.csv"))


if __name__ == "__main__":
    main()
