#!/usr/bin/env python3
"""
Overlay of core PCE measures and their "less info-processing equipment" variants.

Lines:
  - Core PCE (ex food & energy)                         BEA line 374
  - Market-based core PCE (ex food & energy)            BEA line 402
  - Core PCE less info proc equip (Tornqvist removal)   374 less 48
  - Market-based core PCE less info proc equip          402 less 48  (Omair's series)

Component removal uses a Tornqvist re-aggregation:
    dln(P_rest) = ( dln(P_agg) - s_ipe * dln(P_ipe) ) / (1 - s_ipe)
with s_ipe the average nominal expenditure share of info-processing equipment
in the aggregate over months t-1, t.

Data: BEA NIPA Underlying Detail, monthly (Tables 2.4.4U prices / 2.4.5U nominal).
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
CORE = "374"   # PCE excluding food and energy
MKT = "402"    # Market-based PCE excluding food and energy
IPE = "48"     # Information processing equipment
START = "1996-05-01"


def fetch(table, key):
    r = requests.get(
        BASE,
        params={
            "UserID": key, "method": "GetData",
            "datasetname": "NIUnderlyingDetail", "TableName": table,
            "Frequency": "M", "Year": "ALL", "ResultFormat": "JSON",
        },
        timeout=60,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["BEAAPI"]["Results"]["Data"])
    df["value"] = pd.to_numeric(df["DataValue"].str.replace(",", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["TimePeriod"], format="%YM%m")
    return df


def line(df, ln):
    return df[df["LineNumber"] == ln].set_index("date")["value"].sort_index()


def remove_component(p_agg, n_agg, p_ipe, n_ipe):
    """Tornqvist removal of one component from a chain-type price index."""
    d = pd.DataFrame({"p_agg": p_agg, "n_agg": n_agg,
                      "p_ipe": p_ipe, "n_ipe": n_ipe}).dropna()
    share = d["n_ipe"] / d["n_agg"]
    avg_share = 0.5 * (share + share.shift(1))
    dln = (np.log(d["p_agg"]).diff() - avg_share * np.log(d["p_ipe"]).diff()) / (1 - avg_share)
    return np.exp(dln.fillna(0).cumsum()) * 100.0


def yoy(s):
    return (s / s.shift(12) - 1) * 100.0


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    px, nom = fetch("U20404", key), fetch("U20405", key)

    p_core, n_core = line(px, CORE), line(nom, CORE)
    p_mkt, n_mkt = line(px, MKT), line(nom, MKT)
    p_ipe, n_ipe = line(px, IPE), line(nom, IPE)

    series = {
        "Core PCE": yoy(p_core),
        "Market-based core PCE": yoy(p_mkt),
        "Core PCE less info proc equip": yoy(remove_component(p_core, n_core, p_ipe, n_ipe)),
        "Mkt-based core PCE less info proc equip": yoy(remove_component(p_mkt, n_mkt, p_ipe, n_ipe)),
    }
    series = {k: v[v.index >= START].dropna() for k, v in series.items()}

    print("Latest (%s):" % series["Core PCE"].index[-1].strftime("%b-%Y"))
    for k, v in series.items():
        print(f"  {k:42s} {v.iloc[-1]:.2f}%")

    colors = {
        "Core PCE": "#111111",
        "Market-based core PCE": "#c0392b",
        "Core PCE less info proc equip": "#2e8b57",
        "Mkt-based core PCE less info proc equip": "#1a2a6c",
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for k, v in series.items():
        ax.plot(v.index, v.values, label=k, color=colors[k],
                linewidth=1.7 if "Mkt-based" in k else 1.2,
                alpha=1.0 if "Mkt-based" in k else 0.85)
        ax.annotate(f"{v.iloc[-1]:.2f}", xy=(v.index[-1], v.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=colors[k])

    ax.set_title("Core PCE measures, with/without info-processing equipment (YoY%)",
                 fontsize=14, color="#333333", pad=12)
    ax.set_ylim(0, 7)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
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
            "Source: BEA (NIPA Underlying Detail Tables 2.4.4U / 2.4.5U), author's calculations",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_overlay.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)

    pd.DataFrame(series).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pce_overlay.csv"))


if __name__ == "__main__":
    main()
