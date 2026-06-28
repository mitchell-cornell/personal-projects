#!/usr/bin/env python3
"""
Reproduce Omair Sharif's (@fcastofthemonth) chart:
    "Mkt-Based Core PCE less Info Proc Equip (YoY%)"

Methodology
-----------
The series is the market-based core PCE price index (PCE ex food & energy,
market-based) with the "information processing equipment" component stripped out.
You cannot subtract chain-type index levels directly, so we remove the component
with a Tornqvist re-aggregation:

    dln(P_rest) = ( dln(P_agg) - s_ipe * dln(P_ipe) ) / (1 - s_ipe)

where s_ipe is the average nominal expenditure share of info-processing equipment
in the aggregate across months t-1 and t. We then chain the residual index and
take 12-month percent changes.

Data: BEA NIPA Underlying Detail, monthly
    Table 2.4.4U (U20404) - chain-type price indexes
    Table 2.4.5U (U20405) - nominal levels (for weights)
    Line 402 - Market-based PCE excluding food and energy
    Line  48 - Information processing equipment

Requires env var BEA_API_KEY (free key from https://apps.bea.gov/API/signup/).
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
AGG_LINE = "402"   # Market-based PCE excluding food and energy
IPE_LINE = "48"    # Information processing equipment


def get_key():
    k = os.environ.get("BEA_API_KEY")
    if not k:
        sys.exit("Set BEA_API_KEY in the environment.")
    return k


def fetch(table, key):
    """Fetch a monthly underlying-detail table as a long DataFrame."""
    r = requests.get(
        BASE,
        params={
            "UserID": key,
            "method": "GetData",
            "datasetname": "NIUnderlyingDetail",
            "TableName": table,
            "Frequency": "M",
            "Year": "ALL",
            "ResultFormat": "JSON",
        },
        timeout=60,
    )
    r.raise_for_status()
    res = r.json()["BEAAPI"]["Results"]
    if "Error" in res:
        sys.exit(f"BEA error for {table}: {res['Error']}")
    df = pd.DataFrame(res["Data"])
    df["value"] = pd.to_numeric(df["DataValue"].str.replace(",", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["TimePeriod"], format="%YM%m")
    return df


def line_series(df, line):
    s = (
        df[df["LineNumber"] == line]
        .set_index("date")["value"]
        .sort_index()
    )
    if s.empty:
        sys.exit(f"Line {line} not found.")
    return s


def main():
    key = get_key()
    px = fetch("U20404", key)    # price indexes
    nom = fetch("U20405", key)   # nominal levels

    p_agg = line_series(px, AGG_LINE)
    p_ipe = line_series(px, IPE_LINE)
    n_agg = line_series(nom, AGG_LINE)
    n_ipe = line_series(nom, IPE_LINE)

    data = pd.DataFrame(
        {"p_agg": p_agg, "p_ipe": p_ipe, "n_agg": n_agg, "n_ipe": n_ipe}
    ).dropna()

    # Nominal share of info-processing equipment in market-based core PCE.
    share = data["n_ipe"] / data["n_agg"]
    avg_share = 0.5 * (share + share.shift(1))   # Tornqvist average weight

    # Monthly log changes.
    dln_agg = np.log(data["p_agg"]).diff()
    dln_ipe = np.log(data["p_ipe"]).diff()

    # Residual (ex-IPE) monthly log change, then chain to an index.
    dln_rest = (dln_agg - avg_share * dln_ipe) / (1.0 - avg_share)
    p_rest = np.exp(dln_rest.fillna(0).cumsum()) * 100.0

    # 12-month percent change.
    yoy = (p_rest / p_rest.shift(12) - 1.0) * 100.0
    yoy = yoy[yoy.index >= "1996-05-01"].dropna()

    latest = yoy.iloc[-1]
    print(f"Latest point: {yoy.index[-1]:%b-%Y} = {latest:.2f}%")
    print(yoy.tail(6).to_string())

    # ---- Plot to match the original styling ----
    fig, ax = plt.subplots(figsize=(11, 6))
    navy = "#1a2a6c"
    ax.plot(yoy.index, yoy.values, color=navy, linewidth=1.6)

    ax.set_title("Mkt-Based Core PCE less Info Proc Equip (YoY%)",
                 fontsize=15, color="#444444", pad=14)
    ax.set_ylim(0, 7)
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)

    # Annotate the latest value, like the original.
    ax.annotate(f"{latest:.2f}", xy=(yoy.index[-1], latest),
                xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=11, color="#333333")

    # Sparse year-ish ticks similar to the source.
    ticks = pd.date_range("1996-05-01", yoy.index[-1], freq="12MS")
    ax.set_xticks(ticks)
    ax.set_xticklabels([d.strftime("%b-%y") for d in ticks], rotation=90, fontsize=8)
    ax.margins(x=0.01)

    ax.text(0.0, -0.22,
            "Source: BEA (NIPA Underlying Detail Tables 2.4.4U / 2.4.5U), "
            "author's calculations",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "mkt_based_core_less_ipe.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Also save the underlying data.
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "mkt_based_core_less_ipe.csv")
    yoy.rename("yoy_pct").to_csv(csv)
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
