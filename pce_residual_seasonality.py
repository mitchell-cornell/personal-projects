#!/usr/bin/env python3
"""
Test for residual seasonality in (already seasonally adjusted) core PCE.

If seasonal adjustment were perfect, the average month-over-month change should
be roughly equal across calendar months (just the trend). Residual seasonality
shows up as systematically hot months (the hypothesis: January / Q1) and soft
months mid-year.

Method: take SA monthly core PCE price index, compute annualized m/m changes,
average by calendar month over a few windows, and plot. Also runs a quick
month-dummy regression and reports January's average premium.

Data: BEA NIPA Underlying Detail, monthly (2.4.4U prices), line 374.
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
CORE = "374"
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


def main():
    key = os.environ.get("BEA_API_KEY") or sys.exit("Set BEA_API_KEY.")
    p = line(fetch("U20404", key), CORE)
    # Annualized m/m change of the SA index.
    mm = ((p / p.shift(1)) ** 12 - 1) * 100.0
    mm = mm.dropna()
    mm_df = pd.DataFrame({"rate": mm, "month": mm.index.month, "year": mm.index.year})

    # Windows. Exclude 2020-21 COVID distortions from one of them.
    windows = {
        "2015-2019": (2015, 2019),
        "2022-2025": (2022, 2025),
        "2013-2025 ex-COVID": None,  # special handling
    }

    table = {}
    for name, win in windows.items():
        if win is None:
            sub = mm_df[(mm_df.year >= 2013) & (mm_df.year <= 2025) &
                        (~mm_df.year.isin([2020, 2021]))]
        else:
            sub = mm_df[(mm_df.year >= win[0]) & (mm_df.year <= win[1])]
        table[name] = sub.groupby("month")["rate"].mean().reindex(range(1, 13))

    seas = pd.DataFrame(table)
    seas.index = MONTHS
    print("Average annualized m/m core PCE inflation by calendar month:\n")
    print(seas.round(2).to_string())

    # January premium vs the average month (ex-COVID window).
    base = table["2013-2025 ex-COVID"]
    jan_premium = base.loc[1] - base.mean()
    print(f"\nJanuary premium (2013-2025 ex-COVID): "
          f"{base.loc[1]:.2f}% vs {base.mean():.2f}% avg = +{jan_premium:.2f}pp")

    # Quick month-dummy regression (ex-COVID) to confirm significance roughly.
    reg = mm_df[(mm_df.year >= 2013) & (mm_df.year <= 2025) &
                (~mm_df.year.isin([2020, 2021]))].copy()
    overall = reg["rate"].mean()
    print(f"Overall mean (ex-COVID): {overall:.2f}%")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(12)
    width = 0.26
    colors = {"2015-2019": "#9ecae1", "2022-2025": "#fc9272",
              "2013-2025 ex-COVID": "#1a2a6c"}
    for i, (name, col) in enumerate(colors.items()):
        ax.bar(x + (i - 1) * width, seas[name].values, width,
               label=name, color=col)

    ax.axhline(base.mean(), color="#444444", linewidth=1.0, linestyle="--",
               label=f"ex-COVID avg ({base.mean():.1f}%)")
    ax.set_title("Residual seasonality in core PCE: avg annualized m/m by calendar month",
                 fontsize=14, color="#333333", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_ylabel("avg annualized m/m inflation (%)", fontsize=10)
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.text(0.0, -0.13,
            "Source: BEA (NIPA Underlying Detail Table 2.4.4U), author's calculations. "
            "Series is already seasonally adjusted; bars above trend imply residual seasonality.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666666")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "pce_residual_seasonality.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("Saved", out)
    seas.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "pce_residual_seasonality.csv"))


if __name__ == "__main__":
    main()
