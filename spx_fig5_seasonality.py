#!/usr/bin/env python3
"""
Figure 5 by sector + seasonality diagnostic for the Q1 (Dec->March) EPS-growth
metric used in the YTD chart.

The Fig-5 x-axis is Q1'26 diluted-EPS YoY (vs Q1'25). YoY already differences
out *calendar* seasonality (Q1 vs Q1). The remaining question: does Q1's YoY
growth systematically over/under-state the full-year growth trend (a "Q1 tilt")?
We test it two ways with history, quantify it, and if material, rebuild Fig 5
with a de-seasonalized (annual-equivalent) EPS-growth axis.

  Part 1  cross-sectional median YoY EPS growth by calendar quarter, 2023-2026,
          then the quarter-of-year effect (Q1 vs Q2 vs Q3).
  Part 2  per-company gap  (Q1 YoY growth  -  same-year FY YoY growth), 2023-25,
          overall and by sector -> the systematic Q1 tilt.
  Output  spx_fig5_sector.png            (raw, by sector)
          spx_fig5_seasonality.png       (the diagnostic)
          spx_fig5_sector_deseason.png   (de-seasonalized, by sector)

Requires DATABENTO_API_KEY (for the cached price series). SEC EDGAR is free.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import spx_earnings_vs_price as base
import spx_earnings_price_suite as suite

HERE = os.path.dirname(os.path.abspath(__file__))


def eps_yoy_map(cur, prv):
    """Per-cik YoY EPS growth (%), positive base only."""
    out = {}
    for cik, a in cur.items():
        b = prv.get(cik)
        if b is not None and b > 0:
            out[cik] = (a - b) / abs(b) * 100.0
    return out


def median_spx(growth, ciks):
    vals = [growth[c] for c in ciks if c in growth]
    return np.median(vals) if vals else np.nan


def main():
    con = base.constituents()
    ciks = set(con["cik"])

    # ---- price series (cached) for YTD price return ----
    prices = base.dedupe_primary(base.pull_prices(con["dbsym"].tolist()))
    n_days = prices["date"].nunique()
    series = {s: suite.clean_series(g, n_days) for s, g in prices.groupby("symbol")}
    dmax = prices["date"].max()
    con["px_ytd"] = con["dbsym"].map(
        {s: suite.wret(v, "2025-12-31", dmax) for s, v in series.items()})

    # ---- fetch EPS frames ----
    Q = {}
    for q in ("Q1", "Q2", "Q3"):
        for y in range(2021, 2027):
            per = f"CY{y}{q}"
            if q == "Q1" and y <= 2026 or q in ("Q2", "Q3") and y <= 2025:
                try:
                    Q[per] = base.edgar_eps(per)
                except Exception:
                    Q[per] = {}
    FY = {y: base.edgar_eps(f"CY{y}") for y in range(2021, 2026)}

    # ---- Part 1: cross-sectional median YoY growth by calendar quarter ----
    rows = []
    for q in ("Q1", "Q2", "Q3"):
        for y in range(2023, 2027):
            cur, prv = Q.get(f"CY{y}{q}"), Q.get(f"CY{y-1}{q}")
            if cur and prv:
                rows.append({"quarter": q, "year": y,
                             "med_yoy": median_spx(eps_yoy_map(cur, prv), ciks)})
    ts = pd.DataFrame(rows).dropna()
    qeff = ts.groupby("quarter")["med_yoy"].mean()
    print("Cross-sectional median YoY EPS growth by quarter (SPX), 2023-2026:")
    print(ts.pivot(index="year", columns="quarter", values="med_yoy").round(1).to_string())
    print("\nQuarter-of-year averages (pp):")
    print(qeff.round(2).to_string())
    q1_tilt_agg = qeff.get("Q1", np.nan) - ts[ts.quarter != "Q1"]["med_yoy"].mean()
    print(f"Q1 vs (Q2,Q3) average tilt: {q1_tilt_agg:+.2f}pp")

    # ---- Part 2: per-company Q1-minus-FY gap ----
    gaps = []
    for y in (2023, 2024, 2025):
        q1 = eps_yoy_map(Q[f"CY{y}Q1"], Q[f"CY{y-1}Q1"])
        fy = {}
        for cik, a in FY[y].items():
            b = FY[y-1].get(cik)
            if b is not None and b > 0:
                fy[cik] = (a - b) / abs(b) * 100.0
        for cik in ciks:
            if cik in q1 and cik in fy:
                gaps.append({"cik": cik, "year": y,
                             "gap": q1[cik] - fy[cik]})
    gdf = pd.DataFrame(gaps)
    gap_mean, gap_med = gdf["gap"].mean(), gdf["gap"].median()
    print(f"\nPer-company Q1-YoY minus FY-YoY gap: n={len(gdf)}, "
          f"mean={gap_mean:+.2f}pp, median={gap_med:+.2f}pp")
    # per-sector median gap (for a sector-aware adjustment)
    cik2sec = dict(zip(con["cik"], con["sector"]))
    gdf["sector"] = gdf["cik"].map(cik2sec)
    sec_gap = gdf.groupby("sector")["gap"].median()
    print("Per-sector median Q1 tilt (pp):")
    print(sec_gap.round(1).to_string())

    # ---- current-year metrics ----
    q1_26 = eps_yoy_map(Q["CY2026Q1"], Q["CY2025Q1"])
    con["eps_q"] = con["cik"].map(q1_26)
    con["eps_q_deseason"] = con.apply(
        lambda r: r["eps_q"] - sec_gap.get(r["sector"], gap_med)
        if pd.notna(r["eps_q"]) else np.nan, axis=1)

    sectors = sorted(con["sector"].unique())
    suite.SECTOR_COLORS.update(
        {s: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
             "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"][i % 11]
         for i, s in enumerate(sectors)})

    # ---- Fig 5 by sector (raw) ----
    def facet(xcol, fname, xlabel, title):
        d = con.dropna(subset=[xcol, "px_ytd"])
        fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
        for ax, s in zip(axes.flat, sectors):
            sub = d[d["sector"] == s]
            rr = sub[xcol].corr(sub["px_ytd"]) if len(sub) > 3 else np.nan
            rho = spearmanr(sub[xcol], sub["px_ytd"])[0] if len(sub) > 3 else np.nan
            ax.scatter(sub[xcol].clip(-80, 150), sub["px_ytd"].clip(-80, 150),
                       s=22, alpha=0.8, color=suite.SECTOR_COLORS[s],
                       edgecolor="white", linewidth=0.3)
            if len(sub) > 3:
                b, a = np.polyfit(sub[xcol].clip(-80, 150), sub["px_ytd"].clip(-80, 150), 1)
                ax.plot([-80, 150], [a - 80 * b, a + 150 * b], "--", color="#111", lw=1)
            ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
            ax.set_title(f"{s}  (n={len(sub)}, r={rr:.2f}, ρ={rho:.2f})", fontsize=9.5)
            ax.set_xlim(-80, 150); ax.set_ylim(-80, 150)
            ax.grid(True, color="#eee", lw=0.5); ax.set_axisbelow(True)
            for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for ax in axes.flat[len(sectors):]:
            ax.axis("off")
        fig.suptitle(title, fontsize=15, y=0.995)
        fig.supxlabel(xlabel, fontsize=11)
        fig.supylabel("YTD-2026 price return (Dec 31 -> now, %)", fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{HERE}/{fname}", dpi=150, bbox_inches="tight")
        return d[xcol].corr(d["px_ytd"])

    r_raw = facet("eps_q", "spx_fig5_sector.png",
                  "YTD-2026 EPS growth (Q1'26 vs Q1'25, %)",
                  "Fig 5 by sector: YTD EPS growth vs YTD price return")
    r_des = facet("eps_q_deseason", "spx_fig5_sector_deseason.png",
                  "de-seasonalized YTD EPS growth (Q1 tilt removed, %)",
                  "Fig 5 by sector, DE-SEASONALIZED: YTD EPS growth vs YTD price return")
    print(f"\nOverall r (YTD price vs EPS growth):  raw={r_raw:.3f}  deseason={r_des:.3f}")

    # ---- seasonality diagnostic figure ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    cmap = {"Q1": "#d62728", "Q2": "#1f77b4", "Q3": "#2ca02c"}
    for q in ("Q1", "Q2", "Q3"):
        sub = ts[ts.quarter == q]
        ax1.plot(sub["year"], sub["med_yoy"], "o-", color=cmap[q], label=q)
    ax1.set_title("Median YoY EPS growth by quarter (SPX)")
    ax1.set_xlabel("year"); ax1.set_ylabel("median YoY EPS growth (%)")
    ax1.legend(frameon=False); ax1.grid(True, color="#eee")
    for sp in ["top", "right"]: ax1.spines[sp].set_visible(False)

    ax2.bar(qeff.index, qeff.values, color=[cmap[q] for q in qeff.index])
    ax2.axhline(ts["med_yoy"].mean(), color="#111", ls="--", lw=1,
                label=f"all-qtr avg {ts['med_yoy'].mean():.1f}%")
    ax2.set_title("Quarter-of-year average\n(seasonal tilt in the growth reading)")
    ax2.set_ylabel("avg median YoY growth (%)"); ax2.legend(frameon=False)
    for sp in ["top", "right"]: ax2.spines[sp].set_visible(False)

    ax3.hist(gdf["gap"].clip(-40, 40), bins=30, color="#9467bd", alpha=0.8)
    ax3.axvline(gap_med, color="#111", ls="--", lw=1.2,
                label=f"median gap {gap_med:+.1f}pp")
    ax3.axvline(0, color="#888", lw=0.8)
    ax3.set_title("Per-company Q1 tilt\n(Q1 YoY growth  -  full-year YoY growth)")
    ax3.set_xlabel("percentage points"); ax3.legend(frameon=False)
    for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_fig5_seasonality.png", dpi=150, bbox_inches="tight")
    print("Saved 3 figures.")


if __name__ == "__main__":
    main()
