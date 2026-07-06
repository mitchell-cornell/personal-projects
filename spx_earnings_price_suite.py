#!/usr/bin/env python3
"""
S&P 500 earnings-vs-price suite (extends spx_earnings_vs_price.py).

Figures:
  1. main_ttm      - TTM diluted-EPS growth vs trailing-12m price return
  2. sector_facet  - the same relationship, one panel per GICS sector
  3. sector_median - one point per sector (median x vs median y) + per-sector r
  4. last_quarter  - latest-quarter YoY EPS growth vs that quarter's price return
  5. ytd           - YTD-2026 price return vs YTD-2026 EPS growth

Data notes / constraints (as of Jul 2026):
  * Q2-2026 EPS is not filed yet (~27 filers), so "this year" earnings = Q1-2026.
  * Standalone Q4 EPS is sparse in XBRL frames (folded into the annual 10-K), so
    TTM is reconstructed as  FY - Q1_prior + Q1_latest  (annual + Q1 frames only):
        TTM( end Q1'26) = FY2025 - Q1'25 + Q1'26
        TTM(end Q1'25)  = FY2024 - Q1'24 + Q1'25
  * Price windows use the cached daily bars (2025-07 .. 2026-07); finer
    per-filing-date alignment would need a wider Databento pull.

Reuses helpers from spx_earnings_vs_price.py. Requires DATABENTO_API_KEY.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spx_earnings_vs_price as base   # constituents, pull_prices, dedupe, edgar_eps

HERE = os.path.dirname(os.path.abspath(__file__))
FWD = base.FWD_SPLITS
REV = [s for s in base.SPLIT_RATIOS if s > 1]
SECTOR_COLORS = {}  # filled after we know sectors


def clean_series(g, n_days):
    """Split-adjusted, de-spiked daily close series for one symbol, or None."""
    g = g.sort_values("date")
    dates = pd.to_datetime(g["date"].to_numpy())
    c = g["close"].to_numpy(dtype=float).copy()
    if len(c) < 0.95 * n_days:
        return None
    for i in range(1, len(c) - 1):
        step, back = c[i] / c[i - 1], c[i + 1] / c[i]
        if abs(step * back - 1) < 0.12 and (step < 0.7 or step > 1.43):
            c[i] = np.sqrt(c[i - 1] * c[i + 1])
    ratio = c[1:] / c[:-1]
    factor = np.ones(len(c))
    for i, rr in enumerate(ratio):
        if rr > 1.4 and any(abs(rr / s - 1) < 0.04 for s in REV):
            return None
        for sr in FWD:
            if abs(rr / sr - 1) < 0.04:
                factor[:i + 1] *= sr
                break
    adj = c * factor
    resid = adj[1:] / adj[:-1]
    if np.nanmax(np.abs(np.log(resid))) > np.log(2.4):
        return None
    return pd.Series(adj, index=dates).sort_index()


def wret(s, d0, d1):
    """Percent return of a clean series between the closes on/at-or-before d0,d1."""
    if s is None:
        return np.nan
    a = s.asof(pd.Timestamp(d0))
    b = s.asof(pd.Timestamp(d1))
    if pd.isna(a) or pd.isna(b) or a <= 0:
        return np.nan
    return (b / a - 1) * 100.0


def scatter(ax, x, y, sectors, title, xlab, ylab, lim=(-100, 200), fit=True):
    fx, fy = x.clip(*lim), y.clip(*lim)
    for s in sorted(sectors.unique()):
        m = sectors == s
        ax.scatter(fx[m], fy[m], s=26, alpha=0.8, color=SECTOR_COLORS[s],
                   edgecolor="white", linewidth=0.3, label=s)
    ax.axhline(0, color="#999", lw=0.7)
    ax.axvline(0, color="#999", lw=0.7)
    r = x.corr(y)
    if fit and x.notna().sum() > 3:
        b, a = np.polyfit(x.clip(*lim), y.clip(*lim), 1)
        xs = np.array(lim)
        ax.plot(xs, a + b * xs, "--", color="#111", lw=1.3,
                label=f"slope={b:.2f}, r={r:.2f}")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    ax.grid(True, color="#eee", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    return r


def main():
    con = base.constituents()
    prices = base.dedupe_primary(base.pull_prices(con["dbsym"].tolist()))
    n_days = prices["date"].nunique()
    series = {sym: clean_series(g, n_days) for sym, g in prices.groupby("symbol")}
    dmin, dmax = prices["date"].min(), prices["date"].max()
    print(f"Prices {dmin.date()}..{dmax.date()} ({n_days} days); "
          f"clean series: {sum(v is not None for v in series.values())}")

    # ---- price windows ----
    ytd_start = "2025-12-31"
    q1_start, q1_end = "2025-12-31", "2026-03-31"
    def px_map(d0, d1):
        return con["dbsym"].map({s: wret(v, d0, d1) for s, v in series.items()})
    con["px_12m"] = px_map(dmin, dmax)
    con["px_ytd"] = px_map(ytd_start, dmax)
    con["px_q1"] = px_map(q1_start, q1_end)

    # ---- earnings frames ----
    q1_24, q1_25, q1_26 = (base.edgar_eps("CY2024Q1"), base.edgar_eps("CY2025Q1"),
                           base.edgar_eps("CY2026Q1"))
    fy24, fy25 = base.edgar_eps("CY2024"), base.edgar_eps("CY2025")

    def ttm_growth(cik):
        try:
            t26 = fy25[cik] - q1_25[cik] + q1_26[cik]
            t25 = fy24[cik] - q1_24[cik] + q1_25[cik]
        except KeyError:
            return np.nan
        return (t26 - t25) / abs(t25) * 100 if t25 > 0 else np.nan

    def q_yoy(cik):
        a, b = q1_26.get(cik), q1_25.get(cik)
        return (a - b) / abs(b) * 100 if (a is not None and b and b > 0) else np.nan

    con["eps_ttm"] = con["cik"].map(ttm_growth)
    con["eps_q"] = con["cik"].map(q_yoy)     # latest quarter YoY == YTD-2026 (only Q1 out)

    sectors = sorted(con["sector"].unique())
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"]
    SECTOR_COLORS.update({s: pal[i % len(pal)] for i, s in enumerate(sectors)})

    # ============ Figure 1: main TTM ============
    d = con.dropna(subset=["eps_ttm", "px_12m"])
    print(f"\n[Fig1] TTM: {len(d)} names")
    fig, ax = plt.subplots(figsize=(11, 8))
    r = scatter(ax, d["eps_ttm"], d["px_12m"], d["sector"],
                "S&P 500: TTM EPS growth vs trailing-12m price return",
                "TTM diluted-EPS growth (%)", "trailing-12m price return (%)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax.text(0, -0.09, "TTM = FY - Q1_prior + Q1_latest (EDGAR frames). "
            "Price: Databento DBEQ.BASIC, split-adjusted. Axes clipped +/-200%.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_fig1_ttm.png", dpi=150, bbox_inches="tight")
    print(f"       overall r={r:.3f}")

    # ============ Figure 2: sector facet ============
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    persec = {}
    for ax, s in zip(axes.flat, sectors):
        sub = d[d["sector"] == s]
        rr = sub["eps_ttm"].corr(sub["px_12m"]) if len(sub) > 3 else np.nan
        persec[s] = (len(sub), rr)
        ax.scatter(sub["eps_ttm"].clip(-100, 200), sub["px_12m"].clip(-100, 200),
                   s=22, alpha=0.8, color=SECTOR_COLORS[s], edgecolor="white", linewidth=0.3)
        if len(sub) > 3:
            b, a = np.polyfit(sub["eps_ttm"].clip(-100, 200), sub["px_12m"].clip(-100, 200), 1)
            ax.plot([-100, 200], [a - 100 * b, a + 200 * b], "--", color="#111", lw=1)
        ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
        ax.set_title(f"{s}  (n={len(sub)}, r={rr:.2f})", fontsize=10)
        ax.set_xlim(-100, 200); ax.set_ylim(-100, 200)
        ax.grid(True, color="#eee", lw=0.5); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for ax in axes.flat[len(sectors):]:
        ax.axis("off")
    fig.suptitle("TTM EPS growth vs trailing-12m price return, by GICS sector",
                 fontsize=15, y=0.995)
    fig.supxlabel("TTM diluted-EPS growth (%)", fontsize=11)
    fig.supylabel("trailing-12m price return (%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_fig2_sector_facet.png", dpi=150, bbox_inches="tight")

    # ============ Figure 3: sector medians ============
    med = (d.groupby("sector")[["eps_ttm", "px_12m"]].median()
           .join(pd.Series({s: persec[s][1] for s in persec}, name="r")))
    fig, ax = plt.subplots(figsize=(11, 8))
    for s, row in med.iterrows():
        ax.scatter(row["eps_ttm"], row["px_12m"], s=140, color=SECTOR_COLORS[s],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(s, (row["eps_ttm"], row["px_12m"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=9)
    b, a = np.polyfit(med["eps_ttm"], med["px_12m"], 1)
    xs = np.array([med["eps_ttm"].min() - 3, med["eps_ttm"].max() + 3])
    ax.plot(xs, a + b * xs, "--", color="#111", lw=1.2,
            label=f"slope={b:.2f}, r={med['eps_ttm'].corr(med['px_12m']):.2f}")
    ax.axhline(0, color="#999", lw=0.7); ax.axvline(0, color="#999", lw=0.7)
    ax.set_title("Sector medians: TTM EPS growth vs trailing-12m price return", fontsize=13)
    ax.set_xlabel("median TTM diluted-EPS growth (%)", fontsize=10)
    ax.set_ylabel("median trailing-12m price return (%)", fontsize=10)
    ax.grid(True, color="#eee", lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_fig3_sector_median.png", dpi=150, bbox_inches="tight")

    # ============ Figure 4: last quarter ============
    dq = con.dropna(subset=["eps_q", "px_q1"])
    print(f"[Fig4] last-quarter: {len(dq)} names")
    fig, ax = plt.subplots(figsize=(11, 8))
    rq = scatter(ax, dq["eps_q"], dq["px_q1"], dq["sector"],
                 "S&P 500: Q1-2026 YoY EPS growth vs Q1-2026 price return",
                 "Q1-2026 YoY diluted-EPS growth (%)", "price return during Q1-2026 (Jan-Mar, %)",
                 lim=(-80, 120))
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax.text(0, -0.09, "Earnings period Jan-Mar 2026 paired with price return over the "
            "same quarter. Latest fully-reported quarter (Q2-2026 not yet filed).",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_fig4_last_quarter.png", dpi=150, bbox_inches="tight")
    print(f"       r={rq:.3f}")

    # ============ Figure 5: YTD ============
    dy = con.dropna(subset=["eps_q", "px_ytd"])
    print(f"[Fig5] YTD: {len(dy)} names")
    fig, ax = plt.subplots(figsize=(11, 8))
    ry = scatter(ax, dy["eps_q"], dy["px_ytd"], dy["sector"],
                 "S&P 500: YTD-2026 EPS growth vs YTD-2026 price return",
                 "YTD-2026 EPS growth (%)  [= Q1'26 vs Q1'25; Q2 not filed]",
                 "YTD-2026 price return (Dec 31 -> now, %)", lim=(-80, 150))
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)
    ax.text(0, -0.09, "YTD earnings currently = Q1-2026 (only reported 2026 quarter). "
            "Will extend to Q2 once it is filed.", transform=ax.transAxes,
            fontsize=8, style="italic", color="#666")
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_fig5_ytd.png", dpi=150, bbox_inches="tight")
    print(f"       r={ry:.3f}")

    # ---- summary table ----
    print("\nPer-sector correlation (TTM EPS growth vs 12m price return):")
    for s in sectors:
        n, rr = persec[s]
        print(f"  {s:26s} n={n:3d}  r={rr:+.2f}")
    con.to_csv(f"{HERE}/spx_earnings_price_suite.csv", index=False)
    print("\nSaved 5 figures + spx_earnings_price_suite.csv")


if __name__ == "__main__":
    main()
