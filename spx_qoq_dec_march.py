#!/usr/bin/env python3
"""
Sequential Q4->Q1 (Dec->March) EPS growth vs YTD-2026 price return.

x = Q4'25 -> Q1'26 sequential diluted-EPS growth (%).  Q4 standalone EPS is
    sparse in XBRL frames (folded into the 10-K), so Q4(y) is reconstructed as
        Q4(y) = FY(y) - Q1(y) - Q2(y) - Q3(y).
    Sequential Q4->Q1 is highly seasonal (Q4 is the big quarter for many
    names), so we also build a DE-SEASONALIZED version that removes each
    company's typical Q4->Q1 change, estimated from prior years:
        season_i = mean over y in {2024,2025} of  log(Q1_y / Q4_{y-1})
        deseason growth = exp( log(Q1'26/Q4'25) - season_i ) - 1
y = YTD-2026 price return (Dec 31 2025 -> latest), split-adjusted (Databento).

Outputs:
  spx_qoq_main.png          raw vs de-seasonalized (2 panels), overall
  spx_qoq_sector.png        de-seasonalized, by GICS sector
  spx_qoq_seasonality.png   distribution of the Q4->Q1 seasonal factor
Requires DATABENTO_API_KEY.
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


def q4(fy, q1, q2, q3, cik):
    try:
        return fy[cik] - q1[cik] - q2[cik] - q3[cik]
    except KeyError:
        return None


def main():
    con = base.constituents()
    prices = base.dedupe_primary(base.pull_prices(con["dbsym"].tolist()))
    n_days = prices["date"].nunique()
    series = {s: suite.clean_series(g, n_days) for s, g in prices.groupby("symbol")}
    dmax = prices["date"].max()
    con["px_ytd"] = con["dbsym"].map(
        {s: suite.wret(v, "2025-12-31", dmax) for s, v in series.items()})

    # EPS frames
    F = {}
    for y in range(2022, 2026):
        F[("FY", y)] = base.edgar_eps(f"CY{y}")
        for q in ("Q1", "Q2", "Q3"):
            F[(q, y)] = base.edgar_eps(f"CY{y}{q}")
    F[("Q1", 2026)] = base.edgar_eps("CY2026Q1")

    def qoq_log(y):
        """log(Q1_y / Q4_{y-1}) per cik, both positive."""
        q4d = {c: q4(F[("FY", y-1)], F[("Q1", y-1)], F[("Q2", y-1)], F[("Q3", y-1)], c)
               for c in set(F[("FY", y-1)])}
        out = {}
        for c, a in F[("Q1", y)].items():
            b = q4d.get(c)
            if a is not None and a > 0 and b is not None and b > 0:
                out[c] = np.log(a / b)
        return out

    cur = qoq_log(2026)                     # Q4'25 -> Q1'26
    hist = [qoq_log(y) for y in (2024, 2025)]  # prior Q4->Q1

    def raw_growth(cik):
        return (np.exp(cur[cik]) - 1) * 100 if cik in cur else np.nan

    def season(cik):
        vals = [h[cik] for h in hist if cik in h]
        return np.mean(vals) if vals else np.nan

    def deseason_growth(cik):
        if cik not in cur:
            return np.nan
        s = season(cik)
        if np.isnan(s):
            return np.nan
        return (np.exp(cur[cik] - s) - 1) * 100

    con["qoq_raw"] = con["cik"].map(raw_growth)
    con["qoq_des"] = con["cik"].map(deseason_growth)
    con["season_pct"] = con["cik"].map(
        lambda c: (np.exp(season(c)) - 1) * 100 if not np.isnan(season(c)) else np.nan)

    # ---- how big is the seasonality? ----
    sk = con.dropna(subset=["season_pct"])
    print(f"Typical Q4->Q1 seasonal EPS change (n={len(sk)}): "
          f"median={sk['season_pct'].median():.1f}%, mean={sk['season_pct'].mean():.1f}%")
    print("Raw Q4'25->Q1'26 growth:   median={:.1f}%".format(
        con["qoq_raw"].median()))
    print("Deseasonalized growth:     median={:.1f}%".format(
        con["qoq_des"].median()))

    sectors = sorted(con["sector"].unique())
    suite.SECTOR_COLORS.update(
        {s: ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
             "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79"][i % 11]
         for i, s in enumerate(sectors)})

    def stats(x, y):
        m = pd.concat([x, y], axis=1).dropna()
        return (m.iloc[:, 0].corr(m.iloc[:, 1]),
                spearmanr(m.iloc[:, 0], m.iloc[:, 1])[0], len(m))

    # ---- Fig 1: raw vs deseasonalized (overall) ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, col, lab in [(axes[0], "qoq_raw", "RAW Q4'25->Q1'26 EPS growth (%)"),
                         (axes[1], "qoq_des", "DE-SEASONALIZED Q4->Q1 EPS growth (%)")]:
        d = con.dropna(subset=[col, "px_ytd"])
        r, rho, n = stats(d[col], d["px_ytd"])
        lim = (-100, 150)
        for s in sectors:
            sub = d[d["sector"] == s]
            ax.scatter(sub[col].clip(*lim), sub["px_ytd"].clip(-80, 150), s=20,
                       alpha=0.8, color=suite.SECTOR_COLORS[s],
                       edgecolor="white", linewidth=0.3, label=s)
        b, a = np.polyfit(d[col].clip(*lim), d["px_ytd"].clip(-80, 150), 1)
        ax.plot(lim, [a + b * lim[0], a + b * lim[1]], "--", color="#111", lw=1.3,
                label=f"winsor slope={b:.2f}\nr={r:.2f}  ρ={rho:.2f}  (n={n})")
        ax.axhline(0, color="#999", lw=0.7); ax.axvline(0, color="#999", lw=0.7)
        ax.set_xlim(*lim); ax.set_ylim(-80, 150)
        ax.set_xlabel(lab, fontsize=10)
        ax.grid(True, color="#eee", lw=0.6); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.legend(loc="upper left", fontsize=7, frameon=False, ncol=2)
    axes[0].set_ylabel("YTD-2026 price return (Dec 31 -> now, %)", fontsize=10)
    fig.suptitle("S&P 500: sequential Q4'25->Q1'26 EPS growth vs YTD price return "
                 "(r=Pearson, ρ=Spearman)", fontsize=13)
    fig.text(0.5, 0.005, "Q4 reconstructed = FY - Q1 - Q2 - Q3. De-seasonalized removes each "
             "firm's typical Q4->Q1 change (prior 2 yrs). Positive-EPS names only.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(f"{HERE}/spx_qoq_main.png", dpi=150, bbox_inches="tight")

    # ---- Fig 2: de-seasonalized by sector ----
    d = con.dropna(subset=["qoq_des", "px_ytd"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, sectors):
        sub = d[d["sector"] == s]
        rr = sub["qoq_des"].corr(sub["px_ytd"]) if len(sub) > 3 else np.nan
        rho = spearmanr(sub["qoq_des"], sub["px_ytd"])[0] if len(sub) > 3 else np.nan
        ax.scatter(sub["qoq_des"].clip(-100, 150), sub["px_ytd"].clip(-80, 150),
                   s=22, alpha=0.8, color=suite.SECTOR_COLORS[s],
                   edgecolor="white", linewidth=0.3)
        if len(sub) > 3:
            b, a = np.polyfit(sub["qoq_des"].clip(-100, 150), sub["px_ytd"].clip(-80, 150), 1)
            ax.plot([-100, 150], [a - 100 * b, a + 150 * b], "--", color="#111", lw=1)
        ax.axhline(0, color="#bbb", lw=0.6); ax.axvline(0, color="#bbb", lw=0.6)
        ax.set_title(f"{s}  (n={len(sub)}, r={rr:.2f}, ρ={rho:.2f})", fontsize=9.5)
        ax.set_xlim(-100, 150); ax.set_ylim(-80, 150)
        ax.grid(True, color="#eee", lw=0.5); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for ax in axes.flat[len(sectors):]:
        ax.axis("off")
    fig.suptitle("De-seasonalized Q4'25->Q1'26 EPS growth vs YTD price return, by sector",
                 fontsize=14, y=0.997)
    fig.supxlabel("de-seasonalized Q4->Q1 EPS growth (%)", fontsize=11)
    fig.supylabel("YTD-2026 price return (%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_qoq_sector.png", dpi=150, bbox_inches="tight")

    # ---- Fig 3: seasonality magnitude ----
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(sk["season_pct"].clip(-80, 80), bins=40, color="#8c564b", alpha=0.8)
    ax.axvline(sk["season_pct"].median(), color="#111", ls="--", lw=1.4,
               label=f"median {sk['season_pct'].median():.0f}%")
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_title("How seasonal is Q4->Q1?  Typical firm-level Q4->Q1 EPS change "
                 "(prior 2 years)", fontsize=12)
    ax.set_xlabel("typical Q4 -> Q1 sequential EPS change (%)")
    ax.set_ylabel("number of S&P 500 firms")
    ax.legend(frameon=False)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{HERE}/spx_qoq_seasonality.png", dpi=150, bbox_inches="tight")

    r0 = stats(con["qoq_raw"], con["px_ytd"])
    r1 = stats(con["qoq_des"], con["px_ytd"])
    print(f"\nvs YTD price return:")
    print(f"  RAW Q4->Q1:          Pearson r={r0[0]:.3f}  Spearman ρ={r0[1]:.3f}  n={r0[2]}")
    print(f"  DE-SEASONALIZED:     Pearson r={r1[0]:.3f}  Spearman ρ={r1[1]:.3f}  n={r1[2]}")
    con.to_csv(f"{HERE}/spx_qoq_dec_march.csv", index=False)
    print("Saved 3 figures + csv.")


if __name__ == "__main__":
    main()
