# PCE Inflation Analysis (market-based core, info-proc-equip, seasonality)

This project reproduces and extends an inflation chart originally posted by
Omair Sharif (@fcastofthemonth): **"Mkt-Based Core PCE less Info Proc Equip
(YoY%)."** Everything is built from public **BEA** data via the BEA API, with
the component-removal math done explicitly so the series can be rebuilt and
audited.

> **For a future session picking this up:** read this whole file first. It
> explains the data source, the line numbers, the formulas, what each script
> produces, the headline findings, and the known caveats. All scripts are
> self-contained and read `BEA_API_KEY` from the environment.

---

## Quick start

```bash
export BEA_API_KEY="<your 36-char BEA key>"   # free: https://apps.bea.gov/API/signup/
pip install pandas matplotlib requests statsmodels
python3 mkt_based_core_less_ipe.py            # the original chart reproduction
```

Each script prints its latest values and writes a `.png` + `.csv` next to itself.
The BEA key is **never** stored in any file — scripts read it from the env only.

---

## Data source

**BEA NIPA Underlying Detail tables** (monthly, seasonally adjusted), pulled
from `https://apps.bea.gov/api/data`, dataset `NIUnderlyingDetail`:

| BEA table | API `TableName` | Contents |
|-----------|-----------------|----------|
| Table 2.4.4U | `U20404` | Chain-type **price indexes** for PCE by type of product |
| Table 2.4.5U | `U20405` | **Nominal** PCE by type of product (used for weights) |

Why the *Underlying Detail* tables (not the standard PCE tables on FRED): they
carry the granular **market-based** and **information-processing-equipment**
lines at **monthly** frequency, which FRED does not. FRED has the backbone
(`DPCXRG3M086SBEA` = market-based core PCE) but not the monthly component detail
needed to strip out info-proc equipment with proper weights.

### Key line numbers (`LineNumber` in both tables)

| Line | Description | Used as |
|------|-------------|---------|
| `1`   | Personal consumption expenditures | Headline PCE |
| `48`  | Information processing equipment | Component removed ("IPE") |
| `374` | PCE excluding food and energy | **Core PCE** |
| `402` | Market-based PCE excluding food and energy | **Market-based core PCE** |

To re-discover line numbers, call `GetData` on `U20404` for one year and inspect
`LineNumber` / `LineDescription` (see the exploration pattern in any script's
`fetch`/`line` helpers).

---

## Methodology

### 1. Removing a component from a chain-type index (Törnqvist)

You **cannot** subtract chain-type price index *levels* to get an "ex-X" series.
We remove a component (info-proc equipment) from an aggregate (market-based or
core) using a Törnqvist re-aggregation on monthly log changes:

```
dln(P_rest) = ( dln(P_agg) - s_X * dln(P_X) ) / (1 - s_X)
```

where `s_X` is the **average nominal expenditure share** of the component across
months `t-1` and `t` (`s_X = 0.5*(n_X/n_agg at t + n_X/n_agg at t-1)`). The
residual log changes are chained back into an index, then turned into rates.
Implemented as `remove_component(...)` in the scripts.

### 2. Rates

- **YoY:** `P_t / P_{t-12} - 1`
- **3-month annualized (SAAR):** `(P_t / P_{t-3})^4 - 1` — equals annualizing
  the average of the trailing 3 monthly changes. Valid because the BEA monthly
  price indexes are already seasonally adjusted.

### 3. Residual seasonality

Even though the data is seasonally adjusted, core PCE shows leftover seasonality.
We test it two ways:
- **Descriptive:** average annualized m/m by calendar month across windows.
- **Formal:** OLS of annualized m/m on a linear trend + sum-to-zero month
  dummies, with HAC (Newey-West, 12 lags) standard errors, plus a joint F-test.
- **Adjustment:** additive seasonal factors from a classic decomposition
  (deviation of log m/m from a centered 12-term moving average), averaged by
  month over 2013–2025 ex-COVID and centered to sum to zero, then subtracted.

---

## Scripts and outputs

| Script | What it makes | Headline result (May 2026) |
|--------|---------------|----------------------------|
| `mkt_based_core_less_ipe.py` | Reproduction of Omair's chart, YoY, since 1996 | mkt-based core less IPE **3.01%** (his print: 3.34%) |
| `pce_overlay.py` | 4-line YoY overlay (core, mkt-based core, each less IPE) | core 3.41 / core-less-IPE 3.29 / mkt-core 3.16 / mkt-core-less-IPE 3.01 |
| `pce_3m_annualized.py` | Same 4 measures on 3m-annualized (full history) | mkt-based core less IPE **2.81%** |
| `pce_yoy_vs_3m.py` | YoY vs 3m-annualized for the headline measure | YoY 3.01 vs 3m 2.81 |
| `pce_3m_last3y.py` | 3m-annualized, last 3 years, incl. headline PCE | headline **6.28%**, core 3.52, core-less-IPE 3.06, mkt-core-less-IPE 2.81 |
| `pce_imputed_wedge.py` | Core minus market-based core (the imputed wedge) | wedge **+0.25pp** |
| `pce_residual_seasonality.py` | Avg annualized m/m by calendar month, 3 windows | Jan premium **+1.03pp** (descriptive) |
| `pce_deseasonalize.py` | Month-dummy regression + de-seasonalized 3y chart | Jan **+1.14pp (p=0.008)**; joint F p=0.012 |

Each writes a matching `.png` and `.csv`.

---

## What it means (interpretation)

**Reproducing the original.** Our pipeline nails headline core PCE (3.41% vs his
cited 3.4%), confirming the plumbing. The exact "less IPE" print differs (we get
3.01% vs his 3.34%) because his Inflation Insights / Haver "market-based"
construction differs from BEA's published line 402, and because of data vintage.
The full historical *shape* matches closely. Notably **core-less-IPE = 3.29%**
lands closest to his 3.34%, suggesting his series tracks nearer the
core-less-IPE level than BEA's published market-based line would imply.

**Market-based vs core.** Two independent exclusions:
- *Core* = total minus **food & energy** (cut by category).
- *Market-based* = total minus **imputed / non-market-priced** components (cut by
  price source): FISIM and other "financial services furnished without payment,"
  portfolio-management/insurance-type items, most gambling, used-vehicle margins,
  spending abroad. It *keeps* owners' equivalent rent. This is ~16% of core PCE
  by dollar value.
- The **imputed wedge** (core − market-based core) is usually mildly positive
  (mean +0.19pp since 1996) and pro-cyclical with asset markets — it swung to
  **−1.27pp** in 2008–09 when financial-services values collapsed. Currently
  +0.25pp.

**Info-processing equipment** is currently *inflating* (~+10% YoY, a tariff-era
hardware-price surge after years of deflation), so removing it now *lowers* the
inflation read — the opposite of its historical effect.

**Lens matters.** On YoY the less-IPE measures look stuck near 3%; on 3m
annualized they run cooler (mkt-based core less IPE at 2.81%, nearest to the 2%
target). 3m SAAR leads YoY at turning points but is far noisier.

**Residual seasonality (confirmed statistically).** January prints ~+1.1pp hot
and November ~−0.85pp cold even in seasonally adjusted data; the joint F-test
rejects no-seasonality (p≈0.012). Practical upshot: discount hot Q1 inflation
prints. But de-seasonalizing only flattens the *shape* — the early-2026
reacceleration survives the adjustment, so that pickup is largely genuine signal,
not just a calendar artifact.

---

## Caveats

- **Vintage:** BEA revises. Re-running later will shift recent values.
- **Törnqvist removal** is the standard analyst approximation, not BEA's exact
  Fisher re-aggregation from full underlying detail — very close, not identical.
- **Seasonal adjustment** here is a transparent additive decomposition with
  factors fixed from 2013–2025 ex-COVID, **not** a full X-13ARIMA re-run (what
  BEA uses). The significance test and qualitative conclusions are robust; treat
  the exact adjusted values as indicative.
- **COVID (2020–21)** is excluded from seasonal-factor estimation to avoid
  distortion, but included in the plotted series.

---

## Reproducing / extending

- All series come from `U20404` (prices) and `U20405` (nominal). Add a measure by
  finding its `LineNumber` and reusing `fetch` / `line` / `remove_component`.
- Common helpers are duplicated across scripts intentionally so each runs
  standalone; if consolidating, factor them into a small `pce_lib.py`.
- Branch: `claude/fred-data-chart-r6oqkp`.
