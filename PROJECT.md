# Estonian Population Projection — Project Reference

## Overview
Streamlit app projecting Estonia's population using the cohort-component method with a female-dominant Leslie matrix model. UI at `entry.py`; model logic in `projection.py`; Streamlit pages in `streamlit_proj.py`, `meetod.py`, `allikad.py`.

## Running
```bash
streamlit run entry.py
```

## Architecture

| File | Purpose |
|---|---|
| `entry.py` | Multi-page navigation |
| `streamlit_proj.py` | UI, sidebar inputs, charts |
| `projection.py` | All model logic (no Streamlit) |
| `meetod.py` | Methodology page (Estonian), includes migration rates chart |
| `allikad.py` | Data sources page |
| `fetch_immig_data.py` | One-time script: fetch migration data from stat.ee, save CSVs to data repo |
| `plot_migration_rates.py` | Diagnostic script: plot in/out rates by age (not part of the app) |

## Key Constants (`projection.py`)

| Constant | Value | Meaning |
|---|---|---|
| `BASE_YEAR` | 2023 | Projection base year |
| `MAX_AGE` | 110 | Open age group upper bound |
| `MAB_BASE` | 30.62 | Mean age at birth (base year) |
| `SD_MAB_BASE` | 5.62 | Std dev of MAB |
| `SEX_RATIO_AT_BIRTH` | 1.05 | Male births per female birth |
| `RADIX` | 200000 | Life table radix (l₀) |
| `RAMP_SHAPE` | 5 | Gompertz shape for TFR ramp |
| `KNOWN_TFR` | `{2024: 1.18, 2025: 1.18}` | Hardcoded observed TFR values |
| `HISTORY_START` | 2010 | First year shown on TFR chart |
| `AGE_WORK_MIN/MAX` | 18 / 65 | Working-age bounds |
| `AGE_SCHOOL_MIN/MAX` | 0 / 18 | Child age bounds (ages 0–17) |
| `AGE_OLD_MIN` | 65 | Old-age lower bound |
| `IMMIG_FEMALE_SHARE` | 0.3942 | Female share of scenario inflow (RVR09 2017–2019 avg) |

## Data Sources (GitHub: klesment/PopProj)

### Demographic base data
| Variable | File | Content |
|---|---|---|
| `URL_ASFR` | `ESTasfrRR_2023.txt` | Age-specific fertility rates |
| `URL_LT` | `LT_Female_2024.txt` | Female life table (e₀ ≈ 83) |
| `URL_LT_MALE` | `LT_Male_2024.txt` | Male life table |
| `URL_POP` | `Population_2025.txt` | Population by age and sex |
| `URL_TFRMAB` | `EST_TFRMAB_2023.txt` | TFR/MAB history |

### Migration data (stat.ee sources, saved as static CSVs)
| Variable | File | Content | Source |
|---|---|---|---|
| `URL_MT_STOCK` | `mt_stock_2021.csv` | Population by mother tongue (Estonian/Other), sex and 5-yr age group, 2021 census | RL21434 |
| `URL_IMMIG_INFLOW` | `immig_inflow_dist.csv` | Normalised inflow age distribution (shares), 2017–2019 avg, excl. Estonian returnees | RVR09 |
| `URL_EMIG_RATES` | `emig_rates.csv` | Annual emigration rates by 5-yr age group and sex, 2017–2019 avg | RVR03 + RVR10 |
| `URL_IMMIG_BASELINE` | `immig_baseline.csv` | Baseline annual inflow counts by 5-yr age group and sex, 2017–2019 avg | RVR03 + RVR10 |

## Model Logic

### Female projection
1. `build_scenario_df()` — TFR path via Gompertz ramp, known years overridden
2. `asfr_gamma()` — TFR + MAB → age-specific fertility (Gamma model)
3. `build_leslie_base()` + `leslie()` — build one Leslie matrix per year
4. `project_both_sexes()` — iterate over projection years

### Male projection
Derived from female: male births = female newborns × SRB × (L₀_male / L₀_female). Survival via `build_male_survival()`.

### Immigration model
Two parallel population stocks — native (Estonian mother tongue) and immigrant (other mother tongue) — projected simultaneously with the **same Leslie matrix**. Each year:

1. **Survival + births** — Leslie matrix applied to both stocks; births from immigrant mothers join immigrant stock
2. **Emigration** — age-specific rates applied multiplicatively to post-survival vectors:
   - Estonian-citizen emigration rates → native stock
   - Non-Estonian-citizen emigration rates → immigrant stock
3. **Immigration inflow** — two components added each year:
   - **Baseline** (`immig_baseline.csv`): historical average inflow (2017–2019) by nationality, always on; Estonian nationals (returnees) → native stock, non-Estonians → immigrant stock
   - **Scenario** (user-controlled slider): additional non-Estonian inflow above baseline → immigrant stock only

#### Key design decisions
- Same fertility and mortality schedules for native and immigrant (single Leslie matrix)
- Children of immigrants join immigrant stock
- Grouping variable is mother tongue (immutable, inherited from mother)
- Emigration implemented as rates (scales with projected population size)
- Baseline inflow implemented as fixed annual counts (from RVR03 total − RVR10 Estonian); same nationality proxy as emigration side
- Slider = additional non-Estonian inflow above historical baseline; at 0 the baseline continues
- Initial population split: 2021 census mother-tongue proportions applied to 2023 total population
- Citizenship used as proxy for native/immigrant on both inflow and outflow sides (known limitation: ~50k naturalized Estonians misclassified)

#### Functions in `projection.py`
| Function | Purpose |
|---|---|
| `_disaggregate_5yr(counts_5yr, lx)` | PCHIP + life-table tail: 5-yr groups → single years |
| `load_mt_stock(lt_f, lt_m)` | Load and disaggregate initial population by mother tongue |
| `load_immig_inflow_dist(lt_f, lt_m)` | Load and disaggregate normalised inflow age distribution |
| `load_emig_rates()` | Load emigration rates, expand to single-year step function |
| `load_immig_baseline(lt_f, lt_m)` | Load and disaggregate baseline annual inflow counts |
| `build_immig_vectors(annual_total, dist_f, dist_m, per)` | Build per-year scenario inflow arrays |

#### `project_both_sexes()` — signature
```python
project_both_sexes(
    lmat_female, subd_male, l0_ratio,
    pop_native_f, pop_native_m,       # initial native vectors
    pop_immig_f,  pop_immig_m,        # initial immigrant vectors
    nat_inflow_f, nat_inflow_m,       # baseline inflow arrays, native (per × MAX_AGE)
    imm_inflow_f, imm_inflow_m,       # total inflow arrays, immigrant (baseline + scenario)
    emig_nat_f, emig_nat_m,           # emigration rate vectors, native
    emig_imm_f, emig_imm_m,           # emigration rate vectors, immigrant
    per
) → (native_f, native_m, immig_f, immig_m)
```

### Structural indicators (`compute_indicators`)
- Working-age: ages 18–64
- Child age: ages 0–17
- Old-age: ages 65+

## Caching
`@st.cache_data` on `load_and_clean()`, `load_immig_data()`, and `run_projection()`. The `_lt_female`, `_lt_male` parameters are prefixed with `_` to exclude DataFrames from the cache key.

## User Inputs (Sidebar)
- TFR at end of period (absolute value, slider 0.5–3.0, default = base year TFR)
- Ramp speed: ▁ / ▄ / █ (Gompertz shape 5/6/8)
- Mean age at birth (MAB) at end of period
- Projection length: 0–100 years in 5-year steps
- Additional non-Estonian inflow above baseline (persons/year, slider 0–20,000)

## Population Pyramid
- Native (Estonian MT): red (female) / blue (male)
- Immigrant (other MT): orange (female) / green (male), stacked
- Dashed line marks projection boundary (cohorts born during projection period)

## TFR Chart
- Solid blue line: observed history from 2010 onward
- Dashed orange line: scenario, starts from last observed value
- 2024–2025 values hardcoded at 1.18; scenario starts 2026

## Known Issues / Design Decisions
- Mortality fixed at base year — no mortality improvement scenario
- `OPEN_AGE_SENTINEL = '110+'` replaced with `MAX_AGE = 110` during data load
- `qx == 0` set to NaN in life tables to avoid division issues
- Population file uses `Population_2025.txt` but base year is 2023 (file contains multi-year data; filtered by `BASE_YEAR`)
