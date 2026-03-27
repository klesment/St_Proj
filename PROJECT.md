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
| `meetod.py` | Methodology page (Estonian) |
| `allikad.py` | Data sources page |
| `fetch_immig_data.py` | One-time script: fetch migration data from stat.ee, save CSVs to data repo |

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
| `AGE_SCHOOL_MIN/MAX` | 7 / 18 | School-age bounds |
| `AGE_OLD_MIN` | 65 | Old-age lower bound |
| `IMMIG_FEMALE_SHARE` | 0.3942 | Female share of annual inflow (RVR09 2019–2021 avg) |

## Data Sources (GitHub: klesment/PopProj)

### Existing
| Variable | File | Content |
|---|---|---|
| `URL_ASFR` | `ESTasfrRR_2023.txt` | Age-specific fertility rates |
| `URL_LT` | `LT_Female_2024.txt` | Female life table (e₀ ≈ 83) |
| `URL_LT_MALE` | `LT_Male_2024.txt` | Male life table |
| `URL_POP` | `Population_2025.txt` | Population by age and sex |
| `URL_TFRMAB` | `EST_TFRMAB_2023.txt` | TFR/MAB history |

### Immigration branch (stat.ee sources, saved as static CSVs)
| Variable | File | Content | Source |
|---|---|---|---|
| `URL_IMMIG_STOCK` | `immig_stock_2025.csv` | Foreign-origin stock (1st+2nd gen) by 5-yr age group and sex, Jan 2025 | RV071 |
| `URL_IMMIG_INFLOW` | `immig_inflow_dist.csv` | Normalised inflow age distribution (shares), 2019–2021 avg, excl. Estonian returnees | RVR09 |
| `URL_EMIG_RATES` | `emig_rates.csv` | Annual emigration rates by 5-yr age group and sex, 2019–2021 avg | RVR08 + RV071 |

## Model Logic

### Female projection
1. `build_scenario_df()` — TFR path via Gompertz ramp, known years overridden
2. `asfr_gamma()` — TFR + MAB → age-specific fertility (Gamma model)
3. `build_leslie_base()` + `leslie()` — build one Leslie matrix per year
4. `project_both_sexes()` — iterate over projection years

### Male projection
Derived from female: male births = female newborns × SRB × (L₀_male / L₀_female). Survival via `build_male_survival()`.

### Immigration model (branch: `immigration`)
Two parallel population stocks — native and immigrant — projected simultaneously with the **same Leslie matrix**. Each year:

1. **Survival + births** — Leslie matrix applied to both stocks; births from immigrant mothers join immigrant stock
2. **Emigration** — age-specific rates (from `emig_rates.csv`) applied multiplicatively to post-survival vectors:
   - Estonian-citizen emigration rates → native stock
   - Non-Estonian-citizen emigration rates → immigrant stock
3. **Immigration inflow** — annual inflow vector (annual total × normalised age distribution) added to immigrant stock

#### Key design decisions
- Same fertility and mortality schedules for native and immigrant (single Leslie matrix)
- Children of immigrants join immigrant stock
- 3rd-generation foreign-origin treated as native; 1st+2nd generation as immigrant
- Emigration implemented as rates (not absolute counts) so it scales with the projected population
- Initial immigrant stock: RV071 1st+2nd generation, January 2025, disaggregated from 5-yr to single-year ages via PCHIP + life-table tail (85+)
- Inflow age distribution: RVR09 2019–2021 average, excluding Estonian-born returnees
- Emigration rates: RVR08 2019–2021 average; Estonian-citizen emigrants / native pop; non-Estonian emigrants / immigrant pop (denominator from RV071)
- Citizenship used as proxy for native/immigrant on the emigration side (known limitation: ~50k naturalized Estonians misclassified as native emigrants)

#### New functions in `projection.py`
| Function | Purpose |
|---|---|
| `_disaggregate_5yr(counts_5yr, lx)` | PCHIP + life-table tail: 5-yr groups → single years |
| `load_immig_stock(lt_f, lt_m)` | Load and disaggregate initial immigrant stock |
| `load_immig_inflow_dist(lt_f, lt_m)` | Load and disaggregate normalised inflow distribution |
| `load_emig_rates()` | Load emigration rates, expand to single-year step function |
| `build_immig_vectors(annual_total, dist_f, dist_m, per)` | Build per-year inflow arrays from scenario total |

#### `project_both_sexes()` — updated signature
```python
project_both_sexes(
    lmat_female, subd_male, l0_ratio,
    pop_native_f, pop_native_m,     # initial native vectors
    pop_immig_f,  pop_immig_m,      # initial immigrant vectors
    imm_f, imm_m,                   # inflow arrays (per × MAX_AGE)
    emig_nat_f, emig_nat_m,         # emigration rate vectors, native
    emig_imm_f, emig_imm_m,         # emigration rate vectors, immigrant
    per
) → (native_f, native_m, immig_f, immig_m)
```

### Structural indicators (`compute_indicators`)
- OADR: 65+ / working-age × 100
- Working-age: ages 18–64
- School-age: ages 7–17
- Old-age: ages 65+

## Caching
`@st.cache_data` on `load_and_clean()` and `run_projection()`. The `_lt_female`, `_lt_male` parameters are prefixed with `_` to exclude DataFrames from the cache key.

## User Inputs (Sidebar)
- TFR change (%) relative to 2023
- Ramp speed: Slow / Medium / Fast (Gompertz shape 5/6/8)
- Mean age at birth (MAB) at end of period
- Projection length: 0–100 years in 5-year steps
- *(immigration branch)* Annual immigration total — **not yet wired into Streamlit**

## Population Pyramid
- Darker shades = cohorts from observed data (survived)
- Lighter shades = cohorts born during projection
- Split by `ages >= period` (data) vs `ages < period` (projected)

## TFR Chart
- Solid blue line: observed history from 2010 onward
- Dashed orange line: scenario, starts from last observed value
- 2024–2025 values hardcoded at 1.18; scenario starts 2026

## Branch: `immigration` — remaining work
- [ ] Update `streamlit_proj.py`: load immigration data, add sidebar slider, wire 4-vector output into indicators and pyramid
- [ ] Update `meetod.py` and `allikad.py` to document the migration model and new data sources

## Known Issues / Design Decisions
- Mortality fixed at base year — no mortality improvement scenario
- `OPEN_AGE_SENTINEL = '110+'` replaced with `MAX_AGE = 110` during data load
- `qx == 0` set to NaN in life tables to avoid division issues
- Population file uses `Population_2025.txt` but base year is 2023 (file contains multi-year data; filtered by `BASE_YEAR`)
