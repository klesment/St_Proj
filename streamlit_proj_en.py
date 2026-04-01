# v2 — baseline inflow by nationality (English UI)
# © 2026 Martin Klesment. Licensed under CC BY-NC 4.0.
import streamlit as st

st.set_page_config(page_title="Interactive Population Projection")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Patch, Rectangle

from projection import (
    URL_ASFR, URL_LT, URL_LT_MALE, URL_POP,
    BASE_YEAR, MAB_BASE, SD_MAB_BASE, MAX_AGE, OPEN_AGE_SENTINEL, SEX_RATIO_AT_BIRTH,
    AGE_SCHOOL_MIN, AGE_SCHOOL_MAX, AGE_WORK_MIN, AGE_WORK_MAX, AGE_OLD_MIN,
    KNOWN_TFR,
    load_data, load_tfr_history, compute_tfr_start, build_scenario_df,
    asfr_gamma, build_leslie_base, leslie, build_male_survival, improve_lt,
    load_mt_stock, load_immig_inflow_dist, load_emig_rates, load_immig_baseline,
    build_immig_vectors, project_both_sexes, compute_indicators,
)

FIGURE_SIZE   = (8, 4.5)
AGE_TICK_STEP = 5


@st.cache_data
def load_and_clean():
    asfr = load_data(URL_ASFR).apply(pd.to_numeric, errors='coerce')

    lt = load_data(URL_LT)
    lt['Age'] = lt['Age'].replace(OPEN_AGE_SENTINEL, str(MAX_AGE))
    lt = lt.apply(pd.to_numeric)
    lt.loc[lt['qx'] == 0, 'qx'] = np.nan

    lt_male = load_data(URL_LT_MALE)
    lt_male['Age'] = lt_male['Age'].replace(OPEN_AGE_SENTINEL, str(MAX_AGE))
    lt_male = lt_male.apply(pd.to_numeric)
    lt_male.loc[lt_male['qx'] == 0, 'qx'] = np.nan

    pop = load_data(URL_POP)
    pop['Age'] = pop['Age'].replace(OPEN_AGE_SENTINEL, str(MAX_AGE))
    pop = pop.apply(pd.to_numeric)

    tfr_history = load_tfr_history()

    return asfr, lt, lt_male, pop, tfr_history


@st.cache_data
def load_immig_data(_lt_female, _lt_male):
    est_f, est_m, oth_f, oth_m     = load_mt_stock(_lt_female, _lt_male)
    inflow_dist_f, inflow_dist_m   = load_immig_inflow_dist(_lt_female, _lt_male)
    emig_nat_f, emig_nat_m, emig_imm_f, emig_imm_m = load_emig_rates()
    base_nat_f, base_nat_m, base_imm_f, base_imm_m = load_immig_baseline(_lt_female, _lt_male)
    return (est_f, est_m, oth_f, oth_m,
            inflow_dist_f, inflow_dist_m,
            emig_nat_f, emig_nat_m, emig_imm_f, emig_imm_m,
            base_nat_f, base_nat_m, base_imm_f, base_imm_m)


@st.cache_data
def run_projection(tfr_start, tfr_change, ramp_speed, mab_stop, period, extra_immig,
                   mort_improvement, baseline_inflow, emigration_scale,
                   _lt_female, _lt_male,
                   n0_nat_f, n0_nat_m, n0_imm_f, n0_imm_m,
                   inflow_dist_f, inflow_dist_m,
                   emig_nat_f, emig_nat_m, emig_imm_f, emig_imm_m,
                   base_nat_f, base_nat_m, base_imm_f, base_imm_m,
                   osc_amp=0.0, osc_cycle=10):
    e0_base  = _lt_female['Tx'].iloc[0] / _lt_female['lx'].iloc[0]
    e0m_base = _lt_male['Tx'].iloc[0]   / _lt_male['lx'].iloc[0]
    if period == 0:
        d = pd.DataFrame({'tfr': [tfr_start], 'mab': [mab_stop], 'sd_mab': [SD_MAB_BASE]})
        return d, np.array(n0_nat_f), np.array(n0_nat_m), np.array(n0_imm_f), np.array(n0_imm_m), e0_base, e0m_base, {}

    d = build_scenario_df(tfr_start, tfr_change, ramp_speed, mab_stop, period, osc_amp, osc_cycle)
    F = np.array([asfr_gamma(r.tfr, r.mab, r.sd_mab) for r in d.itertuples(index=False)])

    if mort_improvement == 0:
        # Fast path: single life table for all years
        base_leslie  = build_leslie_base(_lt_female)
        LL           = [leslie(f, base_leslie) for f in F]
        subd_m_arr   = np.tile(build_male_survival(_lt_male), (period, 1))
        l0_r_arr     = np.full(period, _lt_male['Lx'].iloc[0] / _lt_female['Lx'].iloc[0])
        e0_end       = e0_base
        e0m_end      = e0m_base
    else:
        # Per-year life tables with mortality improvement
        LL         = []
        subd_m_arr = np.zeros((period, MAX_AGE))
        l0_r_arr   = np.zeros(period)
        for t in range(period):
            lt_f_t = improve_lt(_lt_female, mort_improvement, t + 1)
            lt_m_t = improve_lt(_lt_male,   mort_improvement, t + 1)
            base_t = build_leslie_base(lt_f_t)
            LL.append(leslie(F[t], base_t))
            subd_m_arr[t] = build_male_survival(lt_m_t)
            l0_r_arr[t]   = lt_m_t['Lx'].iloc[0] / lt_f_t['Lx'].iloc[0]
        lt_end   = improve_lt(_lt_female, mort_improvement, period)
        lt_m_end = improve_lt(_lt_male,   mort_improvement, period)
        e0_end   = lt_end['Tx'].iloc[0]   / lt_end['lx'].iloc[0]
        e0m_end  = lt_m_end['Tx'].iloc[0] / lt_m_end['lx'].iloc[0]

    # Scenario inflow: user-controlled additional non-Estonian immigrants
    scen_imm_f, scen_imm_m = build_immig_vectors(
        extra_immig,
        np.array(inflow_dist_f), np.array(inflow_dist_m),
        period,
    )

    # Total inflow: baseline (always on) + scenario (user-controlled, immigrant stock only)
    nat_inflow_f = np.tile(np.array(base_nat_f) * baseline_inflow, (period, 1))
    nat_inflow_m = np.tile(np.array(base_nat_m) * baseline_inflow, (period, 1))
    imm_inflow_f = np.tile(np.array(base_imm_f) * baseline_inflow, (period, 1)) + scen_imm_f
    imm_inflow_m = np.tile(np.array(base_imm_m) * baseline_inflow, (period, 1)) + scen_imm_m

    nat_f, nat_m, out_imm_f, out_imm_m, snapshots = project_both_sexes(
        np.array(LL), subd_m_arr, l0_r_arr,
        n0_nat_f, n0_nat_m, n0_imm_f, n0_imm_m,
        nat_inflow_f, nat_inflow_m,
        imm_inflow_f, imm_inflow_m,
        np.array(emig_nat_f) * emigration_scale, np.array(emig_nat_m) * emigration_scale,
        np.array(emig_imm_f) * emigration_scale, np.array(emig_imm_m) * emigration_scale,
        period,
        snapshot_years=(2050, 2075, 2100),
    )
    return d, nat_f, nat_m, out_imm_f, out_imm_m, e0_end, e0m_end, snapshots


# --- Data ---
try:
    with st.spinner('Loading data...'):
        asfr, lt, lt_male, pop, tfr_history = load_and_clean()
except Exception as e:
    st.error(
        'Failed to load data. '
        'Please check your internet connection and reload the page.'
    )
    st.stop()

lt_base      = lt[lt['Year'] == BASE_YEAR]
lt_male_base = lt_male[lt_male['Year'] == BASE_YEAR]
pop_base     = pop[pop['Year'] == BASE_YEAR]
e0_base      = lt_base['Tx'].iloc[0]      / lt_base['lx'].iloc[0]
e0m_base     = lt_male_base['Tx'].iloc[0] / lt_male_base['lx'].iloc[0]
N0           = pop_base.Female[0:MAX_AGE].tolist()
N0_male      = pop_base.Male[0:MAX_AGE].tolist()
tfr_start    = compute_tfr_start(asfr, BASE_YEAR)

try:
    with st.spinner('Loading migration data...'):
        (census_est_f, census_est_m, census_oth_f, census_oth_m,
         inflow_dist_f, inflow_dist_m,
         emig_nat_f, emig_nat_m, emig_imm_f, emig_imm_m,
         base_nat_f, base_nat_m, base_imm_f, base_imm_m) = load_immig_data(
            _lt_female=lt_base, _lt_male=lt_male_base
        )
except Exception:
    st.error(
        'Failed to load migration data. '
        'Please check your internet connection and reload the page.'
    )
    st.stop()

# Split 2023 population using 2021 census mother-tongue proportions
census_total_f = census_est_f + census_oth_f
census_total_m = census_est_m + census_oth_m
oth_share_f = np.where(census_total_f > 0, census_oth_f / census_total_f, 0.0)
oth_share_m = np.where(census_total_m > 0, census_oth_m / census_total_m, 0.0)
N0_imm_f = np.array(N0)      * oth_share_f
N0_imm_m = np.array(N0_male) * oth_share_m
N0_nat_f = np.array(N0)      - N0_imm_f
N0_nat_m = np.array(N0_male) - N0_imm_m

# --- Sidebar ---
st.sidebar.markdown('Select projection assumptions')

option_map = {2: "▁", 6: "▄", 20: "█"}

def user_input_features(tfr_start):
    target_year = st.sidebar.slider(
        "Target year", min_value=BASE_YEAR, max_value=BASE_YEAR + 100, step=1, value=2026)
    Years = target_year - BASE_YEAR
    tfr_end = st.sidebar.slider(
        f"Fertility at end of period (TFR, {BASE_YEAR} = {tfr_start:.2f})",
        min_value=0.5, max_value=3.0, step=0.05, value=float(round(tfr_start * 20) / 20))
    TFR_Change = tfr_end / tfr_start - 1
    _ramp = st.sidebar.segmented_control(
        "Speed of TFR change",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode='single', default=6)
    Ramp = _ramp if _ramp is not None else 6
    Osc_amp = st.sidebar.slider(
        "TFR oscillation amplitude", min_value=0.0, max_value=0.5, step=0.05, value=0.0)
    Osc_cycle = st.sidebar.slider(
        "TFR oscillation period (yr)", min_value=5, max_value=30, step=5, value=10,
        disabled=(Osc_amp == 0.0))
    MAB_end = st.sidebar.slider(
        "Mean age at birth at end of period", min_value=MAB_BASE - 5, max_value=MAB_BASE + 5,
        step=0.5, value=MAB_BASE)
    Mort_impr = st.sidebar.slider(
        "Mortality decline (%/yr)", min_value=0.0, max_value=2.0, step=0.1, value=0.0) / 100
    _bi = st.sidebar.segmented_control(
        "Baseline immigration (2017–2019 average)", options=[0, 50, 100],
        format_func=lambda x: f"{x}%",
        selection_mode='single', default=100)
    Baseline_inflow = (_bi if _bi is not None else 100) / 100
    _em = st.sidebar.segmented_control(
        "Baseline emigration (2017–2019 average)", options=[0, 50, 100],
        format_func=lambda x: f"{x}%",
        selection_mode='single', default=100)
    Emigration = (_em if _em is not None else 100) / 100
    Annual_immig = st.sidebar.slider(
        "Additional immigration, non-Estonian (persons/yr)", min_value=0, max_value=20_000, step=500, value=0)
    return TFR_Change, Ramp, MAB_end, Years, Annual_immig, Mort_impr, Baseline_inflow, Emigration, Osc_amp, Osc_cycle


TFR_Change, Ramp, mab_stop, period, extra_immig, mort_improvement, baseline_inflow, emigration_scale, osc_amp, osc_cycle = user_input_features(tfr_start)

# --- Projection ---
d, nat_f, nat_m, imm_f, imm_m, e0_end, e0m_end, snapshots = run_projection(
    tfr_start, TFR_Change, Ramp, mab_stop, period, extra_immig,
    mort_improvement, baseline_inflow, emigration_scale,
    _lt_female=lt_base, _lt_male=lt_male_base,
    n0_nat_f=N0_nat_f.tolist(), n0_nat_m=N0_nat_m.tolist(),
    n0_imm_f=N0_imm_f.tolist(), n0_imm_m=N0_imm_m.tolist(),
    inflow_dist_f=inflow_dist_f.tolist(), inflow_dist_m=inflow_dist_m.tolist(),
    emig_nat_f=emig_nat_f.tolist(), emig_nat_m=emig_nat_m.tolist(),
    emig_imm_f=emig_imm_f.tolist(), emig_imm_m=emig_imm_m.tolist(),
    base_nat_f=base_nat_f.tolist(), base_nat_m=base_nat_m.tolist(),
    base_imm_f=base_imm_f.tolist(), base_imm_m=base_imm_m.tolist(),
    osc_amp=osc_amp, osc_cycle=osc_cycle,
)

out      = nat_f + imm_f
out_male = nat_m + imm_m

p_size_start = sum(pop['Total'][pop['Year'] == BASE_YEAR])
p_size_end   = sum(out) + sum(out_male)
total_births = round(out[0] * (1 + SEX_RATIO_AT_BIRTH))

_lt_f_end = improve_lt(lt_base,      mort_improvement, period) if mort_improvement > 0 and period > 0 else lt_base
_lt_m_end = improve_lt(lt_male_base, mort_improvement, period) if mort_improvement > 0 and period > 0 else lt_male_base
qx_f_end  = np.nan_to_num(_lt_f_end.sort_values('Age')['qx'].values[:MAX_AGE])
qx_m_end  = np.nan_to_num(_lt_m_end.sort_values('Age')['qx'].values[:MAX_AGE])
qx_f_base = np.nan_to_num(lt_base.sort_values('Age')['qx'].values[:MAX_AGE])
qx_m_base = np.nan_to_num(lt_male_base.sort_values('Age')['qx'].values[:MAX_AGE])
total_deaths       = round((out           * qx_f_end).sum()  + (out_male          * qx_m_end).sum())
total_deaths_start = round((np.array(N0)  * qx_f_base).sum() + (np.array(N0_male) * qx_m_base).sum())

# Net migration: inflow (fixed counts) minus emigration rates × projected population
inflow_nat_end   = (base_nat_f.sum() + base_nat_m.sum()) * baseline_inflow
inflow_imm_end   = (base_imm_f.sum() + base_imm_m.sum()) * baseline_inflow + extra_immig
inflow_total_end = inflow_nat_end + inflow_imm_end
emig_nat_end     = (nat_f * emig_nat_f).sum() * emigration_scale + (nat_m * emig_nat_m).sum() * emigration_scale
emig_imm_end     = (imm_f * emig_imm_f).sum() * emigration_scale + (imm_m * emig_imm_m).sum() * emigration_scale
net_migration_end  = inflow_total_end - emig_nat_end - emig_imm_end
imm_inflow_pct_end = inflow_imm_end / inflow_total_end * 100 if inflow_total_end > 0 else np.nan

inflow_total_base   = (base_nat_f.sum() + base_nat_m.sum() + base_imm_f.sum() + base_imm_m.sum()) * baseline_inflow
inflow_imm_base     = (base_imm_f.sum() + base_imm_m.sum()) * baseline_inflow
emig_nat_base       = ((N0_nat_f * emig_nat_f).sum() + (N0_nat_m * emig_nat_m).sum()) * emigration_scale
emig_imm_base       = ((N0_imm_f * emig_imm_f).sum() + (N0_imm_m * emig_imm_m).sum()) * emigration_scale
net_migration_base  = inflow_total_base - emig_nat_base - emig_imm_base
imm_inflow_pct_base = inflow_imm_base / inflow_total_base * 100 if inflow_total_base > 0 else np.nan

# --- TFR chart + metrics ---
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

with col1:
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=2)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    last_known_year = max(tfr_history.keys())
    last_known_idx  = last_known_year - BASE_YEAR - 1  # row index in d

    # Actual line: full history from HISTORY_START through last known year
    actual_x = sorted(tfr_history.keys())
    actual_y = [tfr_history[y] for y in actual_x]
    ax.plot(actual_x, actual_y, linewidth=2.5, color='#2196F3', label='Actual')

    # Scenario line: starts at last known value and continues into the projection
    if period > last_known_idx + 1:
        proj_years = list(range(BASE_YEAR + 1, BASE_YEAR + 1 + period))
        scenario_x = [last_known_year] + proj_years[last_known_idx + 1:]
        scenario_y = [tfr_history[last_known_year]] + d['tfr'].iloc[last_known_idx + 1:].tolist()
        ax.plot(scenario_x, scenario_y, linewidth=2.5, linestyle='--', color='#FF5722', label='Scenario')

    ax.set(ylabel='Births per woman', xlabel='Year')
    ax.legend()
    st.caption(f"Fertility change {min(tfr_history.keys())} – {BASE_YEAR + period}")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.caption(f"{BASE_YEAR + period}")
    col_c, col_d = st.columns(2)
    col_e, col_f = st.columns(2)
    col_c.metric("Life expectancy (F)",
                 round(e0_end, 1), round(e0_end - e0_base, 1), border=False)
    col_d.metric("Life expectancy (M)",
                 round(e0m_end, 1), round(e0m_end - e0m_base, 1), border=False)
    col_e.metric("Annual net migration",
                 f"{round(net_migration_end):,}", f"{round(net_migration_end - net_migration_base):,}", border=False)
    col_f.metric("Estonian share of immigration",
                 f"{100 - imm_inflow_pct_end:.1f}%", f"{imm_inflow_pct_base - imm_inflow_pct_end:.1f}pp", border=False)

# --- Population pyramid ---
ind_end   = compute_indicators(out, out_male)
ind_start = compute_indicators(N0, N0_male)

st.divider()

ages = np.arange(MAX_AGE)

class HandlerSplitPatch(HandlerBase):
    def __init__(self, color_left, color_right):
        self.color_left  = color_left
        self.color_right = color_right
        super().__init__()
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        r1 = Rectangle((xdescent,            ydescent), width / 2, height, facecolor=self.color_left,  transform=trans)
        r2 = Rectangle((xdescent + width / 2, ydescent), width / 2, height, facecolor=self.color_right, transform=trans)
        return [r1, r2]

_C_EST_F, _C_EST_M = '#e05252', '#2e86c1'
_C_IMM_F, _C_IMM_M = '#e8923a', '#3aae82'

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1)
fig2, ax2 = plt.subplots(figsize=(10, 8))

ax2.barh(ages,  nat_f, color=_C_EST_F)
ax2.barh(ages, -nat_m, color=_C_EST_M)
ax2.barh(ages,  imm_f, left= nat_f,  color=_C_IMM_F)
ax2.barh(ages, -imm_m, left=-nat_m, color=_C_IMM_M)

if period > 0:
    hline = ax2.axhline(period - 0.5, color='black', linewidth=1.2, linestyle='--')

ax2.set_xlabel('People per age group')
ax2.set_ylabel('Age')
ax2.set_yticks(np.arange(0, MAX_AGE - 10, AGE_TICK_STEP))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(int(x)):,}'))
ax2.axvline(0, color='black', linewidth=0.8)

h_est = Patch()
h_muu = Patch()
legend_handles = [h_est, h_muu]
legend_labels  = ['Estonian mother tongue', 'Other mother tongue']
legend_handler_map = {
    h_est: HandlerSplitPatch(_C_EST_M, _C_EST_F),
    h_muu: HandlerSplitPatch(_C_IMM_M, _C_IMM_F),
}
if period > 0:
    legend_handles.append(hline)
    legend_labels.append(f'Projection boundary\n(b. {BASE_YEAR + 1}–{BASE_YEAR + period})')
ax2.legend(legend_handles, legend_labels, handler_map=legend_handler_map, loc='upper left')

xlim = ax2.get_xlim()
ylim = ax2.get_ylim()
ax2.text(xlim[0] * 0.97, ylim[0] + 1, '← Men',   ha='left',  va='bottom', fontsize=12, color='#555555')
ax2.text(xlim[1] * 0.97, ylim[0] + 1, 'Women →', ha='right', va='bottom', fontsize=12, color='#555555')

st.caption(f"Population pyramid {BASE_YEAR + period}, by mother tongue  |  Dashed line separates data-based and projected cohorts")
st.pyplot(fig2, width='stretch')
plt.close(fig2)

# --- Non-Estonian summary metrics ---
imm_total_end   = int(round(imm_f.sum()    + imm_m.sum()))
imm_newborn_end = int(round(imm_f[0]       + imm_m[0]))
imm_total_start = int(round(N0_imm_f.sum() + N0_imm_m.sum()))
imm_newborn_start = int(round(N0_imm_f[0]  + N0_imm_m[0]))

# --- Structural indicators ---
work_pct_end     = ind_end['working_age']  / p_size_end   * 100
work_pct_start   = ind_start['working_age'] / p_size_start * 100
school_pct_end   = ind_end['school_age']   / p_size_end   * 100
school_pct_start = ind_start['school_age'] / p_size_start * 100
old_pct_end      = ind_end['old_age']      / p_size_end   * 100
old_pct_start    = ind_start['old_age']    / p_size_start * 100

imm_school_end   = (imm_f[AGE_SCHOOL_MIN:AGE_SCHOOL_MAX].sum() + imm_m[AGE_SCHOOL_MIN:AGE_SCHOOL_MAX].sum())
imm_work_end     = (imm_f[AGE_WORK_MIN:AGE_WORK_MAX].sum()     + imm_m[AGE_WORK_MIN:AGE_WORK_MAX].sum())
imm_old_end      = (imm_f[AGE_OLD_MIN:].sum()                   + imm_m[AGE_OLD_MIN:].sum())
imm_school_pct_end  = imm_school_end / ind_end['school_age']  * 100 if ind_end['school_age']  > 0 else np.nan
imm_work_pct_end    = imm_work_end   / ind_end['working_age'] * 100 if ind_end['working_age'] > 0 else np.nan
imm_old_pct_end     = imm_old_end    / ind_end['old_age']     * 100 if ind_end['old_age']     > 0 else np.nan

imm_school_start = (N0_imm_f[AGE_SCHOOL_MIN:AGE_SCHOOL_MAX].sum() + N0_imm_m[AGE_SCHOOL_MIN:AGE_SCHOOL_MAX].sum())
imm_work_start   = (N0_imm_f[AGE_WORK_MIN:AGE_WORK_MAX].sum()     + N0_imm_m[AGE_WORK_MIN:AGE_WORK_MAX].sum())
imm_old_start    = (N0_imm_f[AGE_OLD_MIN:].sum()                   + N0_imm_m[AGE_OLD_MIN:].sum())
imm_school_pct_start = imm_school_start / ind_start['school_age']  * 100 if ind_start['school_age']  > 0 else np.nan
imm_work_pct_start   = imm_work_start   / ind_start['working_age'] * 100 if ind_start['working_age'] > 0 else np.nan
imm_old_pct_start    = imm_old_start    / ind_start['old_age']     * 100 if ind_start['old_age']     > 0 else np.nan

st.divider()
st.caption(f"Projected absolute figures {BASE_YEAR + period}")

r1a, r1b, r1c, r1d = st.columns(4)
r1a.metric("Total population",        f"{round(p_size_end):,}", round(p_size_end - p_size_start),        border=False)
r1b.metric("Other mother tongue",     f"{imm_total_end:,}",    imm_total_end - imm_total_start,          border=False)
r1c.metric("Births",                  f"{total_births:,}",     "",                                       border=False)
r1d.metric("Deaths",                  f"{total_deaths:,}",     round(total_deaths - total_deaths_start), border=False)

st.caption("Projected age group shares in total population")
r3a, r3b, r3c = st.columns(3)
r3a.metric("0–17",  f"{school_pct_end:.1f}%", f"{school_pct_end - school_pct_start:.1f}pp", border=False)
r3b.metric("18–64", f"{work_pct_end:.1f}%",   f"{work_pct_end - work_pct_start:.1f}pp",     border=False)
r3c.metric("65+",   f"{old_pct_end:.1f}%",    f"{old_pct_end - old_pct_start:.1f}pp",       border=False)

st.caption("of which Estonian mother tongue")
r4a, r4b, r4c = st.columns(3)
r4a.metric("0–17",  f"{100 - imm_school_pct_end:.1f}%", f"{imm_school_pct_start - imm_school_pct_end:.1f}pp", border=False)
r4b.metric("18–64", f"{100 - imm_work_pct_end:.1f}%",   f"{imm_work_pct_start - imm_work_pct_end:.1f}pp",     border=False)
r4c.metric("65+",   f"{100 - imm_old_pct_end:.1f}%",    f"{imm_old_pct_start - imm_old_pct_end:.1f}pp",       border=False)

# --- Snapshot tables ---
available_snaps = [yr for yr in (2050, 2075, 2100) if yr in snapshots]
st.divider()
st.caption("Population age distribution at projection snapshots — pre-selected years")
all_snap_years = [BASE_YEAR] + available_snaps
all_snap_data  = {BASE_YEAR: (N0_nat_f, N0_nat_m, N0_imm_f, N0_imm_m), **snapshots}
tabs = st.tabs([str(yr) for yr in all_snap_years])
for tab, yr in zip(tabs, all_snap_years):
    with tab:
        sn_nat_f, sn_nat_m, sn_imm_f, sn_imm_m = all_snap_data[yr]
        snap_df = pd.DataFrame({
            'Estonian F': np.round(sn_nat_f).astype(int),
            'Estonian M': np.round(sn_nat_m).astype(int),
            'Other F':    np.round(sn_imm_f).astype(int),
            'Other M':    np.round(sn_imm_m).astype(int),
        }, index=pd.Index(np.arange(MAX_AGE), name='Age'))
        st.dataframe(snap_df, width='stretch')
