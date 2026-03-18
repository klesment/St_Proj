import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from projection import (
    URL_ASFR, URL_LT, URL_LT_MALE, URL_POP,
    BASE_YEAR, MAB_BASE, SD_MAB_BASE, MAX_AGE, OPEN_AGE_SENTINEL, SEX_RATIO_AT_BIRTH,
    AGE_SCHOOL_MIN, AGE_SCHOOL_MAX, AGE_WORK_MIN, AGE_WORK_MAX, AGE_OLD_MIN,
    KNOWN_TFR,
    load_data, load_tfr_history, compute_tfr_start, build_scenario_df,
    asfr_gamma, build_leslie_base, leslie, build_male_survival,
    project_both_sexes, compute_indicators,
)

FIGURE_SIZE   = (8, 4.5)
AGE_TICK_STEP = 5


@st.cache_data
def load_and_clean():
    asfr = load_data(URL_ASFR).apply(pd.to_numeric, errors='coerce')

    lt = load_data(URL_LT)
    lt.loc[lt['Age'] == OPEN_AGE_SENTINEL, 'Age'] = MAX_AGE
    lt = lt.apply(pd.to_numeric)
    lt.loc[lt['qx'] == 0, 'qx'] = np.nan

    lt_male = load_data(URL_LT_MALE)
    lt_male.loc[lt_male['Age'] == OPEN_AGE_SENTINEL, 'Age'] = MAX_AGE
    lt_male = lt_male.apply(pd.to_numeric)
    lt_male.loc[lt_male['qx'] == 0, 'qx'] = np.nan

    pop = load_data(URL_POP)
    pop.loc[pop['Age'] == OPEN_AGE_SENTINEL, 'Age'] = MAX_AGE
    pop = pop.apply(pd.to_numeric)

    tfr_history = load_tfr_history()

    return asfr, lt, lt_male, pop, tfr_history


@st.cache_data
def run_projection(tfr_start, tfr_change, ramp_speed, mab_stop, period,
                   _lt_female, _lt_male, n0_female, n0_male):
    if period == 0:
        d = pd.DataFrame({'tfr': [tfr_start], 'mab': [mab_stop], 'sd_mab': [SD_MAB_BASE]})
        return d, np.array(n0_female), np.array(n0_male)

    d         = build_scenario_df(tfr_start, tfr_change, ramp_speed, mab_stop, period)
    base      = build_leslie_base(_lt_female)
    F         = np.array([asfr_gamma(r.tfr, r.mab, r.sd_mab) for r in d.itertuples(index=False)])
    LL        = np.array([leslie(f, base) for f in F])
    subd_male = build_male_survival(_lt_male)
    l0_ratio  = _lt_male['Lx'].iloc[0] / _lt_female['Lx'].iloc[0]
    out_female, out_male = project_both_sexes(LL, subd_male, l0_ratio, n0_female, n0_male, period)
    return d, out_female, out_male


# --- Data ---
try:
    with st.spinner('Laadin andmeid...'):
        asfr, lt, lt_male, pop, tfr_history = load_and_clean()
except Exception as e:
    st.error(
        'Andmete laadimine ebaõnnestus. '
        'Palun kontrollige internetiühendust ja proovige lehte uuesti laadida.'
    )
    st.stop()

lt_base      = lt[lt['Year'] == BASE_YEAR]
lt_male_base = lt_male[lt_male['Year'] == BASE_YEAR]
pop_base     = pop[pop['Year'] == BASE_YEAR]
N0           = pop_base.Female[0:MAX_AGE].tolist()
N0_male      = pop_base.Male[0:MAX_AGE].tolist()
tfr_start    = compute_tfr_start(asfr, BASE_YEAR)

# --- Sidebar ---
st.sidebar.markdown(f'Prognoosi alusandmed on {BASE_YEAR}')
st.sidebar.markdown('''Allpool vali prognoosi eeldused:

1. sündimustaseme muutus (%) perioodi lõpuks 2023. a suhtes ja muutuse kiirus \
(kuvatakse ülemisel väiksel joonisel).
2. keskmine sünnitusvanus perioodi lõpus
3. prognoosi pikkus aastates
''')
st.sidebar.divider()

option_map = {5: "Aeglasem", 6: "Keskmine", 8: "Kiirem"}

def user_input_features():
    TFR_Change = st.sidebar.number_input(
        "Sündimuse muutus protsentides 2023.a suhtes",
        min_value=-50, max_value=70, step=10, value=0) / 100
    Ramp = st.sidebar.segmented_control(
        "Muutuse kiirus",
        options=option_map.keys(),
        format_func=lambda option: option_map[option],
        selection_mode='single', default=6)
    MAB_end = st.sidebar.number_input(
        "Keskmine sünnitusvanus", min_value=27.0, max_value=33.0, step=0.5, value=MAB_BASE)
    Years = st.sidebar.slider(
        "Prognoosi pikkus (aastat)", min_value=0, max_value=100, step=5, value=0)
    return TFR_Change, Ramp, MAB_end, Years


TFR_Change, Ramp, mab_stop, period = user_input_features()

# --- Projection ---
d, out, out_male = run_projection(
    tfr_start, TFR_Change, Ramp, mab_stop, period,
    _lt_female=lt_base, _lt_male=lt_male_base,
    n0_female=N0, n0_male=N0_male,
)

tfr_last     = round(d['tfr'].iloc[-1], 2)
p_size_start = sum(pop['Total'][pop['Year'] == BASE_YEAR])
p_size_end   = sum(out) + sum(out_male)
total_births = round(out[0] * (1 + SEX_RATIO_AT_BIRTH))

# --- TFR chart + metrics ---
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
    ax.plot(actual_x, actual_y, linewidth=2.5, color='#2196F3', label='Tegelik')

    # Scenario line: starts at last known value and continues into the projection
    if period > last_known_idx + 1:
        proj_years = list(range(BASE_YEAR + 1, BASE_YEAR + 1 + period))
        scenario_x = [last_known_year] + proj_years[last_known_idx + 1:]
        scenario_y = [tfr_history[last_known_year]] + d['tfr'].iloc[last_known_idx + 1:].tolist()
        ax.plot(scenario_x, scenario_y, linewidth=2.5, linestyle='--', color='#FF5722', label='Stsenaarium')

    ax.set(ylabel='Last naise kohta', xlabel='Aasta')
    ax.legend()
    st.caption(f"Sündimuse muutus {BASE_YEAR} - {BASE_YEAR + period}")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)
    col_a.metric(f"TFR {BASE_YEAR + period}",
                 tfr_last, round(tfr_last - tfr_start, 1), border=False)
    col_b.metric(f"Sünnitusvanus {BASE_YEAR + period}",
                 round(mab_stop, 1), round(mab_stop - MAB_BASE, 1), border=False)
    col_c.metric(f"Rahvaarv (milj.)  {BASE_YEAR + period}",
                 round(p_size_end / 1_000_000, 3), round(p_size_end - p_size_start), border=False)
    col_d.metric(f"Sündide arv {BASE_YEAR + period}",
                 total_births, "", border=False)

# --- Structural indicators ---
ind_end   = compute_indicators(out, out_male)
ind_start = compute_indicators(N0, N0_male)

st.divider()
i1, i2, i3, i4 = st.columns(4)
i1.metric(
    f"Vanussõltuvusmäär {BASE_YEAR + period}",
    f"{ind_end['oadr']:.1f}%",
    f"{ind_end['oadr'] - ind_start['oadr']:.1f}pp",
    border=False,
)
i2.metric(
    f"Tööealised (18–64)  {BASE_YEAR + period}",
    f"{round(ind_end['working_age']):,}",
    round(ind_end['working_age'] - ind_start['working_age']),
    border=False,
)
i3.metric(
    f"Kooliealised (7–17)  {BASE_YEAR + period}",
    f"{round(ind_end['school_age']):,}",
    round(ind_end['school_age'] - ind_start['school_age']),
    border=False,
)
i4.metric(
    f"65+  {BASE_YEAR + period}",
    f"{round(ind_end['old_age']):,}",
    round(ind_end['old_age'] - ind_start['old_age']),
    border=False,
)

# --- Population pyramid ---
st.divider()

ages = np.arange(MAX_AGE)
data_mask = ages >= period   # born before projection — from data (survived)
proj_mask = ages <  period   # born during projection — from model

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1)
fig2, ax2 = plt.subplots(figsize=(10, 8))

ax2.barh(ages[data_mask],  out[data_mask],           color='#e05252', label='Naised (andmed)')
ax2.barh(ages[proj_mask],  out[proj_mask],            color='#f5b7b1', label='Naised (prognoos)')
ax2.barh(ages[data_mask], -out_male[data_mask],       color='#2e86c1', label='Mehed (andmed)')
ax2.barh(ages[proj_mask], -out_male[proj_mask],       color='#a9cce3', label='Mehed (prognoos)')

ax2.set_xlabel('Inimesi vanusrühmas')
ax2.set_ylabel('Vanus')
ax2.set_yticks(np.arange(0, MAX_AGE - 10, AGE_TICK_STEP))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(int(x)):,}'))
ax2.axvline(0, color='black', linewidth=0.8)
ax2.legend(loc='upper left')

st.caption(f"Rahvastikupüramiid {BASE_YEAR + period} aastal  |  Tumedam = andmepõhised kohordid, heledam = prognoositud kohordid")
st.pyplot(fig2)
plt.close(fig2)
