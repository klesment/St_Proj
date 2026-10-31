# © 2026 Martin Klesment. Licensed under CC BY-NC 4.0.
"""
Streamlit-cached data loading and scenario runs shared by the app pages
(model pages in Estonian and English, productivity page).
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from projection import (
    URL_LT, URL_LT_MALE, URL_POP,
    BASE_YEAR, LT_YEAR, SD_MAB_BASE, MAX_AGE, OPEN_AGE_SENTINEL,
    load_data, load_tfr_history, build_scenario_df,
    asfr_gamma, build_leslie_base, leslie, build_male_survival, improve_lt,
    load_mt_stock, load_immig_inflow_dist, load_emig_rates, load_immig_baseline,
    build_immig_vectors, project_both_sexes,
)


@st.cache_data
def load_and_clean():
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

    return lt, lt_male, pop, tfr_history


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
        return d, np.array(n0_nat_f), np.array(n0_nat_m), np.array(n0_imm_f), np.array(n0_imm_m), e0_base, e0m_base, {}, None, None

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

    nat_f, nat_m, out_imm_f, out_imm_m, snapshots, yearly, by_age = project_both_sexes(
        np.array(LL), subd_m_arr, l0_r_arr,
        n0_nat_f, n0_nat_m, n0_imm_f, n0_imm_m,
        nat_inflow_f, nat_inflow_m,
        imm_inflow_f, imm_inflow_m,
        np.array(emig_nat_f) * emigration_scale, np.array(emig_nat_m) * emigration_scale,
        np.array(emig_imm_f) * emigration_scale, np.array(emig_imm_m) * emigration_scale,
        period,
        snapshot_years=(2050, 2075, 2100),
    )
    return d, nat_f, nat_m, out_imm_f, out_imm_m, e0_end, e0m_end, snapshots, yearly, by_age


def load_base():
    """
    Load and prepare all base-year inputs. Returns a SimpleNamespace with the
    life tables, population, TFR history, the base population split by mother
    tongue (2021 census proportions) and the migration inputs.
    """
    lt, lt_male, pop, tfr_history = load_and_clean()
    lt_base      = lt[lt['Year'] == LT_YEAR]
    lt_male_base = lt_male[lt_male['Year'] == LT_YEAR]
    pop_base     = pop[pop['Year'] == BASE_YEAR]
    N0           = pop_base.Female[0:MAX_AGE].tolist()
    N0_male      = pop_base.Male[0:MAX_AGE].tolist()

    (census_est_f, census_est_m, census_oth_f, census_oth_m,
     inflow_dist_f, inflow_dist_m,
     emig_nat_f, emig_nat_m, emig_imm_f, emig_imm_m,
     base_nat_f, base_nat_m, base_imm_f, base_imm_m) = load_immig_data(
        _lt_female=lt_base, _lt_male=lt_male_base
    )

    # Split base-year population using 2021 census mother-tongue proportions
    census_total_f = census_est_f + census_oth_f
    census_total_m = census_est_m + census_oth_m
    oth_share_f = np.where(census_total_f > 0, census_oth_f / census_total_f, 0.0)
    oth_share_m = np.where(census_total_m > 0, census_oth_m / census_total_m, 0.0)
    N0_imm_f = np.array(N0)      * oth_share_f
    N0_imm_m = np.array(N0_male) * oth_share_m

    return SimpleNamespace(
        lt=lt, lt_male=lt_male, pop=pop, tfr_history=tfr_history,
        lt_base=lt_base, lt_male_base=lt_male_base,
        e0_base=lt_base['Tx'].iloc[0] / lt_base['lx'].iloc[0],
        e0m_base=lt_male_base['Tx'].iloc[0] / lt_male_base['lx'].iloc[0],
        N0=N0, N0_male=N0_male, tfr_start=tfr_history[BASE_YEAR],
        N0_nat_f=np.array(N0) - N0_imm_f, N0_nat_m=np.array(N0_male) - N0_imm_m,
        N0_imm_f=N0_imm_f, N0_imm_m=N0_imm_m,
        inflow_dist_f=inflow_dist_f, inflow_dist_m=inflow_dist_m,
        emig_nat_f=emig_nat_f, emig_nat_m=emig_nat_m, emig_imm_f=emig_imm_f, emig_imm_m=emig_imm_m,
        base_nat_f=base_nat_f, base_nat_m=base_nat_m, base_imm_f=base_imm_f, base_imm_m=base_imm_m,
    )


def run_scenario(b, sc):
    """
    Run run_projection for scenario dict sc (keys: tfr_change, ramp, mab_stop, period,
    extra_immig, mort_improvement, baseline_inflow, emigration_scale, osc_amp, osc_cycle)
    on base inputs b from load_base().
    """
    return run_projection(
        b.tfr_start, sc['tfr_change'], sc['ramp'], sc['mab_stop'], sc['period'], sc['extra_immig'],
        sc['mort_improvement'], sc['baseline_inflow'], sc['emigration_scale'],
        _lt_female=b.lt_base, _lt_male=b.lt_male_base,
        n0_nat_f=b.N0_nat_f.tolist(), n0_nat_m=b.N0_nat_m.tolist(),
        n0_imm_f=b.N0_imm_f.tolist(), n0_imm_m=b.N0_imm_m.tolist(),
        inflow_dist_f=b.inflow_dist_f.tolist(), inflow_dist_m=b.inflow_dist_m.tolist(),
        emig_nat_f=b.emig_nat_f.tolist(), emig_nat_m=b.emig_nat_m.tolist(),
        emig_imm_f=b.emig_imm_f.tolist(), emig_imm_m=b.emig_imm_m.tolist(),
        base_nat_f=b.base_nat_f.tolist(), base_nat_m=b.base_nat_m.tolist(),
        base_imm_f=b.base_imm_f.tolist(), base_imm_m=b.base_imm_m.tolist(),
        osc_amp=sc['osc_amp'], osc_cycle=sc['osc_cycle'],
    )
