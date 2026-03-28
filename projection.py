# v2 — baseline inflow by nationality
import numpy as np
import pandas as pd
import requests
import io
from scipy.special import gamma
from scipy.interpolate import PchipInterpolator

# Data URLs
URL_ASFR = 'https://raw.githubusercontent.com/klesment/PopProj/main/ESTasfrRR_2023.txt'
URL_LT        = 'https://raw.githubusercontent.com/klesment/PopProj/main/LT_Female_2024.txt'
URL_LT_MALE   = 'https://raw.githubusercontent.com/klesment/PopProj/main/LT_Male_2024.txt'
URL_POP     = 'https://raw.githubusercontent.com/klesment/PopProj/main/Population_2025.txt'
URL_TFRMAB  = 'https://raw.githubusercontent.com/klesment/PopProj/main/EST_TFRMAB_2023.txt'
URL_MT_STOCK       = 'https://raw.githubusercontent.com/klesment/PopProj/main/mt_stock_2021.csv'
URL_IMMIG_INFLOW   = 'https://raw.githubusercontent.com/klesment/PopProj/main/immig_inflow_dist.csv'
URL_EMIG_RATES     = 'https://raw.githubusercontent.com/klesment/PopProj/main/emig_rates.csv'
URL_IMMIG_BASELINE = 'https://raw.githubusercontent.com/klesment/PopProj/main/immig_baseline.csv'

# Demographic constants
BASE_YEAR           = 2023
MAB_BASE            = 30.62   # mean age at birth in base year
SD_MAB_BASE         = 5.62    # std dev of mean age at birth in base year
MAX_AGE             = 110     # open-interval upper age group
ASFR_FERTILE_MIN    = 12      # youngest fertile age in gamma model
ASFR_FERTILE_MAX    = 50      # oldest fertile age (exclusive) in gamma model
SEX_RATIO_AT_BIRTH  = 1.05    # male births per female birth
RADIX               = 200000  # life table radix (l0)
RAMP_SHAPE          = 5       # fixed shape parameter for Gompertz TFR ramp
OPEN_AGE_SENTINEL   = '110+'  # string value replaced during data cleaning

# TFR values not yet in the data file — override the ramp for those years
KNOWN_TFR     = {2024: 1.18, 2025: 1.18}
HISTORY_START = 2010   # first year shown on the TFR chart

# Age range boundaries (inclusive lower, exclusive upper)
AGE_SCHOOL_MIN =  7   # start of compulsory education
AGE_SCHOOL_MAX = 18   # end of compulsory education (ages 7–17)
AGE_WORK_MIN   = 18   # working-age lower bound
AGE_WORK_MAX   = 65   # standard working-age upper bound (ages 15–64)
AGE_OLD_MIN    = 65   # old-age lower bound

# Immigration constants — derived from RVR09 2019-2021 average, excl. Estonian returnees
IMMIG_FEMALE_SHARE = 0.3942   # share of annual inflow that is female


def load_data(url):
    s = requests.get(url).content
    return pd.read_csv(io.StringIO(s.decode('utf-8')), sep=r'\s+', header=1)


def load_tfr_history():
    """
    Load TFR from HISTORY_START onward from the data file,
    then merge with KNOWN_TFR for years not yet in the file.
    Returns a dict {year: TFR}.
    """
    s  = requests.get(URL_TFRMAB).content
    df = pd.read_csv(io.StringIO(s.decode('utf-8')), skipinitialspace=True)
    df.columns = df.columns.str.strip()
    history = {
        int(row['Year1']): float(row['TFR'])
        for _, row in df.iterrows()
        if int(row['Year1']) >= HISTORY_START
    }
    history.update(KNOWN_TFR)   # add/override with values not yet in the file
    return history


def compute_tfr_start(asfr_df, year):
    return round(sum(asfr_df['ASFR'][asfr_df['Year'] == year]), 2)


def ramp_fun(tfr_chng, speed, pr_per):
    x = np.linspace(0, 1, pr_per + 1)[1:]  # skip x=0 so period=1 applies the full change
    y = tfr_chng * np.exp(-RAMP_SHAPE * np.exp(-speed * x))
    return np.add(y, 1)


def build_scenario_df(tfr_start, tfr_change, ramp_speed, mab_stop, period):
    d = pd.DataFrame({
        'tfr':    np.multiply(np.full(period, tfr_start), ramp_fun(tfr_change, ramp_speed, period)),
        'mab':    np.linspace(MAB_BASE, mab_stop, period),
        'sd_mab': np.full(period, SD_MAB_BASE),
    })
    for year, tfr in KNOWN_TFR.items():
        idx = year - BASE_YEAR - 1  # row 0 = first projection year (BASE_YEAR + 1)
        if 0 <= idx < period:
            d.loc[idx, 'tfr'] = tfr
    return d


def asfr_gamma(tfr, mab, sd_mab):
    """Vectorised gamma model — no Python loop over ages."""
    a2   = mab / (sd_mab ** 2)
    a3   = (mab / sd_mab) ** 2
    ages = np.arange(ASFR_FERTILE_MIN, ASFR_FERTILE_MAX, dtype=float)
    return (1 / gamma(a3)) * tfr * (a2 ** a3) * (ages ** (a3 - 1)) * np.exp(-a2 * ages)


def build_leslie_base(lt):
    """
    Precompute the time-invariant structure of the Leslie matrix from a life table.
    Returns a dict of components reused across every projection year.
    """
    SUBD = lt.Lx.iloc[1:].values / lt.Lx.iloc[0:MAX_AGE].values
    SUBD[-1:] = 0
    k    = (1 / (1 + SEX_RATIO_AT_BIRTH)) * (lt.Lx.iloc[0] / RADIX)

    L_base = np.zeros((MAX_AGE, MAX_AGE))
    np.fill_diagonal(L_base[1:], SUBD[:MAX_AGE - 1])   # subdiagonal
    L_base[MAX_AGE - 1, MAX_AGE - 1] = (                # open age group
        lt['Tx'].values[MAX_AGE] / lt['Tx'].values[MAX_AGE - 1]
    )
    return {'SUBD': SUBD, 'k': k, 'L_base': L_base}


def leslie(fert, base):
    """
    Build one Leslie matrix from fertility rates and precomputed base components.
    fert - ASFR array (length ASFR_FERTILE_MAX - ASFR_FERTILE_MIN)
    base - dict returned by build_leslie_base()
    """
    Fx       = np.zeros(MAX_AGE + 1)
    Fx[ASFR_FERTILE_MIN:ASFR_FERTILE_MAX] = fert
    R1       = base['k'] * (Fx[:MAX_AGE] + Fx[1:MAX_AGE + 1] * base['SUBD'])
    L        = base['L_base'].copy()
    L[0]     = R1
    return L


def project_dyn(lmat, pop, per):
    N_0 = pop
    for i in range(per):
        N_0 = np.dot(lmat[i], N_0)
    return N_0


def build_male_survival(lt_male):
    """Male survival vector from life table, including open age group correction."""
    SUBD_m = lt_male.Lx.iloc[1:].values / lt_male.Lx.iloc[0:MAX_AGE].values
    SUBD_m[-1] = lt_male['Tx'].values[MAX_AGE] / lt_male['Tx'].values[MAX_AGE - 1]
    return SUBD_m


def build_immig_vectors(annual_total, dist_female, dist_male, per):
    """
    Build per-year immigration vectors from a constant annual total and
    pre-computed single-year age distributions.

    annual_total - scalar: total immigrants per year (both sexes)
    dist_female  - normalised age distribution for females (length MAX_AGE)
    dist_male    - normalised age distribution for males  (length MAX_AGE)
    per          - number of projection years

    Returns (imm_f, imm_m): arrays of shape (per, MAX_AGE).
    """
    female_annual = annual_total * IMMIG_FEMALE_SHARE
    male_annual   = annual_total * (1 - IMMIG_FEMALE_SHARE)
    imm_f = np.outer(np.full(per, female_annual), dist_female)
    imm_m = np.outer(np.full(per, male_annual),   dist_male)
    return imm_f, imm_m


def project_both_sexes(lmat_female, subd_male, l0_ratio,
                       pop_native_f, pop_native_m,
                       pop_immig_f,  pop_immig_m,
                       nat_inflow_f, nat_inflow_m,
                       imm_inflow_f, imm_inflow_m,
                       emig_nat_f, emig_nat_m,
                       emig_imm_f, emig_imm_m,
                       per):
    """
    Project native and immigrant populations simultaneously.

    lmat_female  - array of yearly female Leslie matrices (per, MAX_AGE, MAX_AGE)
    subd_male    - male survival vector (length MAX_AGE)
    l0_ratio     - L0_male / L0_female: first-year survival ratio for male births
    pop_native_f - initial native female vector
    pop_native_m - initial native male vector
    pop_immig_f  - initial immigrant female vector
    pop_immig_m  - initial immigrant male vector
    nat_inflow_f - annual inflow vectors, native female  (per, MAX_AGE)
    nat_inflow_m - annual inflow vectors, native male    (per, MAX_AGE)
    imm_inflow_f - annual inflow vectors, immigrant female (per, MAX_AGE)
    imm_inflow_m - annual inflow vectors, immigrant male   (per, MAX_AGE)
    emig_nat_f   - annual emigration rates, native female  (length MAX_AGE)
    emig_nat_m   - annual emigration rates, native male    (length MAX_AGE)
    emig_imm_f   - annual emigration rates, immigrant female (length MAX_AGE)
    emig_imm_m   - annual emigration rates, immigrant male   (length MAX_AGE)
    per          - number of years to project

    Order each year: survival/births → emigration → immigration inflow.
    Children of immigrant mothers join the immigrant stock.
    Returns (native_f, native_m, immig_f, immig_m).
    """
    N_nat_f = np.array(pop_native_f, dtype=float)
    N_nat_m = np.array(pop_native_m, dtype=float)
    N_imm_f = np.array(pop_immig_f,  dtype=float)
    N_imm_m = np.array(pop_immig_m,  dtype=float)
    male_birth_factor = SEX_RATIO_AT_BIRTH * l0_ratio

    for i in range(per):
        # --- survival + births ---
        N_nat_f_new  = np.dot(lmat_female[i], N_nat_f)
        leslie_imm_f = np.dot(lmat_female[i], N_imm_f)

        N_nat_m_new       = np.zeros(MAX_AGE)
        N_nat_m_new[1:]   = N_nat_m[:-1] * subd_male[:-1]
        N_nat_m_new[-1]  += N_nat_m[-1]  * subd_male[-1]
        N_nat_m_new[0]    = N_nat_f_new[0] * male_birth_factor

        N_imm_m_new       = np.zeros(MAX_AGE)
        N_imm_m_new[1:]   = N_imm_m[:-1] * subd_male[:-1]
        N_imm_m_new[-1]  += N_imm_m[-1]  * subd_male[-1]
        N_imm_m_new[0]    = leslie_imm_f[0] * male_birth_factor

        # --- emigration (applied as rates to post-survival stock) ---
        N_nat_f_new  *= (1 - emig_nat_f)
        N_nat_m_new  *= (1 - emig_nat_m)
        N_imm_f_new   = leslie_imm_f * (1 - emig_imm_f)
        N_imm_m_new  *= (1 - emig_imm_m)

        # --- immigration inflow: baseline (both stocks) + scenario (immigrant only) ---
        N_nat_f_new  += nat_inflow_f[i]
        N_nat_m_new  += nat_inflow_m[i]
        N_imm_f_new  += imm_inflow_f[i]
        N_imm_m_new  += imm_inflow_m[i]

        N_nat_f, N_nat_m = N_nat_f_new, N_nat_m_new
        N_imm_f, N_imm_m = N_imm_f_new, N_imm_m_new

    return N_nat_f, N_nat_m, N_imm_f, N_imm_m


def _disaggregate_5yr(counts_5yr, lx):
    """
    Disaggregate 18 five-year age group counts (0-4 … 80-84, 85+) to
    MAX_AGE single-year values.

    Closed groups (0–84): PCHIP interpolation on the cumulative distribution.
    Open group  (85+)   : life-table Lx column used as proportional weights.

    counts_5yr : array-like, length 18
    lx         : Lx column of the relevant life table (length >= MAX_AGE)
    """
    counts_5yr = np.asarray(counts_5yr, dtype=float)
    closed     = counts_5yr[:17]          # groups 0-4 … 80-84
    open_total = counts_5yr[17]           # 85+

    # PCHIP on cumulative counts at 5-year boundaries 0, 5, …, 85
    boundaries = np.arange(0, 90, 5, dtype=float)          # length 18
    cumul      = np.concatenate([[0.0], np.cumsum(closed)]) # length 18
    single_cumul = PchipInterpolator(boundaries, cumul)(np.arange(86, dtype=float))
    single_closed = np.maximum(np.diff(single_cumul), 0)   # length 85, ages 0–84

    # Life-table tail for ages 85 … MAX_AGE-1
    lx_tail = np.asarray(lx, dtype=float)[85:MAX_AGE]
    lx_sum  = lx_tail.sum()
    single_open = (open_total * lx_tail / lx_sum) if lx_sum > 0 else np.zeros(MAX_AGE - 85)

    return np.concatenate([single_closed, single_open])   # length MAX_AGE


def load_mt_stock(lt_female, lt_male):
    """
    Load the 2021 census population by mother tongue and disaggregate from
    5-year groups to single years.

    Returns (estonian_female, estonian_male, other_female, other_male),
    each of length MAX_AGE.
    """
    df = pd.read_csv(io.StringIO(requests.get(URL_MT_STOCK).content.decode()))
    est_f = _disaggregate_5yr(df['estonian_female'].values, lt_female['Lx'].values)
    est_m = _disaggregate_5yr(df['estonian_male'].values,   lt_male['Lx'].values)
    oth_f = _disaggregate_5yr(df['other_female'].values,    lt_female['Lx'].values)
    oth_m = _disaggregate_5yr(df['other_male'].values,      lt_male['Lx'].values)
    return est_f, est_m, oth_f, oth_m


def load_immig_inflow_dist(lt_female, lt_male):
    """
    Load the normalised inflow age distribution (2019-2021 average, foreign-born
    arrivals excluding Estonian returnees) and disaggregate to single years.

    Returns (female_dist, male_dist), each summing to 1.0, length MAX_AGE.
    """
    df = pd.read_csv(io.StringIO(requests.get(URL_IMMIG_INFLOW).content.decode()))
    female = _disaggregate_5yr(df['female_share'].values, lt_female['Lx'].values)
    male   = _disaggregate_5yr(df['male_share'].values,   lt_male['Lx'].values)
    return female / female.sum(), male / male.sum()


def load_emig_rates():
    """
    Load annual emigration rates by 5-year age group and sex (2019-2021 average).
    Expand to single-year vectors using a step function (each age takes its
    5-year group's rate; the open 85+ group extends to MAX_AGE).

    Returns (native_female_rate, native_male_rate, immig_female_rate, immig_male_rate),
    each of length MAX_AGE.
    """
    df = pd.read_csv(io.StringIO(requests.get(URL_EMIG_RATES).content.decode()))

    def expand(rates_5yr):
        closed = np.repeat(rates_5yr[:17], 5)          # ages 0–84
        open_  = np.full(MAX_AGE - 85, rates_5yr[17])  # ages 85–109
        return np.concatenate([closed, open_])

    return (
        expand(df['estonian_female_rate'].values),
        expand(df['estonian_male_rate'].values),
        expand(df['other_female_rate'].values),
        expand(df['other_male_rate'].values),
    )


def load_immig_baseline(lt_female, lt_male):
    """
    Load baseline annual inflow counts by 5-year age group (2019-2021 average)
    and disaggregate to single years using life-table weights for the open group.

    Returns (nat_f, nat_m, imm_f, imm_m): annual inflow vectors for the native
    (Estonian-citizen) and immigrant (non-Estonian-citizen) stocks, each length MAX_AGE.
    """
    df = pd.read_csv(io.StringIO(requests.get(URL_IMMIG_BASELINE).content.decode()))
    nat_f = _disaggregate_5yr(df['estonian_female'].values, lt_female['Lx'].values)
    nat_m = _disaggregate_5yr(df['estonian_male'].values,   lt_male['Lx'].values)
    imm_f = _disaggregate_5yr(df['other_female'].values,    lt_female['Lx'].values)
    imm_m = _disaggregate_5yr(df['other_male'].values,      lt_male['Lx'].values)
    return nat_f, nat_m, imm_f, imm_m


def compute_indicators(out_female, out_male):
    """
    Calculate structural demographic indicators from projected age vectors.
    Returns a dict with counts and the old-age dependency ratio.
    """
    total = np.array(out_female) + np.array(out_male)

    school_age  = total[AGE_SCHOOL_MIN:AGE_SCHOOL_MAX].sum()
    working_age = total[AGE_WORK_MIN:AGE_WORK_MAX].sum()
    old_age     = total[AGE_OLD_MIN:].sum()
    oadr        = old_age / working_age * 100 if working_age > 0 else np.nan

    return {
        'school_age':  school_age,
        'working_age': working_age,
        'old_age':     old_age,
        'oadr':        oadr,
    }
