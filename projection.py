import numpy as np
import pandas as pd
import requests
import io
from scipy.special import gamma

# Data URLs
URL_ASFR = 'https://raw.githubusercontent.com/klesment/PopProj/main/ESTasfrRR_2023.txt'
URL_LT        = 'https://raw.githubusercontent.com/klesment/PopProj/main/LT_Female_2024.txt'
URL_LT_MALE   = 'https://raw.githubusercontent.com/klesment/PopProj/main/LT_Male_2024.txt'
URL_POP     = 'https://raw.githubusercontent.com/klesment/PopProj/main/Population_2025.txt'
URL_TFRMAB  = 'https://raw.githubusercontent.com/klesment/PopProj/main/EST_TFRMAB_2023.txt'

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


def project_both_sexes(lmat_female, subd_male, l0_ratio, pop_female, pop_male, per):
    """
    Project female and male populations simultaneously.

    lmat_female - list of yearly female Leslie matrices
    subd_male   - male survival vector (length MAX_AGE)
    l0_ratio    - L0_male / L0_female: first-year survival ratio used to scale male births
    pop_female  - initial female population vector
    pop_male    - initial male population vector
    per         - number of years to project
    """
    N_f = np.array(pop_female, dtype=float)
    N_m = np.array(pop_male,   dtype=float)

    male_birth_factor = SEX_RATIO_AT_BIRTH * l0_ratio

    for i in range(per):
        N_f_new = np.dot(lmat_female[i], N_f)

        N_m_new = np.zeros(MAX_AGE)
        N_m_new[1:]  = N_m[:-1] * subd_male[:-1]   # survival of each cohort
        N_m_new[-1] += N_m[-1]  * subd_male[-1]    # open age group
        N_m_new[0]   = N_f_new[0] * male_birth_factor  # new male births

        N_f = N_f_new
        N_m = N_m_new

    return N_f, N_m


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
