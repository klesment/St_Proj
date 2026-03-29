"""
One-time script to fetch immigration data from stat.ee and save as CSV files
for use in the population projection model.

Outputs:
  mt_stock_2021.csv     -- population by mother tongue (Estonian / Other),
                           sex and 5-year age group, 2021 census (RL21434)
  immig_inflow_dist.csv -- normalised inflow age distribution (shares summing
                           to 1.0) averaged over 2019-2021, by 5-year age
                           group and sex, excl. Estonian-born returnees (RVR09)
  emig_rates.csv        -- annual emigration rates by 5-year age group and sex,
                           averaged over 2019-2021 (RVR08 + RV071 denominator):
                           native stock: Estonian-citizen emigrants / native pop
                           immigrant stock: non-Estonian emigrants / immig pop
  immig_baseline.csv    -- baseline annual inflow counts by 5-year age group and sex,
                           averaged over 2019-2021 (RVR03 total − RVR10 Estonian):
                           estonian: Estonian-citizen immigrants (returnees)
                           other: non-Estonian-citizen immigrants

Run once, commit all CSVs to the data repo.
"""

import requests
import numpy as np
import pandas as pd

STAT_EE = 'https://andmed.stat.ee/api/v1/et/stat/'

AGE_LABELS = [
    '0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39',
    '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79',
    '80-84', '85+',
]


def post(table, query):
    r = requests.post(
        STAT_EE + table,
        json={'query': query, 'response': {'format': 'json-stat2'}},
    )
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 1. Mother-tongue stock -- RL21434 (2021 Census)
#    Population by mother tongue (Estonian / Other), sex and 5-year age group.
#    Dimensions: Year(1) x AgeGroup(18) x Place(1) x Sex(2) x MotherTongue(2)
#    Sex: 2=Males, 3=Females  |  MotherTongue: 1=Total, 2=Estonian
# ---------------------------------------------------------------------------
print('Fetching RL21434 (mother-tongue stock, 2021 census)...')
mt_raw = requests.post(
    'https://andmed.stat.ee/api/v1/en/stat/RL21434',
    json={'query': [
        {'code': 'Aasta',      'selection': {'filter': 'item', 'values': ['2021']}},
        {'code': 'Vanuserühm', 'selection': {'filter': 'item', 'values': [
            '5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22'
        ]}},
        {'code': 'Elukoht',    'selection': {'filter': 'item', 'values': ['1']}},
        {'code': 'Sugu',       'selection': {'filter': 'item', 'values': ['2','3']}},
        {'code': 'Emakeel',    'selection': {'filter': 'item', 'values': ['1','2']}},
    ], 'response': {'format': 'json-stat2'}},
).json()

# After squeezing year and place dims: shape (18, 2, 2) = age x sex x tongue
# tongue index 0 = total, 1 = Estonian  →  other = total - Estonian
mt_arr = np.array(mt_raw['value']).reshape(18, 2, 2)

male_est   = mt_arr[:, 0, 1]
male_other = mt_arr[:, 0, 0] - male_est
fem_est    = mt_arr[:, 1, 1]
fem_other  = mt_arr[:, 1, 0] - fem_est

mt_df = pd.DataFrame({
    'age_group':       AGE_LABELS,
    'estonian_male':   male_est,
    'other_male':      male_other,
    'estonian_female': fem_est,
    'other_female':    fem_other,
})
mt_df.to_csv('mt_stock_2021.csv', index=False)
print(f'  Saved mt_stock_2021.csv')
print(f'  Estonian: male={int(male_est.sum()):,}  female={int(fem_est.sum()):,}')
print(f'  Other MT: male={int(male_other.sum()):,}  female={int(fem_other.sum()):,}')


# ---------------------------------------------------------------------------
# 2. Inflow age distribution -- RVR09
#    Immigrants 2017-2019, excluding Estonian-born (country code 1),
#    averaged across years, normalised to proportions summing to 1.0
#    per sex.
#    Same period as emigration rates (RVR03/RVR10) to avoid COVID-year distortion.
#    Dimensions: Year(3) x AgeGroup(18) x Sex(2) x Country(9)
# ---------------------------------------------------------------------------
print('Fetching RVR09 (inflow 2017-2019)...')
inflow_raw = post('RVR09', [
    {'code': 'Aasta',      'selection': {'filter': 'item', 'values': ['2017','2018','2019']}},
    {'code': 'Vanuserühm', 'selection': {'filter': 'item', 'values': [
        '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'
    ]}},
    {'code': 'Sugu',       'selection': {'filter': 'item', 'values': ['2','3']}},
])

# shape: 3(year) x 18(age) x 2(sex) x 9(country)
inflow_arr = np.array(inflow_raw['value']).reshape(3, 18, 2, 9)

# exclude country index 0 (Estonian-born returnees), sum remaining countries
foreign_arr = inflow_arr[:, :, :, 1:].sum(axis=3)  # shape: 3(year) x 18(age) x 2(sex)

# average over years, then normalise each sex to sum to 1.0
avg = foreign_arr.mean(axis=0)                       # shape: 18(age) x 2(sex)
dist_male   = avg[:, 0] / avg[:, 0].sum()
dist_female = avg[:, 1] / avg[:, 1].sum()

inflow_df = pd.DataFrame({
    'age_group':    AGE_LABELS,
    'male_share':   dist_male,
    'female_share': dist_female,
})
inflow_df.to_csv('immig_inflow_dist.csv', index=False)

avg_annual_male   = int(avg[:, 0].sum())
avg_annual_female = int(avg[:, 1].sum())
print(f'  Saved immig_inflow_dist.csv  '
      f'(avg annual: {avg_annual_male + avg_annual_female:,}  '
      f'male: {avg_annual_male:,}  female: {avg_annual_female:,})')


# ---------------------------------------------------------------------------
# 3. Emigration rates -- RVR03 (total) + RVR10 (Estonians) + RL21434 denominator
#    Non-Estonian mother tongue = total − Estonian mother tongue (RVR10).
#    Average over 2017-2019; denominator from 2021 census (mt_arr from Section 1).
#    RVR03 shape: Year(3) x Sex(2) x AgeGroup(18) x Indicator(2) x MigrationType(1)
#    RVR10 shape: Year(3) x AgeGroup(18) x Indicator(2) x Sex(2)
#    Indicator: index 0=Immigration, 1=Emigration  |  Sex: index 0=Males, 1=Females
# ---------------------------------------------------------------------------
print('Fetching RVR03 (total migration 2017-2019)...')
rvr03_raw = requests.post(
    'https://andmed.stat.ee/api/v1/en/stat/RVR03',
    json={'query': [
        {'code': 'Aasta',      'selection': {'filter': 'item', 'values': ['2017','2018','2019']}},
        {'code': 'Sugu',       'selection': {'filter': 'item', 'values': ['2','3']}},
        {'code': 'Vanuserühm', 'selection': {'filter': 'item', 'values': [
            '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']}},
        {'code': 'Näitaja',    'selection': {'filter': 'item', 'values': ['1','2']}},
        {'code': 'Rände liik', 'selection': {'filter': 'item', 'values': ['2']}},
    ], 'response': {'format': 'json-stat2'}},
).json()

print('Fetching RVR10 (Estonian migration 2017-2019)...')
rvr10_raw = requests.post(
    'https://andmed.stat.ee/api/v1/en/stat/RVR10',
    json={'query': [
        {'code': 'Aasta',      'selection': {'filter': 'item', 'values': ['2017','2018','2019']}},
        {'code': 'Vanuserühm', 'selection': {'filter': 'item', 'values': [
            '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']}},
        {'code': 'Näitaja',    'selection': {'filter': 'item', 'values': ['1','2']}},
        {'code': 'Sugu',       'selection': {'filter': 'item', 'values': ['2','3']}},
    ], 'response': {'format': 'json-stat2'}},
).json()

# RVR03: reshape to [3(year), 2(sex), 18(age), 2(indicator)] — migration type dim squeezed
total_arr = np.array(rvr03_raw['value']).reshape(3, 2, 18, 2)

# RVR10: reshape to [3(year), 18(age), 2(indicator), 2(sex)]
# transpose to [3(year), 2(sex), 18(age), 2(indicator)] for consistent indexing
est_arr = np.array(rvr10_raw['value']).reshape(3, 18, 2, 2).transpose(0, 3, 1, 2)

# Emigration = indicator index 1; average over years → shape: 2(sex) x 18(age)
total_emig_avg = total_arr[:, :, :, 1].mean(axis=0)
est_emig_avg   = est_arr[:, :, :, 1].mean(axis=0)
oth_emig_avg   = total_emig_avg - est_emig_avg

# Denominators from 2021 census: mt_arr shape [18(age), 2(sex), 2(tongue)]
# tongue index 0=total, 1=estonian → transpose to [2(sex), 18(age)]
est_pop = mt_arr[:, :, 1].T
oth_pop = (mt_arr[:, :, 0] - mt_arr[:, :, 1]).T

def safe_rate(emig, pop):
    return np.where(pop > 0, np.clip(emig / pop, 0, 1), 0.0)

# sex index 0=male, 1=female
rates_df = pd.DataFrame({
    'age_group':            AGE_LABELS,
    'estonian_male_rate':   safe_rate(est_emig_avg[0], est_pop[0]),
    'estonian_female_rate': safe_rate(est_emig_avg[1], est_pop[1]),
    'other_male_rate':      safe_rate(oth_emig_avg[0], oth_pop[0]),
    'other_female_rate':    safe_rate(oth_emig_avg[1], oth_pop[1]),
})
rates_df.to_csv('emig_rates.csv', index=False)

print(f'  Saved emig_rates.csv')
print(f'  Avg annual Estonian emigrants: male={int(est_emig_avg[0].sum()):,}  female={int(est_emig_avg[1].sum()):,}')
print(f'  Avg annual other-MT emigrants: male={int(oth_emig_avg[0].sum()):,}  female={int(oth_emig_avg[1].sum()):,}')
print(f'  Peak Estonian emig rate (male): age {AGE_LABELS[est_emig_avg[0].argmax()]}  '
      f'= {safe_rate(est_emig_avg[0], est_pop[0]).max():.4f}')
print(f'  Peak other-MT emig rate (male): age {AGE_LABELS[oth_emig_avg[0].argmax()]}  '
      f'= {safe_rate(oth_emig_avg[0], oth_pop[0]).max():.4f}')


# ---------------------------------------------------------------------------
# 4. Baseline inflow counts -- RVR03 (total) + RVR10 (Estonians)
#    Same source arrays as Section 3; immigration indicator index 0.
#    Non-Estonian inflow = total − Estonian (same subtraction as emigration).
#    Absolute counts (not rates), averaged over 2017-2019.
# ---------------------------------------------------------------------------
print('Deriving baseline inflow counts from RVR03 / RVR10...')

# immigration indicator index 0; shape: 2(sex) x 18(age)
total_immig_avg = total_arr[:, :, :, 0].mean(axis=0)
est_immig_avg   = est_arr[:, :, :, 0].mean(axis=0)
oth_immig_avg   = total_immig_avg - est_immig_avg

baseline_df = pd.DataFrame({
    'age_group':        AGE_LABELS,
    'estonian_male':    est_immig_avg[0].round(1),
    'estonian_female':  est_immig_avg[1].round(1),
    'other_male':       oth_immig_avg[0].round(1),
    'other_female':     oth_immig_avg[1].round(1),
})
baseline_df.to_csv('immig_baseline.csv', index=False)

print(f'  Saved immig_baseline.csv')
print(f'  Avg annual Estonian immigrants:     male={int(est_immig_avg[0].sum()):,}  female={int(est_immig_avg[1].sum()):,}')
print(f'  Avg annual non-Estonian immigrants: male={int(oth_immig_avg[0].sum()):,}  female={int(oth_immig_avg[1].sum()):,}')

print('\nDone. Commit all CSV files to the data repo.')
