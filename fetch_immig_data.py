"""
One-time script to fetch immigration data from stat.ee and save as CSV files
for use in the population projection model.

Outputs:
  immig_stock_2025.csv  -- foreign-origin stock (1st + 2nd generation) by
                           5-year age group and sex, January 2025 (RV071)
  immig_inflow_dist.csv -- normalised inflow age distribution (shares summing
                           to 1.0) averaged over 2019-2021, by 5-year age
                           group and sex, excl. Estonian-born returnees (RVR09)
  emig_rates.csv        -- annual emigration rates by 5-year age group and sex,
                           averaged over 2019-2021 (RVR08 + RV071 denominator):
                           native stock: Estonian-citizen emigrants / native pop
                           immigrant stock: non-Estonian emigrants / immig pop

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
# 1. Initial immigrant stock -- RV071
#    Foreign-origin population 1 January 2025, 1st + 2nd generation,
#    by 5-year age group and sex.
#    Dimensions: Year(1) x County(1) x Põlisus(5) x Sex(2) x AgeGroup(18)
# ---------------------------------------------------------------------------
print('Fetching RV071 (stock)...')
stock_raw = post('RV071', [
    {'code': 'Aasta',    'selection': {'filter': 'item', 'values': ['2025']}},
    {'code': 'Maakond',  'selection': {'filter': 'item', 'values': ['1']}},
    {'code': 'Põlisus',  'selection': {'filter': 'item', 'values': ['3','4','5','6']}},
    {'code': 'Sugu',     'selection': {'filter': 'item', 'values': ['2','3']}},
    {'code': 'Vanuserühm', 'selection': {'filter': 'item', 'values': [
        '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'
    ]}},
])

# shape: 1(year) x 1(county) x 4(põlisus: total,1st,2nd,3rd) x 2(sex) x 18(age)
stock_arr = np.array(stock_raw['value']).reshape(1, 1, 4, 2, 18)

# indices: 0=foreign-origin total, 1=1st gen, 2=2nd gen, 3=3rd gen
# sum 1st (idx 1) + 2nd generation (idx 2)
stock_12 = stock_arr[0, 0, 1, :, :] + stock_arr[0, 0, 2, :, :]  # shape: sex(2) x age(18)

stock_df = pd.DataFrame({
    'age_group': AGE_LABELS,
    'male':   stock_12[0],   # sex index 0 = Males
    'female': stock_12[1],   # sex index 1 = Females
})
stock_df.to_csv('immig_stock_2025.csv', index=False)
print(f'  Saved immig_stock_2025.csv  '
      f'(total: {int(stock_12.sum()):,}  '
      f'male: {int(stock_12[0].sum()):,}  '
      f'female: {int(stock_12[1].sum()):,})')


# ---------------------------------------------------------------------------
# 2. Inflow age distribution -- RVR09
#    Immigrants 2019-2021, excluding Estonian-born (country code 1),
#    averaged across years, normalised to proportions summing to 1.0
#    per sex.
#    Dimensions: Year(3) x AgeGroup(18) x Sex(2) x Country(9)
# ---------------------------------------------------------------------------
print('Fetching RVR09 (inflow 2019-2021)...')
inflow_raw = post('RVR09', [
    {'code': 'Aasta',      'selection': {'filter': 'item', 'values': ['2019','2020','2021']}},
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
# 3. Emigration rates -- RVR08 + RV071
#    Emigration 2019-2021 averaged, split by Estonian / non-Estonian citizen.
#    Rates = avg annual emigrants / avg annual population (RV071 2019-2021).
#    Dimensions RVR08: Year(3) x Indicator(2) x AgeGroup(18) x Sex(2) x Citizenship(10)
#    Indicator index 1 = Emigration; citizenship index 0 = Estonian citizen.
# ---------------------------------------------------------------------------
print('Fetching RVR08 (emigration 2019-2021)...')
# Note: RVR08 uses the /en/ endpoint
emig_raw = requests.post(
    'https://andmed.stat.ee/api/v1/en/stat/RVR08',
    json={'query': [
        {'code': 'Aasta',        'selection': {'filter': 'item', 'values': ['2019','2020','2021']}},
        {'code': 'Vanuserühm',   'selection': {'filter': 'item', 'values': [
            '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']}},
        {'code': 'Sugu',         'selection': {'filter': 'item', 'values': ['2','3']}},
        {'code': 'Kodakondsus',  'selection': {'filter': 'item', 'values': [
            '1','6','7','8','9','2','10','11','3','4']}},
    ], 'response': {'format': 'json-stat2'}},
).json()

# shape: 3(year) x 2(indicator) x 18(age) x 2(sex) x 10(citizenship)
emig_arr = np.array(emig_raw['value']).reshape(3, 2, 18, 2, 10)

# Emigration = indicator index 1; average over years; shape: 18(age) x 2(sex) x 10(citizenship)
emig_avg = emig_arr[:, 1, :, :, :].mean(axis=0)

# Estonian-citizen emigrants → native stock (citizenship index 0)
emig_nat_male   = emig_avg[:, 0, 0]   # age x male, Estonian citizen
emig_nat_female = emig_avg[:, 1, 0]   # age x female, Estonian citizen

# Non-Estonian emigrants → immigrant stock (citizenship indices 1-9)
emig_imm_male   = emig_avg[:, 0, 1:].sum(axis=1)
emig_imm_female = emig_avg[:, 1, 1:].sum(axis=1)

print('Fetching RV071 (population denominator 2019-2021)...')
denom_raw = post('RV071', [
    {'code': 'Aasta',    'selection': {'filter': 'item', 'values': ['2019','2020','2021']}},
    {'code': 'Maakond',  'selection': {'filter': 'item', 'values': ['1']}},
    {'code': 'Põlisus',  'selection': {'filter': 'item', 'values': ['2','4','5']}},
    {'code': 'Sugu',     'selection': {'filter': 'item', 'values': ['2','3']}},
    {'code': 'Vanuserühm', 'selection': {'filter': 'item', 'values': [
        '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19'
    ]}},
])

# shape: 3(year) x 1(county) x 3(põlisus: native,1st,2nd) x 2(sex) x 18(age)
denom_arr = np.array(denom_raw['value']).reshape(3, 1, 3, 2, 18)

# Average over years; põlisus indices: 0=native, 1=1st gen, 2=2nd gen
denom_avg = denom_arr[:, 0, :, :, :].mean(axis=0)   # shape: 3(põlisus) x 2(sex) x 18(age)

pop_nat_male   = denom_avg[0, 0, :]   # native male
pop_nat_female = denom_avg[0, 1, :]   # native female
pop_imm_male   = denom_avg[1, 0, :] + denom_avg[2, 0, :]   # 1st+2nd gen male
pop_imm_female = denom_avg[1, 1, :] + denom_avg[2, 1, :]   # 1st+2nd gen female

# Compute rates; clip to [0, 1] to guard against zeros in tiny age groups
def safe_rate(emig, pop):
    return np.where(pop > 0, np.clip(emig / pop, 0, 1), 0.0)

rates_df = pd.DataFrame({
    'age_group':          AGE_LABELS,
    'native_male_rate':   safe_rate(emig_nat_male,   pop_nat_male),
    'native_female_rate': safe_rate(emig_nat_female, pop_nat_female),
    'immig_male_rate':    safe_rate(emig_imm_male,   pop_imm_male),
    'immig_female_rate':  safe_rate(emig_imm_female, pop_imm_female),
})
rates_df.to_csv('emig_rates.csv', index=False)

print(f'  Saved emig_rates.csv')
print(f'  Avg annual native  emigrants: male={int(emig_nat_male.sum()):,}  female={int(emig_nat_female.sum()):,}')
print(f'  Avg annual immig   emigrants: male={int(emig_imm_male.sum()):,}  female={int(emig_imm_female.sum()):,}')
print(f'  Peak native  emig rate (male):  age {AGE_LABELS[emig_nat_male.argmax()]}  '
      f'= {safe_rate(emig_nat_male, pop_nat_male).max():.4f}')
print(f'  Peak immig   emig rate (male):  age {AGE_LABELS[emig_imm_male.argmax()]}  '
      f'= {safe_rate(emig_imm_male, pop_imm_male).max():.4f}')

print('\nDone. Commit all CSV files to the data repo.')
