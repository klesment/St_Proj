"""
One-time script to fetch labour market data from Statistics Estonia for the
productivity page.

Outputs:
  emp_rates_2025.csv        -- employment rate (%) by sex and 5-year age group
                               15-19 … 70-74, 75+ (TT330, labour force survey)
  productivity_growth.csv   -- annual real change (%) of labour productivity
                               per employed person (RAL0013, national accounts)

Usage: python fetch_labour_data.py [year]   (default 2025)
Commit both CSVs to the data repo (klesment/PopProj).
"""

import sys

import numpy as np
import pandas as pd
import requests

STAT_EE = 'https://andmed.stat.ee/api/v1/en/stat/'

AGE_GROUPS = ['Y15-19', 'Y20-24', 'Y25-29', 'Y30-34', 'Y35-39', 'Y40-44', 'Y45-49',
              'Y50-54', 'Y55-59', 'Y60-64', 'Y65-69', 'Y70-74', 'Y_GE75']
AGE_LABELS = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
              '50-54', '55-59', '60-64', '65-69', '70-74', '75+']


def post(table, query):
    r = requests.post(STAT_EE + table,
                      json={'query': query, 'response': {'format': 'json-stat2'}},
                      timeout=60)
    r.raise_for_status()
    return r.json()


def to_array(d):
    """json-stat2 response → ndarray in the response's own dimension order, with category codes."""
    codes = []
    for dim in d['id']:
        idx = d['dimension'][dim]['category']['index']
        codes.append(sorted(idx, key=idx.get))
    values = np.array([np.nan if v is None else v for v in d['value']], dtype=float)
    return values.reshape(d['size']), d['id'], codes


year = sys.argv[1] if len(sys.argv) > 1 else '2025'

# ---------------------------------------------------------------------------
# 1. Employment rates -- TT330 (labour force survey)
# ---------------------------------------------------------------------------
print(f'Fetching TT330 (employment rate by sex and age group, {year})...')
arr, dims, codes = to_array(post('TT330', [
    {'code': 'Näitaja',        'selection': {'filter': 'item', 'values': ['EMPRATE']}},
    {'code': 'Sugu',           'selection': {'filter': 'item', 'values': ['M', 'F']}},
    {'code': 'Vanuserühm',     'selection': {'filter': 'item', 'values': AGE_GROUPS}},
    {'code': 'Vaatlusperiood', 'selection': {'filter': 'item', 'values': [year]}},
]))
# reorder to (sex, age) using the category codes, whatever order the API returns
arr = arr.transpose([dims.index(d) for d in ('Sugu', 'Vanuserühm', 'Näitaja', 'Vaatlusperiood')])[:, :, 0, 0]
sex_codes, age_codes = codes[dims.index('Sugu')], codes[dims.index('Vanuserühm')]
rate = {s: [arr[sex_codes.index(s), age_codes.index(g)] for g in AGE_GROUPS] for s in ('M', 'F')}

emp = pd.DataFrame({'age_group': AGE_LABELS, 'male_rate': rate['M'], 'female_rate': rate['F']})
assert emp[['male_rate', 'female_rate']].notna().all().all(), 'missing employment rates'
emp.to_csv(f'emp_rates_{year}.csv', index=False)
print(f'  Saved emp_rates_{year}.csv')
print(emp.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Labour productivity growth -- RAL0013 (national accounts)
#    Productivity per employed person, 1st-4th quarters (annual),
#    real change compared with previous year, %.
# ---------------------------------------------------------------------------
print('Fetching RAL0013 (real labour productivity growth per employed person)...')
arr, dims, codes = to_array(post('RAL0013', [
    {'code': 'Suhtarv', 'selection': {'filter': 'item', 'values': ['1']}},  # per employed person
    {'code': 'Kvartal', 'selection': {'filter': 'item', 'values': ['1']}},  # 1st-4th quarters
    {'code': 'Näitaja', 'selection': {'filter': 'item', 'values': ['6']}},  # real change vs. same period of previous year
]))
years = codes[dims.index('Aasta')]
prod = pd.DataFrame({'year': [int(y) for y in years], 'growth_pct': arr.reshape(-1)}).dropna()
prod.to_csv('productivity_growth.csv', index=False)
print(f'  Saved productivity_growth.csv ({prod.year.min()}–{prod.year.max()})')
