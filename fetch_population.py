"""
One-time script to fetch the 1 January population by sex and single year of age
from Statistics Estonia (RV0212) and save it in the HMD Population.txt layout
used by the projection (ages 0 … 109, 110+).

Statistics Estonia publishes ages 0–99 and 100+. The 100+ group is split over
ages 100 … 110+ in proportion to the life-table person-years (Lx, Tx for 110+)
of the life table year LT_YEAR.

Usage: python fetch_population.py 2025   →  Population_2025_RV0212.txt
Commit the output to the data repo (klesment/PopProj).
"""

import io
import sys

import numpy as np
import pandas as pd
import requests

from projection import URL_LT, URL_LT_MALE, LT_YEAR, MAX_AGE, OPEN_AGE_SENTINEL

RV0212 = 'https://andmed.stat.ee/api/v1/en/stat/RV0212'
OPEN_STAT = 100   # Statistics Estonia open age group (100+)


def fetch_rv0212(year):
    """Return (male, female) arrays for ages 0 … 99, 100+ on 1 January of year."""
    ages = [str(a) for a in range(OPEN_STAT)] + [f'{OPEN_STAT}+']
    r = requests.post(RV0212, json={'query': [
        {'code': 'Aasta',   'selection': {'filter': 'item', 'values': [str(year)]}},
        {'code': 'Vanus',   'selection': {'filter': 'item', 'values': ages}},
        {'code': 'Näitaja', 'selection': {'filter': 'item', 'values': ['1']}},     # Population, 1 January
        {'code': 'Sugu',    'selection': {'filter': 'item', 'values': ['2', '3']}}, # Males, Females
    ], 'response': {'format': 'json-stat2'}}, timeout=60)
    r.raise_for_status()
    d = r.json()
    assert d['id'] == ['Aasta', 'Vanus', 'Näitaja', 'Sugu'], d['id']
    arr = np.array(d['value'], dtype=float).reshape(len(ages), 2)
    return arr[:, 0], arr[:, 1]


def open_age_weights(url):
    """Life-table weights for ages 100 … 110+ (Lx for closed ages, Tx for the open group)."""
    s  = requests.get(url, timeout=60)
    s.raise_for_status()
    lt = pd.read_csv(io.StringIO(s.text), sep=r'\s+', header=1)
    lt['Age'] = lt['Age'].replace(OPEN_AGE_SENTINEL, str(MAX_AGE))
    lt = lt.apply(pd.to_numeric)
    lt = lt[lt['Year'] == LT_YEAR].set_index('Age')
    w  = np.append(lt.loc[OPEN_STAT:MAX_AGE - 1, 'Lx'].values, lt.loc[MAX_AGE, 'Tx'])
    return w / w.sum()


def expand(counts, weights):
    """Ages 0 … 99, 100+  →  ages 0 … 109, 110+."""
    return np.concatenate([counts[:OPEN_STAT], counts[OPEN_STAT] * weights])


if __name__ == '__main__':
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    male, female = fetch_rv0212(year)
    female = expand(female, open_age_weights(URL_LT))
    male   = expand(male,   open_age_weights(URL_LT_MALE))

    out = f'Population_{year}_RV0212.txt'
    with open(out, 'w') as f:
        f.write(f'Estonia, Population size (1-year), 1 January\tSource: Statistics Estonia RV0212; '
                f'100+ split by life table {LT_YEAR}\n\n')
        f.write(f"  {'Year':<14}{'Age':<16}{'Female':>6}{'Male':>16}{'Total':>16}\n")
        for age in range(MAX_AGE + 1):
            label = OPEN_AGE_SENTINEL if age == MAX_AGE else str(age)
            f.write(f'  {year:<14}{label:<12}{female[age]:>14.2f}{male[age]:>16.2f}'
                    f'{female[age] + male[age]:>16.2f}\n')
    print(f'Saved {out}: total {female.sum() + male.sum():,.0f}  '
          f'(female {female.sum():,.0f}, male {male.sum():,.0f})')
