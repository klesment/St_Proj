"""Quick diagnostic: in/out migration rates by age, Estonian vs non-Estonian, male vs female."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

emig   = pd.read_csv('emig_rates.csv')
inflow = pd.read_csv('immig_baseline.csv')
pop    = pd.read_csv('mt_stock_2021.csv')

# inflow counts → rates using 2021 census population as denominator
inflow_rates = pd.DataFrame({'age_group': inflow['age_group']})
for group, sex in [('estonian', 'male'), ('estonian', 'female'),
                   ('other', 'male'),    ('other', 'female')]:
    pop_col    = f'{group}_{sex}'
    inflow_col = f'{group}_{sex}'
    denom = pop[pop_col].replace(0, np.nan)
    inflow_rates[f'{group}_{sex}_rate'] = inflow[inflow_col] / denom

ages = np.arange(len(emig))
labels = emig['age_group'].tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
fig.suptitle('Annual migration rates by age  (2019–2021 avg)', fontsize=13)

panels = [
    ('estonian', 'male',   'Estonian — Male',   axes[0, 0]),
    ('estonian', 'female', 'Estonian — Female', axes[0, 1]),
    ('other',    'male',   'Non-Estonian — Male',   axes[1, 0]),
    ('other',    'female', 'Non-Estonian — Female', axes[1, 1]),
]

for group, sex, title, ax in panels:
    col = f'{group}_{sex}'
    e_rate = emig[f'{col}_rate'].values
    i_rate = inflow_rates[f'{col}_rate'].values

    ax.plot(ages, e_rate * 100, color='#c0392b', marker='o', markersize=3, label='Emigration')
    ax.plot(ages, i_rate * 100, color='#2980b9', marker='s', markersize=3, label='Immigration')
    ax.set_title(title)
    ax.set_xticks(ages)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Rate (%)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('migration_rates_by_age.png', dpi=150)
print('Saved migration_rates_by_age.png')
