"""
Generate a realistic sample of the Adult Income dataset for demo purposes.
Run this once: python generate_sample_data.py
"""

import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

np.random.seed(42)
n = 1000

ages = np.random.randint(18, 70, n)
education_nums = np.random.randint(5, 16, n)
hours_per_week = np.random.randint(20, 60, n)

genders = np.random.choice(['Male', 'Female'], n, p=[0.67, 0.33])
races = np.random.choice(
    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
    n, p=[0.85, 0.095, 0.03, 0.01, 0.015]
)

marital_status = []
for g in genders:
    if g == 'Male':
        m = np.random.choice(
            ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed'],
            p=[0.55, 0.25, 0.12, 0.04, 0.04]
        )
    else:
        m = np.random.choice(
            ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed'],
            p=[0.40, 0.30, 0.18, 0.05, 0.07]
        )
    marital_status.append(m)

occupations = []
for g in genders:
    if g == 'Male':
        o = np.random.choice(
            ['Exec-managerial', 'Craft-repair', 'Prof-specialty', 'Sales',
             'Transport-moving', 'Handlers-cleaners', 'Tech-support', 'Other-service'],
            p=[0.17, 0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.10]
        )
    else:
        o = np.random.choice(
            ['Exec-managerial', 'Craft-repair', 'Prof-specialty', 'Sales',
             'Transport-moving', 'Handlers-cleaners', 'Tech-support', 'Other-service'],
            p=[0.12, 0.04, 0.18, 0.10, 0.02, 0.03, 0.12, 0.39]
        )
    occupations.append(o)

relationship = []
for g in genders:
    if g == 'Male':
        r = np.random.choice(
            ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'],
            p=[0.50, 0.25, 0.10, 0.06, 0.01, 0.08]
        )
    else:
        r = np.random.choice(
            ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'],
            p=[0.01, 0.25, 0.12, 0.25, 0.30, 0.07]
        )
    relationship.append(r)

capital_gain = np.where(np.random.rand(n) < 0.1, np.random.randint(1000, 50000, n), 0)
capital_loss = np.where(np.random.rand(n) < 0.05, np.random.randint(100, 4000, n), 0)

workclasses = np.random.choice(
    ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
     'Local-gov', 'State-gov', 'Without-pay'],
    n, p=[0.70, 0.08, 0.05, 0.04, 0.06, 0.05, 0.02]
)

income_prob = (
    0.10
    + 0.18 * (np.array(genders) == 'Male')
    + 0.05 * (np.array(races) == 'White')
    + 0.02 * (education_nums - 9)
    + 0.003 * hours_per_week
    + 0.001 * ages
)
income_prob = np.clip(income_prob, 0.02, 0.95)
income = np.where(np.random.rand(n) < income_prob, '>50K', '<=50K')

df = pd.DataFrame({
    'age': ages,
    'workclass': workclasses,
    'education_num': education_nums,
    'marital_status': marital_status,
    'occupation': occupations,
    'relationship': relationship,
    'race': races,
    'sex': genders,
    'capital_gain': capital_gain,
    'capital_loss': capital_loss,
    'hours_per_week': hours_per_week,
    'income': income
})

output_path = os.path.join('data', 'adult_income_sample.csv')
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
print(df.head())
