"""
generate_data.py
----------------
Genera un dataset realista de riesgo crediticio y lo guarda como CSV.
En un proyecto real, aquí cargarías tus datos desde una base de datos o archivo.
"""

import pandas as pd
import numpy as np


def generate_credit_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Genera n registros de clientes con variables financieras.
    Target: 1 = buen pagador, 0 = mal pagador
    """
    np.random.seed(seed)

    age              = np.random.randint(18, 75, n)
    duration         = np.random.choice([6, 12, 18, 24, 36, 48, 60], n)
    amount           = np.random.randint(500, 15000, n)
    savings          = np.random.choice(
        ['none', 'little', 'moderate', 'quite_rich', 'rich'], n,
        p=[0.20, 0.30, 0.25, 0.15, 0.10])
    employment       = np.random.choice(
        ['unemployed', '<1yr', '1-4yr', '4-7yr', '>7yr'], n,
        p=[0.05, 0.15, 0.35, 0.25, 0.20])
    credit_history   = np.random.choice(
        ['no_credits', 'paid', 'existing_paid', 'delayed', 'critical'], n,
        p=[0.04, 0.05, 0.53, 0.09, 0.29])
    purpose          = np.random.choice(
        ['car', 'furniture', 'education', 'business', 'repairs', 'other'], n)
    housing          = np.random.choice(
        ['own', 'free', 'rent'], n, p=[0.55, 0.10, 0.35])
    existing_credits = np.random.choice([1, 2, 3, 4], n, p=[0.40, 0.35, 0.15, 0.10])

    risk = (
        (age > 30) * 0.3
        + (duration < 24) * 0.4
        + (amount < 5000) * 0.3
        + np.isin(savings, ['moderate', 'quite_rich', 'rich']) * 0.5
        + np.isin(employment, ['4-7yr', '>7yr']) * 0.4
        + np.isin(credit_history, ['paid', 'existing_paid']) * 0.5
        + (housing == 'own') * 0.3
        + np.random.normal(0, 0.3, n)
    )
    target = (risk > risk.mean()).astype(int)

    return pd.DataFrame({
        'age': age, 'duration': duration, 'amount': amount,
        'savings': savings, 'employment': employment,
        'credit_history': credit_history, 'purpose': purpose,
        'housing': housing, 'existing_credits': existing_credits,
        'target': target
    })


if __name__ == '__main__':
    df = generate_credit_data(n=1000)
    df.to_csv('data/credit_data.csv', index=False)
    print(f"✓ Dataset guardado: {df.shape[0]} registros")
    print(f"  Buenos pagadores: {df['target'].mean():.1%}")
    print(f"  Malos pagadores:  {1 - df['target'].mean():.1%}")
    print(df.head())