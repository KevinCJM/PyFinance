from __future__ import annotations
import numpy as np
import pandas as pd

from opt.numba_accel import true_range, atr_sma


def make_df(n=200, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    price = np.cumsum(rng.normal(0, 1, size=n)).astype(float) + 50
    high = price + rng.uniform(0, 1, size=n)
    low = price - rng.uniform(0, 1, size=n)
    close = price + rng.normal(0, 0.2, size=n)
    return pd.DataFrame({'date': dates, 'high': high, 'low': low, 'close': close})


def atr_pandas(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        (df['high']-df['low']).abs(),
        (df['high']-prev_close).abs(),
        (df['low']-prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def run_case(n=200, seed=7, window=14):
    df = make_df(n=n, seed=seed)
    hi = df['high'].to_numpy(dtype=float)
    lo = df['low'].to_numpy(dtype=float)
    cl = df['close'].to_numpy(dtype=float)
    # numba ATR
    atr_nb = atr_sma(hi, lo, cl, window)
    # pandas ATR
    atr_pd = atr_pandas(df, window).to_numpy(dtype=float)
    # compare with tolerance
    # allow NaN in head until window-1
    mask = ~np.isnan(atr_pd)
    diff = np.abs(atr_nb[mask] - atr_pd[mask])
    assert (diff < 1e-8).all(), f"ATR mismatch max diff={diff.max()}"


if __name__ == '__main__':
    for seed in (0, 1, 7, 42):
        run_case(n=256, seed=seed, window=14)
        run_case(n=512, seed=seed, window=20)
    print('ATR numba tests: OK')
