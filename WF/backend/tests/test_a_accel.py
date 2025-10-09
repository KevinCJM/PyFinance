from __future__ import annotations
import numpy as np
import pandas as pd

import os, sys
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from services.analysis_service import find_a_points
from opt.a_core import compute_a_core_single_code


def make_df(n=300, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='B')
    price = np.cumsum(rng.normal(0, 1, size=n)).astype(float) + 100
    high = price + rng.uniform(0, 1, size=n)
    low = price - rng.uniform(0, 1, size=n)
    close = price + rng.normal(0, 0.2, size=n)
    openp = close + rng.normal(0, 0.2, size=n)
    vol = rng.integers(1000, 100000, size=n).astype(float)
    df = pd.DataFrame({
        'code': '000001',
        'date': dates,
        'open': openp,
        'high': high,
        'low': low,
        'close': close,
        'volume': vol,
    })
    return df


def compare(a: pd.Series, b: pd.Series, name: str):
    if a.dtype == bool or b.dtype == bool:
        assert (a.fillna(False).astype(bool).values == b.fillna(False).astype(bool).values).all(), f"Mismatch in {name}"
    else:
        diff = (a.astype(float) - b.astype(float)).abs()
        # 允许少量 NaN & 数值微小差值
        ok = ((a.isna() & b.isna()) | (diff.fillna(0) < 1e-8))
        assert ok.all(), f"Mismatch in {name} with max diff {diff.max()}"


def run_case(n=300, seed=42):
    df = make_df(n=n, seed=seed)
    # 参数：覆盖多种窗口
    c1 = { 'enabled': True, 'long_window': 60, 'down_lookback': 30 }
    c2 = { 'enabled': True, 'short_windows': (5, 10), 'long_window': 60, 'cross_window': 3, 'require_all': True }
    c3 = { 'enabled': True, 'confirm_lookback_days': 10, 'confirm_ma_window': 20, 'confirm_price_col': 'high' }

    out_ref = find_a_points(df, code_col='code', date_col='date', close_col='close', volume_col='volume',
                            with_explain_strings=False, cond1=c1, cond2=c2, cond3=c3)
    out_new = compute_a_core_single_code(df[['date','open','high','low','close','volume']].copy(), c1, c2, c3, vr1_lookback=10, eps=0.0)

    # 对齐
    out_ref = out_ref.sort_values('date').reset_index(drop=True)
    out_new = out_new.sort_values('date').reset_index(drop=True)

    # 关键条件一致性
    compare(out_ref['cond1_ok'], out_new['cond1_ok'], 'cond1_ok')
    compare(out_ref['cond2_ok'], out_new['cond2_ok'], 'cond2_ok')
    compare(out_ref['cond3_ok'], out_new['cond3_ok'], 'cond3_ok')
    compare(out_ref['A_point'], out_new['A_point'], 'A_point')

    # 均线值一致性（抽查）
    for k in (5, 10, 20, 60):
        if f'ma_{k}' in out_ref.columns and f'ma_{k}' in out_new.columns:
            compare(out_ref[f'ma_{k}'], out_new[f'ma_{k}'], f'ma_{k}')


if __name__ == '__main__':
    # 多组不同种子/长度覆盖
    for seed in (0, 1, 42, 2024):
        run_case(n=280, seed=seed)
        run_case(n=365, seed=seed)
    print('A-accel tests: OK')
