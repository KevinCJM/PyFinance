from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .numba_accel import (
    rolling_mean,
    rolling_bool_sum,
    rolling_max_prev,
    cross_up_bool,
    above_bool,
)


def compute_a_core_single_code(df: pd.DataFrame, c1: Dict[str, Any], c2: Dict[str, Any], c3: Dict[str, Any],
                               vr1_lookback: int = 10, eps: float = 0.0) -> pd.DataFrame:
    """
    加速版 A 点核心逻辑（单只股票）。
    输入 df 必须包含列：date, open, high, low, close, volume。
    参数 c1/c2/c3 与 app.compute_a_points_v2 内部构造的结构一致。
    返回与 services.analysis_service.find_a_points 兼容的关键列：
      - ma_{k}, ma_long_t1, ma_long_t1_prev
      - today_any_cross, recent_all_cross, today_all_above, cond1_ok, cond2_ok, cond3_ok
      - A_point, vr1
    """
    d = df.copy()
    d = d.sort_values('date').reset_index(drop=True)

    n = len(d)
    close = d['close'].to_numpy(dtype=np.float64)
    high = d['high'].to_numpy(dtype=np.float64)
    low = d['low'].to_numpy(dtype=np.float64)
    volume = d['volume'].to_numpy(dtype=np.float64)

    # ---- 均线集合 ----
    shorts: Tuple[int, ...] = tuple(c2.get('short_windows') or (5, 10))
    long1 = int(c1.get('long_window', 60))
    long2 = int(c2.get('long_window', long1))
    confirm_ma = c3.get('confirm_ma_window', None)
    needed = set(shorts) | {long1, long2}
    if confirm_ma:
        needed.add(int(confirm_ma))

    ma_map: Dict[int, np.ndarray] = {}
    for k in sorted(needed):
        if k and k > 0:
            ma_map[k] = rolling_mean(close, int(k))
        else:
            ma_map[k] = np.full(n, np.nan, dtype=np.float64)
        d[f'ma_{k}'] = ma_map[k]

    # ---- 条件1：长均线下跌（t-1 与 t-1-L 比较）----
    c1_enabled = bool(c1.get('enabled', True))
    lookback = int(c1.get('down_lookback', 30))
    maL = ma_map[long1]
    maL_t1 = np.empty(n, dtype=np.float64);
    maL_t1[:] = np.nan
    maL_t1_prev = np.empty(n, dtype=np.float64);
    maL_t1_prev[:] = np.nan
    for i in range(n):
        if i - 1 >= 0:
            maL_t1[i] = maL[i - 1]
        if i - 1 - lookback >= 0:
            maL_t1_prev[i] = maL[i - 1 - lookback]
    cond1_ok = ((~np.isnan(maL_t1)) & (~np.isnan(maL_t1_prev)) & ((maL_t1 + eps) < maL_t1_prev))

    d['ma_long_t1'] = maL_t1
    d['ma_long_t1_prev'] = maL_t1_prev
    d['cond1_ok'] = cond1_ok if c1_enabled else np.ones(n, dtype=bool)

    # ---- 条件2：短均线上穿（当日触发）----
    c2_enabled = bool(c2.get('enabled', True))
    cross_window = int(c2.get('cross_window', 3))
    require_all = bool(c2.get('require_all', True))
    required_shorts = tuple(c2.get('required_shorts') or shorts)

    ma_long2 = ma_map[long2]
    today_any_cross = np.zeros(n, dtype=np.uint8)
    recent_all = np.ones(n, dtype=np.uint8)
    all_above_today = np.ones(n, dtype=np.uint8)

    # 逐条短均线计算上方与上穿
    for k in required_shorts:
        ma_s = ma_map[int(k)]
        above = above_bool(ma_s, ma_long2, float(eps))  # uint8
        cross = cross_up_bool(ma_s, ma_long2, float(eps))  # uint8 基于均线与其前一日
        today_any_cross = np.maximum(today_any_cross, cross)
        if require_all:
            # 最近 cross_window 天至少一次上穿
            cnt = rolling_bool_sum(cross.astype(np.float64), max(1, int(cross_window)))
            rc = (cnt > 0).astype(np.uint8)
            recent_all = np.minimum(recent_all, rc)
            all_above_today = np.minimum(all_above_today, above)
        # 写出便于表格诊断的列
        d[f'above_{k}'] = above.astype(bool)
        d[f'crossup_{k}'] = cross.astype(bool)

    if require_all:
        cond2_core = (today_any_cross.astype(bool) & recent_all.astype(bool) & all_above_today.astype(bool))
    else:
        cond2_core = today_any_cross.astype(bool)
        recent_all = np.ones(n, dtype=np.uint8)
        all_above_today = np.ones(n, dtype=np.uint8)

    d['today_any_cross'] = today_any_cross.astype(bool)
    d['recent_all_cross'] = recent_all.astype(bool)
    d['today_all_above'] = all_above_today.astype(bool)
    d['cond2_ok'] = cond2_core if c2_enabled else np.ones(n, dtype=bool)

    # ---- 条件3：确认（A点前 N 日价格上穿 MA_confirm ≥ 1 次）----
    c3_enabled = bool(c3.get('enabled', True))
    look = int(c3.get('confirm_lookback_days', 0) or 0)
    ma_c = int(confirm_ma) if confirm_ma else None
    price_col = str(c3.get('confirm_price_col', 'high'))
    if price_col not in ('high', 'close', 'low'):
        price_col = 'high'
    price = {'high': high, 'close': close, 'low': low}[price_col]

    if c3_enabled and ma_c and look > 0:
        m = ma_map[ma_c]
        cross_p = cross_up_bool(price, m, float(eps)).astype(np.float64)
        # 统计 [t-look, t-1] 上穿次数：即 cross_p.shift(1).rolling(look).sum()
        # 先生成 shift(1)
        cross_shift1 = np.empty(n, dtype=np.float64)
        cross_shift1[0] = np.nan
        for i in range(1, n):
            cross_shift1[i] = cross_p[i - 1]
        cnt = rolling_bool_sum(cross_shift1, int(look))
        cond3_ok = (cnt >= 1)
        d['confirm_cross_cnt'] = cnt
        d['cond3_ok'] = cond3_ok
    else:
        d['confirm_cross_cnt'] = np.full(n, np.nan, dtype=np.float64)
        d['cond3_ok'] = np.ones(n, dtype=bool)

    # ---- VR1：volume / max(prev N) ----
    prev_max = rolling_max_prev(volume, int(max(1, int(vr1_lookback))))
    with np.errstate(divide='ignore', invalid='ignore'):
        vr1 = volume / prev_max
    d['vr1'] = vr1

    # ---- 综合 A 点（不含 cond4 组）----
    d['A_point'] = ((np.where(c1_enabled, d['cond1_ok'].to_numpy(), True)) &
                    (np.where(c2_enabled, d['cond2_ok'].to_numpy(), True)) &
                    (np.where(c3_enabled, d['cond3_ok'].to_numpy(), True))).astype(bool)

    return d
