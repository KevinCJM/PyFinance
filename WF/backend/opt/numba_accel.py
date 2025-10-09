from __future__ import annotations
import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    def njit(signature_or_function=None, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        if callable(signature_or_function):
            return signature_or_function
        return wrapper


    def prange(*args, **kwargs):  # type: ignore
        return range(*args)


    HAS_NUMBA = False


@njit(cache=True)
def _rolling_mean_nb(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if window <= 1:
        for i in prange(n):
            out[i] = arr[i]
        return out
    csum = 0.0
    valid_count = 0
    # initialize first window progressively
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            # if nan appears, mark nan when window is complete
            pass
        else:
            csum += v
            valid_count += 1
        if i >= window:
            old = arr[i - window]
            if not np.isnan(old):
                csum -= old
                valid_count -= 1
            else:
                # recompute for stability if old was nan
                csum = 0.0
                valid_count = 0
                for k in range(i - window + 1, i + 1):
                    vv = arr[k]
                    if not np.isnan(vv):
                        csum += vv
                        valid_count += 1
        if i >= window - 1:
            if valid_count == window:
                out[i] = csum / window
            else:
                out[i] = np.nan
    return out


@njit(cache=True)
def _rolling_sum_nb_bool(arr01: np.ndarray, window: int) -> np.ndarray:
    n = arr01.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if window <= 0:
        return out
    s = 0.0
    for i in range(n):
        v = arr01[i]
        if not np.isnan(v):
            s += v
        if i >= window:
            prev = arr01[i - window]
            if not np.isnan(prev):
                s -= prev
            else:
                s = 0.0
                for k in range(i - window + 1, i + 1):
                    vv = arr01[k]
                    if not np.isnan(vv):
                        s += vv
        if i >= window - 1:
            out[i] = s
    return out


@njit(cache=True)
def _rolling_max_prev_nb(arr: np.ndarray, window: int) -> np.ndarray:
    """返回 prev-window 的滚动最大值：位置 i 输出 max(arr[i-window:i])。
    当 i=0 时无前值，返回 nan；i<window 时取已有的 [0:i) 区间最大。
    """
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    if window <= 0:
        return out
    for i in range(n):
        start = i - window
        if start < 0:
            start = 0
        if i == 0:
            out[i] = np.nan
            continue
        m = arr[start]
        for k in range(start + 1, i):
            if arr[k] > m:
                m = arr[k]
        out[i] = m
    return out


@njit(cache=True)
def _cross_up_bool(price: np.ndarray, ma: np.ndarray, eps: float) -> np.ndarray:
    n = price.shape[0]
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        p = price[i]
        m = ma[i]
        pm1 = price[i - 1] if i - 1 >= 0 else np.nan
        mm1 = ma[i - 1] if i - 1 >= 0 else np.nan
        if (not np.isnan(p)) and (not np.isnan(m)) and (not np.isnan(pm1)) and (not np.isnan(mm1)):
            if (p + eps) >= m and (pm1 + eps) < mm1:
                out[i] = 1
    return out


@njit(cache=True)
def _above_bool(ma_short: np.ndarray, ma_long: np.ndarray, eps: float) -> np.ndarray:
    n = ma_short.shape[0]
    out = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        a = ma_short[i]
        b = ma_long[i]
        if (not np.isnan(a)) and (not np.isnan(b)):
            if (a + eps) > b:
                out[i] = 1
    return out


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return _rolling_mean_nb(arr.astype(np.float64), int(window))


def rolling_bool_sum(arr01: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum_nb_bool(arr01.astype(np.float64), int(window))


def rolling_max_prev(arr: np.ndarray, window: int) -> np.ndarray:
    return _rolling_max_prev_nb(arr.astype(np.float64), int(window))


def cross_up_bool(price: np.ndarray, ma: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return _cross_up_bool(price.astype(np.float64), ma.astype(np.float64), float(eps))


def above_bool(ma_short: np.ndarray, ma_long: np.ndarray, eps: float = 0.0) -> np.ndarray:
    return _above_bool(ma_short.astype(np.float64), ma_long.astype(np.float64), float(eps))


@njit(cache=True)
def _true_range_nb(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    n = high.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        h = high[i]
        l = low[i]
        pc = prev_close[i]
        if np.isnan(h) or np.isnan(l):
            out[i] = np.nan
            continue
        v1 = h - l
        if np.isnan(pc):
            out[i] = v1
        else:
            v2 = abs(h - pc)
            v3 = abs(l - pc)
            m = v1
            if v2 > m:
                m = v2
            if v3 > m:
                m = v3
            out[i] = m
    return out


def true_range(high: np.ndarray, low: np.ndarray, prev_close: np.ndarray) -> np.ndarray:
    return _true_range_nb(high.astype(np.float64), low.astype(np.float64), prev_close.astype(np.float64))


def atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n = high.shape[0]
    pc = np.empty(n, dtype=np.float64)
    pc.fill(np.nan)
    for i in range(1, n):
        pc[i] = close[i - 1]
    tr = true_range(high, low, pc)
    return rolling_mean(tr, int(window))
