# -*- encoding: utf-8 -*-
"""
Tushare data fetch utilities.

Provides:
  - fetch_stock_daily: Fetch single stock daily kline via Tushare `pro.daily` and save as Parquet
  - fetch_many_daily: Fetch multiple stocks with a thread pool and basic rate limiting

Notes:
  - Uses token from config.TUSHARE_TOKEN
  - Saves Parquet files under backend/data with filename `{code}.parquet`
  - Parquet column names align with data_service.py and analysis expectations:
      日期(YYYY-MM-DD), 开盘, 收盘, 最高, 最低, 成交量
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:  # Lazy import to allow tests to mock
    import tushare as ts  # type: ignore
except Exception:  # pragma: no cover - tests mock get_pro
    ts = None  # type: ignore

import os

def _load_token() -> str:
    """Load Tushare token with multiple fallbacks:
    - top-level config.py:TUSHARE_TOKEN (repo根目录)
    - environment variable TUSHARE_TOKEN
    """
    # 1) 尝试从顶层 config.py 读取
    try:
        from config import TUSHARE_TOKEN  # type: ignore
        if TUSHARE_TOKEN:
            return str(TUSHARE_TOKEN)
    except Exception:
        pass
    # 2) 环境变量
    token = os.environ.get("TUSHARE_TOKEN", "")
    if token:
        return token
    raise RuntimeError("未找到 TUSHARE_TOKEN，请在 config.py 或环境变量中配置")


# Keep consistent with data_service.py
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


_PRO = None


def get_pro():
    """Return a cached Tushare Pro client with token set."""
    global _PRO
    if _PRO is not None:
        return _PRO
    if ts is None:
        raise RuntimeError("tushare is not available in this environment")
    token = _load_token()
    ts.set_token(token)
    _PRO = ts.pro_api()
    return _PRO


def _guess_ts_code(code: str) -> str:
    """Best-effort mapping from 6-digit code to Tushare ts_code with exchange suffix.

    Rules (common for A股):
      - 6xxxxxx -> .SH
      - 0xxxxxx / 3xxxxx -> .SZ
    """
    s = str(code).strip()
    if len(s) == 6 and s.isdigit():
        if s.startswith("6"):
            return f"{s}.SH"
        return f"{s}.SZ"  # 0/2/3 typically SZ
    # If already has suffix, return as-is
    if s.endswith(".SH") or s.endswith(".SZ"):
        return s
    return s


def _normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Tushare `daily` dataframe to our Chinese-named schema and sort by 日期 asc."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"])
    d = pd.DataFrame({
        "日期": pd.to_datetime(df.get("trade_date"), errors="coerce"),
        "开盘": pd.to_numeric(df.get("open"), errors="coerce"),
        "收盘": pd.to_numeric(df.get("close"), errors="coerce"),
        "最高": pd.to_numeric(df.get("high"), errors="coerce"),
        "最低": pd.to_numeric(df.get("low"), errors="coerce"),
        "成交量": pd.to_numeric(df.get("vol"), errors="coerce"),
    })
    d = d.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    d["日期"] = d["日期"].dt.strftime("%Y-%m-%d")
    return d


def fetch_stock_daily(
    code: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ts_code: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily kline for a single stock and save to Parquet under backend/data/{code}.parquet.

    Args:
      code: 6-digit stock code used for parquet filename (e.g., '300008')
      start_date: YYYYMMDD; defaults to none (Tushare returns recent period if omitted)
      end_date: YYYYMMDD
      ts_code: Optional explicit Tushare ts_code (e.g., '300008.SZ'). If not given, inferred.

    Returns: normalized dataframe with columns: 日期, 开盘, 收盘, 最高, 最低, 成交量
    """
    pro = get_pro()
    ts_c = ts_code or _guess_ts_code(code)
    params: Dict[str, Optional[str]] = {"ts_code": ts_c}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    raw = pro.daily(**params)
    d = _normalize_daily_df(raw)
    out_path = DATA_DIR / f"{code}.parquet"
    d.to_parquet(out_path, index=False)
    return d


@dataclass
class FetchResult:
    code: str
    ok: bool
    rows: int = 0
    error: Optional[str] = None


class RateLimiter:
    """Simple time-based rate limiter: allow up to `max_calls_per_second` across threads."""

    def __init__(self, max_calls_per_second: float):
        self.interval = 1.0 / max(float(max_calls_per_second), 0.1)
        self._last = 0.0

    def acquire(self):
        now = time.time()
        delta = now - self._last
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self._last = time.time()


def fetch_many_daily(
    codes: Iterable[str],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_workers: int = 4,
    max_calls_per_second: float = 8.0,
) -> Dict[str, FetchResult]:
    """Batch fetch daily kline for multiple stocks with basic rate limiting.

    Returns dict of code -> FetchResult.
    """
    limiter = RateLimiter(max_calls_per_second)

    results: Dict[str, FetchResult] = {}

    def _task(c: str) -> FetchResult:
        try:
            limiter.acquire()
            df = fetch_stock_daily(c, start_date=start_date, end_date=end_date)
            return FetchResult(code=c, ok=True, rows=int(len(df)))
        except Exception as e:  # pragma: no cover - error path covered in separate test
            return FetchResult(code=c, ok=False, error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_task, str(c)): str(c) for c in codes}
        for fut in as_completed(future_map):
            res = fut.result()
            results[res.code] = res
    return results


__all__ = [
    "fetch_stock_daily",
    "fetch_many_daily",
    "get_pro",
]
