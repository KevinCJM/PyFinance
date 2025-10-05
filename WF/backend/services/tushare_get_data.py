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
from typing import Dict, Iterable, List, Optional, Tuple, Callable, Any

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
META_DIR = DATA_DIR / ".meta"
META_DIR.mkdir(parents=True, exist_ok=True)
META_INDEX = META_DIR / "meta_index.json"


_PRO = None
_TRADE_CAL_CACHE: Optional[pd.DataFrame] = None
_TRADE_CAL_LAST_FETCH: Optional[str] = None  # YYYYMMDD


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


def _today_yyyymmdd() -> str:
    return pd.to_datetime("today").strftime("%Y%m%d")


def get_trade_calendar(pro=None) -> pd.DataFrame:
    """Get and cache trade calendar from Tushare (SSE)."""
    global _TRADE_CAL_CACHE, _TRADE_CAL_LAST_FETCH
    today = _today_yyyymmdd()
    if _TRADE_CAL_CACHE is not None and _TRADE_CAL_LAST_FETCH == today:
        return _TRADE_CAL_CACHE
    pro = pro or get_pro()
    df = pro.trade_cal(exchange='SSE', start_date='20100101', end_date=today)
    # Expect columns: cal_date, is_open
    _TRADE_CAL_CACHE = df
    _TRADE_CAL_LAST_FETCH = today
    return df


def get_last_trading_day(pro=None) -> str:
    """Return last trading day as YYYY-MM-DD (based on SSE calendar)."""
    cal = get_trade_calendar(pro)
    d = cal[cal['is_open'] == 1]['cal_date'].max()
    if not isinstance(d, str):
        d = str(d)
    # to YYYY-MM-DD
    return pd.to_datetime(d).strftime('%Y-%m-%d')


def _read_parquet_last_date_fast(code: str) -> Optional[str]:
    p = DATA_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=['日期'])
        if df.empty:
            return None
        return str(df['日期'].iloc[-1])
    except Exception:
        return None


def load_meta_index() -> Dict[str, Dict[str, Any]]:
    """Load meta index using json to avoid pandas datetime inference warnings."""
    if not META_INDEX.exists():
        return {}
    import json
    try:
        return json.loads(META_INDEX.read_text(encoding='utf-8'))
    except Exception:
        return {}


def save_meta_index(meta: Dict[str, Dict[str, Any]]):
    try:
        import json, tempfile, os
        tmp = META_INDEX.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding='utf-8')
        os.replace(tmp, META_INDEX)
    except Exception:
        pass


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
    """Adaptive time-based rate limiter shared across threads."""

    def __init__(self, max_calls_per_second: float, min_cps: float = 1.0, max_cps: float | None = None):
        import threading
        self._lock = threading.Lock()
        self._cps0 = float(max(0.1, max_calls_per_second))
        self._cps = float(max(0.1, max_calls_per_second))
        self._min = float(max(0.1, min_cps))
        self._max = float(max_cps) if max_cps else float(max_calls_per_second)
        self._last = 0.0
        self._penalty = 0  # error bursts lower CPS

    def acquire(self):
        with self._lock:
            interval = 1.0 / max(self._min, min(self._cps, self._max))
            last = self._last
        now = time.time()
        delta = now - last
        if delta < interval:
            time.sleep(interval - delta)
        with self._lock:
            self._last = time.time()

    def feedback(self, ok: bool):
        with self._lock:
            if ok:
                # slowly increase
                self._penalty = max(0, self._penalty - 1)
                self._cps = min(self._max, self._cps * (1.0 + 0.02))
            else:
                # shrink quickly when errors (e.g., rate limit)
                self._penalty += 1
                factor = 0.5 if self._penalty >= 2 else 0.7
                self._cps = max(self._min, self._cps * factor)


@dataclass
class IncrementalResult(FetchResult):
    action: str = 'skip'  # 'skip' | 'incremental' | 'full'
    added_rows: int = 0
    last_date: Optional[str] = None


def fetch_stock_daily_incremental(
    code: str,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resume: bool = True,
    force: bool = False,
    limiter: Optional[RateLimiter] = None,
    ts_code: Optional[str] = None,
) -> IncrementalResult:
    """Incremental fetch for a single stock using meta_index + last trading day.

    - If resume and data up-to-date to last trading day: skip
    - Else, fetch from max(meta_last_date+1, start_date) to end_date (or today)
    - Merge and write atomically; update meta_index
    """
    pro = get_pro()
    ts_c = ts_code or _guess_ts_code(code)
    meta = load_meta_index()
    last_trading = get_last_trading_day(pro)
    end_ymd = (end_date or _today_yyyymmdd())
    # convert for comparison
    last_trading_iso = last_trading
    mi = meta.get(code, {}) if resume else {}
    meta_last = mi.get('last_date')

    # Up-to-date check
    if resume and not force and meta_last:
        try:
            if pd.to_datetime(meta_last) >= pd.to_datetime(last_trading_iso):
                return IncrementalResult(code=code, ok=True, rows=int(mi.get('rows', 0)), action='skip', added_rows=0, last_date=meta_last)
        except Exception:
            pass

    # Determine fetch start
    if resume and meta_last:
        s_dt = (pd.to_datetime(meta_last) + pd.Timedelta(days=1)).strftime('%Y%m%d')
    else:
        s_dt = start_date
    params: Dict[str, Optional[str]] = {"ts_code": ts_c}
    if s_dt:
        params['start_date'] = s_dt
    if end_ymd:
        params['end_date'] = end_ymd

    # Call API
    if limiter:
        limiter.acquire()
    try:
        raw = pro.daily(**params)
        if limiter:
            limiter.feedback(True)
    except Exception as e:
        if limiter:
            limiter.feedback(False)
        return IncrementalResult(code=code, ok=False, rows=int(mi.get('rows', 0)), error=f"{type(e).__name__}: {e}")

    new_df = _normalize_daily_df(raw)

    p = DATA_DIR / f"{code}.parquet"
    old_df = None
    if p.exists():
        try:
            old_df = pd.read_parquet(p)
        except Exception:
            old_df = None

    if (old_df is None or old_df.empty) and (resume and meta_last) and new_df.empty:
        # meta 声称有数据，但文件丢失且没有新数据；降级为读历史全量
        params_f = {"ts_code": ts_c}
        if start_date:
            params_f['start_date'] = start_date
        if end_ymd:
            params_f['end_date'] = end_ymd
        if limiter:
            limiter.acquire()
        try:
            raw_full = pro.daily(**params_f)
            if limiter:
                limiter.feedback(True)
        except Exception as e:
            if limiter:
                limiter.feedback(False)
            return IncrementalResult(code=code, ok=False, rows=0, error=f"{type(e).__name__}: {e}")
        new_df = _normalize_daily_df(raw_full)
        action = 'full'
        merged = new_df
    else:
        if old_df is None or old_df.empty:
            merged = new_df
            action = 'full'
        else:
            if new_df.empty:
                # nothing to add
                cur_rows = len(old_df)
                last = str(old_df['日期'].iloc[-1]) if cur_rows else None
                meta[code] = {
                    'last_date': last,
                    'rows': cur_rows,
                    'updated_at': pd.Timestamp.utcnow().isoformat(),
                    'source': 'tushare_daily',
                    'schema_version': 1,
                }
                save_meta_index(meta)
                return IncrementalResult(code=code, ok=True, rows=cur_rows, action='skip', added_rows=0, last_date=last)
            merged = pd.concat([old_df, new_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)
            action = 'incremental'

    rows = len(merged)
    last = str(merged['日期'].iloc[-1]) if rows else None
    # atomic write
    tmp = p.with_suffix('.parquet.tmp')
    merged.to_parquet(tmp, index=False)
    import os as _os
    _os.replace(tmp, p)

    meta[code] = {
        'last_date': last,
        'rows': rows,
        'updated_at': pd.Timestamp.utcnow().isoformat(),
        'source': 'tushare_daily',
        'schema_version': 1,
    }
    save_meta_index(meta)

    added = 0 if action == 'full' else (0 if old_df is None else max(0, rows - len(old_df)))
    return IncrementalResult(code=code, ok=True, rows=rows, action=action, added_rows=int(added), last_date=last)


def fetch_many_daily(
    codes: Iterable[str],
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_workers: int = 4,
    max_calls_per_second: float = 8.0,
    resume: bool = True,
    force: bool = False,
    on_progress: Optional[Callable[[IncrementalResult], None]] = None,
    codes_ts_map: Optional[Dict[str, str]] = None,
) -> Dict[str, FetchResult]:
    """Batch fetch daily kline (incremental) with adaptive rate limiting.

    Returns dict of code -> FetchResult.
    """
    limiter = RateLimiter(max_calls_per_second)

    results: Dict[str, FetchResult] = {}

    def _task(c: str) -> FetchResult:
        try:
            res = fetch_stock_daily_incremental(
                c,
                start_date=start_date,
                end_date=end_date,
                resume=resume,
                force=force,
                limiter=limiter,
                ts_code=(codes_ts_map.get(c) if codes_ts_map else None),
            )
            if on_progress:
                try:
                    on_progress(res)
                except Exception:
                    pass
            return FetchResult(code=res.code, ok=res.ok, rows=res.rows, error=res.error)
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
    "fetch_stock_daily_incremental",
    "fetch_many_daily",
    "get_pro",
    "get_trade_calendar",
    "get_last_trading_day",
    "load_meta_index",
    "save_meta_index",
]


def fetch_stock_basic_and_save(list_status: str = 'L') -> pd.DataFrame:
    """Fetch stock_basic from Tushare and save to backend/data/stock_basic.parquet.

    Args:
      list_status: 'L' 上市, 'D' 退市, 'P' 暂停上市。默认仅拉取在市股票。
    """
    pro = get_pro()
    # 选择常用字段，减少体积
    fields = 'ts_code,symbol,name,fullname,market,exchange,area,industry,list_status,list_date'
    df = pro.stock_basic(exchange='', list_status=list_status, fields=fields)
    # 统一字段类型与格式：list_date -> YYYY-MM-DD
    if 'list_date' in df.columns:
        s = pd.to_datetime(df['list_date'], errors='coerce')
        df['list_date'] = s.dt.strftime('%Y-%m-%d')
    out = df.copy()
    out_path = DATA_DIR / 'stock_basic.parquet'
    out.to_parquet(out_path, index=False)
    return out
