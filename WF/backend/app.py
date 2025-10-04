from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException, Body
import threading, time, random, uuid, datetime
import difflib
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

# Assuming data_service.py is in the services directory
from services.data_service import get_stock_data, get_stock_info
from services.tushare_get_data import fetch_stock_daily
from services.analysis_service import find_a_points, find_b_points, find_c_points
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

app = FastAPI(title="Stock Data API")

# Allow CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, can be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the path to the stock basic data
# The app.py is in WF/backend, and the parquet file is in WF/
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = DATA_DIR / "stock_basic.parquet"
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ---- 批量拉取任务（全量获取） ----
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _log(job: Dict[str, Any], msg: str):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    job['logs'].append(f"[{ts}] {msg}")


# ---------------- 批量 ABC 择时（进程池） ----------------

def _abc_worker(symbol: str, a_params: Dict[str, Any], b_params: Dict[str, Any], c_params: Dict[str, Any]) -> Dict[str, Any]:
    """子进程执行：单只股票的 A/B/C 分析与结果落盘。
    返回：{symbol, a_count, b_count, c_count, b_recent: bool}
    """
    # 延迟导入，避免主进程启动时产生代价，并确保在子进程中可加载本模块依赖
    import pandas as pd
    from pathlib import Path
    from typing import Dict, Any
    # 引入本模块内函数
    from app import compute_a_points_v2, compute_b_points, compute_c_points, DATA_DIR, ANALYSIS_DIR

    out_dir = (ANALYSIS_DIR / symbol)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A 点
    a_res = compute_a_points_v2(symbol, a_params)
    a_pts = a_res.get('points', []) or []
    a_tbl = pd.DataFrame(a_res.get('table', []) or [])
    if not a_tbl.empty:
        a_tbl.to_excel(out_dir / 'A_point.xlsx', index=False)
    a_dates = [p.get('date') for p in a_pts if p.get('date')]

    # B 点
    bp = dict(b_params)
    bp['a_points_dates'] = a_dates
    b_res = compute_b_points(symbol, bp)
    b_pts = b_res.get('points', []) or []
    b_tbl = pd.DataFrame(b_res.get('table', []) or [])
    if not b_tbl.empty:
        b_tbl.to_excel(out_dir / 'B_point.xlsx', index=False)
    b_dates = [p.get('date') for p in b_pts if p.get('date')]

    # 近5天是否有 B 点
    b_recent = False
    try:
        if b_dates:
            today = pd.Timestamp.today().normalize()
            thresh = today - pd.Timedelta(days=5)
            for ds in b_dates:
                dt = pd.to_datetime(ds, errors='coerce')
                if pd.notna(dt) and dt.normalize() >= thresh:
                    b_recent = True
                    break
    except Exception:
        b_recent = False

    # C 点
    cp = dict(c_params)
    cp['a_points_dates'] = a_dates
    cp['b_points_dates'] = b_dates
    c_res = compute_c_points(symbol, cp)
    c_tbl = pd.DataFrame(c_res.get('table', []) or [])
    if not c_tbl.empty:
        c_tbl.to_excel(out_dir / 'C_point.xlsx', index=False)

    return {
        'symbol': symbol,
        'a_count': int(len(a_pts)),
        'b_count': int(len(b_pts)),
        'c_count': int(len(c_res.get('points', []) or [])),
        'b_recent': bool(b_recent),
    }


def _run_abc_batch(job_id: str):
    """主线程中启动进程池，批量执行 ABC 择时。"""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return
    try:
        if not DATA_FILE.exists():
            raise RuntimeError("stock_basic.parquet 不存在，无法获取股票列表")
        base = pd.read_parquet(DATA_FILE)
        # 过滤：symbols 或 market
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            job_symbols = set([str(s) for s in (job.get('symbols') or [])])
            job_market = (job.get('market') or '').strip()
        df = base.copy()
        if job_market:
            df = df[df['market'].astype(str) == job_market]
        if job_symbols:
            df = df[df['symbol'].astype(str).isin(job_symbols)]
        symbols: List[str] = [str(s) for s in df['symbol'].dropna().astype(str).tolist()]
        # 映射：代码->名称/市场
        name_map = {str(r['symbol']): r.get('name') for _, r in base.iterrows()}
        market_map = {str(r['symbol']): r.get('market') for _, r in base.iterrows()}

        with JOBS_LOCK:
            job['total'] = len(symbols)
        a_params = job.get('a_params', {}) or {}
        b_params = job.get('b_params', {}) or {}
        c_params = job.get('c_params', {}) or {}

        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_abc_worker, sym, a_params, b_params, c_params) for sym in symbols]
            for fut in as_completed(futures):
                res = None
                try:
                    res = fut.result()
                    sym = res.get('symbol')
                    with JOBS_LOCK:
                        job = JOBS.get(job_id)
                        job['done'] += 1
                        job['success'] += 1
                        job['last_symbol'] = sym
                        if res.get('b_recent'):
                            entry = {
                                'symbol': sym,
                                'name': name_map.get(sym, ''),
                                'market': market_map.get(sym, ''),
                            }
                            job.setdefault('b_recent', [])
                            job['b_recent'].append(entry)
                    _log(job, f"完成 {sym}")
                except Exception as e:
                    with JOBS_LOCK:
                        job = JOBS.get(job_id)
                        job['done'] += 1
                        job['fail'] += 1
                        job.setdefault('failed', []).append('unknown')
                    _log(job, f"失败（子进程）：{type(e).__name__} {e}")
        with JOBS_LOCK:
            job['status'] = 'finished'
            job['finished_at'] = datetime.datetime.now().isoformat()
        _log(job, "ABC 择时批量完成")
    except Exception as e:
        with JOBS_LOCK:
            job['status'] = 'error'
            job['error'] = f"{type(e).__name__}: {e}"
        _log(job, f"任务异常: {type(e).__name__}: {e}")


def _run_fetch_all(job_id: str, sleep_max: float = 1.0):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return
    try:
        # 读取股票清单
        if not DATA_FILE.exists():
            raise RuntimeError("stock_basic.parquet 不存在，无法获取股票列表")
        df = pd.read_parquet(DATA_FILE)
        # 支持 job 参数过滤
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            job_symbols = set([str(s) for s in (job.get('symbols') or [])])
            job_market = (job.get('market') or '').strip()

        if job_market:
            # 使用 stock_basic 中的 market 列过滤板块
            df = df[df['market'].astype(str) == job_market]
        if job_symbols:
            df = df[df['symbol'].astype(str).isin(job_symbols)]

        if 'symbol' in df.columns:
            symbols = [str(s) for s in df['symbol'].tolist() if pd.notna(s)]
        elif '代码' in df.columns:
            symbols = [str(s) for s in df['代码'].tolist() if pd.notna(s)]
        else:
            raise RuntimeError("股票列表缺少 symbol 列")
        with JOBS_LOCK:
            job['total'] = len(symbols)
        for i, sym in enumerate(symbols, start=1):
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job or job.get('status') == 'cancelled':
                    return
            try:
                _log(job, f"开始获取 {sym} (tushare) ...")
                # 使用 Tushare 接口抓取并保存到 backend/data/{code}.parquet
                data = fetch_stock_daily(sym)
                if data is None or data.empty:
                    raise RuntimeError("返回空数据")
                with JOBS_LOCK:
                    job['done'] += 1
                    job['success'] += 1
                    job['last_symbol'] = sym
                _log(job, f"成功 {sym}（{i}/{len(symbols)}）")
            except Exception as e:
                with JOBS_LOCK:
                    job['done'] += 1
                    job['fail'] += 1
                    job.setdefault('failed', []).append(sym)
                    job['last_symbol'] = sym
                _log(job, f"失败 {sym}: {type(e).__name__} {e}")
            # 0~1 秒随机休眠
            time.sleep(random.random() * float(sleep_max))
        with JOBS_LOCK:
            job['status'] = 'finished'
            job['finished_at'] = datetime.datetime.now().isoformat()
        _log(job, "全量获取完成")
    except Exception as e:
        with JOBS_LOCK:
            job['status'] = 'error'
            job['error'] = f"{type(e).__name__}: {e}"
        _log(job, f"任务异常: {type(e).__name__}: {e}")


@app.post("/api/fetch_all/start")
def start_fetch_all(payload: Dict[str, Any] = Body(default=None), sleep_max: float = 1.0):
    job_id = uuid.uuid4().hex[:8]
    job = {
        'job_id': job_id,
        'status': 'running',
        'started_at': datetime.datetime.now().isoformat(),
        'finished_at': None,
        'total': 0,
        'done': 0,
        'success': 0,
        'fail': 0,
        'failed': [],
        'logs': [],
        'last_symbol': None,
    }
    # 接收过滤参数：symbols(列表)、market(板块)
    if isinstance(payload, dict):
        syms = payload.get('symbols')
        mkt = payload.get('market')
    else:
        syms, mkt = None, None
    with JOBS_LOCK:
        JOBS[job_id] = job
        if syms:
            JOBS[job_id]['symbols'] = [str(s).strip() for s in syms if str(s).strip()]
        if mkt:
            JOBS[job_id]['market'] = str(mkt).strip()
    t = threading.Thread(target=_run_fetch_all, args=(job_id, sleep_max), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.get("/api/fetch_all/{job_id}/status")
def fetch_all_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='job not found')
        # 返回浅拷贝，避免并发修改
        return dict(job)


# ---- ABC 批量分析 API ----

@app.post("/api/abc_batch/start")
def start_abc_batch(params: Dict[str, Any]):
    job_id = uuid.uuid4().hex[:8]
    job = {
        'job_id': job_id,
        'status': 'running',
        'started_at': datetime.datetime.now().isoformat(),
        'finished_at': None,
        'total': 0,
        'done': 0,
        'success': 0,
        'fail': 0,
        'failed': [],
        'logs': [],
        'last_symbol': None,
        # 记录近5天存在B点的股票
        'b_recent': [],
        # 记录参数以传递给子进程
        'a_params': params.get('a_params', {}),
        'b_params': params.get('b_params', {}),
        'c_params': params.get('c_params', {}),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
        # 过滤参数（可选）：symbols / market
        syms = params.get('symbols')
        mkt = params.get('market')
        if syms:
            JOBS[job_id]['symbols'] = [str(s).strip() for s in syms if str(s).strip()]
        if mkt:
            JOBS[job_id]['market'] = str(mkt).strip()
    t = threading.Thread(target=_run_abc_batch, args=(job_id,), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.get("/api/abc_batch/{job_id}/status")
def abc_batch_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail='job not found')
        return dict(job)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stocks")
def get_stocks(
        page: int = 1,
        page_size: int = 10,
        sort_by: Optional[str] = None,
        sort_dir: str = 'asc',
        market: Optional[str] = None,
        industry: Optional[str] = None,
        code: Optional[str] = Query(None, description="按股票代码模糊匹配（substring）"),
        name: Optional[str] = Query(None, description="按股票名称匹配（可模糊）"),
        name_fuzzy: bool = Query(True, description="名称是否使用相似度匹配"),
        name_sim_threshold: float = Query(0.55, ge=0.0, le=1.0, description="相似度阈值，仅在 name_fuzzy 时生效"),
):
    """
    Get a list of stocks with pagination, sorting, and filtering.
    """
    if not DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Stock data file not found.")

    try:
        # 优先从 data 目录读取；兼容旧路径（WF 根目录）
        df = pd.read_parquet(DATA_FILE) if DATA_FILE.exists() else pd.read_parquet(
            Path(__file__).parent.parent / "stock_basic.parquet")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read stock data: {e}")

    # Filtering
    if market:
        df = df[df['market'].str.contains(market, case=False, na=False)]
    if industry:
        df = df[df['industry'].str.contains(industry, case=False, na=False)]
    if code:
        # 股票代码模糊匹配（包含即可）
        if 'symbol' in df.columns:
            df = df[df['symbol'].astype(str).str.contains(str(code), case=False, na=False)]
    if name:
        # 名称匹配：支持 contains 或 模糊相似度
        if 'name' in df.columns:
            name_q = str(name).strip()
            if name_fuzzy and name_q:
                # 计算相似度，并按阈值过滤
                def _sim(s: str) -> float:
                    try:
                        return difflib.SequenceMatcher(None, str(s).lower(), name_q.lower()).ratio()
                    except Exception:
                        return 0.0

                df = df.copy()
                df['__similarity__'] = df['name'].astype(str).map(_sim)
                df = df[df['__similarity__'] >= float(name_sim_threshold)]
            else:
                df = df[df['name'].astype(str).str.contains(name_q, case=False, na=False)]

    # Sorting
    if name and name_fuzzy and '__similarity__' in df.columns and (not sort_by):
        # 名称相似度搜索时，默认按相似度倒序
        df = df.sort_values(by='__similarity__', ascending=False)
    elif sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=(sort_dir == 'asc'))

    # Pagination
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    paginated_df = df.iloc[start:end]

    result = paginated_df.to_dict(orient='records')
    # 清理临时列
    if '__similarity__' in paginated_df.columns:
        for r in result:
            if '__similarity__' in r:
                r.pop('__similarity__', None)
    return {
        "items": result,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/stocks/{symbol}")
def get_stock_kline(symbol: str):
    """
    Get K-line data for a specific stock.
    Prefer reading local parquet if present; otherwise fetch via get_stock_data.
    """
    try:
        # Prefer reading from backend/data
        existing = DATA_DIR / f"{symbol}.parquet"
        if existing.exists():
            df = pd.read_parquet(existing)
            df = _normalize_kline_df(df)
            # 仅返回前端绘图所需列，避免列名干扰
            needed = [c for c in ['日期', '开盘', '收盘', '最高', '最低', '成交量'] if c in df.columns]
            if needed:
                df = df[needed]
            return df.to_dict(orient='records')

        # Fallback: fetch using akshare and save parquet inside get_stock_data
        stock_df = get_stock_data(symbol)
        if stock_df is None or stock_df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found for the given symbol.")
        stock_df = _normalize_kline_df(stock_df)
        needed = [c for c in ['日期', '开盘', '收盘', '最高', '最低', '成交量'] if c in stock_df.columns]
        if needed:
            stock_df = stock_df[needed]
        print('----')
        print(stock_df)
        return stock_df.to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/{symbol}/refresh")
def refresh_stock_kline(symbol: str):
    """
    Force refresh: actively call get_stock_data to fetch latest and persist parquet, then return data.
    """
    try:
        stock_df = get_stock_data(symbol)
        if stock_df is None or stock_df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found for the given symbol.")
        stock_df = _normalize_kline_df(stock_df)
        needed = [c for c in ['日期', '开盘', '收盘', '最高', '最低', '成交量'] if c in stock_df.columns]
        if needed:
            stock_df = stock_df[needed]
        print('----')
        print(stock_df)
        return stock_df.to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/{symbol}/a_points")
def compute_a_points(symbol: str, params: Dict[str, Any]):
    """
    计算并返回 A 点列表（按用户定义的四个条件）。
    Body 示例：
    {
      "cond1": {"enabled": true, "Y": 60, "M": 60, "slope_max": 0.0},
      "cond2": {"enabled": true, "X": 5, "Y": 50},
      "cond3": {"enabled": true, "J": 10, "K": 2.0},
      "cond4": {"enabled": true, "D": 5, "F": 10}
    }
    返回：{"points": [{"date": "YYYY-MM-DD", "price_low": float, "price_close": float}], "count": int}
    """
    try:
        # 读取本地数据，不存在则拉取
        path = DATA_DIR / f"{symbol}.parquet"
        if path.exists():
            df_cn = pd.read_parquet(path)
        else:
            # 兼容旧路径（WF 根目录曾保存过）
            legacy = Path(__file__).parent.parent / f"{symbol}.parquet"
            if legacy.exists():
                df_cn = pd.read_parquet(legacy)
            else:
                # 最后才拉取，避免依赖外网
                df_cn = get_stock_data(symbol)
        if df_cn is None or df_cn.empty:
            raise HTTPException(status_code=404, detail="No kline data for symbol")

        # 取字段
        d = pd.DataFrame({
            "date": pd.to_datetime(df_cn["日期"], errors="coerce"),
            "open": pd.to_numeric(df_cn.get("开盘"), errors="coerce"),
            "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
            "low": pd.to_numeric(df_cn.get("最低"), errors="coerce"),
            "close": pd.to_numeric(df_cn.get("收盘"), errors="coerce"),
            "volume": pd.to_numeric(df_cn.get("成交量"), errors="coerce"),
        })
        d = d.sort_values("date").reset_index(drop=True)

        # 解析参数
        cond1 = params.get("cond1", {}) or {}
        cond2 = params.get("cond2", {}) or {}
        cond3 = params.get("cond3", {}) or {}
        cond4 = params.get("cond4", {}) or {}

        eps = 1e-9  # 固定容差

        # --- 条件1：MA_Y 的最近 M 天线性回归 slope < slope_max（在 t-1 上评估） ---
        c1_enabled = bool(cond1.get("enabled", True))
        Y = int(cond1.get("Y", 60))
        M = int(cond1.get("M", 60))
        slope_max = float(cond1.get("slope_max", 0.0))

        maY = d["close"].rolling(Y, min_periods=Y).mean()
        # 回归 slope：rolling 窗口为 M，使用等间隔 x
        import numpy as np
        if M > 1:
            x = np.arange(M, dtype=float)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()

            def _slope(arr: np.ndarray) -> float:
                y = arr.astype(float)
                y_mean = y.mean()
                cov = ((x - x_mean) * (y - y_mean)).sum()
                return float(cov / x_var) if x_var != 0 else 0.0

            slope_series = maY.rolling(M, min_periods=M).apply(lambda a: _slope(a), raw=True)
        else:
            slope_series = pd.Series(0.0, index=d.index)
        cond1_ok = (slope_series.shift(1) < slope_max)  # 在 t-1 时刻的斜率
        if not c1_enabled:
            cond1_ok[:] = True

        # --- 条件2：短均线 X 上穿 长均线 Y（当日 t 触发） ---
        c2_enabled = bool(cond2.get("enabled", True))
        Y2 = int(cond2.get("Y", 50))
        hold_days = int(cond2.get("hold_days", 1))
        shorts_raw = cond2.get("shorts")
        if shorts_raw is None:
            # 兼容旧入参：X 作为单一短均线
            shorts_list = [int(cond2.get("X", 5))]
        else:
            if isinstance(shorts_raw, str):
                shorts_list = [int(s) for s in str(shorts_raw).split(',') if s.strip().isdigit()]
            else:
                shorts_list = [int(x) for x in (shorts_raw or [])]
        shorts_list = [k for k in shorts_list if k > 0]
        if not shorts_list:
            shorts_list = [5]

        maY2 = d["close"].rolling(Y2, min_periods=Y2).mean()
        above_all = None
        ma_shorts = {}
        for k in shorts_list:
            m = d["close"].rolling(k, min_periods=k).mean()
            ma_shorts[k] = m
            a = (m + eps) >= maY2
            above_all = a if above_all is None else (above_all & a)
        if above_all is None:
            above_all = pd.Series(False, index=d.index)
        if hold_days > 1:
            above_hold_ok = (above_all.astype("int8").rolling(hold_days, min_periods=hold_days).min() == 1)
        else:
            above_hold_ok = above_all
        cross_up = above_hold_ok
        if not c2_enabled:
            cross_up[:] = True

        # --- 条件3：放量（当日）：Vol(t) ≥ K × max(Vol[t-J..t-1]) ---
        c3_enabled = bool(cond3.get("enabled", True))
        J = int(cond3.get("J", 10))
        K = float(cond3.get("K", 2.0))
        prevJmax = d["volume"].shift(1).rolling(J, min_periods=1).max()
        cond3_ok = (d["volume"] >= (K * prevJmax))
        if not c3_enabled:
            cond3_ok[:] = True

        # --- 条件4：当日 VMA_D 上穿 VMA_F ---
        c4_enabled = bool(cond4.get("enabled", True))
        D = int(cond4.get("D", 5))
        F = int(cond4.get("F", 10))
        vmaD = d["volume"].rolling(D, min_periods=D).mean()
        vmaF = d["volume"].rolling(F, min_periods=F).mean()
        v_above_today = (vmaD + eps) >= vmaF
        v_above_yday = (vmaD.shift(1) + eps) >= vmaF.shift(1)
        cond4_ok = v_above_today & (~v_above_yday.fillna(False))
        if not c4_enabled:
            cond4_ok[:] = True

        # A 点：四个条件合取（各自可独立关闭）且“发生在同一天 t”
        A_mask = cond1_ok & cross_up & cond3_ok & cond4_ok

        # 导出点
        pts = []
        for i, ok in enumerate(A_mask):
            if bool(ok):
                dt = d.at[i, "date"]
                price_low = d.at[i, "low"] if pd.notna(d.at[i, "low"]) else None
                price_close = d.at[i, "close"] if pd.notna(d.at[i, "close"]) else None
                pts.append({
                    "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "price_low": None if price_low is None else float(price_low),
                    "price_close": None if price_close is None else float(price_close),
                })

        # 组装每日诊断表（用于前端表格展示）
        table = []
        # 为避免键名混淆，cond1 使用 maY_long (Y)，cond2 使用 maY2 (Y2)
        for i in range(len(d)):
            row = {
                "date": pd.to_datetime(d.at[i, "date"]).strftime("%Y-%m-%d"),
                "open": None if pd.isna(d.at[i, "open"]) else float(d.at[i, "open"]),
                "high": None if pd.isna(d.at[i, "high"]) else float(d.at[i, "high"]),
                "low": None if pd.isna(d.at[i, "low"]) else float(d.at[i, "low"]),
                "close": None if pd.isna(d.at[i, "close"]) else float(d.at[i, "close"]),
                "volume": None if pd.isna(d.at[i, "volume"]) else float(d.at[i, "volume"]),
                "maY2": None if pd.isna(maY2.iat[i]) else float(maY2.iat[i]),
                "maY_long": None if pd.isna(maY.iat[i]) else float(maY.iat[i]),
                "slope_t1": None if pd.isna(slope_series.shift(1).iat[i]) else float(slope_series.shift(1).iat[i]),
                "prevJmax": None if pd.isna(prevJmax.iat[i]) else float(
                    prevJmax.iat[i]) if 'prevJmax' in locals() else None,
                "vr": None if ('prevJmax' not in locals() or pd.isna(prevJmax.iat[i]) or pd.isna(
                    d.at[i, "volume"])) else float(d.at[i, "volume"]) / float(prevJmax.iat[i]) if float(
                    prevJmax.iat[i]) != 0 else None,
                "vmaD": None if pd.isna(vmaD.iat[i]) else float(vmaD.iat[i]),
                "vmaF": None if pd.isna(vmaF.iat[i]) else float(vmaF.iat[i]),
                "above_all": bool(above_all.iat[i]) if not pd.isna(above_all.iat[i]) else False,
                "above_hold_ok": bool(above_hold_ok.iat[i]) if not pd.isna(above_hold_ok.iat[i]) else False,
                "cond1": bool(cond1_ok.iat[i]) if not pd.isna(cond1_ok.iat[i]) else False,
                "cond2": bool(cross_up.iat[i]) if not pd.isna(cross_up.iat[i]) else False,
                "cond3": bool(cond3_ok.iat[i]) if not pd.isna(cond3_ok.iat[i]) else False,
                "cond4": bool(cond4_ok.iat[i]) if not pd.isna(cond4_ok.iat[i]) else False,
                "a_point": bool(A_mask.iat[i]) if not pd.isna(A_mask.iat[i]) else False,
            }
            table.append(row)

        return {"points": pts, "count": len(pts), "table": table}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/{symbol}/b_points")
def compute_b_points(symbol: str, params: Dict[str, Any]):
    """
    计算并返回 B 点列表（B 点依赖 A 点分段；本接口内部先用默认/轻量参数求 A_point 分段，再按传入 cond1..cond6 计算 B_point）。

    Body 示例：
    {
      "cond1": {"enabled": true, "min_days_from_a": 60, "max_days_from_a": null, "allow_multi_b_per_a": true},
      "cond2": {"enabled": true, "above_maN_window": 5, "long_ma_window": 60, "above_maN_days": 15, "above_maN_consecutive": false, "max_maN_below_days": 5},
      "cond3": {"enabled": true, "touch_price": "low", "touch_relation": "le", "require_bearish": false, "require_close_le_prev": false, "long_ma_days": 60},
      "cond4": {"enabled": true, "vr1_max": 1.2, "recent_max_vol_window": 10},
      "cond5": {"enabled": true, "dryness_ratio_max": 0.8, "require_vol_le_vma10": true, "dryness_recent_window": 0, "dryness_recent_min_days": 0, "vma_short_window": 5, "vma_long_window": 10},
      "cond6": {"enabled": false, "price_stable_mode": "no_new_low", "max_drop_ratio": 0.03, "use_atr_window": 14, "atr_buffer": 0.5}
    }
    返回：{"points": [{"date": "YYYY-MM-DD", "price_low": float, "price_close": float}], "count": int, "table": [...诊断列...]}
    """
    try:
        path = DATA_DIR / f"{symbol}.parquet"
        if path.exists():
            df_cn = pd.read_parquet(path)
        else:
            legacy = Path(__file__).parent.parent / f"{symbol}.parquet"
            if legacy.exists():
                df_cn = pd.read_parquet(legacy)
            else:
                df_cn = get_stock_data(symbol)
        if df_cn is None or df_cn.empty:
            raise HTTPException(status_code=404, detail="No kline data for symbol")

        # 基础字段
        d = pd.DataFrame({
            "code": symbol,
            "date": pd.to_datetime(df_cn["日期"], errors="coerce"),
            "open": pd.to_numeric(df_cn.get("开盘"), errors="coerce"),
            "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
            "low": pd.to_numeric(df_cn.get("最低"), errors="coerce"),
            "close": pd.to_numeric(df_cn.get("收盘"), errors="coerce"),
            "volume": pd.to_numeric(df_cn.get("成交量"), errors="coerce"),
        })
        d = d.sort_values("date").reset_index(drop=True)
        # 计算“长期平均线”供 B 点判定，默认60天，可由前端传参覆盖（取 cond3.long_ma_days，用于触及/击破）
        p2 = params.get("cond3", {}) or {}
        try:
            long_ma_days = int(p2.get("long_ma_days", 60)) if p2.get("long_ma_days", None) not in ("", None) else 60
        except Exception:
            long_ma_days = 60
        d["ma_long_b"] = d["close"].rolling(long_ma_days, min_periods=long_ma_days).mean()

        # 根据前端传入的 A 点集合进行分段；若未提供，则回退到 find_a_points 的默认逻辑
        a_dates = set()
        if isinstance(params.get("a_points_dates"), list):
            try:
                a_dates = {pd.to_datetime(x).normalize() for x in params.get("a_points_dates")}
            except Exception:
                a_dates = set()
        if a_dates:
            d_a = d.copy()
            d_a["A_point"] = d_a["date"].map(lambda x: pd.to_datetime(x).normalize() in a_dates).astype(bool)
        else:
            # 回退：使用默认的 A 点查找以确保分段存在
            d_a = find_a_points(d, code_col="code", date_col="date", close_col="close", volume_col="volume",
                                with_explain_strings=False)

        # 条件4：生成/覆盖 VR1（今天成交量/最近窗口内最大成交量），若 find_b_points 内未自动生成
        p4 = params.get("cond4", {}) or {}
        recent_max_win = int(p4.get("recent_max_vol_window", 10) or 10)
        prev_max = d_a.groupby("code")["volume"].shift(1).rolling(recent_max_win, min_periods=1).max()
        d_a["vr1"] = d_a["volume"] / prev_max

        # 解析 B 条件参数
        cond1 = params.get("cond1", {}) or {}
        cond2 = params.get("cond2", {}) or {}
        cond3 = params.get("cond3", {}) or {}
        cond4 = params.get("cond4", {}) or {}
        cond5 = params.get("cond5", {}) or {}
        cond6 = params.get("cond6", {}) or {}

        out = find_b_points(
            d_a,
            code_col="code", date_col="date",
            open_col="open", high_col="high", low_col="low", close_col="close",
            volume_col="volume", ma60_col="ma_long_b",
            cond1=cond1, cond2=cond2, cond3=cond3, cond4=cond4, cond5=cond5, cond6=cond6,
            with_explain_strings=False,
        )

        # 输出 B 点坐标
        bmask = out["B_point"].astype(bool)
        pts = []
        for i, ok in enumerate(bmask):
            if bool(ok):
                dt = out.at[i, "date"]
                price_low = out.at[i, "low"] if pd.notna(out.at[i, "low"]) else None
                price_close = out.at[i, "close"] if pd.notna(out.at[i, "close"]) else None
                pts.append({
                    "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "price_low": None if price_low is None else float(price_low),
                    "price_close": None if price_close is None else float(price_close),
                })

        # 组装诊断表格（仅输出基础列 + 已启用条件/子模块的相关列）
        # 读取前端开关，决定列的裁剪
        _c1 = cond1 or {}
        _c2 = cond2 or {}
        _c3 = cond3 or {}
        _c4 = cond4 or {}
        _c5 = cond5 or {}
        _c6 = cond6 or {}

        def _bool(v, default=False):
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            return s in ("1", "true", "yes", "y", "on") if s != '' else default

        c1_enabled = _bool(_c1.get("enabled", True), True)
        c2_enabled = _bool(_c2.get("enabled", True), True)
        c3_enabled = _bool(_c3.get("enabled", True), True)
        c4_enabled = _bool(_c4.get("enabled", True), True)
        c5_enabled = _bool(_c5.get("enabled", True), True)
        c6_enabled = _bool(_c6.get("enabled", False), False)

        # cond5 子模块开关（新式）
        has_new_flags = any(k in _c5 for k in ("vr1_enabled", "recent_n", "vr1_max", "vma_rel_enabled", "vol_down_enabled", "ratio_enabled", "vol_cmp_enabled"))
        vr1_sub_enabled = _bool(_c5.get("vr1_enabled", False), False)
        vma_rel_enabled = _bool(_c5.get("vma_rel_enabled", False), False)
        vol_down_enabled = _bool(_c5.get("vol_down_enabled", False), False)
        ratio_enabled = _bool(_c5.get("ratio_enabled", (False if has_new_flags else True)), (False if has_new_flags else True))
        vol_cmp_enabled = _bool(_c5.get("vol_cmp_enabled", (_c5.get("require_vol_le_vma10", True) if not has_new_flags else False)), (not has_new_flags))

        table = []
        for i in range(len(out)):
            row = out.iloc[i]
            rec = {
                "date": pd.to_datetime(row.get("date")).strftime("%Y-%m-%d") if pd.notna(row.get("date")) else None,
                "open": None if pd.isna(row.get("open")) else float(row.get("open")),
                "close": None if pd.isna(row.get("close")) else float(row.get("close")),
                "low": None if pd.isna(row.get("low")) else float(row.get("low")),
                "high": None if pd.isna(row.get("high")) else float(row.get("high")),
                "volume": None if pd.isna(row.get("volume")) else float(row.get("volume")),
            }

            if c1_enabled:
                rec.update({
                    "days_since_A": None if pd.isna(row.get("days_since_A")) else int(row.get("days_since_A")),
                    "cond1": bool(row.get("cond1_ok")) if pd.notna(row.get("cond1_ok")) else False,
                })

            if c2_enabled:
                rec.update({
                    "cond2_ratio_pct": (None if pd.isna(row.get("maN_above_ratio")) else float(row.get("maN_above_ratio") * 100.0)),
                    "cond2": bool(row.get("cond2_ok")) if pd.notna(row.get("cond2_ok")) else False,
                })

            if c3_enabled:
                rec.update({
                    "ma_long": None if pd.isna(row.get("ma_long_b")) else float(row.get("ma_long_b")),
                    "cond3": bool(row.get("cond3_ok")) if pd.notna(row.get("cond3_ok")) else False,
                    "bearish": bool(row.get("cond3_bearish_ok")) if pd.notna(row.get("cond3_bearish_ok")) else True,
                    "close_le_prev": bool(row.get("cond3_close_le_prev_ok")) if pd.notna(row.get("cond3_close_le_prev_ok")) else True,
                    "touch_ma60": bool(row.get("cond3_touch_ok")) if pd.notna(row.get("cond3_touch_ok")) else True,
                })

            if c4_enabled:
                # 同时输出 vr1 数值便于校验
                rec.update({
                    "vr1": None if pd.isna(row.get("vr1")) else float(row.get("vr1")),
                    "cond4": bool(row.get("cond4_ok")) if pd.notna(row.get("cond4_ok")) else False,
                    "vr1_ok": bool(row.get("cond4_vr1_ok")) if pd.notna(row.get("cond4_vr1_ok")) else True,
                })

            if c5_enabled:
                # cond5 总体结果
                rec.update({"cond5": bool(row.get("cond5_ok")) if pd.notna(row.get("cond5_ok")) else False})
                # 子模块明细（仅开启的才输出）
                if vr1_sub_enabled:
                    rec.update({
                        "c5_vr1_ok": bool(row.get("c5_vr1_ok")) if "c5_vr1_ok" in row.index and pd.notna(row.get("c5_vr1_ok")) else False,
                    })
                if vma_rel_enabled:
                    rec.update({
                        "c5_vma_rel_ok": bool(row.get("c5_vma_rel_ok")) if "c5_vma_rel_ok" in row.index and pd.notna(row.get("c5_vma_rel_ok")) else False,
                    })
                if ratio_enabled:
                    rec.update({
                        "c5_ratio_ok": bool(row.get("c5_ratio_ok")) if "c5_ratio_ok" in row.index and pd.notna(row.get("c5_ratio_ok")) else False,
                        "dryness_ratio": None if pd.isna(row.get("dryness_ratio")) else float(row.get("dryness_ratio")),
                    })
                if vol_cmp_enabled:
                    rec.update({
                        "c5_vol_cmp_ok": bool(row.get("c5_vol_cmp_ok")) if "c5_vol_cmp_ok" in row.index and pd.notna(row.get("c5_vol_cmp_ok")) else False,
                    })
                if vol_down_enabled:
                    rec.update({
                        "c5_down_ok": bool(row.get("c5_down_ok")) if "c5_down_ok" in row.index and pd.notna(row.get("c5_down_ok")) else False,
                        "vol_down_streak": None if ("vol_down_streak" not in row.index or pd.isna(row.get("vol_down_streak"))) else int(row.get("vol_down_streak")),
                    })

            if c6_enabled:
                rec.update({
                    "cond6": bool(row.get("cond6_ok")) if pd.notna(row.get("cond6_ok")) else False,
                    "cond6_metric": None if pd.isna(row.get("cond6_metric")) else float(row.get("cond6_metric")),
                })

            rec["B_point"] = bool(row.get("B_point")) if pd.notna(row.get("B_point")) else False
            table.append(rec)

        return {"points": pts, "count": len(pts), "table": table}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/{symbol}/c_points")
def compute_c_points(symbol: str, params: Dict[str, Any]):
    """
    计算并返回 C 点列表。

    条件：
    - cond1: { enabled, max_days_from_b }
    - cond2: { enabled, recent_n, vol_multiple }
    - cond3: { enabled, price_field('close'|'high'|'low'), ma_days, relation('ge'|'gt') }

    可选：
    - a_points_dates: ["YYYY-MM-DD", ...] 用于分段
    - b_points_dates: ["YYYY-MM-DD", ...] 用于最近B点定位
    """
    try:
        path = DATA_DIR / f"{symbol}.parquet"
        if path.exists():
            df_cn = pd.read_parquet(path)
        else:
            legacy = Path(__file__).parent.parent / f"{symbol}.parquet"
            if legacy.exists():
                df_cn = pd.read_parquet(legacy)
            else:
                df_cn = get_stock_data(symbol)
        if df_cn is None or df_cn.empty:
            raise HTTPException(status_code=404, detail="No kline data for symbol")

        d = pd.DataFrame({
            "code": symbol,
            "date": pd.to_datetime(df_cn["日期"], errors="coerce"),
            "open": pd.to_numeric(df_cn.get("开盘"), errors="coerce"),
            "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
            "low": pd.to_numeric(df_cn.get("最低"), errors="coerce"),
            "close": pd.to_numeric(df_cn.get("收盘"), errors="coerce"),
            "volume": pd.to_numeric(df_cn.get("成交量"), errors="coerce"),
        })
        d = d.sort_values("date").reset_index(drop=True)

        # 标记 A/B 点（若前端传入），确保分段与“最近B点”准确
        a_dates = set()
        b_dates = set()
        try:
            if isinstance(params.get("a_points_dates"), list):
                a_dates = {pd.to_datetime(x).normalize() for x in params.get("a_points_dates")}
            if isinstance(params.get("b_points_dates"), list):
                b_dates = {pd.to_datetime(x).normalize() for x in params.get("b_points_dates")}
        except Exception:
            a_dates, b_dates = set(), set()

        if a_dates:
            d["A_point"] = d["date"].map(lambda x: pd.to_datetime(x).normalize() in a_dates).astype(bool)
        else:
            d = find_a_points(d, code_col="code", date_col="date", close_col="close", volume_col="volume",
                              with_explain_strings=False)
        if b_dates:
            d["B_point"] = d["date"].map(lambda x: pd.to_datetime(x).normalize() in b_dates).astype(bool)
        else:
            # 确保存在 B_point 列，若未传则置 False（C 的 cond1 会据此不满足）
            if "B_point" not in d.columns:
                d["B_point"] = False

        cond1 = params.get("cond1", {}) or {}
        cond2 = params.get("cond2", {}) or {}
        cond3 = params.get("cond3", {}) or {}

        out = find_c_points(
            d,
            code_col="code", date_col="date",
            open_col="open", high_col="high", low_col="low", close_col="close",
            volume_col="volume",
            cond1=cond1, cond2=cond2, cond3=cond3,
            with_explain_strings=False,
        )

        # C 点
        cmask = out["C_point"].astype(bool)
        pts = []
        for i, ok in enumerate(cmask):
            if bool(ok):
                dt = out.at[i, "date"]
                price_low = out.at[i, "low"] if pd.notna(out.at[i, "low"]) else None
                price_close = out.at[i, "close"] if pd.notna(out.at[i, "close"]) else None
                pts.append({
                    "date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "price_low": None if price_low is None else float(price_low),
                    "price_close": None if price_close is None else float(price_close),
                })

        # 诊断表（基础列 + 已启用条件/子模块）
        _c1 = cond1 or {}
        _c2 = cond2 or {}
        _c3 = cond3 or {}
        def _bool(v, default=False):
            if v is None: return default
            if isinstance(v, bool): return v
            s = str(v).strip().lower()
            return s in ("1","true","yes","y","on") if s!='' else default

        c1_enabled = _bool(_c1.get("enabled", True), True)
        c2_enabled = _bool(_c2.get("enabled", True), True)
        c3_enabled = _bool(_c3.get("enabled", True), True)
        vr1_enabled = _bool(_c2.get("vr1_enabled", True), True)
        vma_cmp_enabled = _bool(_c2.get("vma_cmp_enabled", False), False)
        vol_up_enabled = _bool(_c2.get("vol_up_enabled", False), False)

        table = []
        for i in range(len(out)):
            r = out.iloc[i]
            rec = {
                "date": pd.to_datetime(r.get("date")).strftime("%Y-%m-%d") if pd.notna(r.get("date")) else None,
                "open": None if pd.isna(r.get("open")) else float(r.get("open")),
                "close": None if pd.isna(r.get("close")) else float(r.get("close")),
                "low": None if pd.isna(r.get("low")) else float(r.get("low")),
                "high": None if pd.isna(r.get("high")) else float(r.get("high")),
                "volume": None if pd.isna(r.get("volume")) else float(r.get("volume")),
            }
            if c1_enabled:
                rec.update({
                    "days_since_B": None if pd.isna(r.get("days_since_B")) else int(r.get("days_since_B")),
                    "cond1": bool(r.get("cond1_ok")) if pd.notna(r.get("cond1_ok")) else False,
                })
            if c2_enabled:
                rec.update({
                    "cond2": bool(r.get("cond2_ok")) if pd.notna(r.get("cond2_ok")) else False,
                })
                if vr1_enabled:
                    rec.update({
                        "vol_ratio": None if pd.isna(r.get("vol_ratio_vs_prevNmax")) else float(r.get("vol_ratio_vs_prevNmax")),
                        "c2_vr1_ok": (None if 'c2_vr1_ok' not in r else bool(r.get("c2_vr1_ok")) if pd.notna(r.get("c2_vr1_ok")) else False),
                    })
                if vma_cmp_enabled:
                    rec.update({
                        "vma_short": None if pd.isna(r.get("vma_short")) else float(r.get("vma_short")),
                        "vma_long": None if pd.isna(r.get("vma_long")) else float(r.get("vma_long")),
                        "c2_vma_ok": (None if 'c2_vma_ok' not in r else bool(r.get("c2_vma_ok")) if pd.notna(r.get("c2_vma_ok")) else False),
                    })
                if vol_up_enabled:
                    rec.update({
                        "c2_up_ok": (None if 'c2_up_ok' not in r else bool(r.get("c2_up_ok")) if pd.notna(r.get("c2_up_ok")) else False),
                    })
            if c3_enabled:
                rec.update({
                    "ma_Y": None if pd.isna(r.get("maY")) else float(r.get("maY")),
                    "cond3": bool(r.get("cond3_ok")) if pd.notna(r.get("cond3_ok")) else False,
                })
            rec["C_point"] = bool(r.get("C_point")) if pd.notna(r.get("C_point")) else False
            table.append(rec)

        return {"points": pts, "count": len(pts), "table": table}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stocks/{symbol}/a_points_v2")
def compute_a_points_v2(symbol: str, params: Dict[str, Any]):
    """
    基于 A02 原始逻辑（analysis_service.find_a_points），且每个条件参数独立（中文/英文键均可）。
    返回：{"points": [...], "count": n, "table": [...]}。
    """
    try:
        print(f"[A_POINTS_V2] symbol={symbol}")
        path = DATA_DIR / f"{symbol}.parquet"
        if not path.exists():
            df_cn = get_stock_data(symbol)
        else:
            df_cn = pd.read_parquet(path)
        if df_cn is None or df_cn.empty:
            raise HTTPException(status_code=404, detail="No kline data for symbol")
        print(f"[A_POINTS_V2] data loaded: shape={df_cn.shape}")

        d = pd.DataFrame({
            "code": str(symbol),
            "date": pd.to_datetime(df_cn["日期"], errors="coerce"),
            "open": pd.to_numeric(df_cn.get("开盘"), errors="coerce"),
            "low": pd.to_numeric(df_cn.get("最低"), errors="coerce"),
            "close": pd.to_numeric(df_cn.get("收盘"), errors="coerce"),
            "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
            "volume": pd.to_numeric(df_cn.get("成交量"), errors="coerce"),
        })

        p1 = params.get("条件1_长期下跌", params.get("cond1", {})) or {}
        p2 = params.get("条件2_短均线上穿", params.get("cond2", {})) or {}
        p3 = params.get("条件3_价格确认", params.get("cond3", {})) or {}
        # 放量确认（模块化，与 C 点 cond2 对齐）：优先读取统一的条件4组；否则回退到旧的 条件4/条件5 组合
        p4_group = params.get("条件4_放量确认", params.get("cond4_vol_group", None))
        p4 = params.get("条件4_放量对比", params.get("cond4_vol", {})) or {}
        p5 = params.get("条件5_量均线比较", params.get("cond5_vma", {})) or {}
        print(f"[A_POINTS_V2] raw params cond1={p1} cond2={p2} cond3={p3}")

        def _bool(v, default=False):
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            s = str(v).strip().lower()
            if s in ("true", "1", "yes", "y", "on"): return True
            if s in ("false", "0", "no", "n", "off"): return False
            return default

        def _int(v, default=None):
            try:
                if v is None:
                    return default
                if isinstance(v, bool):
                    return int(v)
                s = str(v).strip()
                if s == "":
                    return default
                return int(float(s))
            except Exception:
                return default

        def _tuple_of_ints(val, default=()):
            if val is None:
                return tuple(default)
            if isinstance(val, (list, tuple)):
                out = []
                for x in val:
                    iv = _int(x, None)
                    if iv is not None and iv > 0:
                        out.append(iv)
                return tuple(out)
            s = str(val).strip()
            if not s:
                return tuple(default)
            out = []
            for part in s.split(','):
                iv = _int(part, None)
                if iv is not None and iv > 0:
                    out.append(iv)
            return tuple(out)

        c1 = {
            "enabled": _bool(p1.get("启用", p1.get("enabled", True)), True),
            "long_window": _int(p1.get("长均线窗口", p1.get("long_window", 60)), 60),
            "down_lookback": _int(p1.get("下跌跨度", p1.get("down_lookback", 30)), 30),
        }

        shorts_tuple = _tuple_of_ints(p2.get("短均线集合", p2.get("short_windows", (5, 10))), (5, 10))
        req_tuple = None
        req_raw = p2.get("必须满足的短均线", p2.get("required_shorts", None))
        if req_raw is not None:
            rt = _tuple_of_ints(req_raw, ())
            req_tuple = rt if rt else None

        c2 = {
            "enabled": _bool(p2.get("启用", p2.get("enabled", True)), True),
            "short_windows": shorts_tuple if shorts_tuple else (5, 10),
            "long_window": _int(p2.get("长均线窗口", p2.get("long_window", 60)), 60),
            "cross_window": _int(p2.get("上穿完备窗口", p2.get("cross_window", 3)), 3),
            "required_shorts": req_tuple,
            "require_all": _bool(p2.get("全部满足", p2.get("require_all", True)), True),
        }

        cmw_raw = p3.get("确认均线窗口", p3.get("confirm_ma_window", None))
        cmw = _int(cmw_raw, None)
        c3 = {
            "enabled": _bool(p3.get("启用", p3.get("enabled", True)), True),
            "confirm_lookback_days": _int(p3.get("确认回看天数", p3.get("confirm_lookback_days", 0)), 0),
            "confirm_ma_window": cmw,
            "confirm_price_col": str(p3.get("确认价格列", p3.get("confirm_price_col", "high"))),
        }

        out = find_a_points(
            d,
            code_col="code", date_col="date", close_col="close", volume_col="volume",
            with_explain_strings=False,
            cond1=c1, cond2=c2, cond3=c3,
        )
        cnt = int(out["A_point"].astype(bool).sum())
        print(f"[A_POINTS_V2] computed rows={len(out)} A_points_base={cnt}")

        # ---- 条件4（模块化）：放量确认（与 C 点 cond2 对齐）----
        # 统一参数解析：优先使用 p4_group；否则用旧的 p4/p5 生成等价配置
        if isinstance(p4_group, dict):
            g4_enabled = _bool(p4_group.get("启用", p4_group.get("enabled", False)), False)
            vr1_enabled = _bool(p4_group.get("vr1_enabled", True), True)
            recent_n = _int(p4_group.get("对比天数", p4_group.get("recent_n", 10)), 10)
            vol_multiple = float(p4_group.get("倍数", p4_group.get("vol_multiple", 2.0)) or 2.0)
            vma_cmp_enabled = _bool(p4_group.get("vma_cmp_enabled", False), False)
            vma_short_days = _int(p4_group.get("短期天数", p4_group.get("vma_short_days", 5)), 5)
            vma_long_days = _int(p4_group.get("长期天数", p4_group.get("vma_long_days", 10)), 10)
            vol_up_enabled = _bool(p4_group.get("vol_up_enabled", False), False)
            vol_increasing_days = _int(p4_group.get("量连升天数", p4_group.get("vol_increasing_days", 3)), 3)
        else:
            # 旧版兼容：条件4=VR1、条件5=量均比较；量连升默认关闭
            g4_enabled = _bool(p4.get("启用", p4.get("enabled", False)), False) or _bool(
                p5.get("启用", p5.get("enabled", False)), False)
            vr1_enabled = _bool(p4.get("启用", p4.get("enabled", False)), False)
            recent_n = _int(p4.get("对比天数", p4.get("J", 10)), 10)
            vol_multiple = float(p4.get("倍数", p4.get("K", 2.0)) or 2.0)
            vma_cmp_enabled = _bool(p5.get("启用", p5.get("enabled", False)), False)
            vma_short_days = _int(p5.get("短期天数", p5.get("D", 5)), 5)
            vma_long_days = _int(p5.get("长期天数", p5.get("F", 10)), 10)
            vol_up_enabled = False
            vol_increasing_days = 3

        # 模块1：VR1 放量（仅启用时计算）
        if vr1_enabled:
            prevXmax = out["volume"].shift(1).rolling(recent_n if recent_n and recent_n > 0 else 1, min_periods=1).max()
            vr1_ok_series = (out["volume"] >= (vol_multiple * prevXmax))
        else:
            prevXmax = pd.Series([np.nan]*len(out), index=out.index)
            vr1_ok_series = pd.Series(True, index=out.index)
        # 模块2：量均比较（仅启用时计算）
        if vma_cmp_enabled:
            vmaD = out["volume"].rolling(vma_short_days if vma_short_days and vma_short_days > 0 else 1,
                                         min_periods=vma_short_days if vma_short_days and vma_short_days > 0 else 1).mean()
            vmaF = out["volume"].rolling(vma_long_days if vma_long_days and vma_long_days > 0 else 1,
                                         min_periods=vma_long_days if vma_long_days and vma_long_days > 0 else 1).mean()
            vma_ok_series = (vmaD > vmaF)
        else:
            vmaD = pd.Series([np.nan]*len(out), index=out.index)
            vmaF = pd.Series([np.nan]*len(out), index=out.index)
            vma_ok_series = pd.Series(True, index=out.index)
        # 模块3：近X日量严格递增（仅启用时计算）
        if vol_up_enabled and vol_increasing_days and vol_increasing_days > 1:
            win = int(max(1, vol_increasing_days - 1))
            inc = out["volume"].diff(1) > 0
            up_ok = inc.rolling(win, min_periods=win).sum() == win
            up_ok = up_ok.fillna(False)
            up_ok_series = up_ok
        else:
            up_ok_series = pd.Series(True, index=out.index)

        # cond4 汇总：仅对启用的子模块取与；若全部子模块关闭，则视为通过
        any_sub_enabled = (vr1_enabled or vma_cmp_enabled or vol_up_enabled)
        if not any_sub_enabled:
            cond4_ok = pd.Series(True, index=out.index)
        else:
            parts = []
            if vr1_enabled: parts.append(vr1_ok_series)
            if vma_cmp_enabled: parts.append(vma_ok_series)
            if vol_up_enabled: parts.append(up_ok_series)
            cond4_ok = parts[0]
            for p in parts[1:]:
                cond4_ok = cond4_ok & p
        # 总开关：保持为 Series 类型
        cond4_ok = (cond4_ok if g4_enabled else pd.Series(True, index=out.index))

        # ---- 综合 A 点（加入 条件4组）----
        A2 = out["A_point"].astype(bool) & cond4_ok
        cnt2 = int(A2.sum())
        print(f"[A_POINTS_V2] A_points_after_vol={cnt2} (cond4_group_enabled={g4_enabled})")

        pts = []
        a_rows = out[A2]
        for _, r in a_rows.iterrows():
            pts.append({
                "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                "price_low": None,
                "price_close": None if pd.isna(r.get("close")) else float(r["close"]),
            })

        # 诊断表：仅输出“基本信息 + 已开启条件/子模块”的相关列
        merged = out.copy()
        lw = c1["long_window"]
        long2 = c2["long_window"]
        shorts = list(c2["short_windows"]) if isinstance(c2.get("short_windows"), tuple) else []
        table = []
        for i, r in merged.iterrows():
            row = {
                "date": pd.to_datetime(r.get("date")).strftime("%Y-%m-%d") if pd.notna(r.get("date")) else None,
                "open": None if pd.isna(r.get("open")) else float(r.get("open")),
                "close": None if pd.isna(r.get("close")) else float(r.get("close")),
                "low": None if pd.isna(r.get("low")) else float(r.get("low")),
                "high": None if pd.isna(r.get("high")) else float(r.get("high")),
                "volume": None if pd.isna(r.get("volume")) else float(r.get("volume")),
            }

            # 条件1：长均线下跌
            if c1["enabled"]:
                row.update({
                    "ma_long_t1": None if pd.isna(r.get("ma_long_t1")) else float(r.get("ma_long_t1")),
                    "ma_long_t1_prev": None if pd.isna(r.get("ma_long_t1_prev")) else float(r.get("ma_long_t1_prev")),
                    "cond1": bool(r.get("cond1_ok")) if pd.notna(r.get("cond1_ok")) else False,
                })

            # 条件2：短均线上穿
            if c2["enabled"]:
                for k in shorts:
                    key = f"ma_{k}"
                    row[key] = None if pd.isna(r.get(key)) else float(r.get(key))
                if long2 is not None:
                    keyL = f"ma_{int(long2)}"
                    row[keyL] = None if pd.isna(r.get(keyL)) else float(r.get(keyL))
                row["cond2"] = bool(r.get("cond2_ok")) if pd.notna(r.get("cond2_ok")) else False

            # 条件3：价格确认
            if c3["enabled"]:
                row.update({
                    "confirm_cross_cnt": None if pd.isna(r.get("confirm_cross_cnt")) else int(r.get("confirm_cross_cnt")),
                    "cond3": bool(r.get("cond3_ok")) if pd.notna(r.get("cond3_ok")) else False,
                })

            # 条件4组：放量确认（仅开启的子模块输出）
            if g4_enabled:
                if vr1_enabled:
                    pv = prevXmax.iat[i] if i < len(prevXmax) else None
                    vol = r.get("volume")
                    vol_ratio = (float(vol) / float(pv)) if (pd.notna(vol) and pd.notna(pv) and float(pv) != 0.0) else None
                    row.update({
                        "prevXmax": None if pv is None or pd.isna(pv) else float(pv),
                        "vol_ratio": None if vol_ratio is None else float(vol_ratio),
                        "VR1": None if pd.isna(r.get("vr1")) else float(r.get("vr1")),
                        "c4_vr1_ok": bool(vr1_ok_series.iat[i]) if i < len(vr1_ok_series) and not pd.isna(vr1_ok_series.iat[i]) else False,
                    })
                if vma_cmp_enabled:
                    vD = vmaD.iat[i] if i < len(vmaD) else None
                    vF = vmaF.iat[i] if i < len(vmaF) else None
                    row.update({
                        "vmaD": None if vD is None or pd.isna(vD) else float(vD),
                        "vmaF": None if vF is None or pd.isna(vF) else float(vF),
                        "c4_vma_ok": bool(vma_ok_series.iat[i]) if i < len(vma_ok_series) and not pd.isna(vma_ok_series.iat[i]) else False,
                    })
                if vol_up_enabled:
                    row.update({
                        "c4_up_ok": bool(up_ok_series.iat[i]) if i < len(up_ok_series) and not pd.isna(up_ok_series.iat[i]) else False,
                    })
                row["cond4"] = bool(cond4_ok.iat[i]) if i < len(cond4_ok) and not pd.isna(cond4_ok.iat[i]) else False

            row["A_point"] = bool(A2.iat[i]) if i < len(A2) and not pd.isna(A2.iat[i]) else False
            table.append(row)

        print(f"[A_POINTS_V2] returning points={len(pts)} table_rows={len(table)}")
        return {"points": pts, "count": len(pts), "table": table}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print("[A_POINTS_V2_ERROR]", type(e).__name__, str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/api/stocks/{symbol}/info")
def get_stock_info_api(symbol: str):
    """
    获取股票的简介信息（调用 services.data_service.get_stock_info），
    将返回的 item/value 两列转换为 {item: value} 的字典。
    """
    try:
        info_df = get_stock_info(symbol)
        if info_df is None or info_df.empty:
            raise HTTPException(status_code=404, detail="Stock info not found for the given symbol.")
        # 期待列名为 item/value
        cols = [c.lower() for c in info_df.columns]
        if len(info_df.columns) >= 2 and ('item' in cols and 'value' in cols):
            # 通过下标而不是名字，兼容大小写
            item_col = info_df.columns[cols.index('item')]
            value_col = info_df.columns[cols.index('value')]
            data = dict(zip(info_df[item_col].astype(str), info_df[value_col].tolist()))
        else:
            # 兼容性兜底：尝试前两列
            item_col = info_df.columns[0]
            value_col = info_df.columns[1] if len(info_df.columns) > 1 else info_df.columns[0]
            data = dict(zip(info_df[item_col].astype(str), info_df[value_col].tolist()))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- 静态页面托管 ----
DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith("api"):
        return {"detail": "Not Found"}
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        try:
            return HTMLResponse(index_file.read_text(encoding='utf-8'))
        except Exception:
            return HTMLResponse("<h1>Frontend index.html 读取失败</h1>", status_code=500)
    return HTMLResponse("<h1>前端未构建，缺少 dist/index.html</h1>", status_code=404)


def _normalize_kline_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure K-line dataframe has expected columns and types, sorted by 日期 ascending.
    Returns a dataframe with at least: 日期, 开盘, 收盘, 最高, 最低, 成交量.
    Unknown columns are preserved but we coerce numeric types.
    """
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    # Potential alternate column names fallback map
    rename_map = {}
    # Accept English fallback
    if 'date' in cols and '日期' not in cols:
        rename_map['date'] = '日期'
    if 'open' in cols and '开盘' not in cols:
        rename_map['open'] = '开盘'
    if 'close' in cols and '收盘' not in cols:
        rename_map['close'] = '收盘'
    if 'high' in cols and '最高' not in cols:
        rename_map['high'] = '最高'
    if 'low' in cols and '最低' not in cols:
        rename_map['low'] = '最低'
    if 'volume' in cols and '成交量' not in cols:
        rename_map['volume'] = '成交量'
    if rename_map:
        df = df.rename(columns=rename_map)
        cols = list(df.columns)

    # Coerce numeric
    for c in ('开盘', '收盘', '最高', '最低', '成交量'):
        if c in cols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # 日期 as datetime string YYYY-MM-DD
    if '日期' in cols:
        df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
        df = df.sort_values('日期').reset_index(drop=True)
        # Keep date as string for frontend readability
        df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
    return df
