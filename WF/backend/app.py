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
from services.tushare_get_data import fetch_stock_daily, fetch_many_daily, get_last_trading_day, fetch_stock_basic_and_save
from services.analysis_service import find_a_points, find_b_points, find_c_points
from opt.a_core import compute_a_core_single_code
from opt.numba_accel import HAS_NUMBA
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, FileResponse
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

def _abc_worker(symbol: str, a_params: Dict[str, Any], b_params: Dict[str, Any], c_params: Dict[str, Any], b_recent_days: int = 5) -> Dict[str, Any]:
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

    t0 = pd.Timestamp.utcnow()
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

    # 近 N 天是否有 B 点
    b_recent = False
    last_b_date: Optional[str] = None
    try:
        if b_dates:
            # 记录最后一个 B 点日期（用于前端展示）
            try:
                last_b_date = max(pd.to_datetime(ds, errors='coerce') for ds in b_dates if pd.to_datetime(ds, errors='coerce') is not pd.NaT)
                if isinstance(last_b_date, pd.Timestamp):
                    last_b_date = last_b_date.strftime('%Y-%m-%d')
            except Exception:
                last_b_date = None
            today = pd.Timestamp.today().normalize()
            ndays = int(b_recent_days) if isinstance(b_recent_days, (int, float)) else 5
            ndays = max(1, ndays)
            thresh = today - pd.Timedelta(days=ndays)
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

    # 查找最后一个 B 点之后的首/末 C 点
    first_c_after_last_b: Optional[str] = None
    last_c_after_last_b: Optional[str] = None
    if last_b_date:
        try:
            c_pts = c_res.get('points', []) or []
            c_dates = [p.get('date') for p in c_pts if p.get('date')]
            # 筛选出在 last_b_date 之后的所有 C 点日期
            c_dates_after_b = sorted([d for d in c_dates if d > last_b_date])
            if c_dates_after_b:
                first_c_after_last_b = c_dates_after_b[0]
                last_c_after_last_b = c_dates_after_b[-1]
        except Exception:
            pass  # 忽略日期转换或比较中的任何错误

    elapsed = (pd.Timestamp.utcnow() - t0).total_seconds() * 1000.0
    return {
        'symbol': symbol,
        'a_count': int(len(a_pts)),
        'b_count': int(len(b_pts)),
        'c_count': int(len(c_res.get('points', []) or [])),
        'b_recent': bool(b_recent),
        'last_b_date': last_b_date,
        'first_c_after_last_b': first_c_after_last_b,
        'last_c_after_last_b': last_c_after_last_b,
        'elapsed_ms': int(elapsed),
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
        industry_map = {str(r['symbol']): r.get('industry') for _, r in base.iterrows()}

        with JOBS_LOCK:
            job['total'] = len(symbols)
        a_params = job.get('a_params', {}) or {}
        b_params = job.get('b_params', {}) or {}
        c_params = job.get('c_params', {}) or {}
        b_recent_days = int(job.get('b_recent_days', 5) or 5)

        # 读取并发设置；默认 CPU-1，范围 [1, CPU]
        req_workers = None
        with JOBS_LOCK:
            req_workers = JOBS.get(job_id, {}).get('max_workers', None)
        cpu_n = max(1, multiprocessing.cpu_count())
        if isinstance(req_workers, int) and req_workers > 0:
            max_workers = max(1, min(cpu_n, req_workers))
        else:
            max_workers = max(1, cpu_n - 1)
        # 记录启动信息
        _log(job, f"启动ABC批量：目标股票数={len(symbols)} 并发进程={max_workers} 近B点统计天数={b_recent_days}")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_abc_worker, sym, a_params, b_params, c_params, b_recent_days) for sym in symbols]
            for fut in as_completed(futures):
                # 支持取消：检测标志位并尝试取消未开始的任务
                with JOBS_LOCK:
                    job_now = JOBS.get(job_id)
                    cancel_requested = bool(job_now.get('cancel')) if job_now else False
                if cancel_requested:
                    try:
                        ex.shutdown(cancel_futures=True)
                    except Exception:
                        pass
                    with JOBS_LOCK:
                        job = JOBS.get(job_id)
                        if job:
                            job['status'] = 'cancelled'
                            job['finished_at'] = datetime.datetime.now().isoformat()
                    _log(job, "收到停止指令，已取消剩余未开始任务")
                    return
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
                                'industry': industry_map.get(sym, ''),
                                'market': market_map.get(sym, ''),
                                'last_b_date': res.get('last_b_date') or '',
                                'first_c_date': res.get('first_c_after_last_b') or '',
                                'last_c_date': res.get('last_c_after_last_b') or '',
                            }
                            job.setdefault('b_recent', [])
                            job['b_recent'].append(entry)
                    _log(job, f"完成 {sym} A={res.get('a_count')} B={res.get('b_count')} C={res.get('c_count')} 耗时={res.get('elapsed_ms')}ms")
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
            # 同时准备 ts_code 映射，优先使用 tushare 的 ts_code 减少市场猜测
            ts_map = {}
            if 'ts_code' in df.columns:
                for _, r in df[['symbol','ts_code']].dropna().iterrows():
                    ts_map[str(r['symbol'])] = str(r['ts_code'])
        elif '代码' in df.columns:
            symbols = [str(s) for s in df['代码'].tolist() if pd.notna(s)]
            ts_map = {}
        else:
            raise RuntimeError("股票列表缺少 symbol 列")
        with JOBS_LOCK:
            job['total'] = len(symbols)

        # 从 job 读取并发、限频、增量参数
        with JOBS_LOCK:
            max_workers = int(job.get('max_workers', 4) or 4)
            max_cps = float(job.get('max_calls_per_second', 8.0) or 8.0)
            resume = bool(job.get('resume', True))
            force = bool(job.get('force', False))
            start_date = job.get('start_date', None)
            end_date = job.get('end_date', None)

        # 日志：参数与交易日
        try:
            last_trading = get_last_trading_day()
            _log(job, f"启动获取：目标股票数={len(symbols)} 并发线程={max_workers} 限频CPS={max_cps} resume={resume} force={force} 最后交易日={last_trading}")
        except Exception:
            _log(job, f"启动获取：目标股票数={len(symbols)} 并发线程={max_workers} 限频CPS={max_cps} resume={resume} force={force}")

        def _on_progress(res):
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    return
                job['done'] += 1
                if res.ok:
                    job['success'] += 1
                else:
                    job['fail'] += 1
                    job.setdefault('failed', []).append(res.code)
                job['last_symbol'] = res.code
                action = getattr(res, 'action', None)
            _log(job, f"完成 {res.code} ({'ok' if res.ok else 'fail'}) action={action}")

        fetch_many_daily(
            symbols,
            start_date=start_date,
            end_date=end_date,
            max_workers=max_workers,
            max_calls_per_second=max_cps,
            resume=resume,
            force=force,
            on_progress=_on_progress,
            codes_ts_map=ts_map,
        )
        with JOBS_LOCK:
            job['status'] = 'finished'
            job['finished_at'] = datetime.datetime.now().isoformat()
        _log(job, f"全量获取完成：done={job['done']} success={job['success']} fail={job['fail']}")
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
        'type': 'fetch_all',
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
            JOBS[job_id]['market'] = str(mkt).strip()        # 可选参数：并发与限频/增量
        if isinstance(payload, dict):
            if 'max_workers' in payload:
                JOBS[job_id]['max_workers'] = int(payload.get('max_workers') or 4)
            if 'max_calls_per_second' in payload:
                JOBS[job_id]['max_calls_per_second'] = float(payload.get('max_calls_per_second') or 8.0)
            if 'resume' in payload:
                JOBS[job_id]['resume'] = bool(payload.get('resume'))
            if 'force' in payload:
                JOBS[job_id]['force'] = bool(payload.get('force'))
            if 'start_date' in payload:
                JOBS[job_id]['start_date'] = payload.get('start_date')
            if 'end_date' in payload:
                JOBS[job_id]['end_date'] = payload.get('end_date')
    t = threading.Thread(target=_run_fetch_all, args=(job_id, sleep_max), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.get("/api/fetch_all/{job_id}/status")
def fetch_all_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            # 返回未知状态，避免 404 噪音；前端据此停止旧任务轮询
            return {"job_id": job_id, "status": "unknown"}
        # 返回浅拷贝，避免并发修改
        return dict(job)


@app.get("/api/fetch_all/active")
def fetch_all_active():
    """Return the latest running fetch_all job if any."""
    with JOBS_LOCK:
        items = [j for j in JOBS.values() if j.get('type') == 'fetch_all' and j.get('status') == 'running']
    if not items:
        return {'job_id': None}
    # pick the latest by started_at
    def _ts(j):
        try:
            return datetime.datetime.fromisoformat(j.get('started_at') or '')
        except Exception:
            return datetime.datetime.min
    items.sort(key=_ts, reverse=True)
    job = dict(items[0])
    return job


# ---- ABC 批量分析 API ----

@app.post("/api/abc_batch/start")
def start_abc_batch(params: Dict[str, Any]):
    job_id = uuid.uuid4().hex[:8]
    job = {
        'job_id': job_id,
        'type': 'abc_batch',
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
        # 存储符合筛选条件的股票结果
        'results': [],
        # 取消标志位
        'cancel': False,
        # 记录参数以传递给子进程
        'a_params': params.get('a_params', {}),
        'b_params': params.get('b_params', {}),
        'c_params': params.get('c_params', {}),
        'max_workers': params.get('max_workers', None),
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
        # 筛选参数：近期B点/C点
        JOBS[job_id]['filter_params'] = params.get('filter_params', {'point_type': 'B', 'days': 5})

    t = threading.Thread(target=_run_abc_batch, args=(job_id,), daemon=True)
    t.start()
    return {'job_id': job_id}


@app.get("/api/abc_batch/{job_id}/status")
def abc_batch_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {'job_id': job_id, 'status': 'unknown'}
        return dict(job)


@app.get("/api/abc_batch/active")
def abc_batch_active():
    """Return the latest running abc_batch job if any."""
    with JOBS_LOCK:
        items = [j for j in JOBS.values() if j.get('type') == 'abc_batch' and j.get('status') == 'running']
    if not items:
        return {'job_id': None}
    def _ts(j):
        try:
            return datetime.datetime.fromisoformat(j.get('started_at') or '')
        except Exception:
            return datetime.datetime.min
    items.sort(key=_ts, reverse=True)
    return dict(items[0])


@app.post("/api/abc_batch/{job_id}/stop")
def abc_batch_stop(job_id: str):
    """Request to stop a running abc_batch job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return {'job_id': job_id, 'status': 'unknown'}
        if job.get('status') != 'running':
            # 非运行状态，直接返回当前状态
            return dict(job)
        job['cancel'] = True
        job['status'] = 'canceling'
    _log(job, "用户请求停止，任务进入取消流程…")
    return {'job_id': job_id, 'status': 'canceling'}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/stocks_basic/refresh")
def refresh_stock_basic(list_status: str = 'L'):
    """Fetch stock basic list via Tushare and save to data/stock_basic.parquet."""
    try:
        df = fetch_stock_basic_and_save(list_status=list_status)
        return {"count": int(len(df))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新股票基础信息失败: {e}")


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

        include_table = bool(params.get('include_table', True))
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
            if include_table:
                table.append(row)

        if include_table:
            return {"points": pts, "count": len(pts), "table": table}
        else:
            return {"points": pts, "count": len(pts)}
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
        if HAS_NUMBA:
            vol_np = d_a["volume"].to_numpy(dtype=float)
            try:
                from opt.numba_accel import rolling_max_prev as _rmp
                prev_max = _rmp(vol_np, int(recent_max_win))
                import numpy as _np
                with _np.errstate(divide='ignore', invalid='ignore'):
                    d_a["vr1"] = vol_np / prev_max
            except Exception:
                prev_max = d_a.groupby("code")["volume"].shift(1).rolling(recent_max_win, min_periods=1).max()
                d_a["vr1"] = d_a["volume"] / prev_max
        else:
            prev_max = d_a.groupby("code")["volume"].shift(1).rolling(recent_max_win, min_periods=1).max()
            d_a["vr1"] = d_a["volume"] / prev_max

        # 解析 B 条件参数
        cond1 = params.get("cond1", {}) or {}
        cond2 = params.get("cond2", {}) or {}
        cond3 = params.get("cond3", {}) or {}
        cond4 = params.get("cond4", {}) or {}
        cond5 = params.get("cond5", {}) or {}
        cond6 = params.get("cond6", {}) or {}
        # 兼容参数名/取值：前端不同页面可能传入 long_ma_days；这里统一为 long_ma_window
        if isinstance(cond2, dict):
            if 'long_ma_window' not in cond2 and 'long_ma_days' in cond2:
                try:
                    cond2['long_ma_window'] = int(cond2.pop('long_ma_days'))
                except Exception:
                    cond2['long_ma_window'] = cond2.pop('long_ma_days', 60)
            # above_maN_ratio 若以百分比传入（>1），转为 0-1 比例
            try:
                if 'above_maN_ratio' in cond2 and cond2['above_maN_ratio'] is not None:
                    v = float(cond2['above_maN_ratio'])
                    if v > 1.0 and v <= 100.0:
                        cond2['above_maN_ratio'] = v / 100.0
            except Exception:
                pass

        # cond5 兼容：允许前端传 short_days/long_days，则转为 vma_short_window/vma_long_window
        if isinstance(cond5, dict):
            if 'vma_short_window' not in cond5 and 'short_days' in cond5:
                try:
                    cond5['vma_short_window'] = int(cond5.get('short_days'))
                except Exception:
                    pass
            if 'vma_long_window' not in cond5 and 'long_days' in cond5:
                try:
                    cond5['vma_long_window'] = int(cond5.get('long_days'))
                except Exception:
                    pass

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
        pts_df = out.loc[bmask, ['date','low','close']].copy()
        pts_df['date'] = pd.to_datetime(pts_df['date']).dt.strftime('%Y-%m-%d')
        pts = [{ 'date': r['date'], 'price_low': (None if pd.isna(r['low']) else float(r['low'])), 'price_close': (None if pd.isna(r['close']) else float(r['close'])) } for _, r in pts_df.iterrows()]

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

        include_table = bool(params.get('include_table', True))
        compact = bool(params.get('compact', False))
        if not include_table:
            return {"points": pts, "count": len(pts)}
        df_tbl = out.copy()
        df_tbl['date'] = pd.to_datetime(df_tbl['date']).dt.strftime('%Y-%m-%d')
        base_cols = ['date','open','close','low','high','volume']
        cols = base_cols[:]
        if c1_enabled:
            for c in ['days_since_A','cond1_ok']:
                if c in df_tbl.columns: cols.append(c)
        if c2_enabled:
            for c in ['cond2_ok','vol_ratio_vs_prevNmax','c2_vr1_ok','vma_short','vma_long','c2_vma_ok','c2_up_ok']:
                if c in df_tbl.columns: cols.append(c)
        if c3_enabled:
            for c in ['ma_long_b','cond3_ok','cond3_bearish_ok','cond3_close_le_prev_ok','cond3_touch_ok']:
                if c in df_tbl.columns: cols.append(c)
        for c in ['cond4_ok','cond5_ok','cond6_ok','B_point']:
            if c in df_tbl.columns: cols.append(c)
        if compact:
            keep = ['date','open','high','low','close','volume','B_point']
            for c in ['cond1_ok','cond2_ok','cond3_ok','cond4_ok','cond5_ok','cond6_ok']:
                if c in df_tbl.columns: keep.append(c)
            cols = [c for c in keep if c in df_tbl.columns]
        table = df_tbl[cols].replace({np.nan: None}).to_dict(orient='records')
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
        pts_df = out.loc[cmask, ['date','low','close']].copy()
        pts_df['date'] = pd.to_datetime(pts_df['date']).dt.strftime('%Y-%m-%d')
        pts = [{'date': r['date'], 'price_low': (None if pd.isna(r['low']) else float(r['low'])), 'price_close': (None if pd.isna(r['close']) else float(r['close']))} for _, r in pts_df.iterrows()]

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

        include_table = bool(params.get('include_table', True))
        compact = bool(params.get('compact', False))
        if not include_table:
            return {"points": pts, "count": len(pts)}
        df_tbl = out.copy()
        df_tbl['date'] = pd.to_datetime(df_tbl['date']).dt.strftime('%Y-%m-%d')
        base_cols = ['date','open','close','low','high','volume']
        cols = base_cols[:]
        if c1_enabled:
            for c in ['days_since_B','cond1_ok']:
                if c in df_tbl.columns: cols.append(c)
        if c2_enabled:
            for c in ['cond2_ok','vol_ratio_vs_prevNmax','c2_vr1_ok','vma_short','vma_long','c2_vma_ok','c2_up_ok']:
                if c in df_tbl.columns: cols.append(c)
        if c3_enabled:
            for c in ['maY','cond3_ok']:
                if c in df_tbl.columns: cols.append(c)
        for c in ['C_point']:
            if c in df_tbl.columns: cols.append(c)
        if compact:
            keep = ['date','open','high','low','close','volume','C_point']
            for c in ['cond1_ok','cond2_ok','cond3_ok']:
                if c in df_tbl.columns: keep.append(c)
            cols = [c for c in keep if c in df_tbl.columns]
        table = df_tbl[cols].replace({np.nan: None}).to_dict(orient='records')
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

        if HAS_NUMBA:
            out = compute_a_core_single_code(d, c1, c2, c3, vr1_lookback=10, eps=0.0)
            cnt = int(out["A_point"].astype(bool).sum())
            print(f"[A_POINTS_V2] computed rows={len(out)} A_points_base={cnt} (accelerated)")
        else:
            out = find_a_points(
                d,
                code_col="code", date_col="date", close_col="close", volume_col="volume",
                with_explain_strings=False,
                cond1=c1, cond2=c2, cond3=c3,
            )
            cnt = int(out["A_point"].astype(bool).sum())
            print(f"[A_POINTS_V2] computed rows={len(out)} A_points_base={cnt} (fallback)")

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

        # points（向量化）
        pts_df = out.loc[A2, ['date', 'close']].copy()
        pts_df['date'] = pd.to_datetime(pts_df['date']).dt.strftime('%Y-%m-%d')
        pts = [{ 'date': r['date'], 'price_low': None, 'price_close': (None if pd.isna(r['close']) else float(r['close'])) } for _, r in pts_df.iterrows()]

        # include_table/compact 参数，裁剪列
        include_table = bool(params.get('include_table', True))
        compact = bool(params.get('compact', False))

        if not include_table:
            print(f"[A_POINTS_V2] returning points={len(pts)} (no table)")
            return {"points": pts, "count": len(pts)}

        df_tbl = out.copy()
        df_tbl['date'] = pd.to_datetime(df_tbl['date']).dt.strftime('%Y-%m-%d')
        base_cols = ['date', 'open', 'close', 'low', 'high', 'volume']
        cols = base_cols[:]
        # cond1
        if c1.get('enabled', True):
            cols += [c for c in ['ma_long_t1', 'ma_long_t1_prev', 'cond1_ok'] if c in df_tbl.columns]
        # cond2
        if c2.get('enabled', True):
            long2 = c2.get('long_window')
            shorts = list(c2.get('short_windows') or [])
            for k in shorts:
                mk = f'ma_{k}'
                if mk in df_tbl.columns: cols.append(mk)
            if long2 is not None and f'ma_{int(long2)}' in df_tbl.columns:
                cols.append(f'ma_{int(long2)}')
            if 'cond2_ok' in df_tbl.columns: cols.append('cond2_ok')
        # cond3
        if c3.get('enabled', True):
            for c in ['confirm_cross_cnt', 'cond3_ok']:
                if c in df_tbl.columns: cols.append(c)
        # cond4 group
        if g4_enabled:
            if vr1_enabled:
                for c in ['vr1']:
                    if c in df_tbl.columns: cols.append(c)
                df_tbl['prevXmax'] = prevXmax
                df_tbl['vol_ratio'] = df_tbl['volume'] / prevXmax.replace(0, np.nan)
                cols += ['prevXmax', 'vol_ratio']
                if 'c4_vr1_ok' not in df_tbl.columns:
                    df_tbl['c4_vr1_ok'] = vr1_ok_series.astype(bool)
                cols.append('c4_vr1_ok')
            if vma_cmp_enabled:
                df_tbl['vmaD'] = vmaD
                df_tbl['vmaF'] = vmaF
                df_tbl['c4_vma_ok'] = vma_ok_series.astype(bool)
                cols += ['vmaD', 'vmaF', 'c4_vma_ok']
            if vol_up_enabled:
                df_tbl['c4_up_ok'] = up_ok_series.astype(bool)
                cols += ['c4_up_ok']
            df_tbl['cond4'] = cond4_ok.astype(bool)
            cols += ['cond4']

        df_tbl['A_point'] = A2.astype(bool)
        cols += ['A_point']

        if compact:
            keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'A_point']
            for k in ['cond1_ok', 'cond2_ok', 'cond3_ok', 'cond4']:
                if k in df_tbl.columns: keep.append(k)
            cols = [c for c in keep if c in df_tbl.columns]

        table = df_tbl[cols].replace({np.nan: None}).to_dict(orient='records')
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
    # 仅挂载静态资源目录，SPA 路由由下方 fallback 处理
    assets_dir = DIST_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    # Vite 的图标
    vite_svg = DIST_DIR / "vite.svg"
    if vite_svg.exists():
        @app.get("/vite.svg")
        def vite_svg_file():
            return FileResponse(str(vite_svg))


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
