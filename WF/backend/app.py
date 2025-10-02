from __future__ import annotations
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
import difflib
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

# Assuming data_service.py is in the services directory
from services.data_service import get_stock_data, get_stock_info
from services.analysis_service import find_a_points
from starlette.staticfiles import StaticFiles
from starlette.responses import HTMLResponse

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
        df = pd.read_parquet(DATA_FILE) if DATA_FILE.exists() else pd.read_parquet(Path(__file__).parent.parent / "stock_basic.parquet")
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
            return df.to_dict(orient='records')

        # Fallback: fetch using akshare and save parquet inside get_stock_data
        stock_df = get_stock_data(symbol)
        if stock_df is None or stock_df.empty:
            raise HTTPException(status_code=404, detail="Stock data not found for the given symbol.")
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
        return stock_df.to_dict(orient='records')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/stocks/{symbol}/a_points")
def compute_a_points(symbol: str, params: Dict[str, Any]):
    """
    计算并返回 A 点列表。
    参数从 body 传入，映射到 find_a_points 的可选项。
    返回：{"points": [{"date": "YYYY-MM-DD", "price_low": float, "price_close": float}], "count": int}
    """
    try:
        # 读取本地数据，不存在则拉取
        path = DATA_DIR / f"{symbol}.parquet"
        if not path.exists():
            df_cn = get_stock_data(symbol)
        else:
            df_cn = pd.read_parquet(path)
        if df_cn is None or df_cn.empty:
            raise HTTPException(status_code=404, detail="No kline data for symbol")

        # 标准化字段，最小列集合
        df = pd.DataFrame({
            "date": pd.to_datetime(df_cn["日期"]),
            "close": pd.to_numeric(df_cn["收盘"], errors="coerce"),
            "volume": pd.to_numeric(df_cn["成交量"], errors="coerce"),
        })
        df["code"] = str(symbol)

        # 参数映射/默认
        kwargs = dict(
            code_col=params.get("code_col", "code"),
            date_col=params.get("date_col", "date"),
            close_col=params.get("close_col", "close"),
            volume_col=params.get("volume_col", "volume"),
            enable_cond1=bool(params.get("enable_cond1", True)),
            enable_cond2=bool(params.get("enable_cond2", True)),
            enable_cond3=bool(params.get("enable_cond3", True)),
            short_windows=tuple(params.get("short_windows", (5, 10))),
            long_window=int(params.get("long_window", 60)),
            cross_window=int(params.get("cross_window", 3)),
            required_shorts=tuple(params.get("required_shorts")) if params.get("required_shorts") else None,
            require_all=bool(params.get("require_all", True)),
            confirm_lookback_days=int(params.get("confirm_lookback_days", 0)),
            confirm_ma_window=(int(params["confirm_ma_window"]) if params.get("confirm_ma_window") is not None else None),
            confirm_price_col=str(params.get("confirm_price_col", "high")),
            down_lookback=int(params.get("down_lookback", 30)),
            with_explain_strings=bool(params.get("with_explain_strings", True)),
            ma_full_window=bool(params.get("ma_full_window", True)),
            cross_requires_prev_below=bool(params.get("cross_requires_prev_below", True)),
            confirm_min_crosses=int(params.get("confirm_min_crosses", 1)),
            vr1_lookback=int(params.get("vr1_lookback", 10)),
            eps=float(params.get("eps", 0.0)),
        )

        out = find_a_points(df, **kwargs)
        mask = out["A_point"].astype(bool)
        sub = out.loc[mask]
        # 从原中文 df 获取相应价格用于展示
        # 使用日期对齐
        merged = sub[["date"]].merge(df_cn[["日期", "收盘", "最低"]].rename(columns={"日期": "date"}), on="date", how="left")
        pts = []
        for _, r in merged.iterrows():
            pts.append({
                "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                "price_low": None if pd.isna(r.get("最低")) else float(r["最低"]),
                "price_close": None if pd.isna(r.get("收盘")) else float(r["收盘"]),
            })
        return {"points": pts, "count": len(pts)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Frontend not built. Run: cd frontend && npm install && npm run build</h3>",
                        status_code=200)
