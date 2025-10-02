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
                "prevJmax": None if pd.isna(prevJmax.iat[i]) else float(prevJmax.iat[i]) if 'prevJmax' in locals() else None,
                "vr": None if ('prevJmax' not in locals() or pd.isna(prevJmax.iat[i]) or pd.isna(d.at[i, "volume"])) else float(d.at[i, "volume"]) / float(prevJmax.iat[i]) if float(prevJmax.iat[i]) != 0 else None,
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
            "close": pd.to_numeric(df_cn.get("收盘"), errors="coerce"),
            "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
            "volume": pd.to_numeric(df_cn.get("成交量"), errors="coerce"),
        })

        p1 = params.get("条件1_长期下跌", params.get("cond1", {})) or {}
        p2 = params.get("条件2_短均线上穿", params.get("cond2", {})) or {}
        p3 = params.get("条件3_价格确认", params.get("cond3", {})) or {}
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

        # ---- 条件4：当日放量对比（Vol >= K * max(Vol[t-J..t-1])）----
        c4_enabled = _bool(p4.get("启用", p4.get("enabled", False)), False)
        c4_J = _int(p4.get("对比天数", p4.get("J", 10)), 10)
        c4_K = float(p4.get("倍数", p4.get("K", 2.0)) or 2.0)
        prevXmax = out["volume"].shift(1).rolling(c4_J if c4_J and c4_J > 0 else 1, min_periods=1).max()
        cond4_ok = (out["volume"] >= (c4_K * prevXmax)) if c4_enabled else pd.Series(True, index=out.index)

        # ---- 条件5：当日量均线比较（VMA_D >= VMA_F）----
        c5_enabled = _bool(p5.get("启用", p5.get("enabled", False)), False)
        c5_D = _int(p5.get("短期天数", p5.get("D", 5)), 5)
        c5_F = _int(p5.get("长期天数", p5.get("F", 10)), 10)
        vmaD = out["volume"].rolling(c5_D if c5_D and c5_D > 0 else 1, min_periods=c5_D if c5_D and c5_D > 0 else 1).mean()
        vmaF = out["volume"].rolling(c5_F if c5_F and c5_F > 0 else 1, min_periods=c5_F if c5_F and c5_F > 0 else 1).mean()
        cond5_ok = (vmaD >= vmaF) if c5_enabled else pd.Series(True, index=out.index)

        # ---- 综合 A 点（加入 cond4/cond5）----
        A2 = out["A_point"].astype(bool) & cond4_ok & cond5_ok
        cnt2 = int(A2.sum())
        print(f"[A_POINTS_V2] A_points_after_vol={cnt2} (c4_enabled={c4_enabled}, c5_enabled={c5_enabled})")

        pts = []
        a_rows = out[A2]
        for _, r in a_rows.iterrows():
            pts.append({
                "date": pd.to_datetime(r["date"]).strftime("%Y-%m-%d"),
                "price_low": None,
                "price_close": None if pd.isna(r.get("close")) else float(r["close"]),
            })

        merged = out.merge(
            pd.DataFrame({
                "date": pd.to_datetime(df_cn["日期"], errors="coerce"),
                "open": pd.to_numeric(df_cn.get("开盘"), errors="coerce"),
                "low": pd.to_numeric(df_cn.get("最低"), errors="coerce"),
                "high": pd.to_numeric(df_cn.get("最高"), errors="coerce"),
            }),
            on="date", how="left"
        )
        lw = c1["long_window"]
        table = []
        for i, r in merged.iterrows():
            # 补充量能相关列
            pv = prevXmax.iat[i] if i < len(prevXmax) else None
            vol = r.get("volume")
            vol_ratio = (float(vol) / float(pv)) if (pd.notna(vol) and pd.notna(pv) and float(pv) != 0.0) else None
            vD = vmaD.iat[i] if i < len(vmaD) else None
            vF = vmaF.iat[i] if i < len(vmaF) else None
            table.append({
                "date": pd.to_datetime(r.get("date")).strftime("%Y-%m-%d") if pd.notna(r.get("date")) else None,
                "open": None if pd.isna(r.get("open")) else float(r.get("open")),
                "close": None if pd.isna(r.get("close")) else float(r.get("close")),
                "low": None if pd.isna(r.get("low")) else float(r.get("low")),
                "high": None if pd.isna(r.get("high")) else float(r.get("high")),
                "volume": None if pd.isna(r.get("volume")) else float(r.get("volume")),
                "MA_long": None if pd.isna(r.get(f"ma_{lw}")) else float(r.get(f"ma_{lw}")),
                "today_any_cross": bool(r.get("today_any_cross")) if pd.notna(r.get("today_any_cross")) else False,
                "recent_all_cross": bool(r.get("recent_all_cross")) if pd.notna(r.get("recent_all_cross")) else False,
                "today_all_above": bool(r.get("today_all_above")) if pd.notna(r.get("today_all_above")) else False,
                "confirm_cross_cnt": None if pd.isna(r.get("confirm_cross_cnt")) else int(r.get("confirm_cross_cnt")),
                "VR1": None if (pd.isna(r.get("vr1"))) else float(r.get("vr1")),
                "cond1": bool(r.get("cond1_ok")) if pd.notna(r.get("cond1_ok")) else False,
                "cond2": bool(r.get("cond2_ok")) if pd.notna(r.get("cond2_ok")) else False,
                "cond3": bool(r.get("cond3_ok")) if pd.notna(r.get("cond3_ok")) else False,
                "prevXmax": None if pv is None or pd.isna(pv) else float(pv),
                "vol_ratio": None if vol_ratio is None else float(vol_ratio),
                "vmaD": None if vD is None or pd.isna(vD) else float(vD),
                "vmaF": None if vF is None or pd.isna(vF) else float(vF),
                "cond4": bool(cond4_ok.iat[i]) if c4_enabled and i < len(cond4_ok) and not pd.isna(cond4_ok.iat[i]) else (True if not c4_enabled else False),
                "cond5": bool(cond5_ok.iat[i]) if c5_enabled and i < len(cond5_ok) and not pd.isna(cond5_ok.iat[i]) else (True if not c5_enabled else False),
                "A_point": bool(A2.iat[i]) if i < len(A2) and not pd.isna(A2.iat[i]) else False,
            })

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
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Frontend not built. Run: cd frontend && npm install && npm run build</h3>",
                        status_code=200)
