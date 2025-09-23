from __future__ import annotations

from typing import List, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from functools import lru_cache
import pandas as pd
from pathlib import Path
import json
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from fit import ClassSpec, ETFSpec, compute_classes_nav, compute_rolling_corr
from fit import compute_rolling_corr_classes


class ETFIn(BaseModel):
    code: str
    name: str
    riskContribution: float = Field(ge=0, description="风险贡献占比 0~100")


class SolveRequest(BaseModel):
    assetClassId: str
    riskMetric: str = "vol"
    maxLeverage: float = 0.0
    etfs: List[ETFIn]


class SolveResponse(BaseModel):
    weights: List[float]


class FitETFIn(BaseModel):
    code: str
    name: str
    weight: float


class FitClassIn(BaseModel):
    id: str
    name: str
    etfs: List[FitETFIn]


class FitRequest(BaseModel):
    startDate: str
    classes: List[FitClassIn]


class FitResponse(BaseModel):
    dates: List[str]
    navs: dict
    corr: List[List[float]]
    corr_labels: List[str]
    metrics: List[dict]


class RollingRequest(BaseModel):
    startDate: str
    window: int = 60
    targetCode: str
    targetName: str
    etfs: List[FitETFIn]


class RollingResponse(BaseModel):
    dates: List[str]
    series: dict
    metrics: List[dict]


app = FastAPI(title="Risk Parity Backend")

# 允许前端本地开发联调（Vite 默认 5173 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True}


def _round2(n: float) -> float:
    return round(n + 1e-8, 2)


@app.post("/api/risk-parity/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    """
    风险平价求解（使用 data/etf_daily_df.parquet 的 adj_nav 复权净值计算日收益 → 协方差矩阵）：
    - 支持目标风险预算（来自前端 riskContribution，占比合计 100）；若未提供则等预算。
    - 非负权重，权重和=1；若 maxLeverage>0，则线性放大到 1+maxLeverage。
    - 目前仅实现基于波动率的风险度量（riskMetric 参数暂不影响计算）。
    """
    import numpy as np

    # 读取并准备收益序列
    pq = DATA_DIR / "etf_daily_df.parquet"
    if not pq.exists():
        # 回退为占比分配
        rc = [max(0.0, float(x.riskContribution)) for x in req.etfs]
        s = sum(rc)
        if s <= 0:
            return SolveResponse(weights=[0.0 for _ in rc])
        base = np.array(rc, dtype=float) / s
        scale = 1.0 + max(0.0, float(req.maxLeverage))
        return SolveResponse(weights=[_round2(float(w * 100 * scale)) for w in base])

    df = pd.read_parquet(pq)
    # 规范列
    for c in ["adj_nav", "ts_code", "name", "date"]:
        if c not in df.columns:
            raise ValueError(f"parquet 缺少必要列：{c}")
    df = df[["ts_code", "name", "date", "adj_nav"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_nav"]).sort_values("date")

    # 根据前端给的 code/name 匹配 ts_code/name
    series_list: List[pd.Series] = []
    labels: List[str] = []
    def _code_nosfx(x: str) -> str:
        xs = (x or "").strip()
        if "." in xs:
            return xs.split(".")[0]
        return xs

    for item in req.etfs:
        code = (item.code or "").strip()
        code_ns = _code_nosfx(code)
        name = (item.name or "").strip()
        ts = df["ts_code"].astype(str)
        nm = df["name"].astype(str)
        sub = df[(ts == code) | (ts.str.split(".").str[0] == code_ns) | (nm == name)]
        if sub.empty and code_ns:
            # 松匹配：code 片段
            sub = df[ts.str.contains(code_ns, na=False)]
        if sub.empty:
            continue
        # 以日期为索引，计算日收益率（使用复权净值）
        s = (
            sub.sort_values("date")["adj_nav"].reset_index(drop=True).pct_change().dropna()
        )
        if s.empty:
            continue
        # 用原始 ts_code 作为列标签，便于对齐
        label = str(sub.iloc[0]["ts_code"])  # 使用 ts_code 作为内部标签
        s.index = range(len(s))  # 统一索引，后续用等长拼接
        series_list.append(s)
        labels.append(label)

    n = len(series_list)
    if n == 0:
        # 回退：预算分配
        rc = [max(0.0, float(x.riskContribution)) for x in req.etfs]
        s = sum(rc)
        if s <= 0:
            return SolveResponse(weights=[0.0 for _ in rc])
        base = np.array(rc, dtype=float) / s
        scale = 1.0 + max(0.0, float(req.maxLeverage))
        return SolveResponse(weights=[_round2(float(w * 100 * scale)) for w in base])

    # 对齐为相同长度：取最短序列长度
    min_len = min(len(s) for s in series_list)
    X = np.column_stack([s.iloc[-min_len:].to_numpy() for s in series_list])
    # 协方差矩阵
    S = np.cov(X, rowvar=False, ddof=1)

    # 目标预算 b
    # 构建预算向量（与 labels 同序），同时兼容 code 无后缀/带后缀；name 作为兜底
    budget_map = {}
    for it in req.etfs:
        c = (it.code or "").strip()
        budget_map[c] = float(it.riskContribution)
        budget_map[_code_nosfx(c)] = float(it.riskContribution)
        budget_map[(it.name or "").strip()] = float(it.riskContribution)
    b = []
    for lab in labels:
        b.append(float(budget_map.get(lab, 0.0) or 0.0))
    b = np.array(b, dtype=float)
    if b.sum() <= 0:
        b = np.ones(n, dtype=float)
    b = b / b.sum()

    # 数值稳定：若协方差半正定，做轻微对角线缩减
    try:
        # 尝试 Cholesky 以检测正定性
        np.linalg.cholesky(S + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        S = S + 1e-6 * np.eye(n)

    # 固定点迭代：w_new ∝ b / (S w)
    eps = 1e-12
    w = b.copy()
    w = np.maximum(w, eps)
    w = w / w.sum()
    for _ in range(5000):
        Sw = S @ w
        denom = np.maximum(Sw, eps)
        w_new = b / denom
        w_new = np.maximum(w_new, eps)
        w_new = w_new / w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < 1e-8:
            w = w_new
            break
        w = 0.7 * w_new + 0.3 * w  # 适度松弛以增强稳定性
    # 计算风险贡献以检验贴合度（非必须）
    Sw = S @ w
    port_var = float(w @ Sw)
    if port_var > 0:
        rc = (w * Sw) / port_var
        # 若偏差很大，做最后一次牛顿修正（可选，略）

    # Map weights back to original input order; unknown assets get 0
    weight_map = {lab: float(wi) for lab, wi in zip(labels, w)}
    out = []
    for it in req.etfs:
        code = (it.code or "").strip()
        key = code if code in weight_map else _code_nosfx(code)
        if key not in weight_map:
            key = (it.name or "").strip()
        out.append(weight_map.get(key, 0.0))

    # 杠杆缩放
    scale = 1.0 + max(0.0, float(req.maxLeverage))
    out = [float(x * 100 * scale) for x in out]
    return SolveResponse(weights=[_round2(x) for x in out])


@app.post("/api/fit-classes", response_model=FitResponse)
def fit_classes(req: FitRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    if not req.classes:
        raise ValueError("classes 不能为空")
    classes = [
        ClassSpec(
            id=c.id,
            name=c.name,
            etfs=[ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in c.etfs],
        )
        for c in req.classes
    ]
    NAV, corr, metrics = compute_classes_nav(DATA_DIR, classes, start)

    def finite_or_none(x: float):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)) and (not (x != x) and abs(x) != float('inf')):
                return float(x)
        except Exception:
            pass
        # 处理 NaN/Inf
        try:
            import math
            if isinstance(x, (int, float)) and (math.isfinite(x)):
                return float(x)
        except Exception:
            pass
        return None

    dates = [d.strftime("%Y-%m-%d") for d in NAV.index]
    navs = {col: [finite_or_none(float(x)) for x in NAV[col].tolist()] for col in NAV.columns}
    corr_labels = list(corr.columns)
    corr_vals = [[finite_or_none(float(v)) or 0.0 for v in row] for row in corr.values.tolist()]
    metrics_out = []
    for name, row in metrics.iterrows():
        metrics_out.append({
            "name": str(name),
            "annual_return": finite_or_none(row.get("年化收益率", None)),
            "annual_vol": finite_or_none(row.get("年化波动率", None)),
            "sharpe": finite_or_none(row.get("夏普比率", None)),
            "var99": finite_or_none(row.get("99%VaR(日)", None)),
            "es99": finite_or_none(row.get("99%ES(日)", None)),
            "max_drawdown": finite_or_none(row.get("最大回撤", None)),
            "calmar": finite_or_none(row.get("卡玛比率", None)),
        })
    return FitResponse(dates=dates, navs=navs, corr=corr_vals, corr_labels=corr_labels, metrics=metrics_out)


@app.post("/api/rolling-corr", response_model=RollingResponse)
def rolling_corr(req: RollingRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    etfs = [ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in req.etfs]
    idx, series_map, metrics = compute_rolling_corr(DATA_DIR, etfs, start, int(req.window), req.targetCode, req.targetName)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    # 清理 NaN/Inf
    safe_series = {k: [float(x) if isinstance(x, (int, float)) and (x == x) and abs(x) != float('inf') else 0.0 for x in v] for k, v in series_map.items()}
    for m in metrics:
        for k in list(m.keys()):
            if k == 'name':
                continue
            v = m[k]
            try:
                if not (isinstance(v, (int, float)) and v == v and abs(v) != float('inf')):
                    m[k] = 0.0
            except Exception:
                m[k] = 0.0
    return RollingResponse(dates=dates, series=safe_series, metrics=metrics)


class RollingClassesRequest(BaseModel):
    startDate: str
    window: int = 60
    targetClassName: str
    classes: List[FitClassIn]


@app.post("/api/rolling-corr-classes", response_model=RollingResponse)
def rolling_corr_classes(req: RollingClassesRequest):
    try:
        start = pd.to_datetime(req.startDate)
    except Exception:
        raise ValueError("startDate 格式错误，应为 YYYY-MM-DD")
    classes = [
        ClassSpec(
            id=c.id,
            name=c.name,
            etfs=[ETFSpec(code=e.code, name=e.name, weight=float(e.weight)) for e in c.etfs],
        )
        for c in req.classes
    ]
    idx, series_map, metrics = compute_rolling_corr_classes(DATA_DIR, classes, start, int(req.window), req.targetClassName)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    safe_series = {k: [float(x) if isinstance(x, (int, float)) and (x == x) and abs(x) != float('inf') else 0.0 for x in v] for k, v in series_map.items()}
    for m in metrics:
        for k in list(m.keys()):
            if k == 'name':
                continue
            v = m[k]
            try:
                if not (isinstance(v, (int, float)) and v == v and abs(v) != float('inf')):
                    m[k] = 0.0
            except Exception:
                m[k] = 0.0
    return RollingResponse(dates=dates, series=safe_series, metrics=metrics)


# -------------------- ETF Universe from data/ --------------------
DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()


def _load_universe() -> List[dict]:
    """Load ETF universe from JSON or Parquet under data/.
    Priority: etf_universe.json -> etf_info_df.parquet -> empty list
    Expected fields: ts_code/code and name
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DATA_DIR / "etf_universe.json"
    pq_path = DATA_DIR / "etf_info_df.parquet"
    if json_path.exists():
        try:
            arr = json.loads(json_path.read_text(encoding="utf-8"))
            out = []
            for x in arr:
                code = x.get("code") or x.get("ts_code")
                name = x.get("name") or x.get("fund_name") or ""
                mgmt = x.get("management") or x.get("manager")
                fd = x.get("found_date") or x.get("foundation_date")
                if code and name:
                    out.append(
                        {
                            "code": str(code),
                            "name": str(name),
                            "management": None if mgmt is None else str(mgmt),
                            "found_date": _normalize_date(fd),
                        }
                    )
            return out
        except Exception:
            pass
    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
            code_col = "ts_code" if "ts_code" in df.columns else ("code" if "code" in df.columns else None)
            name_col = "name" if "name" in df.columns else ("fund_name" if "fund_name" in df.columns else None)
            mgmt_col = "management" if "management" in df.columns else ("manager" if "manager" in df.columns else None)
            fd_col = "found_date" if "found_date" in df.columns else (
                "foundation_date" if "foundation_date" in df.columns else None
            )
            if code_col and name_col:
                cols = [code_col, name_col]
                if mgmt_col:
                    cols.append(mgmt_col)
                if fd_col:
                    cols.append(fd_col)
                out_df = df[cols].dropna(subset=[code_col, name_col]).drop_duplicates()
                items: List[dict] = []
                for _, r in out_df.iterrows():
                    items.append(
                        {
                            "code": str(r[code_col]),
                            "name": str(r[name_col]),
                            "management": None if not mgmt_col else (None if pd.isna(r[mgmt_col]) else str(r[mgmt_col])),
                            "found_date": _normalize_date(None if not fd_col else r[fd_col]),
                        }
                    )
                return items
        except Exception:
            pass
    return []


def _normalize_date(v) -> Optional[str]:
    if v is None:
        return None
    try:
        # handle 20180101 or '2018-01-01'
        s = str(v)
        if s.isdigit() and len(s) == 8:
            return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            return None
        return str(dt.date())
    except Exception:
        return None


@lru_cache(maxsize=1)
def _cached_universe_with_mtime(mtime: float) -> List[dict]:  # noqa: ARG001
    return _load_universe()


def _get_universe() -> List[dict]:
    # Invalidate cache when files change
    mtimes = []
    for fname in ("etf_universe.json", "etf_info_df.parquet"):
        p = DATA_DIR / fname
        if p.exists():
            mtimes.append(p.stat().st_mtime)
    m = max(mtimes) if mtimes else 0.0
    return _cached_universe_with_mtime(m)


@app.get("/api/etf/search")
def etf_search(
    q: Optional[str] = Query(default=""),
    k: Optional[int] = Query(default=None),  # deprecated by page/page_size
    sort_by: str = Query(default="name"),  # one of: name, code, found_date, management
    sort_dir: str = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=200),
):
    arr = _get_universe()
    qnorm = (q or "").strip().lower()
    filtered: List[dict]
    if not qnorm:
        filtered = arr
    else:
        filtered = []
        for x in arr:
            hay = f"{x.get('code','')} {x.get('name','')} {x.get('management','')}".lower()
            if qnorm in hay:
                filtered.append(x)
    reverse = sort_dir.lower() == "desc"
    key = (lambda x: (x.get(sort_by) or "")) if sort_by in {"name", "code", "management", "found_date"} else (lambda x: x.get("name") or "")
    filtered.sort(key=key, reverse=reverse)
    total = len(filtered)
    if k is not None and k > 0:
        # compatibility: take top-k of filtered then apply pagination
        filtered = filtered[:k]
    # pagination
    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]
    return JSONResponse({"items": items, "total": total, "page": page, "page_size": page_size})


# ---- 静态页面托管（可选：将前端构建产物放到 frontend/dist 下） ----
DIST_DIR = (Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    if full_path.startswith("api"):
        return {"detail": "Not Found"}
    index_file = DIST_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>Frontend not built. Run: cd frontend && npm install && npm run build</h3>", status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
