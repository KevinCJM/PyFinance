from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from functools import lru_cache
import pandas as pd
from pathlib import Path
import json
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles


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
    简化版风险平价：
    - 假设标的风险同权且低相关，权重≈风险贡献占比；
    - 若允许杠杆（maxLeverage>0），线性放大总资金权重：scale=1+maxLeverage。
    """
    rc = [max(0.0, float(x.riskContribution)) for x in req.etfs]
    s = sum(rc)
    if s <= 0:
        return SolveResponse(weights=[0.0 for _ in rc])
    base = [100.0 * x / s for x in rc]
    scale = 1.0 + max(0.0, float(req.maxLeverage))
    weights = [_round2(b * scale) for b in base]
    return SolveResponse(weights=weights)


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
                if code and name:
                    out.append({"code": str(code), "name": str(name)})
            return out
        except Exception:
            pass
    if pq_path.exists():
        try:
            df = pd.read_parquet(pq_path)
            code_col = "ts_code" if "ts_code" in df.columns else ("code" if "code" in df.columns else None)
            name_col = "name" if "name" in df.columns else ("fund_name" if "fund_name" in df.columns else None)
            if code_col and name_col:
                out_df = df[[code_col, name_col]].dropna().drop_duplicates()
                return [
                    {"code": str(c), "name": str(n)} for c, n in zip(out_df[code_col].tolist(), out_df[name_col].tolist())
                ]
        except Exception:
            pass
    return []


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
def etf_search(q: Optional[str] = Query(default=""), k: int = Query(default=10, ge=1, le=200)):
    arr = _get_universe()
    qnorm = (q or "").strip().lower()
    if not qnorm:
        return JSONResponse(arr[:k])
    scored = []
    for x in arr:
        hay = f"{x.get('code','')} {x.get('name','')}".lower()
        idx = hay.find(qnorm)
        if idx >= 0:
            score = idx + len(qnorm) * 0.2
            scored.append((score, x))
    scored.sort(key=lambda t: t[0])
    return JSONResponse([x for _, x in scored[:k]])


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
