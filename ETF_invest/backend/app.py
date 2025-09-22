from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from pathlib import Path


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
