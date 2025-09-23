from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class ETFSpec:
    code: str
    name: str
    weight: float  # percentage 0-100


@dataclass
class ClassSpec:
    id: str
    name: str
    etfs: List[ETFSpec]


def _code_nosfx(x: str) -> str:
    xs = (x or "").strip()
    if "." in xs:
        return xs.split(".")[0]
    return xs


def _load_adj_nav(data_dir: Path) -> pd.DataFrame:
    pq = data_dir / "etf_daily_df.parquet"
    if not pq.exists():
        raise FileNotFoundError("data/etf_daily_df.parquet 不存在")
    df = pd.read_parquet(pq)
    for c in ["ts_code", "name", "date", "adj_nav"]:
        if c not in df.columns:
            raise ValueError(f"parquet 缺少必要列：{c}")
    df = df[["ts_code", "name", "date", "adj_nav"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adj_nav"])  # 保留有效数据
    return df


def _returns_from_adj_nav(series: pd.Series) -> pd.Series:
    r = series.sort_index().pct_change().dropna()
    return r


def _pick_series(df: pd.DataFrame, code: str, name: str) -> Optional[pd.Series]:
    code_ns = _code_nosfx(code)
    ts = df["ts_code"].astype(str)
    nm = df["name"].astype(str)
    sub = df[(ts == code) | (ts.str.split(".").str[0] == code_ns) | (nm == name)]
    if sub.empty and code_ns:
        sub = df[ts.str.contains(code_ns, na=False)]
    if sub.empty:
        return None
    s = sub.groupby("date")["adj_nav"].last().sort_index()
    if s.empty:
        return None
    return s


def compute_classes_nav(
    data_dir: Path,
    classes: List[ClassSpec],
    start_date: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回：(nav_df, corr_df, metrics_df)
    - nav_df: index=日期, columns=大类名称, 值=虚拟净值（起始=1）
    - corr_df: 各大类收益率相关系数矩阵
    - metrics_df: 各大类年化收益、年化波动、夏普
    """
    df = _load_adj_nav(data_dir)

    # 构建每个大类的日收益率（对齐起始日期后共同对齐）
    class_ret: Dict[str, pd.Series] = {}
    for c in classes:
        # 选中资产：需要 weight>0，求归一化权重
        valid = [e for e in c.etfs if e.weight is not None]
        if not valid:
            continue
        w = np.array([max(0.0, float(e.weight)) for e in valid], dtype=float)
        sw = w.sum()
        if sw <= 0:
            continue
        w = w / sw

        # 每个 ETF 的收益序列
        series = []
        for e in valid:
            s = _pick_series(df, e.code, e.name)
            if s is None:
                continue
            r = _returns_from_adj_nav(s)
            series.append(r)
        if not series:
            continue
        # 从 start_date 起裁剪
        series = [s[s.index >= start_date] for s in series]
        # 对齐：内联并填充缺失为 0（也可选择丢弃缺失日，这里用 0 近似）
        ret_df = pd.concat(series, axis=1).fillna(0.0)
        ret_df.columns = [f"a{i}" for i in range(len(ret_df.columns))]
        # 加权求大类收益率
        w_aligned = w[: ret_df.shape[1]]
        agg = (ret_df * w_aligned).sum(axis=1)
        class_ret[c.name] = agg

    if not class_ret:
        raise ValueError("没有可用的大类收益率：请检查权重或数据匹配。")

    # 对齐所有大类的收益率（内联），并据此计算 nav/corr/metrics
    R = pd.DataFrame(class_ret).dropna(how="all")
    R = R.fillna(0.0)
    # nav 从 1 开始
    NAV = (1 + R).cumprod()
    NAV.iloc[0] = 1.0

    # 相关系数
    corr = R.corr()

    # 指标（年化）
    ann_factor = 252.0
    mean = R.mean() * ann_factor
    vol = R.std(ddof=1) * np.sqrt(ann_factor)
    sharpe = mean / vol.replace(0, np.nan)
    metrics = pd.DataFrame({
        "年化收益率": mean,
        "年化波动率": vol,
        "夏普比率": sharpe,
    })
    return NAV, corr, metrics

