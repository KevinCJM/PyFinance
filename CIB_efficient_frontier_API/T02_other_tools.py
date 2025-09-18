# -*- encoding: utf-8 -*-
"""
@File: T02_other_tools.py
@Modify Time: 2025/9/16 10:17       
@Author: Kevin-Chen
@Descriptions: 
"""

import time
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from numba import njit, prange

    HAS_NUMBA = True
    print("Numba 可用，启用 JIT 加速。")
except Exception:
    HAS_NUMBA = False
    print("Numba 不可用，使用纯 Python 版本。")

import numpy as np
import pandas as pd


# 返回当前时间戳字符串
def _ts() -> str:
    """返回当前时间戳字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 标准化日志输出，带时间戳
def log(msg: str) -> None:
    """标准化日志输出，带时间戳。"""
    print(f"[{_ts()}] {msg}")


# 读取excel的数据
def load_returns_from_excel(
        excel_path: str,
        sheet_name: str,
        assets_list: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """从 Excel 读取净值数据，生成日收益二维数组 (T,N)。"""
    log(f"加载数据: {excel_path} | sheet={sheet_name}")
    df = read_excel_compat(excel_path, sheet_name)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True)
    df = df.rename(
        {
            "货基指数": "货币现金类",
            "固收类": "固定收益类",
            "混合类": "混合策略类",
            "权益类": "权益投资类",
            "另类": "另类投资类",
        },
        axis=1,
    )

    # 若部分资产列不存在，报错以避免静默问题
    missing = [c for c in assets_list if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    hist_ret_df = df[assets_list].pct_change().dropna()
    # 全局采用 float32，以获得更好的吞吐（矩阵乘法/缓存）
    arr = hist_ret_df.values.astype(np.float32, copy=False)
    log(f"数据加载完成，样本天数={arr.shape[0]}，资产数={arr.shape[1]}")
    return arr, assets_list


def ann_log_return(
        returns_daily: np.ndarray,
        w: np.ndarray,
        trading_days: float = 252.0,
) -> float:
    """计算组合的年化对数收益率。

    参数:
        returns_daily: (T, N) 日简单收益矩阵
        w: (N,) 组合权重
        trading_days: 年化天数（默认 252）
    公式:
        μ_annual_log = mean(log(1 + R_t)) * trading_days
        其中 R_t = returns_daily @ w
    """
    Rt = returns_daily @ np.asarray(w, dtype=np.float64)
    Xt = np.log1p(Rt)
    return float(Xt.mean()) * float(trading_days)


def ann_log_vol(
        returns_daily: np.ndarray,
        w: np.ndarray,
        trading_days: float = 252.0,
        ddof: int = 1,
) -> float:
    """计算组合年化对数收益波动率。

    参数:
        returns_daily: (T, N) 日简单收益矩阵
        w: (N,) 组合权重
        trading_days: 年化天数（默认 252）
        ddof: 样本标准差自由度（默认 1）
    公式:
        σ_annual = std(log(1 + R_t), ddof) * sqrt(trading_days)
        其中 R_t = returns_daily @ w
    """
    Rt = returns_daily @ np.asarray(w, dtype=np.float64)
    Xt = np.log1p(Rt)
    return float(Xt.std(ddof=int(ddof))) * float(np.sqrt(trading_days))
# 读取 Excel，兼容老环境引擎选择
def read_excel_compat(excel_path: str, sheet_name: str) -> pd.DataFrame:
    """兼容性读取 Excel：
    - .xlsx/.xlsm/.xltx/.xltm 优先使用 engine='openpyxl'（xlrd 新版不支持 xlsx）。
    - .xls 使用 engine='xlrd'。
    - 若引擎缺失，给出清晰错误提示。
    """
    ext = os.path.splitext(str(excel_path))[-1].lower()
    try:
        if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
            return pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
        elif ext == ".xls":
            return pd.read_excel(excel_path, sheet_name=sheet_name, engine="xlrd")
        else:
            # 其他扩展名，尝试默认
            return pd.read_excel(excel_path, sheet_name=sheet_name)
    except ImportError as e:
        raise ImportError(
            "读取 Excel 失败，缺少所需引擎。对于 .xlsx 文件请安装 openpyxl (建议 openpyxl==2.6.4 以兼容 Py3.6)，"
            "对于 .xls 文件请安装 xlrd<2.0。原始错误: %s" % str(e)
        )
    except Exception:
        # 再做一次回退尝试：不指定引擎，让 pandas 自行选择
        try:
            return pd.read_excel(excel_path, sheet_name=sheet_name)
        except Exception as e2:
            raise
