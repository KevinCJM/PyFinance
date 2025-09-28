# -*- encoding: utf-8 -*-
"""
@File: T02_other_tools.py
@Modify Time: 2025/9/16 10:17       
@Author: Kevin-Chen
@Descriptions: 
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple

try:
    from numba import njit, prange, float64, int64, types

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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


if HAS_NUMBA:
    @njit(
        types.UniTuple(float64[:], 4)(float64[:, :], float64[:, :], float64, int64, float64),
        parallel=True, nogil=True, fastmath=True
    )
    def compute_performance_numba(r, w, trading_days, dof, z_score):
        """
        Numba-accelerated function to compute performance metrics for multiple portfolios.
        """
        big_t, big_n = r.shape
        big_m = w.shape[0]

        out_ret = np.empty(big_m, dtype=np.float64)
        out_vol = np.empty(big_m, dtype=np.float64)
        out_sharpe = np.empty(big_m, dtype=np.float64)
        out_var = np.empty(big_m, dtype=np.float64)

        sqrt_td = np.sqrt(trading_days)

        for j in prange(big_m):
            mean = 0.0
            m2 = 0.0
            n = 0
            invalid = False

            for t in range(big_t):
                s = 0.0
                for k in range(big_n):
                    s += r[t, k] * w[j, k]
                if s <= -0.999999999:
                    invalid = True
                    break
                x = np.log1p(s)
                n += 1
                delta = x - mean
                mean += delta / n
                m2 += delta * (x - mean)

            if invalid or n <= 1 or (n - dof) <= 0:
                out_ret[j] = np.nan
                out_vol[j] = np.nan
                out_sharpe[j] = np.nan
                out_var[j] = np.nan
                continue

            var_daily = m2 / (n - dof)
            if var_daily < 0.0:
                var_daily = 0.0

            annual_ret = mean * trading_days
            annual_vol = np.sqrt(var_daily) * sqrt_td

            out_ret[j] = annual_ret
            out_vol[j] = annual_vol

            if annual_vol > 1e-9:
                out_sharpe[j] = annual_ret / annual_vol
            else:
                out_sharpe[j] = np.nan

            out_var[j] = annual_ret - z_score * annual_vol

        return out_ret, out_vol, out_var, out_sharpe
else:
    def compute_performance_numba(r, w, trading_days, dof, z_score):
        raise NotImplementedError("Numba is required for performance calculations.")
