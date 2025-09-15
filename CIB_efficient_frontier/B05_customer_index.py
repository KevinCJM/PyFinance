# -*- encoding: utf-8 -*-
"""
@File: B05_customer_index.py
@Modify Time: 2025/9/11 18:34
@Author: Kevin-Chen
@Descriptions:
  1) 用 QCQP 逐风险扫描，刻准有效前沿（带线性约束与多资产联合约束）
  2) 以前沿锚点为种子，小步随机游走 + POCS 投影，填充前沿之下的可行空间
  3) 支持生成样本的“权重精度”可选（0.1% / 0.2% / 0.5%），并去重
  4) 批量计算绩效，并作图与 Excel 导出（全局 + C1~C6 多 Sheet）
"""
import time
import numpy as np
import cvxpy as cp
import pandas as pd
from enum import Enum
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Literal

''' ========= 工具：加载 & 基础统计 =========
'''


# 从Excel文件中加载资产净值数据，计算日收益率、年化收益和协方差矩阵。
def load_returns_from_excel(
        filepath: str,
        sheet_name: str,
        assets_list: List[str],
        rename_map: Dict[str, str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    从Excel文件中加载资产净值数据，计算日收益率、年化收益和协方差矩阵。

    参数:
        filepath (str): Excel文件路径
        sheet_name (str): Excel工作表名称
        assets_list (List[str]): 资产名称列表
        rename_map (Dict[str, str]): 列名重映射字典

    返回:
        tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
            - 原始净值数据DataFrame
            - 日收益率数组
            - 年化收益率数组
            - 年化协方差矩阵
    """
    print(f"{str_time()} [加载数据] 读取excel文件数据...")
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df = df.set_index('date')
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index(ascending=True).rename(rename_map, axis=1)

    print(f"{str_time()} [加载数据] 计算日涨跌收益率...")
    hist_value_r = df[assets_list].pct_change().dropna()
    daily_returns = hist_value_r.values

    print(f"{str_time()} [加载数据] 将日收益率转换为对数收益...")
    log_r = np.log1p(daily_returns)

    print(f"{str_time()} [加载数据] 计算年化收益与协方差...")
    miu, cov = ann_mu_sigma(log_r)
    return df, daily_returns, miu, cov


# 计算年化的期望收益率和协方差矩阵
def ann_mu_sigma(log_returns: np.ndarray):
    """
    计算年化的期望收益率和协方差矩阵

    参数:
        log_returns (np.ndarray): 对数收益率数组，每一行代表一个时间点，每一列代表一个资产

    返回:
        tuple: 包含两个元素的元组
            - miu (np.ndarray): 年化的期望收益率向量
            - cov (np.ndarray): 年化的协方差矩阵
    """
    # 计算年化的期望收益率，假设252个交易日
    miu = log_returns.mean(axis=0) * 252.0
    # 计算年化的协方差矩阵，使用样本协方差(ddof=1)，假设252个交易日
    cov = np.cov(log_returns, rowvar=False, ddof=1) * 252.0
    return miu, cov


# 获取当前本地时间的字符串表示
def str_time():
    """
    获取当前本地时间的字符串表示

    Returns:
        str: 格式为 "YYYY-MM-DD HH:MM:SS" 的时间字符串
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


''' ========= 指标规格（统一口径） ========= 
'''

TRADING_DAYS = 252.0

ReturnType = Literal[
    "total_cum",  # 总累计收益率: Π(1+R)-1
    "total_geom_annual",  # 总年化复利收益: (Π(1+R))**(252/T)-1
    "total_log",  # 总对数收益: Σ log(1+R)
    "total_log_annual",  # 总年化对数收益: (Σlog(1+R))/T*252
    "ew_roll_cum",  # 指数加权滚动N区间累计收益率
    "ew_roll_log",  # 指数加权滚动N区间对数收益率
    "ew_roll_mean_simple",  # 指数加权普通收益均值
    "mean_simple",  # 普通收益率均值
    "mean_log"  # 对数收益率均值
]

RiskType = Literal[
    "std",  # 总日普通收益标准差
    "std_annual",  # 总日普通收益年化标准差
    "log_std",  # 总日对数收益标准差
    "log_std_annual",  # 总日对数收益年化标准差
    "ew_roll_std",  # 指数加权滚动N区间普通收益率标准差
    "ew_roll_log_std",  # 指数加权滚动N区间对数收益率标准差
    "var_abs",  # x% VaR(取绝对值, 无亏损时=0)
    "es_abs",  # x% ES (取绝对值, 无亏损时=0)
    "max_drawdown_abs"  # 最大回撤率(取绝对值, 无回撤时=0)
]


@dataclass
class ReturnSpec:
    """
    返回规范类，用于定义返回值的类型和相关参数。

    Attributes:
        kind (ReturnType): 返回类型枚举值
        N (int | None): 整数参数，可选
        lam (float | None): 浮点参数，范围在0到1之间，值越大越强调近端，可选
    """
    kind: ReturnType
    N: int | None = 25 * 3  # 滚动窗口大小
    lam: float | None = 0.8  # 0<lam<1, 越大越强调近端


@dataclass
class RiskSpec:
    """
    风险规格配置类

    用于定义不同类型风险计算的参数配置

    Attributes:
        kind: 风险类型，指定要计算的风险种类
        N: 滚动窗口大小，用于指定计算风险时使用的历史数据窗口长度
        lam: 指数加权参数，取值范围(0,1)，数值越大越强调近期数据的重要性
        p: VaR/ES置信度水平，例如0.99表示99%的置信度
    """
    kind: RiskType
    N: int | None = 25 * 3  # 滚动窗口大小
    lam: float | None = 0.8  # 0<lam<1, 越大越强调近端
    p: float | None = 0.99  # VaR/ES 置信度, 例: 0.99


def _safe_one_plus(x: np.ndarray) -> np.ndarray:
    """
    计算1+x的安全版本，避免结果过小导致数值不稳定。

    该函数通过将1+x的结果限制在最小值1e-12以上，防止出现接近零的数值，
    从而提高数值计算的稳定性。

    参数:
        x (np.ndarray): 输入的数值数组

    返回值:
        np.ndarray: 经过安全处理的1+x结果数组，所有元素都大于等于1e-12
    """
    # 使用clip函数限制1+x的最小值为1e-12，防止数值过小
    return np.clip(1.0 + x, 1e-12, None)


def _exp_weights(N: int, lam: float) -> np.ndarray:
    k = np.arange(N, dtype=np.float64)
    w = (1.0 - lam) * np.power(lam, k)
    s = w.sum()
    return (w / s) if s > 0 else np.ones(N, dtype=np.float64) / N


def return_metric(R: np.ndarray, spec: ReturnSpec) -> float:
    T = R.shape[0]
    one_plus = _safe_one_plus(R)
    if spec.kind == "total_cum":
        return float(np.prod(one_plus) - 1.0)
    if spec.kind == "total_geom_annual":
        g = np.prod(one_plus)
        return float(np.power(g, TRADING_DAYS / T) - 1.0)
    if spec.kind == "total_log":
        return float(np.sum(np.log(one_plus)))
    if spec.kind == "total_log_annual":
        return float(np.sum(np.log(one_plus)) / T * TRADING_DAYS)
    if spec.kind in ("ew_roll_cum", "ew_roll_log"):
        assert spec.N and spec.lam is not None
        N = min(spec.N, T)
        r_win = R[-N:]
        w = _exp_weights(N, spec.lam)[::-1]  # 最近一天权重大
        log_win = np.log(_safe_one_plus(r_win))
        log_agg = float(np.dot(w, log_win))
        return float(np.exp(log_agg) - 1.0) if spec.kind == "ew_roll_cum" else float(log_agg)
    if spec.kind == "ew_roll_mean_simple":
        assert spec.N and spec.lam is not None
        N = min(spec.N, T)
        r_win = R[-N:]
        w = _exp_weights(N, spec.lam)[::-1]  # 最近权重大
        return float(np.dot(w, r_win))
    if spec.kind == "mean_simple":
        return float(np.mean(R))  # 普通收益率均值
    if spec.kind == "mean_log":
        return float(np.mean(np.log(one_plus)))  # 对数收益率均值
    raise ValueError(f"Unknown ReturnSpec: {spec}")


def risk_metric(R: np.ndarray, spec: RiskSpec) -> float:
    T = R.shape[0]
    if spec.kind == "std":
        return float(np.std(R, ddof=1))
    if spec.kind == "std_annual":
        return float(np.std(R, ddof=1) * np.sqrt(TRADING_DAYS))
    if spec.kind == "log_std":
        return float(np.std(np.log(_safe_one_plus(R)), ddof=1))
    if spec.kind == "log_std_annual":
        return float(np.std(np.log(_safe_one_plus(R)), ddof=1) * np.sqrt(TRADING_DAYS))
    if spec.kind in ("ew_roll_std", "ew_roll_log_std"):
        assert spec.N and spec.lam is not None
        N = min(spec.N, T)
        r_win = R[-N:]
        w = _exp_weights(N, spec.lam)[::-1]
        x = r_win if spec.kind == "ew_roll_std" else np.log(_safe_one_plus(r_win))
        mu = float(np.dot(w, x))
        var = float(np.dot(w, (x - mu) * (x - mu)))
        return float(np.sqrt(max(var, 0.0)))
    if spec.kind in ("var_abs", "es_abs"):
        assert spec.p is not None and 0 < spec.p < 1
        L = -R
        q = np.quantile(L, spec.p, method="linear")
        if q <= 0:
            return 0.0
        if spec.kind == "var_abs":
            return float(abs(q))
        tail = L[L >= q]
        es = float(tail.mean()) if tail.size else float(q)
        return abs(es)
    if spec.kind == "max_drawdown_abs":
        pv = np.cumprod(_safe_one_plus(R))
        if pv.size == 0:
            return 0.0
        roll_max = np.maximum.accumulate(pv)
        dd = 1.0 - pv / roll_max
        return float(abs(np.max(dd)))
    raise ValueError(f"Unknown RiskSpec: {spec}")


# ========= QCQP 可用性与构造 =========
def _is_qcqp_compatible(ret_spec: ReturnSpec, risk_spec: RiskSpec) -> bool:
    # log 年化 + log_std 年化
    if (ret_spec.kind in ("total_log", "total_log_annual")
            and risk_spec.kind in ("log_std", "log_std_annual")):
        return True
    # EWMA 对数口径（N,λ必须一致）
    if (ret_spec.kind == "ew_roll_log" and risk_spec.kind == "ew_roll_log_std"
            and (ret_spec.N == risk_spec.N) and (ret_spec.lam == risk_spec.lam)):
        return True
    return False


def _mu_Sigma_for_ewma_simple_qcqp(port_daily: np.ndarray, N: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """
    精确 Markowitz（EWMA 简单收益口径）：
      μ = 252 * Σ_t w_t * r_t
      Σ = 252 * Σ_t w_t * (r_t - μ_d)(r_t - μ_d)^T
    其中 r_t 为【普通日收益】向量；w_t 为指数加权（近端权重大）；μ_d 为加权日均值。
    """
    R = port_daily.astype(np.float64)  # [T, n] 普通日收益
    T = R.shape[0]
    N = int(min(N, T))
    w = _exp_weights(N, lam)[::-1]  # 近端权重大，和=1

    Rw = R[-N:]  # 取窗口
    mu_day = (Rw * w[:, None]).sum(axis=0)  # 加权日均
    X = Rw - mu_day  # 去中心
    covariance_day = np.tensordot(w, np.einsum('ti,tj->tij', X, X), axes=(0, 0))  # 加权协方差（日频）

    mu = mu_day * TRADING_DAYS
    covariance_matrix = covariance_day * TRADING_DAYS

    # 轻度 PSD 投影
    S = 0.5 * (covariance_matrix + covariance_matrix.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None)
    covariance_psd = (vecs * vals) @ vecs.T
    return mu.astype(np.float64), covariance_psd.astype(np.float64)


def _mu_Sigma_for_simple_qcqp(port_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    精确 Markowitz (mean_simple + std/std_annual) 的 μ、Σ（基于普通日收益）。
    日→年化：μ*252, Σ*252
    """
    R = port_daily.astype(np.float64)  # [T, n] 普通日收益
    mu = R.mean(axis=0) * TRADING_DAYS
    covariance_matrix = np.cov(R, rowvar=False, ddof=1) * TRADING_DAYS
    # 轻度 PSD 投影
    S = 0.5 * (covariance_matrix + covariance_matrix.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None)
    covariance_psd = (vecs * vals) @ vecs.T
    return mu.astype(np.float64), covariance_psd.astype(np.float64)


def _mu_Sigma_for_log_qcqp(port_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = np.log(_safe_one_plus(port_daily))  # [T,n]
    mu = L.mean(axis=0) * TRADING_DAYS
    covariance_matrix = np.cov(L, rowvar=False, ddof=1) * TRADING_DAYS
    # 轻度 PSD 投影
    S = 0.5 * (covariance_matrix + covariance_matrix.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None)
    covariance_psd = (vecs * vals) @ vecs.T
    return mu.astype(np.float64), covariance_psd.astype(np.float64)


def _mu_Sigma_for_ewma_log_qcqp(port_daily: np.ndarray, N: int, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """
    基于对数收益 L_t = log(1+r_t)，计算 EWMA 加权的年化 μ 与 Σ，并做轻度 PSD 投影。
    与 RET_SPEC=ew_roll_log、RISK_SPEC=ew_roll_log_std 的口径一致。
    """
    L = np.log(_safe_one_plus(port_daily))  # [T, n]
    T = L.shape[0]
    N = int(min(N, T))
    w = _exp_weights(N, lam)  # 和=1，从远到近（下面做反转）
    w = w[::-1]  # 最近权重大；对应 ew_roll* 系列实现

    Lw = L[-N:]  # 取窗口
    # 加权均值（年化）：sum(w * L_t) * 252
    mu = (Lw * w[:, None]).sum(axis=0) * TRADING_DAYS

    # 加权协方差（年化）：EWMA 样式，Σ = 252 * sum(w * (L - μ_d)*(L - μ_d)^T)
    mu_d = (Lw * w[:, None]).sum(axis=0)  # 日频均值
    X = Lw - mu_d  # 去中心
    # 逐日外积的加权和
    covariance_matrix = np.tensordot(w, np.einsum('ti,tj->tij', X, X), axes=(0, 0)) * TRADING_DAYS

    # 轻度 PSD 投影
    S = 0.5 * (covariance_matrix + covariance_matrix.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 1e-12, None)
    covariance_psd = (vecs * vals) @ vecs.T
    return mu.astype(np.float64), covariance_psd.astype(np.float64)


class QCQPTag(str, Enum):
    EXACT = "QCQP可解"
    APPROX = "QCQP可近似"
    NONE = "QCQP无法使用"


def classify_qcqp_combo(ret_spec: ReturnSpec, risk_spec: RiskSpec) -> QCQPTag:
    """
    体系化地判断 (收益, 风险) 组合是否可用 QCQP。
    只对“线性均值 + 二次方差”结构给 EXACT；对“线性化近似可写成二次”的给 APPROX；其余 NONE。
    """

    # 1) 精确 QCQP：mean_simple + {std, std_annual}
    if ret_spec.kind == "mean_simple" and risk_spec.kind in ("std", "std_annual"):
        return QCQPTag.EXACT

    # 1.1) 精确 QCQP ：ew_roll_mean_simple + ew_roll_std，且 N/λ 对齐
    if (ret_spec.kind == "ew_roll_mean_simple" and risk_spec.kind == "ew_roll_std"
            and (ret_spec.N == risk_spec.N) and (ret_spec.lam == risk_spec.lam)):
        return QCQPTag.EXACT

    # 2) 近似 QCQP：基于“把资产某变换当作线性因子”的线性化
    # 2.1 对数口径（近似假设：log(1+r_p) ≈ wᵀlog(1+r_i)）
    if (ret_spec.kind in ("total_log", "total_log_annual", "ew_roll_log")
            and risk_spec.kind in ("log_std", "log_std_annual", "ew_roll_log_std")):
        # 仅当 EWMA 日志两者 N、λ 对齐时，近似更一致
        if ret_spec.kind == "ew_roll_log" and risk_spec.kind == "ew_roll_log_std":
            if (ret_spec.N == risk_spec.N) and (ret_spec.lam == risk_spec.lam):
                return QCQPTag.APPROX
            else:
                return QCQPTag.NONE
        # 非 EWMA 情况也可给“近似”标签（但要提示）
        return QCQPTag.APPROX

    # 3) 其它组合：无法表示成 (线性目标, 二次约束) 的，标记为 NONE
    return QCQPTag.NONE


def build_mu_Sigma_for_qcqp(
        port_daily: np.ndarray,
        ret_spec: ReturnSpec,
        risk_spec: RiskSpec
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], QCQPTag, str]:
    """
    返回 (mu, covariance_matrix, tag, msg)
    tag: EXACT / APPROX / NONE
    msg: 说明信息（用于日志提示）
    """
    tag = classify_qcqp_combo(ret_spec, risk_spec)

    if tag == QCQPTag.EXACT:
        # 精确 Markowitz：mean_simple + std/std_annual
        if ret_spec.kind == "mean_simple" and risk_spec.kind in ("std", "std_annual"):
            mu, covariance_matrix = _mu_Sigma_for_simple_qcqp(port_daily)
            return mu, covariance_matrix, tag, "[QCQP] 精确 Markowitz (mean_simple + std*)"

        # 精确 Markowitz：ew_roll_mean_simple + ew_roll_std (N/λ 对齐)
        if ret_spec.kind == "ew_roll_mean_simple" and risk_spec.kind == "ew_roll_std":
            mu, covariance_matrix = _mu_Sigma_for_ewma_simple_qcqp(port_daily, ret_spec.N, ret_spec.lam)
            return mu, covariance_matrix, tag, "[QCQP] 精确 Markowitz (EWMA simple mean + EWMA std)"

    if tag == QCQPTag.APPROX:
        # 近似一：log 口径（与现有实现保持一致）
        if (ret_spec.kind in ("total_log", "total_log_annual", "ew_roll_log")
                and risk_spec.kind in ("log_std", "log_std_annual", "ew_roll_log_std")):

            if ret_spec.kind == "ew_roll_log" and risk_spec.kind == "ew_roll_log_std":
                if (ret_spec.N == risk_spec.N) and (ret_spec.lam == risk_spec.lam):
                    mu, covariance_matrix = _mu_Sigma_for_ewma_log_qcqp(port_daily, ret_spec.N, ret_spec.lam)
                    return mu, covariance_matrix, tag, "[QCQP-Approx] 采用 EWMA-log 线性化近似 (N/λ 对齐)"
                else:
                    return None, None, QCQPTag.NONE, "[QCQP-Approx] EWMA N/λ 未对齐，放弃近似"
            else:
                mu, covariance_matrix = _mu_Sigma_for_log_qcqp(port_daily)
                return mu, covariance_matrix, tag, "[QCQP-Approx] 采用 log 线性化近似"

    return None, None, QCQPTag.NONE, "[QCQP] 该指标组合无法用 QCQP 表达"


''' ========= SLSQP 路径 ========= 
'''


def _scipy_slsqp_one(
        port_daily: np.ndarray,
        ret_spec: ReturnSpec,
        risk_spec: RiskSpec,
        s_target: float,
        single_limits,
        multi_limits,
        w0: np.ndarray,
        n_starts: int = 6,
        risk_band: float = 5e-7,
        rho_penalty: float = 5e4,
        extra_starts: Optional[List[np.ndarray]] = None,
        rng: Optional[np.random.Generator] = None,
):
    """
    强化版 SLSQP：
      - 多起点：last_w / 中点投影 / 随机可行点 / 可选热启动池 extra_starts
      - 风险同温带：σ(w) ≤ s_target + risk_band（硬约束）
      - 软惩罚：最大化 [收益 - ρ·max(0, σ-s_target)^2]（更贴近风险边界）
    """
    try:
        from scipy.optimize import minimize
    except Exception as e:
        print(f"[WARN] SciPy 不可用，无法走SLQP: {e}")
        return None

    n = port_daily.shape[1]
    if rng is None:
        rng = np.random.default_rng(2024)
    bounds = [tuple(map(float, single_limits[i])) for i in range(n)]

    # 组约束
    cons_base = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    for idxs, (low, up) in multi_limits.items():
        idxs = np.array(idxs, dtype=np.int64)
        cons_base.append({'type': 'ineq', 'fun': (lambda idxs, low: lambda w: np.sum(w[idxs]) - low)(idxs, float(low))})
        cons_base.append({'type': 'ineq', 'fun': (lambda idxs, up: lambda w: up - np.sum(w[idxs]))(idxs, float(up))})

    # 风险同温带（硬约束放宽到 s_target + band）
    def risk_band_con(w):
        R = port_daily @ w
        return float((s_target + risk_band) - risk_metric(R, risk_spec))

    cons_base.append({'type': 'ineq', 'fun': risk_band_con})

    # 目标：-(收益 - 惩罚)  →  最小化
    def obj(w):
        R = port_daily @ w
        ret = return_metric(R, ret_spec)
        sig = risk_metric(R, risk_spec)
        pen = max(0.0, sig - s_target)
        return -(ret - rho_penalty * (pen * pen))

    # ====== 多起点集合 ======
    starts: List[np.ndarray] = [w0]

    # 中点投影
    mids = np.array([(lo + hi) * 0.5 for (lo, hi) in single_limits], dtype=np.float64)
    w_mid = project_to_constraints_pocs(mids, single_limits, multi_limits, max_iter=300, tol=1e-10)
    if w_mid is not None:
        starts.append(w_mid)

    # 随机可行点
    need_rand = max(0, n_starts - len(starts))
    for _ in range(need_rand):
        x = rng.uniform([a for a, _ in single_limits], [b for _, b in single_limits])
        x = project_to_constraints_pocs(x, single_limits, multi_limits, max_iter=300, tol=1e-10)
        if x is not None:
            starts.append(x)

    # 热启动池
    if extra_starts:
        for x in extra_starts:
            if x is not None and np.isfinite(x).all():
                starts.append(x)
    best = None
    best_val = -np.inf

    for w_init in starts:
        res = minimize(obj, w_init, method='SLSQP', bounds=bounds, constraints=cons_base,
                       options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
        if not res.success:
            continue
        w = res.x.astype(np.float64)
        if abs(w.sum() - 1.0) > 1e-6:
            continue
        R = port_daily @ w
        sig = risk_metric(R, risk_spec)
        # 落在同温带内的解才收
        if sig <= s_target + risk_band + 1e-12:
            val = -obj(w)  # 因为 obj 取了负
            if val > best_val:
                best_val = val
                best = w

    return best


def _risk_grid_from_samples(port_daily: np.ndarray, risk_spec: RiskSpec,
                            single_limits, multi_limits, n_grid=300, seed=1234):
    rng = np.random.default_rng(seed)
    n = port_daily.shape[1]
    lows = np.array([a for a, _ in single_limits])
    highs = np.array([b for _, b in single_limits])
    cand = [(lows + highs) * 0.5]
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        e = project_to_constraints_pocs(e, single_limits, multi_limits, max_iter=300, tol=1e-10)
        if e is not None:
            cand.append(e)
    for _ in range(256):
        x = rng.uniform(lows, highs)
        x = project_to_constraints_pocs(x, single_limits, multi_limits, max_iter=300, tol=1e-10)
        if x is not None:
            cand.append(x)
    W = np.vstack(cand)
    risks = np.fromiter((risk_metric(port_daily @ w, risk_spec) for w in W), dtype=np.float64,
                        count=W.shape[0])
    s_min, s_max = float(np.min(risks)), float(np.max(risks))
    if not np.isfinite(s_min) or not np.isfinite(s_max) or s_min >= s_max:
        s_min, s_max = max(s_min, 0.0), max(s_min + 1e-6, s_min * 1.2 + 1e-6)
    return np.linspace(s_min, s_max, n_grid)


def multistage_frontier_scan_slqp(
        port_daily: np.ndarray,
        ret_spec: ReturnSpec,
        risk_spec: RiskSpec,
        single_limits,
        multi_limits,
        *,
        n_grid: int = 300,
        # --- SLSQP 扫描参数 ---
        risk_band: float = 5e-7,
        rho_penalty: float = 5e4,
        n_starts: int = 6,
        seed: int = 2024,
        # --- 随机游走参数（阶段1→2）---
        walk_per_anchor: int = 60,
        walk_step: float = 0.08,
        walk_sigma_tol: float = 1e-4,
        walk_sigma_band: Optional[float] = None,
        walk_precision: Optional[str | float] = None,
        # --- 热启动池筛选 ---
        hot_bins: int = 60,
        hot_top_k_per_bin: int = 3,
        # --- 是否返回“最终前沿”的随机游走填充 ---
        return_fill: bool = True,
        final_fill_per_anchor: Optional[int] = None,  # 若为 None，沿用 walk_per_anchor
        final_fill_step: Optional[float] = None,  # 若为 None，沿用 walk_step
        final_fill_sigma_tol: Optional[float] = None,  # 若为 None，沿用 walk_sigma_tol
        final_fill_precision: Optional[str | float] = None  # 覆盖最终量化精度
):
    # ========== 阶段1：初扫 ==========
    grid1, W1, R1, S1 = sweep_frontier_by_risk_slqp(
        port_daily, ret_spec, risk_spec,
        single_limits, multi_limits,
        n_grid=n_grid, risk_band=risk_band, rho_penalty=rho_penalty,
        n_starts=n_starts, seed=seed, hot_start_pool=None
    )
    idx = np.argsort(S1)
    S_sorted, R_sorted, W_sorted = S1[idx], R1[idx], W1[idx]
    keep = np.isclose(R_sorted, np.maximum.accumulate(R_sorted), atol=1e-10)
    W_anchors = W_sorted[keep]

    # ========== 阶段2：随机游走拿热启动池 ==========
    W_walk = random_walk_frontier_explore(
        W_anchor=W_anchors, port_daily=port_daily,
        single_limits=single_limits, multi_limits=multi_limits,
        per_anchor=walk_per_anchor, step=walk_step,
        sigma_tol=walk_sigma_tol, seed=seed + 1,
        precision=walk_precision,
        ret_spec=ret_spec, risk_spec=risk_spec,
        walk_region="any",
        sigma_band=(2.0 * walk_sigma_tol if walk_sigma_band is None else walk_sigma_band),
        ret_tol_up=1e-6
    )

    hot_pool = None
    if W_walk is not None and W_walk.size:
        perf = generate_alloc_perf_batch(port_daily, W_walk, ret_spec, risk_spec)
        Sseed = perf['vol_annual'].to_numpy();
        Rseed = perf['ret_annual'].to_numpy()
        bins = np.linspace(Sseed.min(), Sseed.max(), max(5, hot_bins))
        pick = []
        for i in range(len(bins) - 1):
            m = (Sseed >= bins[i]) & (Sseed <= bins[i + 1])
            if not m.any():
                continue
            ids = np.where(m)[0]
            top = ids[np.argsort(Rseed[m])][-hot_top_k_per_bin:]
            pick.extend(top)
        hot_pool = np.unique(W_walk[pick], axis=0)

    # ========== 阶段3：热启动再扫 ==========
    grid2, W2, R2, S2 = sweep_frontier_by_risk_slqp(
        port_daily, ret_spec, risk_spec,
        single_limits, multi_limits,
        n_grid=n_grid, risk_band=risk_band, rho_penalty=rho_penalty,
        n_starts=max(n_starts, 8), seed=seed + 7,
        hot_start_pool=hot_pool, hot_k=3
    )

    if not return_fill:
        return grid2, W2, R2, S2

    # ========== 阶段4：基于“新的前沿”做最终随机游走 ==========
    idx2 = np.argsort(S2)
    S2s, R2s, W2s = S2[idx2], R2[idx2], W2[idx2]
    keep2 = np.isclose(R2s, np.maximum.accumulate(R2s), atol=1e-10)
    W_anchors2 = W2s[keep2]

    W_fill = random_walk_frontier_explore(
        W_anchor=W_anchors2, port_daily=port_daily,
        single_limits=single_limits, multi_limits=multi_limits,
        per_anchor=(walk_per_anchor if final_fill_per_anchor is None else final_fill_per_anchor),
        step=(walk_step if final_fill_step is None else final_fill_step),
        sigma_tol=(walk_sigma_tol if final_fill_sigma_tol is None else final_fill_sigma_tol),
        seed=seed + 11,
        precision=(walk_precision if final_fill_precision is None else final_fill_precision),
        ret_spec=ret_spec, risk_spec=risk_spec,
        walk_region="any",
        sigma_band=(2.0 * walk_sigma_tol if walk_sigma_band is None else walk_sigma_band),
        ret_tol_up=1e-6
    )
    return grid2, W2, R2, S2, W_fill


def sweep_frontier_by_risk_slqp(
        port_daily: np.ndarray,
        ret_spec: ReturnSpec,
        risk_spec: RiskSpec,
        single_limits,
        multi_limits,
        n_grid=300,
        risk_band=5e-7,  # 同温带宽度
        rho_penalty=5e4,  # 风险软惩罚系数
        n_starts=6,
        seed=2024,
        hot_start_pool: Optional[np.ndarray] = None,  # 形状 [m, n] 的候选起点（来自随机游走）
        hot_k: int = 3,  # 对每个目标风险挑选的热启动个数
):
    """
    使用 SLSQP 优化器在给定风险水平网格上扫描有效前沿，返回对应的风险、收益和权重。

    参数:
        port_daily : np.ndarray               资产的日收益率矩阵，形状为 [T, n]，T 为时间长度，n 为资产数量。
        ret_spec : ReturnSpec                 收益度量的配置对象。
        risk_spec : RiskSpec                  风险度量的配置对象。
        single_limits : list of tuple         每个资产的上下限约束 [(lo1, hi1), (lo2, hi2), ...]。
        multi_limits : list of tuple          多资产线性约束，格式为 (A, b)，表示 A @ w <= b。
        n_grid : int, optional                在风险维度上划分的网格点数，默认为 300。
        risk_band : float, optional           用于风险软约束的带宽参数，默认为 5e-7。
        rho_penalty : float, optional         风险软约束的惩罚系数，默认为 5e4。
        n_starts : int, optional              每个优化问题尝试的随机初始点数量，默认为 6。
        seed : int, optional                  随机种子，默认为 2024。
        hot_start_pool : Optional[np.ndarray] 可选的热启动候选点池，形状为 [m, n]，默认为 None。
        hot_k : int, optional                 每个目标风险水平选取的热启动点数量，默认为 3。

    返回:
        tuple:
            - grid : np.ndarray     实际有效的风险目标值数组。
            - W : np.ndarray        对应的有效前沿权重矩阵，形状为 [len(grid), n]。
            - R : np.ndarray        对应的年化收益数组。
            - S : np.ndarray        对应的年化风险数组。
    """

    rng = np.random.default_rng(seed)
    # 构造风险目标网格
    grid = _risk_grid_from_samples(port_daily, risk_spec, single_limits, multi_limits, n_grid=n_grid, seed=seed)

    # 默认初始点：取中点并投影到约束空间
    mids = np.array([(lo + hi) * 0.5 for (lo, hi) in single_limits], dtype=np.float64)
    w0 = project_to_constraints_pocs(mids, single_limits, multi_limits, max_iter=300, tol=1e-10)
    if w0 is None:
        w0 = np.full(port_daily.shape[1], 1.0 / port_daily.shape[1], dtype=np.float64)

    # 若提供了热启动池，则预计算其风险与收益
    hot_S = hot_R = None
    if hot_start_pool is not None and hot_start_pool.size:
        perf = generate_alloc_perf_batch(port_daily, hot_start_pool, ret_spec, risk_spec)
        hot_S = perf['vol_annual'].to_numpy()
        hot_R = perf['ret_annual'].to_numpy()

    W_list = []
    last_w = w0

    # 遍历风险目标网格，逐个优化得到前沿点
    for s in grid:
        extra = None
        if hot_start_pool is not None and hot_start_pool.size:
            # 从热启动池中选择风险最接近且收益较高的点作为额外初始点
            d = np.abs(hot_S - s)
            idx = np.argsort(d)[:max(1, hot_k * 2)]  # 先取距离近的
            idx = idx[np.argsort(hot_R[idx])][-hot_k:]  # 再在里面取收益最高
            extra = [hot_start_pool[i] for i in idx]

        # 使用 SLSQP 进行优化
        w = _scipy_slsqp_one(
            port_daily, ret_spec, risk_spec, float(s),
            single_limits, multi_limits, last_w,
            n_starts=n_starts, risk_band=risk_band, rho_penalty=rho_penalty,
            extra_starts=extra, rng=rng
        )
        # 若失败则使用默认初始点重试
        if w is None:
            w = _scipy_slsqp_one(
                port_daily, ret_spec, risk_spec, float(s),
                single_limits, multi_limits, w0,
                n_starts=n_starts + 2, risk_band=risk_band, rho_penalty=rho_penalty,
                extra_starts=extra, rng=rng
            )
        W_list.append(w if w is not None else np.full(port_daily.shape[1], np.nan))
        if w is not None:
            last_w = w

    # 将所有结果堆叠为矩阵
    W = np.vstack(W_list)
    R = np.full(W.shape[0], np.nan)
    S = np.full(W.shape[0], np.nan)

    # 计算每个权重向量对应的实际收益和风险
    for i in range(W.shape[0]):
        wi = W[i]
        if np.any(~np.isfinite(wi)):
            continue
        Ri = port_daily @ wi
        R[i] = return_metric(Ri, ret_spec)
        S[i] = risk_metric(Ri, risk_spec)

    # 过滤掉无效结果
    mask = np.isfinite(R) & np.isfinite(S)
    return grid[mask], W[mask], R[mask], S[mask]


def sweep_frontier_by_risk_unified(
        port_daily: np.ndarray,
        ret_spec: ReturnSpec, risk_spec: RiskSpec,
        single_limits, multi_limits, n_grid=300):
    """
    统一入口：
      - EXACT: 直接用 QCQP (精确 Markowitz) 扫描。
      - APPROX: 先用 QCQP-Approx(线性化 μ、Σ) 扫描拿近似权重，再把这些权重作为 SLSQP 的热启动池做真口径精修。
      - NONE: 使用多阶段 SLSQP（多起点 + 同温带 + 软惩罚 + 随机游走热启动）。
    返回: grid, W, R, S
    """
    # 先尝试构造 (mu, covariance_matrix) 及可解性标记
    mu, covariance_matrix, tag, msg = build_mu_Sigma_for_qcqp(port_daily, ret_spec, risk_spec)
    print(f"{str_time()} [INFO] {msg}")

    # ===== 1) 精确 QCQP：直接返回 =====
    if tag == QCQPTag.EXACT:
        print(f"{str_time()} [INFO] 使用 QCQP (EXACT) 逐风险扫描前沿 ...")
        grid, W, R, S, *_ = sweep_frontier_by_risk(
            mu, covariance_matrix, single_limits, multi_limits, n_grid=n_grid
        )
        return grid, W, R, S

    # ===== 2) 近似 QCQP：先拿近似解，再用 SLSQP 精修 =====
    if tag == QCQPTag.APPROX:
        print(f"{str_time()} [INFO] 使用 QCQP-Approx 先刻画近似前沿，产出热启动权重池 ...")
        # 2.1 用线性化的 μ、Σ 做一遍 QCQP 扫描，得到近似前沿权重
        grid_approx, W_approx, R_approx, S_approx, *_ = sweep_frontier_by_risk(
            mu, covariance_matrix, single_limits, multi_limits, n_grid=n_grid
        )
        # 2.2 用真口径计算近似权重的绩效，以便 SLSQP 选热启动
        perf_approx = generate_alloc_perf_batch(port_daily, W_approx, ret_spec, risk_spec)
        Sa = perf_approx['vol_annual'].to_numpy()
        Ra = perf_approx['ret_annual'].to_numpy()
        print(f"{str_time()} [INFO] QCQP-Approx 热启动池规模: {len(W_approx)}（将用于 SLSQP 精修）")

        # 2.3 用 SLSQP 在真口径上做逐风险扫描；把近似解作为 hot_start_pool
        grid, W, R, S = sweep_frontier_by_risk_slqp(
            port_daily, ret_spec, risk_spec,
            single_limits, multi_limits,
            n_grid=n_grid,
            # 同温带与惩罚参数可按需调；这里沿用默认的稳健配置
            risk_band=5e-7,
            rho_penalty=5e4,
            n_starts=8,  # 适当增大起点数，让 SLSQP 有机会跳出局部
            seed=2024,
            hot_start_pool=W_approx,  # 关键：注入 QCQP-Approx 的前沿权重
            hot_k=5  # 对每个目标风险，最多挑 5 个热启动进行尝试
        )
        print(f"{str_time()} [INFO] SLSQP 已基于 QCQP-Approx 热启动完成精修。")
        return grid, W, R, S

    # ===== 3) 完全不能 QCQP 的：走多阶段 SLSQP =====
    print(f"{str_time()} [INFO] 使用多阶段 SLSQP（多起点 + 同温带 + 软惩罚 + 随机游走热启动）")
    grid, W, R, S = multistage_frontier_scan_slqp(
        port_daily, ret_spec, risk_spec,
        single_limits, multi_limits,
        n_grid=n_grid,
        risk_band=5e-7,
        rho_penalty=5e4,
        n_starts=6, seed=2024,
        walk_per_anchor=60, walk_step=0.08, walk_sigma_tol=1e-4,
        walk_sigma_band=None, walk_precision=None,
        hot_bins=60, hot_top_k_per_bin=3
    )
    return grid, W, R, S


''' ========= 统一指标计算 ========== 
'''


# 对任意权重矩阵 W（m×n）统一计算收益与风险两列
def compute_metrics_for_W(port_daily: np.ndarray, W: np.ndarray,
                          ret_spec: ReturnSpec, risk_spec: RiskSpec,
                          col_ret: str = "ret_annual", col_risk: str = "vol_annual") -> pd.DataFrame:
    """
    对任意权重矩阵 W（m×n）统一计算收益与风险两列，列名默认沿用原脚本，便于无缝替换。

    参数:
        port_daily: 形状为[T, n]的日度收益率矩阵，T为时间维度，n为资产数量
        W: 形状为[m, n]的权重矩阵，m为组合数量，n为资产数量
        ret_spec: 收益指标计算规范对象
        risk_spec: 风险指标计算规范对象
        col_ret: 收益列的列名，默认为"ret_annual"
        col_risk: 风险列的列名，默认为"vol_annual"

    返回:
        pd.DataFrame: 包含权重、收益和风险指标的DataFrame，形状为[m, n+2]
    """
    T, n = port_daily.shape
    assert W.shape[1] == n
    # 计算所有组合在各个时间点的收益率：R_all = port_daily @ W.T
    # 结果矩阵R_all的形状为[T, m]，每一列表示一个组合的收益率时间序列
    R_all = port_daily @ W.T
    # 分别逐列算指标（避免大复制）
    m = W.shape[0]
    rets = np.empty(m, dtype=np.float64)
    risks = np.empty(m, dtype=np.float64)
    for i in range(m):
        Ri = R_all[:, i]
        rets[i] = return_metric(Ri, ret_spec)
        risks[i] = risk_metric(Ri, risk_spec)
    df = pd.DataFrame({col_ret: rets, col_risk: risks})
    wdf = pd.DataFrame(W, columns=[f"w_{j}" for j in range(n)])
    out = pd.concat([wdf, df], axis=1)
    # 清理异常
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


''' ========= 约束 & 投影 =========
'''


# 将边界字典转换为限制列表格式
def bounds_dict_to_limits(assets: List[str],
                          level_bounds: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[
    str, List[Tuple[float, float]]]:
    """
    将边界字典转换为限制列表格式

    参数:
        assets: 资产列表，包含所有需要处理的资产名称
        level_bounds: 等级边界字典，格式为 {等级: {资产名: (下界, 上界)}}

    返回:
        字典格式的限制范围，格式为 {等级: [(资产1下界, 资产1上界), (资产2下界, 资产2上界), ...]}
    """
    out = {}
    # 遍历每个等级的边界映射
    for level, bmap in level_bounds.items():
        print(f"{str_time()} [构造边界] 处理等级:", level)
        # 为每个资产获取对应的边界值，如果不存在则使用默认值(0.0, 1.0)
        out[level] = [(bmap.get(a, (0.0, 1.0))[0], bmap.get(a, (0.0, 1.0))[1]) for a in assets]
    return out


# 计算全局包络边界限制
def global_envelope_limits(per_level_limits: Dict[str, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    """
    计算全局包络边界限制

    该函数接收按层级分类的边界限制字典，计算每个位置上所有层级的最小下界和最大上界，
    从而得到全局的包络边界。

    参数:
        per_level_limits: 字典类型，键为层级标识符，值为该层级的边界限制列表。
                         每个边界限制是一个包含两个浮点数的元组(下界, 上界)。

    返回值:
        List[Tuple[float, float]]: 全局包络边界限制列表，每个元素是一个包含下界和上界的元组。
    """
    print(f"{str_time()} [构造边界] 处理全局边界")
    levels = list(per_level_limits.keys())
    n = len(per_level_limits[levels[0]])

    # 计算每个位置上所有层级的最小下界
    lows = np.min(np.array([[per_level_limits[L][j][0] for L in levels] for j in range(n)]), axis=1)

    # 计算每个位置上所有层级的最大上界
    highs = np.max(np.array([[per_level_limits[L][j][1] for L in levels] for j in range(n)]), axis=1)

    return [(float(l), float(h)) for l, h in zip(lows, highs)]


# 使用 POCS 法将向量投影到多个约束集合的交集上
def project_to_constraints_pocs(v: np.ndarray,
                                single_limits,  # list[(low, high)]
                                multi_limits: dict,  # {(tuple_idx): (low, high)}
                                max_iter=200, tol=1e-9, damping=0.9):
    """
    使用 POCS（Projection Onto Convex Sets）方法将向量投影到多个约束集合的交集上。

    该函数处理以下三类约束：
    1. 每个变量的上下界约束（盒约束）；
    2. 所有变量之和等于 1 的约束；
    3. 多个子集的和的上下界约束（组约束）。

    参数：
        v : np.ndarray
            初始向量，将被投影到满足所有约束的空间中。
        single_limits : list of tuple (low, high)
            每个元素表示对应位置变量的上下界限制。
        multi_limits : dict {tuple of int : (low, high)}
            键为索引元组，表示一个子集；值为该子集和的上下界。
        max_iter : int, optional
            最大迭代次数，默认为 200。
        tol : float, optional
            收敛判断的容忍度，当变量变化小于该值时停止迭代，默认为 1e-9。
        damping : float, optional
            阻尼系数，用于控制组约束投影步长的衰减，默认为 0.9。

    返回：
        np.ndarray or None
            如果成功找到满足所有约束的解，则返回投影后的向量；
            否则返回 None。
    """
    x = v.astype(np.float64).copy()
    n = x.size

    # 提取单变量上下界
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)

    # 解析组约束信息
    groups = []
    for idx_tuple, (low, up) in multi_limits.items():
        idx = np.array(idx_tuple, dtype=np.int64)
        a2 = float(len(idx))
        groups.append((idx, float(low), float(up), a2))

    # 初始投影：裁剪至单变量边界并调整总和为1
    x = np.clip(x, lows, highs)
    x += (1.0 - x.sum()) / n

    # 迭代进行 POCS 投影
    for _ in range(max_iter):
        x_prev = x
        # 投影到单变量边界
        x = np.clip(x, lows, highs)
        # 投影到总和为1的约束
        x += (1.0 - x.sum()) / n
        # 投影到各组约束
        for idx, low, up, a2 in groups:
            s = x[idx].sum()
            if s > up + 1e-12:
                x[idx] -= damping * (s - up) / a2
            elif s < low - 1e-12:
                x[idx] += damping * (low - s) / a2
        # 再次投影到总和为1的约束
        x += (1.0 - x.sum()) / n
        # 判断是否收敛
        if np.max(np.abs(x - x_prev)) < tol:
            break

    # 验证最终结果是否满足所有约束
    if (x < lows - 1e-6).any() or (x > highs + 1e-6).any():
        return None
    for idx, low, up, _ in groups:
        if len(idx) == 0:
            continue
        s = x[idx].sum()
        if s < low - 1e-6 or s > up + 1e-6:
            return None
    if not np.isclose(x.sum(), 1.0, atol=1e-6):
        return None
    return x


# 计算水平中点权重，通过投影到约束集合来获得满足约束条件的权重分布
def level_midpoint_weights(limits_1d: List[Tuple[float, float]]) -> np.ndarray:
    """
    计算水平中点权重，通过投影到约束集合来获得满足约束条件的权重分布。

    参数:
        limits_1d: 一维限制范围列表，每个元素为(l, h)元组，表示下限和上限

    返回:
        np.ndarray: 满足约束条件的权重数组
    """
    # 计算每个限制区间的中点值
    mids = np.array([(l + h) * 0.5 for (l, h) in limits_1d], dtype=np.float64)

    # 将中点值投影到约束集合中，获取满足约束的权重
    w0 = project_to_constraints_pocs(mids, limits_1d, {}, max_iter=500, tol=1e-12, damping=0.9)

    # 如果投影失败，则进行归一化处理后重新投影
    if w0 is None:
        w0 = mids / np.sum(mids) if np.sum(mids) > 0 else np.full_like(mids, 1.0 / len(mids))
        w0 = project_to_constraints_pocs(w0, limits_1d, {}, max_iter=500, tol=1e-12, damping=0.9)
    return w0


# 将基础权重向量投影到指定级别的约束空间中
def project_baseline_to_level(w_base: np.ndarray, level_limits: List[Tuple[float, float]]) -> np.ndarray:
    """
    将基础权重向量投影到指定级别的约束空间中

    该函数首先对输入的权重进行归一化处理，确保其在[0,1]范围内且和为1，
    然后使用POCS算法将权重投影到给定的级别约束空间中。

    参数:
        w_base: 基础权重向量，形状为(n,)的numpy数组
        level_limits: 级别约束限制列表，每个元素为(min_limit, max_limit)元组，
                     表示对应权重的取值范围

    返回:
        投影后的权重向量，形状为(n,)的numpy数组，满足约束条件
    """
    # 对基础权重进行裁剪和归一化处理
    w = np.clip(w_base, 0.0, 1.0).astype(np.float64)
    if not np.isclose(w.sum(), 1.0, atol=1e-12):
        s = w.sum()
        w = w / s if s > 0 else np.full_like(w, 1.0 / w.size)

    # 使用POCS算法将权重投影到约束空间
    w_proj = project_to_constraints_pocs(w, level_limits, {}, max_iter=500, tol=1e-12, damping=0.9)

    # 如果投影失败，则使用级别中点权重作为备选方案
    if w_proj is None:
        w_proj = level_midpoint_weights(level_limits)

    return w_proj


''' ========= 前沿刻画（QCQP） =========
'''


def port_stats(W: np.ndarray, mu: np.ndarray, covariance_matrix: np.ndarray):
    if W.ndim == 1:
        ret = float(W @ mu)
        vol = float(np.sqrt(W @ covariance_matrix @ W))
        return np.array([ret]), np.array([vol])
    rets = W @ mu
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, covariance_matrix, W))
    return rets, vols


def port_RS_by_spec(port_daily: np.ndarray, w: np.ndarray,
                    ret_spec: ReturnSpec | None = None,
                    risk_spec: RiskSpec | None = None) -> tuple[float, float]:
    if ret_spec is None:
        ret_spec = globals().get('RET_SPEC', ReturnSpec(kind="total_log_annual"))
    if risk_spec is None:
        risk_spec = globals().get('RISK_SPEC', RiskSpec(kind="log_std_annual"))
    R = port_daily @ w
    return return_metric(R, ret_spec), risk_metric(R, risk_spec)


def solve_min_variance(covariance_matrix, single_limits, multi_limits):
    """
    求解最小方差投资组合优化问题

    该函数通过凸优化方法求解在给定约束条件下的最小方差投资组合权重分配问题。
    目标是最小化投资组合的方差，即 min w^T * covariance_matrix * w，其中 covariance_matrix 为资产协方差矩阵。

    参数:
        covariance_matrix: numpy.ndarray, shape (n, n)
            资产收益率的协方差矩阵
        single_limits: list of tuples
            单个资产的权重约束，每个元素为(下限, 上限)的元组
        multi_limits: dict
            多个资产组合的权重约束，键为资产索引的可迭代对象，值为(下限, 上限)的元组

    返回值:
        numpy.ndarray
            最优投资组合权重向量
    """
    # 获取资产数量
    n = covariance_matrix.shape[0]

    # 定义优化变量（投资组合权重）
    w = cp.Variable(n)

    # 构建约束条件列表
    cons = [cp.sum(w) == 1]

    # 添加单个资产的边界约束
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]

    # 添加多个资产组合的约束
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]

    # 构建并求解优化问题
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, covariance_matrix)), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-8, reltol=1e-8, feastol=1e-8)

    # 返回最优解
    return w.value


def solve_max_return(mu, single_limits, multi_limits):
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=1e-8, reltol=1e-8, feastol=1e-8)
    return w.value


def solve_max_return_at_risk(mu, covariance_matrix, s_target, single_limits, multi_limits, w0=None):
    n = mu.size
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1, cp.quad_form(w, covariance_matrix) <= float(s_target ** 2)]
    for i, (lo, hi) in enumerate(single_limits):
        cons += [w[i] >= lo, w[i] <= hi]
    for idxs, (low, up) in multi_limits.items():
        cons += [cp.sum(w[list(idxs)]) >= low, cp.sum(w[list(idxs)]) <= up]
    prob = cp.Problem(cp.Maximize(mu @ w), cons)
    if w0 is not None:
        try:
            w.value = w0
        except Exception:
            pass
    prob.solve(solver=cp.ECOS, warm_start=True, abstol=5e-8, reltol=5e-8, feastol=5e-8, max_iters=1000)
    return w.value


def sweep_frontier_by_risk(mu, covariance_matrix, single_limits, multi_limits, n_grid=300):
    """逐风险扫描得到前沿曲线。"""
    w_minv = solve_min_variance(covariance_matrix, single_limits, multi_limits)
    w_maxr = solve_max_return(mu, single_limits, multi_limits)
    _, s_min = port_stats(w_minv, mu, covariance_matrix)
    s_min = s_min[0]
    _, s_max = port_stats(w_maxr, mu, covariance_matrix)
    s_max = np.maximum(s_min, s_max).item()

    grid = np.linspace(s_min, s_max, n_grid)
    W = []
    w0 = w_minv
    for s in grid:
        w = solve_max_return_at_risk(mu, covariance_matrix, s, single_limits, multi_limits, w0=w0)
        W.append(w)
        w0 = w
    W = np.asarray(W)
    R, S = port_stats(W, mu, covariance_matrix)
    return grid, W, R, S, w_minv, w_maxr


def compute_frontier_anchors(mu: np.ndarray, covariance_matrix: np.ndarray,
                             single_limits, multi_limits,
                             n_grid: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (W_anchors, R_anchors, S_anchors)。"""
    _, W_frontier, R_frontier, S_frontier, _, _ = sweep_frontier_by_risk(
        mu, covariance_matrix, single_limits, multi_limits, n_grid=n_grid
    )
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    cummax_R = np.maximum.accumulate(R_sorted)
    keep = np.isclose(R_sorted, cummax_R, atol=1e-10)
    W_anchors = W_sorted[keep]
    R_anchors, S_anchors = R_sorted[keep], S_sorted[keep]
    return W_anchors, R_anchors, S_anchors


''' ========= 网格量化 & 去重 =========
'''


def _parse_precision(choice: str | float) -> float:
    choice = str(choice).strip()
    if choice.endswith('%'):
        val = float(choice[:-1]) / 100.0
    else:
        val = float(choice)
    return float(val)


def _snap_to_grid_simplex(w: np.ndarray, step: float, single_limits) -> Optional[np.ndarray]:
    """
    将权重向量 w 投影到由 step 和 single_limits 定义的网格上，并满足单纯形约束（各分量非负且和为1）。

    参数:
        w (np.ndarray): 输入的权重向量，形状为 (n,)。
        step (float): 网格步长，用于离散化权重空间。
        single_limits: 每个维度的上下限约束，格式为 [(low_0, high_0), ..., (low_n-1, high_n-1)]。

    返回:
        Optional[np.ndarray]: 投影后的量化权重向量，如果无法满足约束则返回 None。
    """

    # 计算单位步长的倒数并四舍五入为整数
    R = int(round(1.0 / step))

    # 将权重裁剪到 [0, 1] 区间内
    w = np.clip(w, 0.0, 1.0)

    # 将权重映射到以 step 为单位的坐标系中
    k_float = w / step
    k_floor = np.floor(k_float).astype(np.int64)
    frac = k_float - k_floor

    # 提取每个维度的上下限，并转换为以 step 为单位的整数坐标
    lows = np.array([a for a, _ in single_limits], dtype=np.float64)
    highs = np.array([b for _, b in single_limits], dtype=np.float64)
    lo_units = np.ceil(lows / step - 1e-12).astype(np.int64)
    hi_units = np.floor(highs / step + 1e-12).astype(np.int64)

    # 初始量化坐标裁剪到合法范围内
    k = np.clip(k_floor, lo_units, hi_units)

    # 计算当前总和与目标总和 R 的差值
    diff = R - int(k.sum())

    # 如果总和不足 R，则从剩余容量较多的位置补充
    if diff > 0:
        cap = hi_units - k
        idx = np.argsort(-frac)  # 按照小数部分降序排序
        for i in idx:
            if diff == 0:
                break
            add = min(cap[i], diff)
            if add > 0:
                k[i] += add
                diff -= add
        if diff != 0:
            return None  # 无法满足总和约束

    # 如果总和超过 R，则从已有量较多的位置减少
    elif diff < 0:
        cap = k - lo_units
        idx = np.argsort(frac)  # 按照小数部分升序排序
        for i in idx:
            if diff == 0:
                break
            sub = min(cap[i], -diff)
            if sub > 0:
                k[i] -= sub
                diff += sub
        if diff != 0:
            return None  # 无法满足总和约束

    # 将整数量化坐标转换回实际权重
    wq = k.astype(np.float64) / R

    # 检查是否满足边界约束和总和约束
    if (wq < lows - 1e-12).any() or (wq > highs + 1e-12).any():
        return None
    if not np.isclose(wq.sum(), 1.0, atol=1e-12):
        return None

    return wq


# 通过投影和量化交替迭代的方法对权重进行量化处理
def quantize_with_projection(w: np.ndarray, step: float,
                             single_limits, multi_limits,
                             rounds: int = 5) -> Optional[np.ndarray]:
    """
    通过投影和量化交替迭代的方法对权重进行量化处理。

    该函数通过多次迭代，交替执行约束投影和网格量化操作，直到满足收敛条件或达到最大迭代轮数。
    在每轮迭代中，先将当前权重投影到约束空间，然后进行量化，最后检查收敛性。

    参数:
        w: 待量化的权重数组
        step: 量化的步长，决定量化精度
        single_limits: 单个元素的约束限制条件
        multi_limits: 多元素间的约束限制条件
        rounds: 最大迭代轮数，默认为5轮

    返回:
        量化后的权重数组，如果在迭代过程中出现错误则返回None
    """
    x = w.copy()
    # 迭代执行投影和量化操作
    for _ in range(rounds):
        # 将当前权重投影到约束空间
        x = project_to_constraints_pocs(x, single_limits, multi_limits,
                                        max_iter=300, tol=1e-10, damping=0.9)
        if x is None:
            return None
        # 对投影后的权重进行网格量化
        xq = _snap_to_grid_simplex(x, step, single_limits)
        if xq is None:
            return None
        # 检查是否满足收敛条件（量化误差小于步长的一半）
        if np.max(np.abs(xq - x)) < step * 0.5:
            return xq
        x = xq
    return x


def dedup_by_grid(W: np.ndarray, step: float) -> np.ndarray:
    if W.size == 0: return W
    K = np.rint(W / step).astype(np.int64)
    _, idx = np.unique(K, axis=0, return_index=True)
    return W[np.sort(idx)]


def quantize_df_for_export(df_in: pd.DataFrame,
                           assets: List[str],
                           step: float,
                           single_limits,
                           multi_limits,
                           port_daily: np.ndarray) -> pd.DataFrame:
    """
    对输入的资产权重进行网格量化处理，并重新计算绩效指标和前沿识别。

    该函数首先对输入 DataFrame 中的资产列执行 POCS 投影与网格量化操作，
    包括边界约束、 simplex 约束（权重和为1）以及组约束。若量化失败，则回退到简单投影和网格化策略。
    随后基于量化后的权重重新计算投资组合的绩效指标（如年化收益、波动率等），
    并重新进行前沿识别。最终返回一个结构与输入一致但内容经过量化处理的 DataFrame。

    参数:
        df_in (pd.DataFrame): 输入的包含资产权重及相关绩效列的 DataFrame。
        assets (List[str]): 资产名称列表，用于指定需要处理的列。
        step (float): 网格量化的步长。
        single_limits: 单资产约束条件（如上下限）。
        multi_limits: 多资产组约束条件（如组权重范围）。
        port_daily (np.ndarray): 日度收益矩阵，用于绩效计算。

    返回:
        pd.DataFrame: 经过量化处理后的 DataFrame，保留原有附加列并更新权重及绩效相关字段。
    """
    W = df_in[assets].to_numpy(dtype=np.float64)
    Wq = []
    for w in W:
        # 尝试使用带投影的量化方法进行处理
        wq = quantize_with_projection(w, step, single_limits, multi_limits, rounds=5)
        if wq is None:
            # 若失败则先投影到约束空间再进行简单网格化
            w_proj = project_to_constraints_pocs(w, single_limits, multi_limits)
            wq = _snap_to_grid_simplex(w_proj, step, single_limits) if w_proj is not None else w
        Wq.append(wq)
    Wq = np.vstack(Wq)

    # 基于量化后的权重批量生成新的绩效数据
    perf_q = generate_alloc_perf_batch(port_daily, Wq)
    rename_map = {f"w_{i}": assets[i] for i in range(len(assets))}
    perf_q = perf_q.rename(columns=rename_map)

    # 保留非资产权重和绩效相关的列
    keep_cols = [c for c in df_in.columns if c not in (assets + ['ret_annual', 'vol_annual', 'on_ef'])]
    out = pd.concat([perf_q[assets + ['ret_annual', 'vol_annual']], df_in[keep_cols].reset_index(drop=True)], axis=1)

    # 重新进行前沿识别
    out = cal_ef2_v4_ultra_fast(out)

    # 对极小权重置零以提升数值稳定性
    eps = step * 1e-6
    for a in assets:
        col = out[a].to_numpy()
        col[np.abs(col) < eps] = 0.0
        out[a] = col
    return out


''' ========= 绩效批量、前沿识别、作图 =========
'''


def spec_axis_labels(ret_spec: ReturnSpec, risk_spec: RiskSpec) -> tuple[str, str]:
    # --- Return label ---
    if ret_spec.kind == "total_cum":
        ret_label = "总累计收益率"
    elif ret_spec.kind == "total_geom_annual":
        ret_label = "年化复利收益"
    elif ret_spec.kind == "total_log":
        ret_label = "总对数收益"
    elif ret_spec.kind == "total_log_annual":
        ret_label = "年化对数收益"
    elif ret_spec.kind == "ew_roll_cum":
        if ret_spec.N is None or ret_spec.lam is None:
            ret_label = "指数加权滚动累计收益率"
        else:
            ret_label = f"指数加权滚动{ret_spec.N}期累计收益率(λ={ret_spec.lam})"
    elif ret_spec.kind == "ew_roll_log":
        if ret_spec.N is None or ret_spec.lam is None:
            ret_label = "指数加权滚动对数收益"
        else:
            ret_label = f"指数加权滚动{ret_spec.N}期对数收益(λ={ret_spec.lam})"
    elif ret_spec.kind == "ew_roll_mean_simple":
        if ret_spec.N is None or ret_spec.lam is None:
            ret_label = "指数加权普通收益均值"
        else:
            ret_label = f"指数加权{ret_spec.N}期普通收益均值(λ={ret_spec.lam})"
    elif ret_spec.kind == "mean_simple":
        ret_label = "普通收益率均值"
    elif ret_spec.kind == "mean_log":
        ret_label = "对数收益率均值"
    else:
        ret_label = "收益"

    # --- Risk label ---
    if risk_spec.kind == "std":
        risk_label = "日收益标准差"
    elif risk_spec.kind == "std_annual":
        risk_label = "年化日收益标准差"
    elif risk_spec.kind == "log_std":
        risk_label = "日对数收益标准差"
    elif risk_spec.kind == "log_std_annual":
        risk_label = "年化日对数收益标准差"
    elif risk_spec.kind == "ew_roll_std":
        if risk_spec.N is None or risk_spec.lam is None:
            risk_label = "指数加权滚动标准差"
        else:
            risk_label = f"指数加权滚动{risk_spec.N}期标准差(λ={risk_spec.lam})"
    elif risk_spec.kind == "ew_roll_log_std":
        if risk_spec.N is None or risk_spec.lam is None:
            risk_label = "指数加权滚动对数收益标准差"
        else:
            risk_label = f"指数加权滚动{risk_spec.N}期对数收益标准差(λ={risk_spec.lam})"
    elif risk_spec.kind == "var_abs":
        risk_label = (f"{int(risk_spec.p * 100)}% VaR(绝对值)"
                      if risk_spec.p is not None else "VaR(绝对值)")
    elif risk_spec.kind == "es_abs":
        risk_label = (f"{int(risk_spec.p * 100)}% ES(绝对值)"
                      if risk_spec.p is not None else "ES(绝对值)")
    elif risk_spec.kind == "max_drawdown_abs":
        risk_label = "最大回撤率(绝对值)"
    else:
        risk_label = "风险"

    return ret_label, risk_label


def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray,
                              ret_spec: ReturnSpec | None = None,
                              risk_spec: RiskSpec | None = None) -> pd.DataFrame:
    if ret_spec is None:
        ret_spec = globals().get('RET_SPEC', ReturnSpec(kind="total_log_annual"))
    if risk_spec is None:
        risk_spec = globals().get('RISK_SPEC', RiskSpec(kind="log_std_annual"))
    return compute_metrics_for_W(port_daily, portfolio_allocs, ret_spec, risk_spec,
                                 col_ret="ret_annual", col_risk="vol_annual")


def cal_ef2_v4_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values
    sorted_idx = np.argsort(ret_values)[::-1]
    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)
    on_ef_sorted = (sorted_vol <= cummin_vol + 1e-6)
    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted
    data['on_ef'] = on_ef
    return data


def plot_efficient_frontier(
        scatter_points_data: List[Dict[str, Any]],
        title: str = '投资组合与有效前沿',
        x_axis_title: str = '年化波动率 (Annual Volatility)',
        y_axis_title: str = '年化收益率 (Annual Return)',
        x_col: str = 'vol_annual',
        y_col: str = 'ret_annual',
        hover_text_col: str = 'hover_text',
        output_filename: Optional[str] = None
):
    fig = go.Figure()
    for point_set in scatter_points_data:
        df = point_set["data"]
        marker_cfg = dict(
            color=point_set["color"],
            size=point_set["size"],
            opacity=point_set["opacity"],
        )
        if "marker_line" in point_set:
            marker_cfg["line"] = point_set["marker_line"]
        if "symbol" in point_set:
            marker_cfg["symbol"] = point_set["symbol"]

        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            hovertext=df[hover_text_col],
            hoverinfo='text',
            mode='markers',
            marker=marker_cfg,
            name=point_set["name"]
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        legend_title="图例",
        hovermode='closest'
    )
    if output_filename:
        fig.write_html(output_filename)
        print(f"{str_time()} 图表已保存到: {output_filename}")
    else:
        fig.show()


''' ========= 其他小工具 ========= 
'''


def dict_alloc_to_vector(assets: List[str], alloc_map: Dict[str, float]) -> np.ndarray:
    return np.array([alloc_map.get(a, 0.0) for a in assets], dtype=np.float64)


def make_upper_envelope_fn(R: np.ndarray, S: np.ndarray):
    order = np.argsort(S)
    S_sorted = S[order];
    R_sorted = R[order]

    def f(sig):
        sig = np.atleast_1d(sig)
        return np.interp(sig, S_sorted, R_sorted, left=R_sorted[0], right=R_sorted[-1])

    return f


def _fmt_return_value(val: float, spec: ReturnSpec) -> str:
    if spec.kind in ("total_cum", "total_geom_annual", "ew_roll_cum"):
        return f"{val:.2%}"
    if spec.kind == "total_log_annual":
        return f"{val:.4f} (≈{np.expm1(val):.2%})"
    if spec.kind == "total_log":
        return f"{val:.4f}"
    if spec.kind == "ew_roll_log":
        return f"{val:.4f}"
    return f"{val:.6g}"


def _fmt_risk_value(val: float, spec: RiskSpec) -> str:
    # 风险大多数口径用百分比展示直观
    return f"{val:.2%}"


def create_hover_text_factory(assets_list: List[str], ret_spec: ReturnSpec, risk_spec: RiskSpec):
    ret_label, risk_label = spec_axis_labels(ret_spec, risk_spec)

    def _create(row: pd.Series) -> str:
        ret_str = _fmt_return_value(float(row['ret_annual']), ret_spec)
        risk_str = _fmt_risk_value(float(row['vol_annual']), risk_spec)
        s = f"{ret_label}: {ret_str}<br>{risk_label}: {risk_str}<br><br><b>权重</b>:<br>"
        for asset in assets_list:
            if asset in row and row[asset] > 1e-4:
                s += f"  {asset}: {row[asset]:.1%}<br>"
        s += f"<br>锚点: {'是' if row.get('is_anchor', False) else '否'}"
        return s

    return _create


def build_export_view(df_named: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
    need_cols = assets + ['ret_annual', 'vol_annual', 'on_ef']
    miss = [c for c in need_cols if c not in df_named.columns]
    if miss:
        raise ValueError(f"导出缺少必要列: {miss}")

    out = df_named[need_cols].copy()

    # 1) 风险/收益列使用“统一口径”的动态中文名
    ret_label, risk_label = spec_axis_labels(RET_SPEC, RISK_SPEC)
    out = out.rename(columns={
        'ret_annual': ret_label,
        'vol_annual': risk_label,
        'on_ef': '是否有效前沿点',
    })

    # 2) 资产列改为“xx比例”
    rename_assets = {
        '货币现金类': '货币现金类比例',
        '固定收益类': '固定收益类比例',
        '混合策略类': '混合策略类比例',
        '权益投资类': '权益投资类比例',
        '另类投资类': '另类投资类比例',
    }
    out = out.rename(columns=rename_assets)

    # 3) 按“风险升序、收益降序”排序（用动态口径名）
    out = out.sort_values(by=[risk_label, ret_label],
                          ascending=[True, False],
                          kind='mergesort').reset_index(drop=True)
    return out


''' ========= 随机游走 =========
'''


# 随机游走: 前沿之下填充
def random_walk_below_frontier(W_anchor: np.ndarray, port_daily: np.ndarray,
                               single_limits, multi_limits,
                               per_anchor: int = 30, step: float = 0.01,
                               sigma_tol: float = 1e-4, seed: int = 123,
                               precision: str | float | None = None,
                               ret_spec: ReturnSpec | None = None,
                               risk_spec: RiskSpec | None = None):
    if ret_spec is None:
        ret_spec = globals().get('RET_SPEC', ReturnSpec(kind="total_log_annual"))
    if risk_spec is None:
        risk_spec = globals().get('RISK_SPEC', RiskSpec(kind="log_std_annual"))
    R_anchor = []
    S_anchor = []
    for w0 in W_anchor:
        r0, s0 = port_RS_by_spec(port_daily, w0, ret_spec, risk_spec)
        R_anchor.append(r0)
        S_anchor.append(s0)
    R_anchor = np.array(R_anchor)
    S_anchor = np.array(S_anchor)
    f_upper = make_upper_envelope_fn(R_anchor, S_anchor)

    rng = np.random.default_rng(seed)
    step_grid = _parse_precision(precision) if precision is not None else None
    collected = []
    for w0 in W_anchor:
        _, s0 = port_RS_by_spec(port_daily, w0, ret_spec, risk_spec)
        s_bar = s0 + sigma_tol
        for _ in range(per_anchor):
            eps = rng.normal(0.0, step, size=w0.size);
            eps -= eps.mean()
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None:
                continue
            if step_grid is not None:
                w_try = quantize_with_projection(w_try, step_grid, single_limits, multi_limits, rounds=5)
                if w_try is None:
                    continue
            r, s = port_RS_by_spec(port_daily, w_try, ret_spec, risk_spec)
            if (s <= s_bar + 1e-12) and (r <= f_upper(s) + 1e-8):
                collected.append(w_try)
    W = np.array(collected) if collected else np.empty((0, W_anchor.shape[1]))
    if (step_grid is not None) and W.size:
        W = dedup_by_grid(W, step_grid)
    return W


# 随机游走
def random_walk_frontier_explore(
        W_anchor: np.ndarray, port_daily: np.ndarray,
        single_limits, multi_limits,
        per_anchor: int = 30, step: float = 0.01,
        sigma_tol: float = 1e-4,  # “below” 模式下的风险上界松弛量
        seed: int = 123,
        precision: str | float | None = None,
        ret_spec: ReturnSpec | None = None,
        risk_spec: RiskSpec | None = None,
        walk_region: str = "below",  # "below" | "around" | "any"
        sigma_band: float | None = None,  # "around" 时的风险带宽；若 None 用 2*sigma_tol
        ret_tol_up: float = 1e-8  # "around" 和 "below" 时生效: 允许略微超过当前上包络的容忍（数值噪声）
):
    """
    使用随机游走方法在有效前沿附近探索新的投资组合权重。

    该函数从给定的锚点（初始权重）出发，通过添加符合约束的小幅扰动，生成新的投资组合权重，
    并根据指定的接受规则筛选出满足条件的新权重点。可用于扩展有效前沿样本或进行稳健性分析。

    参数:
        W_anchor (np.ndarray): 锚点权重矩阵，每一行是一个投资组合权重向量。
        port_daily (np.ndarray): 资产的日收益率矩阵，形状为 (T, N)，T为时间长度，N为资产数。
        single_limits: 单个资产权重的上下限约束。
        multi_limits: 多资产组合的线性约束。
        per_anchor (int): 每个锚点尝试生成的新权重数量，默认为30。
        step (float): 随机扰动的标准差，默认为0.01。
        sigma_tol (float): 在"below"模式下用于风险容忍度的阈值，默认为1e-4。
        seed (int): 随机数种子，默认为123。
        precision (str | float | None): 权重精度控制参数，用于量化权重网格。
        ret_spec (ReturnSpec | None): 收益计算规范对象，默认使用全局RET_SPEC。
        risk_spec (RiskSpec | None): 风险计算规范对象，默认使用全局RISK_SPEC。
        walk_region (str): 控制新点接受区域的策略，可选 "below", "around", "any"。
        sigma_band (float | None): 当walk_region="around"时使用的风险波动带宽。
        ret_tol_up (float): 允许略微超过当前上包络的容忍度（处理数值误差），默认为1e-8。

    返回:
        np.ndarray: 探索得到的新权重矩阵，形状为 (M, N)，M为有效新点数量。
    """
    if ret_spec is None:
        ret_spec = globals().get('RET_SPEC', ReturnSpec(kind="total_log_annual"))
    if risk_spec is None:
        risk_spec = globals().get('RISK_SPEC', RiskSpec(kind="log_std_annual"))

    # 基于当前锚点构建上包络函数，用于后续判断收益是否合理
    R_anchor, S_anchor = [], []
    for w0 in W_anchor:
        r0, s0 = port_RS_by_spec(port_daily, w0, ret_spec, risk_spec)
        R_anchor.append(r0)
        S_anchor.append(s0)
    R_anchor = np.array(R_anchor)
    S_anchor = np.array(S_anchor)
    f_upper = make_upper_envelope_fn(R_anchor, S_anchor)

    rng = np.random.default_rng(seed)
    step_grid = _parse_precision(precision) if precision is not None else None
    if sigma_band is None:
        sigma_band = 2.0 * sigma_tol

    collected = []
    for w0 in W_anchor:
        r0, s0 = port_RS_by_spec(port_daily, w0, ret_spec, risk_spec)
        s_bar = s0 + sigma_tol

        for _ in range(per_anchor):
            # 生成符合均值为0的正态分布扰动，并投影到约束空间
            eps = rng.normal(0.0, step, size=w0.size)
            eps -= eps.mean()
            w_try = project_to_constraints_pocs(w0 + eps, single_limits, multi_limits,
                                                max_iter=200, tol=1e-9, damping=0.9)
            if w_try is None:
                continue

            # 若设置了精度控制，则对权重进行量化处理
            if step_grid is not None:
                w_try = quantize_with_projection(w_try, step_grid,
                                                 single_limits, multi_limits, rounds=5)
                if w_try is None:
                    continue

            r, s = port_RS_by_spec(port_daily, w_try, ret_spec, risk_spec)

            # 根据 walk_region 策略决定是否接受该点
            if walk_region == "below":
                accept = (s <= s_bar + 1e-12) and (r <= f_upper(s) + ret_tol_up)
            elif walk_region == "around":
                in_band = (abs(s - s0) <= sigma_band + 1e-12)
                # “around” 建议仍参考上包络，避免无意义的远离上界的点爆量
                accept = in_band and (r <= f_upper(s) + ret_tol_up)
            elif walk_region == "any":
                accept = True
            else:
                raise ValueError(f"unknown walk_region: {walk_region}")

            if accept:
                collected.append(w_try)

    W = np.array(collected) if collected else np.empty((0, W_anchor.shape[1]))
    # 若启用精度控制，则去除重复点
    if (step_grid is not None) and W.size:
        W = dedup_by_grid(W, step_grid)
    return W


''' ========= 核心流程 (全局 & 等级)  ========= 
'''


def run_global_layer(cfg: Dict[str, Any],
                     assets_list: List[str],
                     mu: np.ndarray, covariance_matrix: np.ndarray,
                     port_daily_returns: np.ndarray,
                     single_limits_global, multi_limits_global) -> tuple[
    pd.DataFrame, List[Dict[str, Any]], pd.DataFrame]:
    """计算全局层：前沿、填充、作图数据与导出 DataFrame（未转中文列名）。"""
    # 前沿（锚点）
    print(f"{str_time()} [资产配置全局层] 计算全局有效前沿锚点...")
    # _, W_frontier, R_frontier, S_frontier, *_ = sweep_frontier_by_risk(
    #     mu, covariance_matrix, single_limits_global, multi_limits_global, n_grid=cfg['n_grid']
    # )
    _, W_frontier, R_frontier, S_frontier = sweep_frontier_by_risk_unified(
        port_daily_returns, RET_SPEC, RISK_SPEC,
        single_limits_global, multi_limits_global, n_grid=cfg['n_grid']
    )

    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    keep = np.isclose(R_sorted, np.maximum.accumulate(R_sorted), atol=1e-10)
    W_anchors = W_sorted[keep]

    # 填充
    print(f"{str_time()} [资产配置全局层] 随机游走填充前沿之下的可行空间...")
    W_below = random_walk_frontier_explore(
        W_anchor=W_anchors,
        port_daily=port_daily_returns,
        single_limits=single_limits_global,
        multi_limits=multi_limits_global,
        per_anchor=cfg['per_anchor'], step=cfg['step_rw'],
        sigma_tol=cfg['sigma_tol'], seed=cfg['seed'],
        precision=cfg['precision_choice'],
        ret_spec=RET_SPEC,
        risk_spec=RISK_SPEC,
        walk_region="any",
        sigma_band=2 * cfg['sigma_tol'],
        ret_tol_up=1e-6
    )

    # 绩效
    print(f"{str_time()} [资产配置全局层] 批量计算绩效指标与识别前沿...")
    perf_df = generate_alloc_perf_batch(
        port_daily_returns, np.vstack([W_anchors, W_below]) if len(W_below) else W_anchors
    )
    # 单独给锚点计算绩效指标
    anchor_perf = generate_alloc_perf_batch(port_daily_returns, W_anchors)
    anchor_perf['is_anchor'] = True
    perf_df['is_anchor'] = False
    full_df = pd.concat([perf_df, anchor_perf], ignore_index=True).drop_duplicates()
    full_df = cal_ef2_v4_ultra_fast(full_df)

    # 作图数据准备
    print(f"{str_time()} [资产配置全局层] 作图数据准备...")
    weight_cols = {f"w_{i}": assets_list[i] for i in range(len(assets_list))}
    full_df_named = full_df.rename(columns=weight_cols)
    create_hover_text = create_hover_text_factory(assets_list, RET_SPEC, RISK_SPEC)
    full_df_named['hover_text'] = full_df_named.apply(create_hover_text, axis=1)

    df_anchor = full_df_named[full_df_named['is_anchor'] == True]
    df_ef = full_df_named[(full_df_named['on_ef'] == True) & (full_df_named['is_anchor'] == False)]
    df_fill = full_df_named[(full_df_named['on_ef'] == False) & (full_df_named['is_anchor'] == False)]

    scatter_data = [
        {"data": df_fill, "name": "全局：前沿之下填充样本", "color": "lightblue", "size": 3, "opacity": 0.45},
        {"data": df_ef, "name": "全局：识别出的有效前沿", "color": "deepskyblue", "size": 3, "opacity": 0.9},
        {"data": df_anchor, "name": "全局：前沿锚点", "color": "crimson", "size": 5, "opacity": 0.9,
         "marker_line": dict(width=1, color='black')},
    ]
    return full_df_named, scatter_data, df_anchor


def run_level_layer(level: str,
                    cfg_lv: Dict[str, Any],
                    limits_lv,
                    multi_limits_lv,
                    assets_list: List[str],
                    mu: np.ndarray, covariance_matrix: np.ndarray,
                    port_daily_returns: np.ndarray,
                    base_alloc_map: Dict[str, Dict[str, float]]) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    """计算单个等级：作图数据与导出 DataFrame（中文列名）。"""
    colors = cfg_lv['color']
    create_hover_text = create_hover_text_factory(assets_list, RET_SPEC, RISK_SPEC)
    weight_cols_map = {f'w_{j}': assets_list[j] for j in range(len(assets_list))}

    # 1) 前沿锚点
    print(f"{str_time()} [资产配置等级层] 处理等级: {level}  计算有效前沿锚点...")
    # W_anchors_lv, _, _ = compute_frontier_anchors(
    #     mu, covariance_matrix, single_limits=limits_lv, multi_limits=multi_limits_lv, n_grid=cfg_lv['n_grid']
    # )
    grid, W_frontier, R_frontier, S_frontier = sweep_frontier_by_risk_unified(
        port_daily_returns, RET_SPEC, RISK_SPEC,
        limits_lv, multi_limits_lv, n_grid=cfg_lv['n_grid']
    )
    idx = np.argsort(S_frontier)
    S_sorted, R_sorted, W_sorted = S_frontier[idx], R_frontier[idx], W_frontier[idx]
    keep = np.isclose(R_sorted, np.maximum.accumulate(R_sorted), atol=1e-10)
    W_anchors_lv = W_sorted[keep]

    # 2) 随机游走 + POCS +（可选）精度
    print(f"{str_time()} [资产配置等级层] 处理等级: {level}  填充前沿下的可配置区域...")
    W_below_lv = random_walk_frontier_explore(
        W_anchor=W_anchors_lv,
        port_daily=port_daily_returns,
        single_limits=limits_lv,
        multi_limits=multi_limits_lv,
        per_anchor=cfg_lv['per_anchor'], step=cfg_lv['step_rw'],
        sigma_tol=cfg_lv['sigma_tol'], seed=cfg_lv['seed'],
        precision=cfg_lv['precision_choice'],
        ret_spec=RET_SPEC, risk_spec=RISK_SPEC,
        walk_region="any",
        sigma_band=2 * cfg_lv['sigma_tol'],
        ret_tol_up=1e-6
    )
    if cfg_lv['precision_choice'] is not None and len(W_below_lv):
        W_below_lv = dedup_by_grid(W_below_lv, _parse_precision(cfg_lv['precision_choice']))

    # 3) 绩效
    print(f"{str_time()} [资产配置等级层] 处理等级: {level}  计算绩效指标与识别前沿...")
    perf_anchor_lv = generate_alloc_perf_batch(port_daily_returns, W_anchors_lv).rename(columns=weight_cols_map)
    perf_anchor_lv['hover_text'] = perf_anchor_lv.apply(create_hover_text, axis=1)

    scatter_data = [{
        "data": perf_anchor_lv, "name": f"{level} 有效前沿",
        "color": colors, "size": 4, "opacity": 0.9, "marker_line": dict(width=1, color='black')
    }]

    if len(W_below_lv) > 0:
        perf_fill_lv = generate_alloc_perf_batch(port_daily_returns, W_below_lv).rename(columns=weight_cols_map)
        perf_fill_lv['hover_text'] = perf_fill_lv.apply(create_hover_text, axis=1)
        scatter_data.append({
            "data": perf_fill_lv, "name": f"{level} 可配置空间",
            "color": colors, "size": 2, "opacity": 0.35
        })
    else:
        perf_fill_lv = pd.DataFrame(columns=list(perf_anchor_lv.columns))

    # 4) 基准点
    if level in base_alloc_map:
        base_w_raw = dict_alloc_to_vector(assets_list, base_alloc_map[level])
    else:
        base_w_raw = level_midpoint_weights(limits_lv)
    base_w = project_baseline_to_level(base_w_raw, limits_lv)
    base_perf_df = generate_alloc_perf_batch(port_daily_returns, base_w.reshape(1, -1)).rename(columns=weight_cols_map)
    base_perf_df['hover_text'] = base_perf_df.apply(create_hover_text, axis=1)
    scatter_data.append({
        "data": base_perf_df, "name": f"{level} 基准点",
        "color": colors, "size": 9, "opacity": 1.0,
        "symbol": "star", "marker_line": dict(width=1.5, color='black')
    })

    # 5) 导出：合并→（若设精度）量化→重算绩效→识别前沿→去重
    frames = [perf_anchor_lv.assign(is_anchor=True)]
    if len(perf_fill_lv):
        frames.append(perf_fill_lv.assign(is_anchor=False))
    level_full_df = pd.concat(frames, ignore_index=True)

    if cfg_lv['precision_choice'] is not None:
        step_val = _parse_precision(cfg_lv['precision_choice'])
        level_full_df = quantize_df_for_export(
            df_in=level_full_df, assets=assets_list, step=step_val,
            single_limits=limits_lv, multi_limits=multi_limits_lv, port_daily=port_daily_returns
        )
    else:
        level_full_df = cal_ef2_v4_ultra_fast(level_full_df)

    level_full_df = level_full_df.drop_duplicates(subset=assets_list, keep='first').reset_index(drop=True)
    export_df = build_export_view(level_full_df, assets_list)

    return scatter_data, export_df


# ========= 主程序 =========

if __name__ == '__main__':
    # ---- 统一配置（可按需调整）----
    RET_SPEC = ReturnSpec(kind="total_log", N=120, lam=0.94)
    RISK_SPEC = RiskSpec(kind="log_std", N=120, lam=0.94, p=0.99)
    ''' ---------------------------------- 可用指标:
"total_cum",  # 总累计收益率: Π(1+R)-1
"total_geom_annual",  # 总年化复利收益: (Π(1+R))**(252/T)-1
"total_log",  # 总对数收益: Σ log(1+R)
"total_log_annual",  # 总年化对数收益: (Σlog(1+R))/T*252
"ew_roll_cum",  # 指数加权滚动N区间累计收益率
"ew_roll_log"  # 指数加权滚动N区间对数收益率
"ew_roll_mean_simple"   # 指数加权普通收益均值
"mean_simple",  # 普通收益率均值
"mean_log"  # 对数收益率均值

"std",  # 总日普通收益标准差
"std_annual",  # 总日普通收益年化标准差
"log_std",  # 总日对数收益标准差
"log_std_annual",  # 总日对数收益年化标准差
"ew_roll_std",  # 指数加权滚动N区间普通收益率标准差
"ew_roll_log_std",  # 指数加权滚动N区间对数收益率标准差
"var_abs",  # x% VaR(取绝对值, 无亏损时=0)
"es_abs",  # x% ES (取绝对值, 无亏损时=0)
"max_drawdown_abs"  # 最大回撤率(取绝对值, 无回撤时=0)
    '''
    if RISK_SPEC.kind in ("var_abs", "es_abs") and RISK_SPEC.p is None:
        raise ValueError("使用 VaR/ES 口径时必须设置 RiskSpec.p（例如 0.95/0.99）")

    CONFIG = {
        # 1) 数据
        "input_excel": "历史净值数据.xlsx",
        "sheet_name": "历史净值数据",
        "rename_map": {
            "货基指数": "货币现金类", "固收类": "固定收益类", "混合类": "混合策略类",
            "权益类": "权益投资类", "另类": "另类投资类",
            "安逸型": "C1", "谨慎型": "C2", "稳健型": "C3",
            "增长型": "C4", "进取型": "C5", "激进型": "C6"
        },
        "assets_list": ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类'],

        # 2) 等级边界（示例；请按业务改数值）
        "proposed_alloc_base": {
            'C1': {'货币现金类': 1.0, '固定收益类': 0.0, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C2': {'货币现金类': 0.2, '固定收益类': 0.8, '混合策略类': 0.0, '权益投资类': 0.0, '另类投资类': 0.0},
            'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35, '权益投资类': 0.0, '另类投资类': 0.0},
            'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
            'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
            'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
        },
        "risk_level_bounds": {
            'C1': {'货币现金类': (1.00, 1.00), '固定收益类': (0.00, 0.00), '混合策略类': (0.00, 0.00),
                   '权益投资类': (0.00, 0.00), '另类投资类': (0.00, 0.00)},
            'C2': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.00),
                   '权益投资类': (0.00, 0.00), '另类投资类': (0.00, 0.00)},
            'C3': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.00), '另类投资类': (0.00, 0.00)},
            'C4': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.2 * 1.2), '另类投资类': (0.00, 0.05 * 1.2)},
            'C5': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.4 * 1.2), '另类投资类': (0.00, 0.1 * 1.2)},
            'C6': {'货币现金类': (0.00, 1.00), '固定收益类': (0.00, 1.00), '混合策略类': (0.00, 0.35 * 1.2),
                   '权益投资类': (0.00, 0.6 * 1.2), '另类投资类': (0.00, 0.1 * 1.2)},
        },

        # 3) 全局层参数
        "global_layer": {
            "n_grid": 300,  # 全局层网格点数量
            "per_anchor": 100,  # 每个锚点的采样数量
            "step_rw": 0.12,  # 随机游走步长
            "sigma_tol": 1e-4,  # 标准差容忍度阈值
            "seed": 123,  # 随机数生成器种子
            "precision_choice": "0.5%",  # 精度选择参数: "0.1%", "0.2%", "0.5%", None
        },
        "global_multi_limits": {  # 例： (assets.index('权益投资类'), assets.index('另类投资类')): (0.0, 0.70),
        },

        # 4) 等级层公共默认参数（各等级可覆盖）
        "level_defaults": {
            "n_grid": 200,  # 网格点数量，默认值为200
            "per_anchor": 80,  # 每个锚点的数量，默认值为80
            "step_rw": 0.08,  # 随机游走步长，默认值为0.08
            "sigma_tol": 1e-4,  # sigma容忍度阈值，默认值为1e-4
            "seed": 2024,  # 随机种子
            "precision_choice": None,  # 精度选择选项, 支持 "0.1%", "0.2%", "0.5%", None
        },
        # 每个等级的颜色（作图用）
        "level_colors": {
            'C1': '#1f77b4', 'C2': '#ff7f0e', 'C3': '#2ca02c',
            'C4': '#d62728', 'C5': '#9467bd', 'C6': '#8c564b'
        },

        # 5) 输出
        "plot_output_html": "efficient_frontier_无调整.html",  # 如: "efficient_frontier.html"
        "export_excel": "前沿与等级可配置空间导出.xlsx",
    }
    s_t = time.time()

    # ---- 运行：加载数据 ----
    print(f"{str_time()} [加载数据] ... ")
    assets_list = CONFIG["assets_list"]
    hist_df, port_daily_returns, mu, covariance_matrix = load_returns_from_excel(
        CONFIG["input_excel"], CONFIG["sheet_name"], assets_list, CONFIG["rename_map"]
    )

    # ---- 构造边界 ----
    print(f"{str_time()} [构造边界] ... ")
    per_level_limits = bounds_dict_to_limits(assets_list, CONFIG["risk_level_bounds"])
    single_limits_global = global_envelope_limits(per_level_limits)
    multi_limits_global = CONFIG["global_multi_limits"]

    # ---- 资产配置全局层 ----
    print(f"{str_time()} [资产配置全局层]")
    glb_cfg = CONFIG["global_layer"] | {"n_grid": CONFIG["global_layer"]["n_grid"]}
    full_df_glb, scatter_data, df_anchor_glb = run_global_layer(
        glb_cfg, assets_list, mu, covariance_matrix,
        port_daily_returns, single_limits_global, multi_limits_global
    )
    print(f"{str_time()} [资产配置全局层] 量化与导出视图构建...")
    export_sheets: Dict[str, pd.DataFrame] = {}
    if glb_cfg["precision_choice"] is not None:
        step_val = _parse_precision(glb_cfg["precision_choice"])
        quantize_df = quantize_df_for_export(
            full_df_glb, assets_list, step_val,
            single_limits_global, multi_limits_global, port_daily_returns
        )
        export_sheets['全局'] = build_export_view(quantize_df, assets_list)
    else:
        export_sheets['全局'] = build_export_view(full_df_glb, assets_list)

    # ---- 资产配置等级层 ----
    print(f"{str_time()} [资产配置等级层]")
    level_defaults = CONFIG["level_defaults"]
    for level in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']:
        print(f"{str_time()} [资产配置等级层] 处理等级:", level)
        lv_cfg = {
            **level_defaults,
            "color": CONFIG["level_colors"][level],
        }
        limits_lv = per_level_limits[level]
        multi_limits_lv = {}  # 如需等级专属组约束，在此填写
        lv_scatter, lv_export_df = run_level_layer(
            level, lv_cfg, limits_lv, multi_limits_lv,
            assets_list, mu, covariance_matrix, port_daily_returns,
            CONFIG["proposed_alloc_base"]
        )
        scatter_data.extend(lv_scatter)
        export_sheets[level] = lv_export_df

    # ---- 作图 ----
    print(f"{str_time()} [作图]")
    xlab, ylab = spec_axis_labels(RET_SPEC, RISK_SPEC)
    plot_efficient_frontier(
        scatter_points_data=scatter_data,
        title="全局与 C1~C6 等级：有效前沿（QCQP）+ 前沿下可行空间（随机游走+POCS）",
        output_filename=CONFIG["plot_output_html"],
        x_axis_title=ylab, y_axis_title=xlab
    )

    # ---- 导出 Excel ----
    print(f"{str_time()} [导出 Excel]")
    with pd.ExcelWriter(CONFIG["export_excel"]) as writer:
        for sheet_name, df_out in export_sheets.items():
            safe_name = sheet_name[:31]
            df_out.to_excel(writer, sheet_name=safe_name, index=False)
    print(f"{str_time()} [完成] ✅ 耗时 {time.time() - s_t:.2f}s")
