# -*- encoding: utf-8 -*-
"""
@File: cal_index_return_risk.py
@Modify Time: 2025/09/11
@Author: Kevin-Chen
@Descriptions:
    - 读取文件夹下所有 Excel（中英混合列名）
    - 规范化 + 去重（按 (date, index_code) ）
    - 可选起始日、最小样本、缺失比例剔除
    - 收益/风险指标可选（向量化计算）
    - 交集不足自动回退为“各自样本期”
    - 画风险-收益散点（点大小一致，hover 显示代码/名称/指标）
    - 画从 1 开始的虚拟净值
"""

import warnings
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ===================== 全局设置 =====================
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

TRADING_DAYS = 252


# ===================== I/O 与预处理 =====================
def get_excel_files(folder_path: str) -> List[str]:
    files = []
    for root, _, fs in os.walk(folder_path):
        for f in fs:
            if f.lower().endswith('.xlsx'):
                files.append(os.path.join(root, f))
    return files


def read_and_combine_excel(file_list: List[str]) -> pd.DataFrame:
    """
    读取所有 Excel，并统一列名 -> 纵表：date, index_code, index_name, close
    """
    cols_map = {
        '日期Date': 'date',
        '指数代码Index Code': 'index_code',
        '指数中文全称Index Chinese Name(Full)': 'index_name',
        '收盘Close': 'close',
    }
    dfs = []
    for fp in file_list:
        df = pd.read_excel(fp)
        df = df.rename(columns=cols_map)
        req = ['date', 'index_code', 'index_name', 'close']
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f'文件缺少列 {missing}: {fp}')
        df = df[req].copy()
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
        df['index_code'] = df['index_code'].astype(str)
        df['index_name'] = df['index_name'].astype(str)
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        dfs.append(df)
    if not dfs:
        raise RuntimeError("未在指定目录发现任何 .xlsx 文件。")
    all_data = pd.concat(dfs, ignore_index=True)
    print(all_data.head(3))
    print(all_data.tail(3))
    return all_data


def normalize_and_deduplicate(all_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    - 规范类型
    - 去重：对 (date, index_code) 保留“最后一条”
    - 返回：规范后的 all_data 与 code->最新名称 映射
    """
    df = all_data.copy()
    df['index_code'] = df['index_code'].astype(str).str.strip()
    df['index_name'] = df['index_name'].astype(str).str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')

    df = df.sort_values(['index_code', 'date'])
    dup = df.duplicated(['date', 'index_code']).sum()
    if dup > 0:
        print(f"[INFO] 检测到重复 (date, index_code) 记录 {dup} 条，已按最后一条保留去重。")
    df = df.drop_duplicates(['date', 'index_code'], keep='last')

    code2name = (df.dropna(subset=['index_name'])
                 .groupby('index_code')['index_name']
                 .last())

    return df, code2name


def build_close_wide(all_data_norm: pd.DataFrame) -> pd.DataFrame:
    return (all_data_norm
            .pivot(index='date', columns='index_name', values='close')
            .sort_index())


# ===================== 过滤器 =====================
def filter_by_start_date(close_df: pd.DataFrame, start_date: str | None) -> pd.DataFrame:
    """
    仅保留：首个有效净值日期 <= start_date 的指数；并从 start_date 起截断
    """
    if start_date is None:
        return close_df
    start_ts = pd.Timestamp(start_date)

    has_data = close_df.notna().sum() > 0
    first_valid = close_df.loc[:, has_data].apply(lambda s: s.first_valid_index())
    keep_cols = first_valid[first_valid <= start_ts].index
    return close_df.loc[close_df.index >= start_ts, keep_cols]


def drop_by_missing_ratio(close_df: pd.DataFrame,
                          max_missing_ratio: float | None = None) -> pd.DataFrame:
    """
    按列缺失比例剔除：缺失占比 > 阈值 的列被剔除
    """
    if max_missing_ratio is None:
        return close_df
    total = close_df.shape[0]
    if total == 0:
        return close_df
    miss_ratio = close_df.isna().sum() / float(total)
    keep = miss_ratio <= float(max_missing_ratio)
    kept = close_df.loc[:, keep]
    if kept.shape[1] == 0:
        # 兜底：保留缺失最少的前若干列
        topk = min(10, miss_ratio.size)
        kept = close_df.loc[:, miss_ratio.nsmallest(topk).index]
        print("[WARN] 所有列缺失占比均超阈值，已兜底保留缺失最少的若干列。")
    return kept


def drop_by_min_obs(close_df: pd.DataFrame, min_obs: int) -> pd.DataFrame:
    if min_obs is None or min_obs <= 0:
        return close_df
    valid_cols = close_df.notna().sum() >= int(min_obs)
    return close_df.loc[:, valid_cols]


# ===================== 序列与通用工具 =====================
def compute_returns(close_df: pd.DataFrame,
                    use_intersection: bool = True) -> tuple[pd.DataFrame, bool]:
    """
    计算日简单收益。若 use_intersection=True，则删除含 NaN 的行以保证横向可比；
    若交集不足（<2 行），自动回退为“各自样本期”。
    返回 (ret_df, intersection_used)
    """
    ret = close_df.pct_change()
    if use_intersection:
        ret_inter = ret.dropna(how='any')
        if ret_inter.shape[0] >= 2 and ret_inter.shape[1] >= 1:
            return ret_inter, True
        print("[WARN] 交集日期不足，自动回退到按列各自样本期。")
    return ret, False


def rolling_cum_return(ret_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算滚动 window 日累计收益 r_(t-window+1:t) = exp( sum(log(1+r)) ) - 1
    （比 rolling.apply(np.prod) 更快更稳）
    """
    log1p = np.log1p(ret_df)
    roll_sum = log1p.rolling(window, min_periods=window).sum()
    return np.expm1(roll_sum)


def ewma_series_last(s_or_df: pd.Series | pd.DataFrame, lam: float) -> pd.Series:
    """
    对时间序列做 EWMA（lambda=衰减因子），返回“最后一个时点”的估计值（按列）
    alpha = 1 - lambda
    """
    alpha = 1.0 - float(lam)
    ewm = s_or_df.ewm(alpha=alpha, adjust=False).mean()
    if isinstance(ewm, pd.DataFrame):
        return ewm.tail(1).iloc[0]
    else:
        return pd.Series({s_or_df.name: ewm.iloc[-1]})


# ===================== 指标计算（可选项） =====================
def compute_return_metric(ret_df: pd.DataFrame,
                          which: str,
                          roll_ret_win: int = 5,
                          lam_ret: float = 0.94,
                          intersection_used: bool = True) -> pd.Series:
    """
    收益指标可选：
    - 'total_return'               累计收益率
    - 'ann_return'                 年化收益率（CAGR）
    - 'daily_mean'                 日收益均值
    - 'roll_mean'                  滚动N日累计收益均值
    - 'ewma_daily_mean'            指数加权平均日收益率均值（取最后 EWMA 均值）
    - 'ewma_roll_mean'             指数加权平均滚动N日累计收益均值（取最后 EWMA 均值）
    """
    which = which.lower()

    if which == 'daily_mean':
        return ret_df.mean()

    if which in ('total_return', 'ann_return', 'cagr'):
        # 统一处理基于净值路径的两类
        if intersection_used:
            wealth = (1.0 + ret_df).cumprod()
            total = wealth.iloc[-1] / wealth.iloc[0] - 1.0
            if which == 'total_return':
                return total
            T = ret_df.shape[0]
            return (wealth.iloc[-1] / wealth.iloc[0]) ** (TRADING_DAYS / T) - 1.0
        else:
            # 各列各自样本
            s_notna = ret_df.notna().sum()
            # total return
            wealth = (1.0 + ret_df).cumprod()
            first = wealth.apply(
                lambda col: col[col.first_valid_index()] if col.first_valid_index() is not None else np.nan)
            last = wealth.apply(
                lambda col: col[col.last_valid_index()] if col.last_valid_index() is not None else np.nan)
            total = last / first - 1.0
            if which == 'total_return':
                return total
            # CAGR
            T = s_notna.clip(lower=1)
            return (last / first) ** (TRADING_DAYS / T) - 1.0

    if which == 'roll_mean':
        roll = rolling_cum_return(ret_df, roll_ret_win)
        return roll.mean()

    if which == 'ewma_daily_mean':
        return ewma_series_last(ret_df, lam_ret)

    if which == 'ewma_roll_mean':
        roll = rolling_cum_return(ret_df, roll_ret_win)
        return ewma_series_last(roll, lam_ret)

    raise ValueError(f"未知收益指标: {which}")


def compute_risk_metric(ret_df: pd.DataFrame,
                        which: str,
                        roll_vol_win: int = 20,
                        lam_vol: float = 0.94) -> pd.Series:
    """
    风险指标可选：
    - 'vol'                        波动率（日）
    - 'ann_vol'                    年化波动率
    - 'roll_vol_mean'              滚动N日波动率均值
    - 'ewma_roll_vol_mean'         指数加权平均滚动N日波动率均值（取最后 EWMA 均值）
    - 'var_99'                     99% VaR（下 1% 分位），无亏损则取 0
    - 'es_99'                      99% ES（VaR 以下均值），无亏损则取 0
    - 'mdd'                        最大回撤（无亏损则取 0）
    """
    which = which.lower()

    if which == 'vol':
        return ret_df.std(ddof=1)

    if which == 'ann_vol':
        return ret_df.std(ddof=1) * np.sqrt(TRADING_DAYS)

    if which == 'roll_vol_mean':
        roll_std = ret_df.rolling(roll_vol_win, min_periods=roll_vol_win).std(ddof=1)
        return roll_std.mean()

    if which == 'ewma_roll_vol_mean':
        roll_std = ret_df.rolling(roll_vol_win, min_periods=roll_vol_win).std(ddof=1)
        return ewma_series_last(roll_std, lam_vol)

    if which in ('var_99', 'es_99'):
        # VaR/ES 按“收益”分布定义：VaR = 下 1% 分位（一般为负数）
        # 无亏损（全 >=0） -> 设 VaR=0, ES=0
        has_loss = ret_df.lt(0).any(axis=0)
        var = ret_df.quantile(0.01, interpolation='linear')
        var = var.where(has_loss, other=0.0)  # 无亏损强制为 0
        if which == 'var_99':
            return var
        mask = ret_df.le(var, axis=1)
        es = ret_df.where(mask).mean()
        es = es.where(has_loss, other=0.0)
        return es

    if which == 'mdd':
        wealth = (1.0 + ret_df).cumprod()
        roll_max = wealth.cummax()
        dd = wealth / roll_max - 1.0
        mdd = dd.min()
        # 无亏损：若所有日收益 >=0，则净值单调不降，回撤为 0
        has_loss = ret_df.lt(0).any(axis=0)
        return mdd.where(has_loss, other=0.0)

    raise ValueError(f"未知风险指标: {which}")


def compute_metrics_table(ret_df: pd.DataFrame,
                          close_df: pd.DataFrame,
                          code2name: pd.Series,
                          return_metric: str,
                          risk_metric: str,
                          roll_ret_win: int = 5,
                          roll_vol_win: int = 20,
                          lam_ret: float = 0.94,
                          lam_vol: float = 0.94,
                          intersection_used: bool = True) -> pd.DataFrame:
    """
    汇总指标表：两列（risk, return），并附上 index_name
    """
    ret_series = compute_return_metric(
        ret_df, close_df, which=return_metric,
        roll_ret_win=roll_ret_win, lam_ret=lam_ret,
        intersection_used=intersection_used
    )
    risk_series = compute_risk_metric(
        ret_df, close_df, which=risk_metric,
        roll_vol_win=roll_vol_win, lam_vol=lam_vol
    )

    metrics = pd.DataFrame({
        'return_metric': ret_series,
        'risk_metric': risk_series
    })
    metrics['index_name'] = metrics.index.map(code2name).fillna('')

    # 清掉全 NaN
    metrics = metrics.dropna(how='all', subset=['return_metric', 'risk_metric'])
    return metrics


# ===================== 绘图 =====================
def plot_risk_return(metrics: pd.DataFrame,
                     return_label: str,
                     risk_label: str,
                     title_suffix: str = "") -> None:
    if metrics.empty:
        raise RuntimeError("指标表为空，无法绘制风险-收益图。")

    marker_size = 10
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics['risk_metric'],
        y=metrics['return_metric'],
        mode='markers',
        marker=dict(size=marker_size, opacity=0.9, line=dict(width=0.5)),
        text=metrics.index.astype(str),
        customdata=np.column_stack([
            metrics['index_name'].to_numpy(),
            metrics['return_metric'].to_numpy(),
            metrics['risk_metric'].to_numpy(),
        ]),
        hovertemplate=(
                "<b>指数代码</b>: %{text}<br>"
                "<b>指数名称</b>: %{customdata[0]}<br>"
                f"{return_label}: " + "%{customdata[1]:.4f}<br>"
                                      f"{risk_label}: " + "%{customdata[2]:.4f}"
                                                          "<extra></extra>"
        )
    ))
    fig.update_layout(
        title=f"指数风险–收益散点图 | 收益={return_label} | 风险={risk_label} {title_suffix}",
        xaxis_title=risk_label,
        yaxis_title=return_label,
        template='plotly_white',
        hovermode='closest'
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dash", opacity=0.6)
    fig.show()


def plot_nav(ret_df: pd.DataFrame,
             code2name: pd.Series,
             intersection_used: bool) -> None:
    fig = go.Figure()
    if intersection_used:
        if ret_df.shape[0] < 1:
            print("[WARN] 交集收益样本过短，跳过 NAV 绘制。")
        else:
            nav = (1.0 + ret_df).cumprod()
            nav = nav / nav.iloc[0]
            for code in nav.columns:
                nm = code2name.get(code, '')
                fig.add_trace(go.Scatter(
                    x=nav.index, y=nav[code], mode='lines',
                    name=str(code),
                    hovertemplate=f"<b>{code} | {nm}</b><br>%{{x|%Y-%m-%d}}  虚拟净值: %{{y:.3f}}<extra></extra>"
                ))
    else:
        for code in ret_df.columns:
            s = ret_df[code].dropna()
            if s.shape[0] < 1:
                continue
            nav = (1.0 + s).cumprod()
            nav = nav / nav.iloc[0]
            nm = code2name.get(code, '')
            fig.add_trace(go.Scatter(
                x=nav.index, y=nav, mode='lines', name=str(code),
                hovertemplate=f"<b>{code} | {nm}</b><br>%{{x|%Y-%m-%d}}  虚拟净值: %{{y:.3f}}<extra></extra>"
            ))

    fig.update_layout(
        title="指数虚拟净值（起点=1）",
        xaxis_title="日期",
        yaxis_title="虚拟净值",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    fig.show()


# ===================== 主流程 =====================
def main(
        folder: str = "./",
        start_date: str | None = None,  # 例如 '2015-01-01'
        min_obs_days: int = 252,  # 价格层面最少有效天数
        use_intersection: bool = True,  # 优先交集；不足自动回退
        max_missing_ratio: float | None = None,  # 例如 0.20；None 不启用
        # ----- 指标选择 -----
        return_metric: str = 'ann_return',  # 见 compute_return_metric 文档
        risk_metric: str = 'ann_vol',  # 见 compute_risk_metric 文档
        # ----- 超参数 -----
        roll_ret_win: int = 5,
        roll_vol_win: int = 20,
        lam_ret: float = 0.94,
        lam_vol: float = 0.94,
):
    files = get_excel_files(folder)
    print(f"[INFO] 找到 Excel 文件 {len(files)} 个")
    if not files:
        raise RuntimeError("未找到任何 Excel 文件。")

    all_data = read_and_combine_excel(files)
    all_data_norm, code2name = normalize_and_deduplicate(all_data)

    close_df = build_close_wide(all_data_norm)
    close_df = filter_by_start_date(close_df, start_date)
    close_df = drop_by_missing_ratio(close_df, max_missing_ratio)
    close_df = drop_by_min_obs(close_df, min_obs_days)

    if close_df.shape[1] == 0:
        raise RuntimeError("没有满足样本长度/起始日/缺失比例条件的指数。")

    ret_df, inter_used = compute_returns(close_df, use_intersection=use_intersection)

    metrics = compute_metrics_table(
        ret_df, close_df, code2name,
        return_metric=return_metric,
        risk_metric=risk_metric,
        roll_ret_win=roll_ret_win,
        roll_vol_win=roll_vol_win,
        lam_ret=lam_ret,
        lam_vol=lam_vol,
        intersection_used=inter_used
    )

    if metrics.empty:
        raise RuntimeError("指标表为空（可能交集或各自样本过短）。请放宽条件或关闭交集。")

    # 标签（用于坐标轴与 hover）
    return_labels = {
        'total_return': '累计收益率',
        'ann_return': '年化收益率(CAGR)',
        'cagr': '年化收益率(CAGR)',
        'daily_mean': '日收益均值',
        'roll5_mean': f'滚动{roll_ret_win}日累计收益均值',
        'ewma_daily_mean': f'EWMA日收益均值(λ={lam_ret})',
        'ewma_roll5_mean': f'EWMA滚动{roll_ret_win}日累计收益均值(λ={lam_ret})',
    }
    risk_labels = {
        'vol': '波动率(日)',
        'ann_vol': '年化波动率',
        'roll20_vol_mean': f'滚动{roll_vol_win}日波动率均值',
        'ewma_roll20_vol_mean': f'EWMA滚动{roll_vol_win}日波动率均值(λ={lam_vol})',
        'var_99': '99%VaR（收益分位）',
        'es_99': '99%ES（尾部均值）',
        'mdd': '最大回撤',
    }
    rlab = return_labels.get(return_metric.lower(), return_metric)
    klab = risk_labels.get(risk_metric.lower(), risk_metric)

    title_suffix = "（交集对齐）" if inter_used else "（各自样本期）"
    plot_risk_return(metrics, rlab, klab, title_suffix=title_suffix)
    plot_nav(ret_df, code2name, intersection_used=inter_used)


if __name__ == '__main__':
    # ============ 参数配置 ============
    FOLDER = "./"  # Excel 所在目录
    START_DATE = None  # 例如 '2015-01-01'；None 表示不限制
    MIN_OBS = 252  # 价格层面最少有效天数
    USE_INTERSECTION = True  # 优先交集；不足自动回退
    MAX_MISSING_RATIO = None  # 例如 0.20；None 不启用

    RETURN_METRIC = 'ewma_roll_mean'  # ['total_return','ann_return','daily_mean','roll_mean','ewma_daily_mean','ewma_roll_mean']
    RISK_METRIC = 'ewma_roll_vol_mean'  # ['vol','ann_vol','roll_vol_mean','ewma_roll_vol_mean','var_99','es_99','mdd']

    ROLL_RET_WIN = 25 * 3  # 用于计算 'roll_mean' 和 'ewma_roll_mean'
    ROLL_VOL_WIN = 25 * 3  # 用于计算 'roll_vol_mean' 和 'ewma_roll_vol_mean'
    LAM_RET = 0.94
    LAM_VOL = 0.94
    # =================================

    main(
        folder=FOLDER,
        start_date=START_DATE,
        min_obs_days=MIN_OBS,
        use_intersection=USE_INTERSECTION,
        max_missing_ratio=MAX_MISSING_RATIO,
        return_metric=RETURN_METRIC,
        risk_metric=RISK_METRIC,
        roll_ret_win=ROLL_RET_WIN,
        roll_vol_win=ROLL_VOL_WIN,
        lam_ret=LAM_RET,
        lam_vol=LAM_VOL,
    )
