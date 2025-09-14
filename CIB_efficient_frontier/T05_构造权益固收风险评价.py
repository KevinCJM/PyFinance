# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal, Tuple
from plotly.subplots import make_subplots


def plot_asset_trends(df, assets_to_plot=None, title='资产历史净值走势'):
    """
    使用 Plotly 绘制资产历史净值的折线图。

    参数:
        df (pd.DataFrame): 包含净值数据的DataFrame，索引必须是日期时间。
        assets_to_plot (list): 需要在图表中绘制的资产列名列表。
        title (str): 图表的标题。
    """
    print(f"\n正在生成图表: {title}...")
    fig = go.Figure()

    # 若未指定，则展示 DataFrame 中的所有资产列
    if assets_to_plot is None:
        assets_to_plot = list(df.columns)

    for asset in assets_to_plot:
        if asset in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[asset],
                mode='lines',
                name=asset
            ))
        else:
            print(f"  (警告: 在数据中未找到资产 '{asset}'，已跳过)")

    fig.update_layout(
        title_text=title,
        xaxis_title='日期',
        yaxis_title='净值',
        legend_title='图例'
    )

    fig.show()


def plot_timeseries_and_corr(df_nv: pd.DataFrame,
                             df_ret: pd.DataFrame,
                             title_ts: str = '净值走势',
                             title_corr: str = '收益率相关性矩阵'):
    """
    在同一网页（单一 Figure）中，第一行展示净值时间序列，第二行展示收益率相关性热力图。
    df_nv: 含所有要展示的净值序列（含组合）
    df_ret: 与 df_nv 列对应的收益率序列（用于计算相关性）
    """
    print(f"\n正在生成图表: {title_ts} + {title_corr}...")
    corr = df_ret.corr().fillna(0.0)
    cols = list(corr.columns)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        subplot_titles=(title_ts, title_corr)
    )
    # Row 1: timeseries
    for col in df_nv.columns:
        fig.add_trace(
            go.Scatter(x=df_nv.index, y=df_nv[col], mode='lines', name=col),
            row=1, col=1
        )
    # Row 2: correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr.values, x=cols, y=cols,
            colorscale='RdBu', zmid=0.0, zmin=-1.0, zmax=1.0,
            colorbar=dict(
                title='corr',
                thickness=10,
                lenmode='fraction', len=0.45,
                y=0.22, yanchor='middle',
                x=1.04, xanchor='left'      # 更紧贴绘图区，给右侧图例留出空间
            )
        ),
        row=2, col=1
    )
    fig.update_layout(
        height=900,
        showlegend=True,
        # 右侧预留空间用于图例（靠近上半页）
        margin=dict(l=60, r=180, t=80, b=80),
        legend=dict(
            orientation='v',
            # 将图例固定在上半页右侧：使用整体坐标的上部区域
            yanchor='top', y=0.98,
            xanchor='left', x=1.02,
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.7)'
        )
    )
    fig.update_xaxes(title_text='日期', row=1, col=1)
    fig.update_yaxes(title_text='净值', row=1, col=1)
    fig.update_xaxes(title_text='资产', row=2, col=1)
    fig.update_yaxes(title_text='资产', row=2, col=1)
    fig.show()


def load_and_process_data(file_path='历史净值数据.xlsx'):
    """
    从指定的 Excel 文件加载和预处理历史净值数据。
    该函数参考 B07_random_weights_reweight.py 中的数据加载逻辑。

    参数:
        file_path (str): Excel文件的路径。

    返回:
        pd.DataFrame: 经过预处理后的数据。
    """
    print(f"正在从 {file_path} 读取数据...")
    # 1. 读取 Excel 文件，指定工作表名称
    hist_value = pd.read_excel(file_path, sheet_name='历史净值数据')

    # 2. 将 'date' 列设为索引，并转换为日期时间格式
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)

    # 3. 删除包含任何缺失值的行，并按时间顺序排序
    hist_value = hist_value.dropna().sort_index(ascending=True)

    # 4. 重命名列以提高可读性
    rename_dict = {
        "货基指数": "货币现金类", '固收类': '固定收益类', '混合类': '混合策略类',
        '权益类': '权益投资类', '另类': '另类投资类', '安逸型': 'C1',
        '谨慎型': 'C2', '稳健型': 'C3', '增长型': 'C4',
        '进取型': 'C5', '激进型': 'C6'
    }
    hist_value = hist_value.rename(columns=rename_dict)

    print("数据读取和预处理完成。")
    return hist_value


def run_custom_portfolio(
        *,
        data_file: str,
        selected_assets: list,
        weight_mode: Literal['equal', 'inverse_vol', 'manual', 'risk_parity'],
        manual_weights: Tuple[float, ...],
        risk_metric: Literal['vol', 'ES', 'VaR'],
        rp_alpha: float,
        rp_tol: float,
        rp_max_iter: int,
        risk_budget: Tuple[float, ...],
):
    """按参数构建“自定义组合”并作图（供 __main__ 调用）。"""
    # 1) 加载与预处理
    hist_value = load_and_process_data(file_path=data_file)
    hist_value = hist_value / hist_value.iloc[0, :]

    # 2) 不单独绘制全资产；仅在最后绘制“全资产 + 自定义组合”一张图

    # 3) 构造“自定义组合”
    sub = hist_value[selected_assets].dropna()
    ret = sub.pct_change().dropna()

    # 3.2 计算权重
    if weight_mode == 'equal':
        w = np.full(len(selected_assets), 1.0 / len(selected_assets))
    elif weight_mode == 'manual':
        if len(manual_weights) != len(selected_assets):
            raise ValueError('manual_weights 长度需与 selected_assets 一致')
        w = np.asarray(manual_weights, dtype=float)
        s = w.sum()
        if s <= 0:
            raise ValueError('手工权重之和必须>0')
        w = w / s
    elif weight_mode == 'inverse_vol':
        # 用全样本波动率做逆波动权重（多资产）
        vol = ret.std().reindex(selected_assets)
        inv = 1.0 / vol.replace(0, np.nan)
        inv = inv.fillna(0.0)
        s = inv.sum()
        if s == 0:
            w = np.full(len(selected_assets), 1.0 / len(selected_assets))
        else:
            w = (inv / s).values.astype(float)
    elif weight_mode == 'risk_parity':
        # 多资产风险平价（支持 vol/ES/VaR）
        if len(risk_budget) != len(selected_assets):
            raise ValueError('risk_budget 长度需与 selected_assets 一致')
        if risk_metric == 'vol':
            cov = ret.cov().reindex(index=selected_assets, columns=selected_assets).values
            w = _erc_vol_n_assets(cov, np.asarray(risk_budget, dtype=float), tol=rp_tol, max_iter=rp_max_iter)
        else:
            R = ret[selected_assets].values
            w = _erc_tail_n_assets(R, metric=risk_metric, alpha=rp_alpha, budget=np.asarray(risk_budget, dtype=float),
                                   tol=rp_tol, max_iter=rp_max_iter)
    else:
        raise ValueError(f'未知的 weight_mode: {weight_mode}')

    mode_str = weight_mode if weight_mode != 'risk_parity' else f"risk_parity[{risk_metric}, alpha={rp_alpha}]"
    pretty = ', '.join([f"{a}={wi:.3f}" for a, wi in zip(selected_assets, w)])
    print(f"使用权重({mode_str}): {pretty}")

    # 3.3 组合日收益与虚拟净值
    port_ret = ret[selected_assets].values @ w
    port_nv = np.cumprod(1.0 + port_ret)
    port_nv = pd.Series(port_nv, index=ret.index, name='自定义组合')

    # 3.4 合并到一个 DataFrame 用于展示（所有资产 + 组合）
    show_df = hist_value.loc[ret.index].copy()
    show_df['自定义组合'] = port_nv

    # 构建与 show_df 对齐的收益率矩阵，用于相关性
    ret_all = hist_value.pct_change().dropna()
    ret_all = ret_all.loc[ret.index]  # 对齐索引
    ret_all['自定义组合'] = pd.Series(port_ret, index=ret.index)
    # 只保留与展示净值同名的列（包括自定义组合）
    ret_all = ret_all.loc[:, [c for c in show_df.columns if c in ret_all.columns]]

    # 3.5 单一页面绘图：上-净值序列，下-相关性矩阵
    plot_timeseries_and_corr(show_df, ret_all,
                             title_ts='全资产 + 自定义组合（净值）',
                             title_corr='全资产 + 自定义组合（收益率相关性）')


def _solve_risk_parity_two_assets(
        ret_e: np.ndarray,
        ret_b: np.ndarray,
        *,
        risk_metric: Literal['vol', 'ES', 'VaR'] = 'vol',
        alpha: float = 0.95,
        tol: float = 1e-6,
        max_iter: int = 50,
        risk_budget: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[float, float]:
    """
    两资产风险平价（Equal Risk Contribution）权重求解。
    - risk_metric='vol'：使用协方差矩阵，最小化 (RC_e - RC_b)^2（鲁棒的一维数值解）。
    - risk_metric='ES'/'VaR'：历史模拟法，使用尾部样本子梯度做固定点迭代 w = g_b / (g_e + g_b)。
    返回: (w_equity, w_bond)
    """
    eps = 1e-9
    # 粗暴防护
    if ret_e.shape[0] != ret_b.shape[0] or ret_e.shape[0] == 0:
        return 0.5, 0.5

    if risk_metric == 'vol':
        # 协方差与相关
        s1 = float(ret_e.std(ddof=1))
        s2 = float(ret_b.std(ddof=1))
        if s1 < eps and s2 < eps:
            return 0.5, 0.5
        rho = float(np.corrcoef(ret_e, ret_b)[0, 1]) if (s1 > eps and s2 > eps) else 0.0
        # 风险预算归一化到占比
        b1, b2 = risk_budget
        bt = b1 + b2 if (b1 + b2) > eps else 2.0
        t1 = float(b1 / bt)

        # ERC 目标：最小化 (RC1_norm - t1)^2，w∈[0,1]
        def rc_diff_sq(w: float) -> float:
            w = min(max(w, 0.0), 1.0)
            u = 1.0 - w
            # sigma_p
            sp2 = (w * w * s1 * s1) + (u * u * s2 * s2) + (2.0 * w * u * rho * s1 * s2)
            sp = np.sqrt(max(sp2, 0.0)) + eps
            # Sigma*w 分量
            a1 = w * s1 * s1 + u * rho * s1 * s2
            a2 = u * s2 * s2 + w * rho * s1 * s2
            RC1 = w * (a1 / sp)
            RC2 = u * (a2 / sp)
            sRC = RC1 + RC2 + eps
            rc1n = RC1 / sRC
            d = rc1n - t1
            return float(d * d)

        # 网格粗扫 + 局部细化（黄金分割）
        grid = np.linspace(0.0, 1.0, 501)
        vals = np.array([rc_diff_sq(x) for x in grid])
        i = int(vals.argmin())
        lo = max(0.0, grid[max(0, i - 1)])
        hi = min(1.0, grid[min(len(grid) - 1, i + 1)])
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        f1 = rc_diff_sq(x1)
        f2 = rc_diff_sq(x2)
        for _ in range(60):
            if f1 > f2:
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + phi * (hi - lo)
                f2 = rc_diff_sq(x2)
            else:
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - phi * (hi - lo)
                f1 = rc_diff_sq(x1)
            if hi - lo < tol:
                break
        w = float((lo + hi) * 0.5)
        return w, 1.0 - w

    # 历史 ES/VaR：固定点迭代（尾部子梯度 + 风险预算）
    tail = max(1e-4, 1.0 - float(alpha))
    w = 0.5
    b1, b2 = risk_budget
    b1 = float(max(b1, 0.0))
    b2 = float(max(b2, 0.0))
    if b1 + b2 < eps:
        b1, b2 = 1.0, 1.0
    for _ in range(max_iter):
        u = 1.0 - w
        rp = w * ret_e + u * ret_b
        q = np.quantile(rp, tail)
        if risk_metric == 'ES':
            idx = rp <= q
            if idx.sum() == 0:
                break
            g_e = -float(ret_e[idx].mean())
            g_b = -float(ret_b[idx].mean())
        else:  # VaR
            j = int(np.abs(rp - q).argmin())
            g_e = -float(ret_e[j])
            g_b = -float(ret_b[j])
        # w_i = (b_i / g_i) / sum_j (b_j / g_j)
        if abs(g_e) < eps and abs(g_b) < eps:
            break
        ge = max(abs(g_e), eps)
        gb = max(abs(g_b), eps)
        w_new = (b1 / ge) / ((b1 / ge) + (b2 / gb))
        w_new = min(max(w_new, 0.0), 1.0)
        if abs(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return w, 1.0 - w


def _erc_vol_n_assets(cov: np.ndarray, budget: np.ndarray, *, tol: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    """
    N资产下基于波动率(协方差)的风险平价（等/配比风险贡献）迭代解：
    固定点迭代： w_i <- (b_i / (Σw)_i)，再归一化到 simplex。
    """
    eps = 1e-12
    N = cov.shape[0]
    w = np.full(N, 1.0 / N, dtype=float)
    b = np.maximum(budget.astype(float), 0.0)
    if b.sum() <= eps:
        b[:] = 1.0
    for _ in range(max_iter):
        sw = cov @ w
        denom = np.maximum(sw, eps)
        w_new = (b / denom)
        s = w_new.sum()
        if s <= eps:
            w_new = np.full(N, 1.0 / N)
        else:
            w_new /= s
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


def _erc_tail_n_assets(R: np.ndarray, *, metric: Literal['ES', 'VaR'] = 'ES', alpha: float = 0.95,
                       budget: np.ndarray, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """
    N资产下基于 ES/VaR 的风险平价近似解：
    固定点迭代： w_i <- (b_i / |g_i|) / sum_j (b_j / |g_j|)，其中 g 为尾部子梯度。
    R: (T, N) 日收益矩阵；metric: 'ES' 或 'VaR'。
    """
    eps = 1e-12
    T, N = R.shape
    w = np.full(N, 1.0 / N, dtype=float)
    b = np.maximum(budget.astype(float), 0.0)
    if b.sum() <= eps:
        b[:] = 1.0
    tail = max(1e-4, 1.0 - float(alpha))
    for _ in range(max_iter):
        rp = R @ w
        q = np.quantile(rp, tail)
        if metric == 'ES':
            idx = rp <= q
            if idx.sum() == 0:
                break
            g = -R[idx].mean(axis=0)
        else:  # VaR
            j = int(np.abs(rp - q).argmin())
            g = -R[j]
        g = np.abs(g)
        g = np.maximum(g, eps)
        w_new = (b / g)
        s = w_new.sum()
        if s <= eps:
            w_new = np.full(N, 1.0 / N)
        else:
            w_new /= s
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


if __name__ == '__main__':
    # ========== 配置参数（仅需修改此处） ==========
    DATA_FILE = '历史净值数据.xlsx'  # 历史净值数据文件路径
    selected_assets = ['权益投资类', '固定收益类', '另类投资类']  # 参与“自定义组合”的资产列表（支持多资产）
    # 权重分配方式: 'equal' - 等权; 'inverse_vol' - 逆波动率; 'manual' - 手工指定; 'risk_parity' - 风险平价
    weight_mode: Literal['equal', 'inverse_vol', 'manual', 'risk_parity'] = 'manual'
    manual_weights: Tuple[float, ...] = (0.2, 0.9, 0.3)  # 与 selected_assets 等长, 选择 weight_mode='manual' 时有效
    risk_metric: Literal['vol', 'ES', 'VaR'] = 'ES'  # 风险平价度量, 选择 weight_mode='risk_parity' 时有效
    rp_alpha: float = 0.95  # ES/VaR 置信度 (左尾 1-alpha), 选择 risk_metric='ES'/'VaR' 时有效
    rp_tol: float = 1e-6  # 迭代收敛阈值, 选择 weight_mode='risk_parity' 时有效
    rp_max_iter: int = 50  # 迭代上限, 选择 weight_mode='risk_parity' 时有效
    risk_budget: Tuple[float, ...] = (9.0, 1.0, 5.0)  # 风险预算比例 (与 selected_assets 等长), 选择 weight_mode='risk_parity' 时有效

    # ========== 执行 ==========
    run_custom_portfolio(
        data_file=DATA_FILE,
        selected_assets=selected_assets,
        weight_mode=weight_mode,
        manual_weights=manual_weights,
        risk_metric=risk_metric,
        rp_alpha=rp_alpha,
        rp_tol=rp_tol,
        rp_max_iter=rp_max_iter,
        risk_budget=risk_budget,
    )
