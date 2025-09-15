import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal, Tuple
from plotly.subplots import make_subplots


def read_data_and_prepare(excel_file_name: str = "万得指数数据.xlsx",
                          excel_sheet_name: str = "万得原始数据") -> pd.DataFrame:
    # 读取万得原始数据
    index_df = pd.read_excel(excel_file_name, sheet_name=excel_sheet_name)

    # 规范日期列
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df[index_df["date"] < pd.to_datetime("2025-09-01")]

    # 识别数值列（除 date 外）
    value_cols = [c for c in index_df.columns if c != "date"]
    print(f"识别到的数值列: \n{value_cols}")

    # 1) 统一为字符串后用正则去除逗号与空白字符
    tmp_vals = index_df[value_cols].astype(str).replace(r'[\s,]+', '', regex=True)
    # 2) 列级别安全转数值（无法转换的置为 NaN）
    index_df[value_cols] = tmp_vals.apply(pd.to_numeric, errors="coerce")

    # 设置索引
    index_df = index_df.set_index("date")

    # 计算日收益率
    ret_df = index_df.pct_change()

    # 对首行（由 pct_change 产生）NaN 用 0 填充；其余 NaN 保留以避免引入伪收益
    if len(ret_df) > 0:
        first_idx = ret_df.index[0]
        ret_df.loc[first_idx] = ret_df.loc[first_idx].fillna(0.0)

    print(ret_df.head())

    # 构造虚拟净值：所有产品起始净值为 1
    nv_df = (1.0 + ret_df).cumprod()
    if len(nv_df) > 0:
        # 明确第一天为 1，并对缺失进行前向填充（缺失视为当日不变）
        nv_df.iloc[0] = 1.0
    nv_df = nv_df.ffill()
    return nv_df


# 画图：净值曲线与日收益率
def plot_lines(df: pd.DataFrame, title: str, y_tick_format: str | None, output_html: str | None):
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))
    fig.update_layout(
        title=title,
        xaxis_title="date",
        yaxis_title="value",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if y_tick_format:
        fig.update_yaxes(tickformat=y_tick_format)
    if output_html:
        fig.write_html(output_html)
        print(f"图已保存: {output_html}")
    fig.show()


def plot_combined_dashboard(original_nv: pd.DataFrame,
                            custom_nv: pd.DataFrame,
                            title: str = "大类资产构建分析"):
    """
    在单一图表中展示：
    1. 原始指数净值走势
    2. 构建的大类资产净值走势
    3. 构建的大类资产收益率相关性
    """
    print(f"\n正在生成组合图表: {title}...")

    # 计算大类资产的收益率和相关性
    custom_ret = custom_nv.pct_change().dropna()
    corr = custom_ret.corr().fillna(0.0)
    corr_labels = list(corr.columns)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=(
            "原始指数净值走势 (虚拟净值, 起始为1)",
            "构建的大类资产净值走势 (虚拟净值, 起时为1)",
            "构建的大类资产收益率相关性"
        )
    )

    # Row 1: 原始指数净值 (不显示图例)
    for col in original_nv.columns:
        fig.add_trace(
            go.Scatter(x=original_nv.index, y=original_nv[col], mode='lines', name=col, showlegend=False),
            row=1, col=1
        )

    # Row 2: 大类资产净值 (显示图例)
    for col in custom_nv.columns:
        fig.add_trace(
            go.Scatter(x=custom_nv.index, y=custom_nv[col], mode='lines', name=col),
            row=2, col=1
        )

    # Row 3: 相关性热力图
    fig.add_trace(
        go.Heatmap(
            z=corr.values, x=corr_labels, y=corr_labels,
            colorscale='RdBu', zmid=0.0, zmin=-1.0, zmax=1.0,
            colorbar=dict(title='corr', len=0.3, y=0.15, yanchor='middle'),
            text=np.around(corr.values, 2),
            texttemplate="%{text}"
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=1500,
        title_text=title,
        showlegend=True,
        margin=dict(l=60, r=200, t=100, b=80),  # 增加右边距以容纳图例
        legend=dict(
            title="大类资产",
            orientation='v',
            yanchor='top', y=0.65,  # 调整图例位置使其在中间图表的旁边
            xanchor='left', x=1.02,
            font=dict(size=9),
            bgcolor='rgba(255,255,255,0.7)'
        )
    )

    fig.update_yaxes(title_text='虚拟净值', row=1, col=1)
    fig.update_yaxes(title_text='虚拟净值', row=2, col=1)

    fig.show()


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


def generate_custom_indices(original_nav_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    根据 config 配置，基于原始指数净值数据，生成自定义的大类资产净值序列。
    """
    print("\n开始生成自定义大类资产...")
    ret_df = original_nav_df.pct_change()
    # 收益率首行为 NaN，用 0 填充
    if not ret_df.empty:
        ret_df.iloc[0] = ret_df.iloc[0].fillna(0.0)

    custom_indices = {}

    for name, params in config.items():
        print(f"  正在处理: {name}")
        method = params.get('method', 'equal')
        index_names = params.get('index_names', [])

        if not index_names:
            print(f"    (警告: '{name}' 未指定 index_names，已跳过)")
            continue

        # 检查数据中是否存在所需指数
        missing_indices = [idx for idx in index_names if idx not in original_nav_df.columns]
        if missing_indices:
            print(f"    (警告: 在数据中找不到以下指数，已跳过 '{name}': {', '.join(missing_indices)})")
            continue

        # 构造该大类资产的收益率子集，并去除全为 NaN 的行
        sub_ret = ret_df[index_names].dropna(how='all')

        # 如果只有一个成分，权重为1
        if len(index_names) == 1:
            w = np.array([1.0])
        else:
            # 计算权重
            if method == 'equal':
                w = np.full(len(index_names), 1.0 / len(index_names))
            elif method == 'manual':
                manual_weights = params.get('manual_weights', [])
                if len(manual_weights) != len(index_names):
                    print(f"    (警告: '{name}' 的 manual_weights 长度与 index_names 不匹配，回退到等权)")
                    w = np.full(len(index_names), 1.0 / len(index_names))
                else:
                    w = np.asarray(manual_weights, dtype=float)
                    s = w.sum()
                    if s <= 0:
                        print(f"    (警告: '{name}' 的 manual_weights 之和为非正数，回退到等权)")
                        w = np.full(len(index_names), 1.0 / len(index_names))
                    else:
                        w = w / s
            elif method == 'inverse_vol':
                # 使用可用时间段的波动率
                vol = sub_ret.std().reindex(index_names).fillna(0)
                inv = 1.0 / vol.replace(0, np.nan)
                inv = inv.fillna(0.0)
                s = inv.sum()
                if s == 0:
                    w = np.full(len(index_names), 1.0 / len(index_names))
                else:
                    w = (inv / s).values.astype(float)
            elif method == 'risk_parity':
                risk_metric = params.get('risk_metric', 'vol')
                risk_budget = params.get('risk_budget', [1.0] * len(index_names))
                if len(risk_budget) != len(index_names):
                    print(f"    (警告: '{name}' 的 risk_budget 长度与 index_names 不匹配，回退到等风险预算)")
                    risk_budget = [1.0] * len(index_names)

                rp_alpha = params.get('rp_alpha', 0.95)
                rp_tol = params.get('rp_tol', 1e-6)
                rp_max_iter = params.get('rp_max_iter', 50)

                # 填充NA以进行矩阵运算
                R_matrix = sub_ret.fillna(0).values

                if risk_metric == 'vol':
                    cov = np.cov(R_matrix, rowvar=False)
                    w = _erc_vol_n_assets(cov, np.asarray(risk_budget, dtype=float), tol=rp_tol, max_iter=rp_max_iter)
                else:  # ES or VaR
                    w = _erc_tail_n_assets(R_matrix, metric=risk_metric, alpha=rp_alpha,
                                           budget=np.asarray(risk_budget, dtype=float),
                                           tol=rp_tol, max_iter=rp_max_iter)
            else:
                print(f"    (警告: 未知方法 '{method}'，回退到等权)")
                w = np.full(len(index_names), 1.0 / len(index_names))

        pretty_weights = ', '.join([f"{asset}={weight:.3f}" for asset, weight in zip(index_names, w)])
        print(f"    方法: {method}, 计算权重: {pretty_weights}")

        # 计算组合日收益与虚拟净值
        port_ret = sub_ret.fillna(0).values @ w
        port_nv = (1.0 + port_ret).cumprod()

        # 对齐到原始索引，并进行前向填充
        port_series = pd.Series(port_nv, index=sub_ret.index, name=name)
        # Reindex and fill, then make sure first value is 1.0 if it's a NaN
        aligned_series = port_series.reindex(original_nav_df.index).ffill()
        if pd.isna(aligned_series.iloc[0]):
            aligned_series.iloc[0] = 1.0
        custom_indices[name] = aligned_series.ffill()

    result_df = pd.DataFrame(custom_indices)
    # 保证第一天净值为1
    if not result_df.empty:
        result_df.iloc[0] = 1.0
        result_df = result_df.ffill()

    print("自定义大类资产生成完毕。")
    return result_df


if __name__ == '__main__':
    # 构建大类的配置
    config = {
        "权益类": {
            # 权重分配方式: 'equal'-等权; 'inverse_vol'-逆波动率; 'manual'-手工指定; 'risk_parity'-风险平价
            'method': 'manual',
            # 指定构建权重的成分名称（必须与数据中的列名一致）
            'index_names': ['万得普通股票型基金指数', '万得股票策略私募指数', '万得QDII股票型基金指数'],
            # 指定手工权重（仅当 method='manual' 时有效）
            'manual_weights': [0.4, 0.4, 0.2],
        },
        "固收类": {
            'method': 'manual',
            'index_names': ['万得短期纯债型基金指数', '万得中长期纯债型指数', '万得QDII债券型基金指数'],
            'manual_weights': [0.25, 0.5, 0.25],
        },
        "另类": {
            'method': 'equal',
            'index_names': ['伦敦金现', '南华商品指数', '万得另类投资基金总指数'],
        },
        "货基指数": {
            'method': 'equal',
            'index_names': ['万得货币市场基金指数'],
        },
        "混合类": {
            'method': 'risk_parity',
            'index_names': ['万得纯债型基金总指数', '万得普通股票型基金指数', '万得另类投资基金总指数'],
            'risk_metric': 'vol',  # 风险平价度量 ['vol', 'ES', 'VaR'], 选择 weight_mode='risk_parity' 时有效
            'rp_alpha': 0.95,  # ES/VaR 置信度 (左尾 1-alpha), 选择 risk_metric='ES'/'VaR' 时有效
            'rp_tol': 1e-6,  # 迭代收敛阈值, 选择 weight_mode='risk_parity' 时有效
            'rp_max_iter': 50,  # 迭代上限, 选择 weight_mode='risk_parity' 时有效
            'risk_budget': (1, 5, 10),  # 风险预算比例 (与 selected_assets 等长), 选择 weight_mode='risk_parity' 时有效
        }
    }
    # 读取并处理数据
    nav_df = read_data_and_prepare()

    # 根据 config 生成大类资产
    custom_nav_df = generate_custom_indices(nav_df, config)

    # 绘制组合图表
    if not custom_nav_df.empty:
        plot_combined_dashboard(nav_df, custom_nav_df, title="大类资产构建分析")
    else:
        print("\n没有生成任何大类资产，仅绘制原始指数图。")
        plot_lines(nav_df, title="万得指数：虚拟净值（起始为1）", y_tick_format=None, output_html=None)

    custom_nav_df.to_excel("历史净值数据_万得指数.xlsx", sheet_name="历史净值数据")
