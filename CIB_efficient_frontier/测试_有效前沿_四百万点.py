import pandas as pd
import numpy as np
import datetime
from copy import deepcopy
from typing import Dict, Optional
from scipy.spatial import ConvexHull
import time
import warnings

warnings.filterwarnings('ignore')
from itertools import combinations
import os
import plotly.graph_objects as go


# 加权求和计算指数组合每日收益数据
def generate_alloc_perf(portfolio_alloc: Dict) -> Dict:
    port_daily = hist_value_r.copy()
    port_daily = port_daily[portfolio_alloc.keys()]

    port_daily.loc[:, 'r'] = port_daily[portfolio_alloc.keys()].values @ np.array(list(portfolio_alloc.values()))

    port_daily['value'] = (1 + port_daily['r']).cumprod()
    port_daily.loc[port_daily.index[0] - datetime.timedelta(days=1), 'value'] = 1
    port_daily = port_daily.sort_index()

    port_vol = np.std(port_daily['r'].apply(lambda r: np.log(r + 1)).dropna(), ddof=1) * np.sqrt(252)
    port_ret = np.log(port_daily['value'].iloc[-1] / port_daily['value'].iloc[0]) / (len(port_daily) - 1) * 252

    return {"ret_annual": float(port_ret), "vol_annual": float(port_vol)}


# 计算不同权重下资产组合的收益波动信息
def generate_alloc_perf_batch(port_daily: np.ndarray, portfolio_allocs: np.ndarray, p95=1.65) -> np.ndarray:
    assert port_daily.shape[1] == portfolio_allocs.shape[1]
    # Step 1: 所有组合收益率（向量化）
    port_return_daily = port_daily @ portfolio_allocs.T

    # Step 2: 累计收益率（log 累乘）
    port_cum_returns = np.cumprod(1 + port_return_daily, axis=0)

    # Step 3: 年化收益率
    log_total_ret = np.log(port_cum_returns[-1, :])
    port_ret_annual = log_total_ret / (port_return_daily.shape[0]) * 252

    # Step 4: 年化波动率（用 log return）
    log_returns = np.log(1 + port_return_daily)
    port_vol_annual = np.std(log_returns, axis=0, ddof=1) * np.sqrt(252)

    sp_annual = port_ret_annual / port_vol_annual
    VaR95_annual = port_ret_annual - port_vol_annual * p95

    # Step 5: 打包结果为 DataFrame
    ret_df = pd.DataFrame({
        "ret_annual": port_ret_annual,
        "vol_annual": port_vol_annual,
        "sp_annual": sp_annual,
        "VaR95_annual": VaR95_annual
    })

    # Step 6: 合并权重数据
    weight_df = pd.DataFrame(portfolio_allocs)

    return pd.concat([weight_df, ret_df], axis=1)


def generate_simplex_grid(n_assets: int, resolution: int):
    """
    n_assets: 几个资产类别（维度）
    resolution: 把 1 划分为 resolution 份，即 Δ=1/resolution
    返回所有加总为1，且每个资产为 Δ 的倍数的组合
    """
    total_slots = resolution + n_assets - 1
    grid = []
    for bars in combinations(range(total_slots), n_assets - 1):
        bars = (-1,) + bars + (total_slots,)
        vec = [bars[i + 1] - bars[i] - 1 for i in range(n_assets)]
        grid.append(np.array(vec) / resolution)
    return np.array(grid)


# 通过近似法求解
def generate_simplex_grid_constraint(n_assets: int, resolution: int, cons: np.array, threshold: float):
    """
    n_assets: 几个资产类别（维度）
    resolution: 把 1 划分为 resolution 份，即 Δ=1/resolution
    返回所有加总为1，且每个资产为 Δ 的倍数的组合
    """
    total_slots = resolution + n_assets - 1
    grid = []
    for bars in combinations(range(total_slots), n_assets - 1):
        bars = (-1,) + bars + (total_slots,)
        vec = [bars[i + 1] - bars[i] - 1 for i in range(n_assets)]
        if abs(np.array(vec) / resolution - cons).max(axis=1).min() <= threshold:
            grid.append(np.array(vec) / resolution)
    return np.array(grid)


def cal_ef2_v4_ultra_fast(data):
    """
    版本4: 超高速版本 - 适用于大数据集
    优点：
    - Pure NumPy操作，避免所有pandas开销
    - 最小内存分配
    - 最高性能
    缺点：
    - 代码相对复杂
    - 失去pandas的易读性下`
    时间复杂度: O(n log n)
    适用场景: 大数据集 (> 100K点)
    """
    # 获取NumPy数组
    ret_values = data['ret_annual'].values
    vol_values = data['vol_annual'].values

    # 排序索引
    sorted_idx = np.argsort(ret_values)[::-1]

    # 向量化计算累计最小值
    sorted_vol = vol_values[sorted_idx]
    cummin_vol = np.minimum.accumulate(sorted_vol)

    # 识别有效前沿点
    on_ef_sorted = (sorted_vol == cummin_vol)

    # 重新映射到原始顺序
    on_ef = np.zeros(len(data), dtype=bool)
    on_ef[sorted_idx] = on_ef_sorted

    data['on_ef'] = on_ef
    return data


def find_best(weights):
    random_port_perf = []
    for w in weights:
        asset_alloc = dict(zip(assets_list, list(w)))
        alloc_perf = generate_alloc_perf(asset_alloc)
        random_port_perf.append(dict(**asset_alloc, **alloc_perf))
    random_port_perf = pd.DataFrame(random_port_perf)
    random_port_perf['on_ef'] = False
    random_port_perf = cal_ef2_v4_ultra_fast(random_port_perf)
    return random_port_perf, random_port_perf[random_port_perf['on_ef'] == True].iloc[:, :5].values


# 绘制有效前沿图
def _build_hover_text(row, assets_cols):
    parts = [
        f"年化收益率: {row['ret_annual']:.2%}",
        f"年化波动率: {row['vol_annual']:.2%}",
        "<br><b>资产权重</b>:"
    ]
    for col in assets_cols:
        if col in row and pd.notna(row[col]) and float(row[col]) > 1e-6:
            parts.append(f"<br>{col}: {float(row[col]):.1%}")
    return "".join(parts)


def plot_efficient_frontier_plotly(df: pd.DataFrame,
                                   assets_cols: list,
                                   title: str = '资产组合有效前沿',
                                   sample_non_ef: int = 50000,
                                   save_html: Optional[str] = 'efficient_frontier.html'):
    """
    使用 plotly.graph_objects 进行交互式展示。
    - x: vol_annual
    - y: ret_annual
    - on_ef: True 的点高亮
    - 为避免内存压力，非前沿点可抽样展示（默认最多 5 万个）
    """
    df = df.copy()
    if 'hover_text' not in df.columns:
        df['hover_text'] = df.apply(lambda r: _build_hover_text(r, assets_cols), axis=1)

    ef_df = df[df['on_ef'] == True]
    non_ef_df = df[df['on_ef'] != True]

    if sample_non_ef is not None and len(non_ef_df) > sample_non_ef:
        non_ef_df = non_ef_df.sample(n=sample_non_ef, random_state=42)

    fig = go.Figure()
    # 非前沿点（灰色、半透明）
    if len(non_ef_df) > 0:
        fig.add_trace(go.Scatter(
            x=non_ef_df['vol_annual'],
            y=non_ef_df['ret_annual'],
            mode='markers',
            marker=dict(color='rgba(150,150,150,0.5)', size=3),
            hovertext=non_ef_df['hover_text'],
            hoverinfo='text',
            name='其他组合'
        ))

    # 有效前沿点（蓝色）
    if len(ef_df) > 0:
        fig.add_trace(go.Scatter(
            x=ef_df['vol_annual'],
            y=ef_df['ret_annual'],
            mode='markers',
            marker=dict(color='rgba(0, 102, 204, 0.9)', size=4),
            hovertext=ef_df['hover_text'],
            hoverinfo='text',
            name='有效前沿'
        ))

    fig.update_layout(
        title=title,
        xaxis_title='年化波动率 (vol_annual)',
        yaxis_title='年化收益率 (ret_annual)',
        hovermode='closest',
        legend_title='图例'
    )

    # 可选输出 HTML，便于在 Docker/无GUI 环境查看
    if save_html:
        try:
            fig.write_html(save_html, include_plotlyjs='cdn')
        except Exception:
            pass

    fig.show()


def read_excel_auto(path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    健壮的 Excel 读取：依据扩展名选择引擎；.xlsx/.xlsm 用 openpyxl；.xls 用 xlrd。
    若缺少对应引擎，会抛出带指引的异常。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.xlsx', '.xlsm', '.xltx', '.xltm'}:
        engine = 'openpyxl'
    elif ext == '.xls':
        engine = 'xlrd'
    else:
        engine = None  # 让 pandas 自行判断
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine=engine)
    except ImportError as e:
        hint = (
            "读取 Excel 失败：缺少引擎。对于 .xlsx/.xlsm 请安装 openpyxl；"
            "对于 .xls 请安装 xlrd<2.0.0。原始错误: %r" % (e,)
        )
        raise RuntimeError(hint)


if __name__ == '__main__':
    # 读取数据
    hist_value = read_excel_auto('历史净值数据.xlsx', sheet_name='历史净值数据')
    # 数据输入处理
    hist_value = hist_value.set_index('date')
    hist_value.index = pd.to_datetime(hist_value.index)
    hist_value = hist_value.dropna().sort_index(ascending=True)
    hist_value = hist_value.rename({
        "货基指数": "货币现金类",
        '固收类': '固定收益类',
        '混合类': '混合策略类',
        '权益类': '权益投资类',
        '另类': '另类投资类',
        '安逸型': 'C1',
        '谨慎型': 'C2',
        '稳健型': 'C3',
        '增长型': 'C4',
        '进取型': 'C5',
        '激进型': 'C6'
    }, axis=1)

    # 预设的风险等级配比(proposed_alloc_df)，前端输入
    proposed_alloc = {
        'C1': {'货币现金类': 1.0},
        'C2': {'货币现金类': 0.2, '固定收益类': 0.8},
        'C3': {'货币现金类': 0.1, '固定收益类': 0.55, '混合策略类': 0.35},
        'C4': {'货币现金类': 0.05, '固定收益类': 0.4, '混合策略类': 0.3, '权益投资类': 0.2, '另类投资类': 0.05},
        'C5': {'货币现金类': 0.05, '固定收益类': 0.2, '混合策略类': 0.25, '权益投资类': 0.4, '另类投资类': 0.1},
        'C6': {'货币现金类': 0.05, '固定收益类': 0.1, '混合策略类': 0.15, '权益投资类': 0.6, '另类投资类': 0.1}
    }
    proposed_alloc_df = pd.DataFrame(proposed_alloc).T
    proposed_alloc_df = proposed_alloc_df.fillna(0)

    # 生成资产大类列表(assets_list)
    assets_list = proposed_alloc_df.columns.tolist()

    # 大类资产净值数据转为每日收益率数据
    hist_value_r = hist_value.pct_change().dropna()

    # 计算大类资产的年化收益率、年化波动率、夏普比率
    VaR95 = 1.96
    hist_perf = pd.concat(
        [
            np.log(hist_value.iloc[-1] / hist_value.iloc[0]) / (hist_value.count() - 1) * 252,
            np.std(np.log(hist_value / hist_value.shift(-1))[1:], axis=0) * np.sqrt(252)
        ],
        axis=1
    )
    print(f"数据形状: {hist_value_r.shape}")
    print(f"时间范围: {hist_value_r.index[0]} 到 {hist_value_r.index[-1]}")
    print("\n各资产年化统计指标:")
    hist_perf.columns = ['Annual Return', 'Annual Volatility']
    hist_perf['Sharpe Ratio'] = hist_perf['Annual Return'] / hist_perf['Annual Volatility']
    hist_perf['VaR95'] = hist_perf['Annual Return'] - hist_perf['Annual Volatility'] * VaR95

    # =========================== 分层网格搜索：100 -> 20 -> 50 -> 100 ===========================

    # 第一层：1% 精度
    print("\nLayer 1: 10% precision (resolution=10)")
    start_time = time.time()
    w_r10 = generate_simplex_grid(n_assets=5, resolution=100)
    print(f"Generated {len(w_r10):,} weight combinations")
    r10, best_r10 = find_best(w_r10)
    time_r10 = time.time() - start_time
    print(f"Found {len(best_r10)} efficient frontier points in {time_r10:.2f} seconds")

    # 使用 Plotly 展示（抽样非前沿点以避免浏览器/内存压力）
    try:
        plot_efficient_frontier_plotly(
            df=r10,
            assets_cols=assets_list,
            title='资产组合有效前沿（分辨率=100，约四百万点，已抽样显示）',
            sample_non_ef=50000,
            save_html='efficient_frontier_4M.html'
        )
        print('交互式图表已生成: efficient_frontier_4M.html')
    except Exception as e:
        print('Plotly 绘图失败:', repr(e))
