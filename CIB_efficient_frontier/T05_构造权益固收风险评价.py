# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Literal, Tuple


def plot_asset_trends(df, assets_to_plot, title='资产历史净值走势'):
    """
    使用 Plotly 绘制资产历史净值的折线图。

    参数:
        df (pd.DataFrame): 包含净值数据的DataFrame，索引必须是日期时间。
        assets_to_plot (list): 需要在图表中绘制的资产列名列表。
        title (str): 图表的标题。
    """
    print(f"\n正在生成图表: {title}...")
    fig = go.Figure()

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


def main():
    """
    主函数：
    1) 加载并预处理历史净值；
    2) 绘制五大类资产净值；
    3) 基于“权益+固收”构造一条虚拟净值曲线（风险评价组合）并展示。
    """
    # 0) 参数区：可按需调整
    equity_col = '权益投资类'
    bond_col = '固定收益类'
    # 组合权重方法：'equal' 等权，'inverse_vol' 逆波动，'manual' 手工
    weight_mode: Literal['equal', 'inverse_vol', 'manual'] = 'inverse_vol'
    manual_weights: Tuple[float, float] = (0.5, 0.5)  # (w_equity, w_bond) 当 weight_mode='manual' 时生效

    # 1) 加载与预处理
    hist_value = load_and_process_data()
    hist_value = hist_value / hist_value.iloc[0, :]

    # 2) 绘制五大类资产净值
    assets_list = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']
    plot_asset_trends(hist_value, assets_list, title='五大类资产历史净值走势')

    # 3) 构造“权益固收风险评价组合”
    # 3.1 计算日收益（按归一化净值的日度变化）
    sub = hist_value[[equity_col, bond_col]].dropna()
    ret = sub.pct_change().dropna()

    # 3.2 计算权重
    if weight_mode == 'equal':
        w_e, w_b = 0.5, 0.5
    elif weight_mode == 'manual':
        w_e, w_b = manual_weights
        s = w_e + w_b
        if s == 0:
            raise ValueError('手工权重之和为0')
        w_e, w_b = w_e / s, w_b / s
    elif weight_mode == 'inverse_vol':
        # 用全样本波动率做逆波动权重（风险均衡近似）
        vol = ret.std()
        inv = 1.0 / vol.replace(0, np.nan)
        inv = inv.fillna(0.0)
        s = inv.sum()
        if s == 0:
            w_e, w_b = 0.5, 0.5
        else:
            w_e = float(inv[equity_col] / s)
            w_b = float(inv[bond_col] / s)
    else:
        raise ValueError(f'未知的 weight_mode: {weight_mode}')

    print(f"使用权重: {equity_col}={w_e:.3f}, {bond_col}={w_b:.3f}")

    # 3.3 组合日收益与虚拟净值
    port_ret = w_e * ret[equity_col].values + w_b * ret[bond_col].values
    port_nv = np.cumprod(1.0 + port_ret)
    port_nv = pd.Series(port_nv, index=ret.index, name='权益固收风险评价组合')

    # 3.4 合并到一个 DataFrame 用于展示
    show_df = pd.concat([sub.loc[ret.index, equity_col], sub.loc[ret.index, bond_col], port_nv], axis=1)

    # 3.5 绘图
    plot_asset_trends(show_df, [equity_col, bond_col, '权益固收风险评价组合'], title='权益固收风险评价组合（虚拟净值）')


if __name__ == '__main__':
    main()
