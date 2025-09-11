# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go


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
    主函数，用于加载、处理数据，并以图表形式展示。
    """
    # 加载和预处理数据
    hist_value_processed = load_and_process_data()

    # 净值归一
    hist_value_processed = hist_value_processed / hist_value_processed.iloc[0, :]

    # 定义要绘图的资产列表
    assets_list = ['货币现金类', '固定收益类', '混合策略类', '权益投资类', '另类投资类']

    # --- 1. 绘制主要资产的净值走势 ---
    plot_asset_trends(
        df=hist_value_processed,
        assets_to_plot=assets_list,
        title='五大类资产历史净值走势'
    )


if __name__ == '__main__':
    main()
