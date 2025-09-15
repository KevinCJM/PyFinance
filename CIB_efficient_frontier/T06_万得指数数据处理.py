import pandas as pd
import numpy as np
import plotly.graph_objects as go


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
    tmp_vals = index_df[value_cols].astype(str).replace({r"[,\s]": ""}, regex=True)
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


if __name__ == '__main__':
    # 构建大类的配置
    config = {
        "权益类": {
            # 权重分配方式: 'equal'-等权; 'inverse_vol'-逆波动率; 'manual'-手工指定; 'risk_parity'-风险平价
            'method': 'manual',
            # 指定构建权重的成分名称（必须与数据中的列名一致）
            'index_names': ['万得普通股票型基金指数', '万得股票策略私募指数', '万得QDII股票型基金指数'],
            # 指定手工权重（仅当 method='manual' 时有效）
            'manual_weights': [0., 0.3, 0.2],
        },
        "固收类": {
            'method': 'equal',
            'index_names': ['万得纯债型基金总指数', '万得短期纯债型基金指数', '万得中长期纯债型指数',
                            '万得QDII债券型基金指数'],
        },
        "另类": {
            'method': 'inverse_vol',
            'index_names': ['伦敦金现', '南华商品指数', '万得管理期货私募指数'],
        },
        "货基指数": {
            'method': 'equal',
            'index_names': ['万得货币市场基金指数'],
        },
        "混合类": {
            'method': 'risk_parity',
            'index_names': ['万得混合型基金指数'],
            'risk_metric': 'vol',  # 风险平价度量 ['vol', 'ES', 'VaR'], 选择 weight_mode='risk_parity' 时有效
            'rp_alpha': 0.95,  # ES/VaR 置信度 (左尾 1-alpha), 选择 risk_metric='ES'/'VaR' 时有效
            'rp_tol': 1e-6,  # 迭代收敛阈值, 选择 weight_mode='risk_parity' 时有效
            'rp_max_iter': 50,  # 迭代上限, 选择 weight_mode='risk_parity' 时有效
            'risk_budget': (4.0, 2.0, 4.0),  # 风险预算比例 (与 selected_assets 等长), 选择 weight_mode='risk_parity' 时有效

        }
    }
    # 读取并处理数据
    nav_df = read_data_and_prepare()
    # 虚拟净值曲线（以 1 为起点）
    plot_lines(nav_df, title="万得指数：虚拟净值（起始为1）", y_tick_format=None, output_html=None)
