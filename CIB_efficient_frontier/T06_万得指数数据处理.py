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
    # 读取并处理数据
    nav_df = read_data_and_prepare()
    # 虚拟净值曲线（以 1 为起点）
    plot_lines(nav_df, title="万得指数：虚拟净值（起始为1）", y_tick_format=None, output_html=None)
