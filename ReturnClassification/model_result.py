# -*- encoding: utf-8 -*-
"""
@File: model_result.py
@Modify Time: 2025/4/23 20:07       
@Author: Kevin-Chen
@Descriptions: 
"""
import joblib
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line, Scatter
from pyecharts.commons.utils import JsCode

from ReturnClassification.train_and_test import main_data_prepare
from ReturnClassification.metrics_data_prepare import get_fund_close_price

n_day = 1
filename = f'random_forest_model_{n_day}.joblib'
loaded_clf = joblib.load(filename)
test_start_date = '2024-04-01'
test_end_date = '2025-04-30'

# 数据读取，包括训练集和测试集，以及 原始指标数据
x_train, y_train, x_test, y_test, metrics_data = main_data_prepare(
    the_fund_code='510050.SH',  # 指定基金代码，此处为 '510050.SH'
    n_days=n_day,  # 预测未来收益的天数，变量 d 在循环中定义，表示不同的预测周期
    folder_path='../Data',  # 基金价格数据的文件夹路径，默认为 '../Data'
    metrics_folder='../Data/Metrics',  # 基金指标数据的文件夹路径，默认为 '../Data/Metrics'
    train_start=None,  # 训练集开始日期，如果为 None，则从数据的最早日期开始
    train_end='2024-03-31',  # 训练集结束日期，指定为 '2024-11-30'
    test_start='2024-04-01',  # 测试集开始日期，指定为 '2024-12-01'
    test_end='2025-04-30',  # 测试集结束日期，指定为 '2025-04-30'
    nan_method='drop',  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'
    standardize_method='zscore',  # 指标标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'。
    basic_data_as_metric=True,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为 True
    return_threshold=0.0,  # 标签生成方法，未来收益率大于 0.01 的样本标记为 1，否则为 0
    dim_reduction=True,  # 是否PCA
    dim_reduction_limit=0.90,  # PCA降维的方差解释比例, 默认为0.9
    n_components=None,  # PCA维度, 写None表示自动选择, 保留90%方差解释比率
)

# 预测数据
y_pred = loaded_clf.predict(x_test)
print(y_pred)

metrics_data = metrics_data[['ts_code', 'date', 'close']]
metrics_data['date'] = pd.to_datetime(metrics_data['date'])
pred_metrics_data = metrics_data[(metrics_data['date'] >= pd.to_datetime(test_start_date))
                                 & (metrics_data['date'] <= pd.to_datetime(test_end_date))]
pred_metrics_data['pred'] = y_pred
print(pred_metrics_data)

''' 画图 '''

# 确保 'date' 列是 datetime 类型
pred_metrics_data['date'] = pd.to_datetime(pred_metrics_data['date'])

# 提取唯一的 ts_code (假设只有一个)
ts_code = pred_metrics_data['ts_code'].iloc[0]

# 分离预测为 1 和 0 的数据点
pred_1_data = pred_metrics_data[pred_metrics_data['pred'] == 1][['date', 'close']].values.tolist()
pred_0_data = pred_metrics_data[pred_metrics_data['pred'] == 0][['date', 'close']].values.tolist()

# 创建 Line 图
line = (
    Line()
    .add_xaxis(xaxis_data=pred_metrics_data['date'].tolist())
    .add_yaxis(
        series_name="Close Price",
        y_axis=pred_metrics_data['close'].tolist(),
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title=f"{ts_code} Close Price with Predictions"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                is_show=True,
                start_value=0,
                end_value=len(pred_metrics_data) - 1,
            ),
            opts.DataZoomOpts(type_="inside"),
        ],
        xaxis_opts=opts.AxisOpts(),  # 不指定 type_="time"
        yaxis_opts=opts.AxisOpts(
            name="Close Price",
            min_=pred_metrics_data['close'].min() - 0.01,  # 设置最小值，可以根据需要调整
            max_=pred_metrics_data['close'].max() + 0.01,  # 设置最大值，可以根据需要调整
        ),
        legend_opts=opts.LegendOpts(is_show=True),
    )
)

# 添加预测为 1 的散点
scatter_pred_1 = (
    Scatter()
    .add_xaxis(xaxis_data=[item[0] for item in pred_1_data])
    .add_yaxis(
        series_name="Prediction = 1",
        y_axis=[item[1] for item in pred_1_data],
        symbol_size=8,
        itemstyle_opts=opts.ItemStyleOpts(color="blue"),
        label_opts=opts.LabelOpts(is_show=False),
    )
)

# 添加预测为 0 的散点
scatter_pred_0 = (
    Scatter()
    .add_xaxis(xaxis_data=[item[0] for item in pred_0_data])
    .add_yaxis(
        series_name="Prediction = 0",
        y_axis=[item[1] for item in pred_0_data],
        symbol_size=8,
        itemstyle_opts=opts.ItemStyleOpts(color="red"),
        label_opts=opts.LabelOpts(is_show=False),
    )
)

# 将 Line 图和 Scatter 图叠加显示
combined_chart = line.overlap(scatter_pred_1).overlap(scatter_pred_0)

combined_chart.render("close_price_with_predictions_sparse_dates.html")
