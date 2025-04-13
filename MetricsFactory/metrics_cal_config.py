# -*- encoding: utf-8 -*-
"""
@File: metrics_cal_config.py
@Modify Time: 2025/4/9 10:52       
@Author: Kevin-Chen
@Descriptions: 金融指标计算参数文件
"""
import numpy as np
from collections import defaultdict

# 年化收益率计算的天数
return_ann_factor = 365
# 年化波动率计算的天数
risk_ann_factor = 252

# 无风险普通年化收益率
no_risk_ann_return = 0.015
# 无风险普通每日收益率 (使用 自然日or交易日 ? 请自由调整)
daily_return = (1 + no_risk_ann_return) ** (1 / return_ann_factor) - 1
# 无风险对数每日收益率
log_daily_return = np.log(1 + daily_return)
# 无风险对数年化收益率
log_ann_return = np.log(1 + no_risk_ann_return)

# 区间指标支持的区间代码
period_list = [
    '2d',  # 2天
    '3d',  # 3天
    '5d',  # 5天
    '10d',  # 10天 (2周)
    '25d',  # 25天 (一个月)
    '50d',  # 50天 (两个月)
    '75d',  # 75天 (三个月)
    '5m',  # 5个月
    '6m',  # 6个月
    '9m',  # 9个月
    '12m',  # 12个月（1年）
    '2y',  # 2年
    '3y',  # 3年
    '5y',  # 5年
    'mtd',  # 本月至今（Month-to-Date）
    'qtd',  # 本季度至今（Quarter-to-Date）
    'ytd',  # 本年至今（Year-to-Date）
    'max',  # 最大
]

# 区间指标
log_return_metrics_dict = {
    'TotalReturn':
        ['总收益率',  # 指标名称
         '总收益率 = (最终价值 - 初始价值) / 初始价值',  # 指标的简要计算方法说明
         # 支持计算的区间
         ['2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AnnualizedReturn':
        ['年化收益率',
         '年化收益率 = (1 + 总收益率)^(return_ann_factor/天数) - 1',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AverageDailyReturn':
        ['日均收益率',
         '日均收益率 = 总收益率 / 交易天数',
         ['2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AvgPositiveReturn':
        ['平均正收益率',
         '平均正收益率 = 所有正收益的平均值',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AvgNegativeReturn':
        ['平均负收益率',
         '平均负收益率 = 所有负收益的平均值',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AvgReturnRatio':
        ['平均盈亏比',
         '平均盈亏比 = 平均正收益率 / 平均负收益率',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'TotalPositiveReturn':
        ['总累计盈利',
         '总累计盈利 = 所有正收益的总和',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'TotalNegativeReturn':
        ['总累计亏损',
         '总累计亏损 = 所有负收益的总和',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'TotalReturnRatio':
        ['累计盈亏比',
         '累计盈亏比 = 总累计盈利 / 总累计亏损',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MedianDailyReturn':
        ['日中位收益率',
         '日中位收益率 = 所有日收益率的中位数',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Volatility':
        ['波动率',
         '波动率 = 收益率的标准差',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AnnualizedVolatility':
        ['年化波动率',
         '年化波动率 = 波动率 * sqrt(risk_ann_factor)',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MeanAbsoluteDeviation':
        ['平均绝对偏差',
         '平均绝对偏差 = 平均每期收益率距离均值的绝对值, 比标准差更不受极端值影响',
         ['3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'ReturnRange':
        ['收益率范围, 也称为极差',
         '收益率范围 = 收益率的最大值 - 收益率的最小值',
         ['3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'RescaledRange':
        ['标准化极差',
         '标准化极差 = (收益率的最大值 - 收益率的最小值) / 波动率',
         ['3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MaxGain':
        ['最大单日收益',
         '最大收益 = 收益率的最大值',
         ['3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MaxLoss':
        ['最大单日亏损',
         '最大亏损 = 收益率的最小值',
         ['3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MaxDrawDown':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['最大回撤',
         '最大回撤 = (峰值 - 谷值) / 峰值',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MaxDrawDownDays':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['最大回撤天数',
         '最大回撤天数 = 回撤开始到谷底的交易日天数',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    # 'MaxDrawDownPaybackDays': # 该指标不适用于强化学习
    #     ['最大回撤恢复天数',
    #      '最大回撤恢复天数 = 从最大回撤底部恢复到历史最高点所需的天数',
    #      ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
    #      ],
    'ReturnDrawDownRatio':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['收益率回撤比',
         '收益率回撤比 = 总收益率 / 最大回撤',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AnnReturnDrawDownRatio':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['年化收益率回撤比, 即 卡尔马比率',
         '年化收益率回撤比/卡尔马比率 = 年化收益率 / 最大回撤',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    # 'RecoveryFactor': # 该指标不适用于强化学习
    #     ['恢复因子',
    #      '恢复因子 = 从最大回撤底部恢复回历史最高点所获得的收益 / 最大回撤',
    #      ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
    #      ],
    'DrawDownSlope':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['回撤斜率',
         '回撤斜率 = 最大回撤幅度 / 回撤所用时间',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'UlcerIndex':  # 最大回撤用的是普通回撤率,而非对数回撤率
        ['溃疡指数 (衡量最大回撤深度和持续时间的指标)',
         '溃疡指数 = 所有 draw down 的均方根（RMS）',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'MartinRatio':
        ['马丁比率 (绩效趋势比率), 衡量趋势的可复制性',
         '马丁比率 = 年化收益率 / 溃疡指数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'SharpeRatio':
        ['夏普比率',
         '夏普比率 = (累计收益率 - 无风险收益) / 投资组合波动率',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'AnnualizedSharpeRatio':
        ['年化夏普比率',
         '年化夏普比率 = 夏普比率 * sqrt(年交易天数)',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'ReturnVolatilityRatio':
        ['收益率波动率比',
         '收益率波动率比 = 总收益率 / 波动率',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'DownsideVolatility':
        ['下行波动率',
         '下行波动率 = 负收益率的标准差',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'UpsideVolatility':
        ['上行波动率',
         '上行波动率 = 正收益率的标准差',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VolatilitySkew':
        ['波动率偏度',
         '波动率偏度 = (上行波动率 - 下行波动率) / 总波动率',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VolatilityRatio':
        ['波动率比, 也叫 上行潜在比率',
         '波动率比 = 上行波动率 / 下行波动率',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'SortinoRatio':
        ['索提诺比率',
         '索提诺比率 = (累计收益率 - 无风险收益率) / 下行波动率',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'GainConsistency':
        ['收益趋势一致性',
         '收益趋势一致性 = 盈利日的标准差(上行波动率) / 盈利日的平均收益(平均正收益率)',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'LossConsistency':
        ['亏损趋势一致性',
         '亏损趋势一致性 = 亏损日的标准差(下行波动率) / 亏损日的平均亏损(平均负收益率)',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'WinningRatio':
        ['胜率',
         '胜率 = 盈利交易次数(不含0) / 总交易次数',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'LosingRatio':
        ['亏损率',
         '亏损率 = 亏损交易次数(不含0) / 总交易次数',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'ReturnSkewness':
        ['收益率偏度',
         '收益率偏度 = 收益率分布的偏度',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'ReturnKurtosis':
        ['收益率峰度',
         '收益率峰度 = 收益率分布的峰度',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    # 'MaxConsecutiveWinsDays': # 不适合强化学习使用
    #     ['最长连续胜利天数',
    #      '最长连续胜利天数 = 连续盈利交易的最长天数',
    #      ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
    #      ],
    # 'MaxConsecutiveLossDays': # 不适合强化学习使用
    #     ['最长连续失败天数',
    #      '最长连续失败天数 = 连续亏损交易的最长天数',
    #      ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
    #      ],
    # 'MaxConsecutiveRate': # 不适合强化学习使用
    #     ['最长连续上涨下跌日比率',
    #      '最长连续上涨下跌日比率 = 最长连续胜利天数 / 最长连续失败天数',
    #      ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
    #      ],
    'VaR-99':
        ['99% VaR',
         '99% VaR = 在99%的置信水平下，投资组合可能的最大损失',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaR-95':
        ['95% VaR',
         '95% VaR = 在95%的置信水平下，投资组合可能的最大损失',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaR-90':
        ['90% VaR',
         '90% VaR = 在90%的置信水平下，投资组合可能的最大损失',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaRSharpe-95':
        ['基于 95% VaR 计算夏普比率',
         '基于 95% VaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% VaR',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaRModified-99':
        ['99% 修正VaR',
         '99% 修正VaR = 在传统正态 VaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaRModified-95':
        ['95% 修正VaR',
         '95% 修正VaR = 在传统正态 VaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaRModified-90':
        ['90% 修正VaR',
         '90% 修正VaR = 在传统正态 VaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'VaRModifiedSharpe-95':
        ['基于 95% 修正VaR 计算夏普比率',
         '基于 95% 修正VaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% 修正VaR',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaR-99':
        ['99% CVaR',
         '99% CVaR = 在99%的置信水平下，投资组合可能的平均损失',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaR-95':
        ['95% CVaR',
         '95% CVaR = 在95%的置信水平下，投资组合可能的平均损失',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaR-90':
        ['90% CVaR',
         '90% CVaR = 在90%的置信水平下，投资组合可能的平均损失',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaRModified-99':
        ['99% 修正CVaR',
         '99% 修正CVaR = 在传统正态 CVaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaRModified-95':
        ['95% 修正CVaR',
         '95% 修正CVaR = 在传统正态 CVaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaRModified-90':
        ['90% 修正CVaR',
         '90% 修正CVaR = 在传统正态 CVaR 的基础上，使用 Cornish-Fisher 展开修正尾部概率',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaRSharpe-95':
        ['基于 95% CVaR 计算夏普比率',
         '基于 95% CVaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% CVaR',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CVaRModifiedSharpe-95':
        ['基于 95% 修正CVaR 计算夏普比率',
         '基于 95% 修正CVaR 计算夏普比率 = (投资组合收益率 - 无风险收益率) / 95% 修正CVaR',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Percentile-1':
        ['所有收益率前1%分位数 (一般是极端亏损部分)',
         '所有收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Percentile-99':
        ['所有收益率前99%分位数 (一般是极端盈利部分)',
         '所有收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Percentile-5':
        ['所有收益率前5%分位数 (一般是极端亏损部分)',
         '所有收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Percentile-95':
        ['所有收益率前95%分位数 (一般是极端盈利部分)',
         '所有收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Percentile-10':
        ['所有收益率前10%分位数 (一般是极端亏损部分)',
         '所有收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'Percentile-90':
        ['所有收益率前90%分位数 (一般是极端盈利部分)',
         '所有收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'PercentileWin-95':
        ['正收益率前5%分位数',
         '只考虑正收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'PercentileLoss-95':
        ['负收益率取绝对值后, 前5%分位数',
         '只考虑负收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'PercentileWin-90':
        ['正收益率前10%分位数',
         '只考虑正收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'PercentileLoss-90':
        ['负收益率取绝对值后, 前10%分位数',
         '只考虑负收益率的分位数',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'TailRatio-90':
        ['尾部比率, 极端正收益与极端负收益的比值',
         'TailRatio-90 = PercentileWin-90 / PercentileLoss-90',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'TailRatio-95':
        ['尾部比率, 极端正收益与极端负收益的比值',
         'TailRatio-95 = PercentileWin-95 / PercentileLoss-95',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'NewHighRatio':
        ['净值新高比率',
         '净值新高比率 = 净值创新高的交易日数 / 总交易日数',
         ['10d', '25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CrossProductRatio-1':
        ['交叉乘积比率 (基于单日收益统计)',
         '交叉乘积比率 = (WW * LL) / (WL * LW)',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CrossProductRatio-5':
        ['交叉乘积比率 (基于5日收益统计)',
         '交叉乘积比率 = (WW * LL) / (WL * LW)',
         ['50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'CrossProductRatio-10':
        ['交叉乘积比率 (基于10日收益统计)',
         '交叉乘积比率 = (WW * LL) / (WL * LW)',
         ['75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'HurstExponent':  # 计算资源较大,涉及线性回归,无法矢量化
        ['赫斯特指数',
         '赫斯特指数 = 对净值构造时间序列，进行 R/S 分析',
         ['70d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'OmegaRatio':
        ['欧米茄比率',
         '欧米茄比率 = 更全面的夏普比率扩展版本（考虑全部分布）',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'ReturnDistributionIntegral':
        ['收益分布积分',
         '收益分布积分 = Omega Ratio 的分子部分',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    "ReturnSlope":
        ['收益斜率',
         '收益斜率 = 收益率拟合直线的 slope',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'KRatio':
        ['K比率 (绩效趋势比率), 衡量趋势的可复制性',
         'K比率 = 净值的 log 值拟合直线的 slope / 标准差',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'SortinoSkewness':
        ['索提诺偏度',
         '索提诺偏度 = 类似 Skewness，但只考虑负收益部分的偏度（衡量尾部亏损分布）',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y'],
         ],
    'NetEquitySlope':
        ['净值增长斜率 (代表了净值随时间的平均上升速度)',
         '净值增长斜率 = 净值的 log 值拟合直线的 slope',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y']
         ],
    'EquitySmoothness':
        ['净值平滑度 (衡量策略净值序列与其线性拟合值之间的拟合程度)',
         '净值平滑度 = 净值的 log 值拟合曲线的 R^2',
         ['25d', '50d', '75d', '5m', '6m', '9m', '12m', '2y', '3y', '5y']
         ],
}

# 历史偏离指标
log_return_relative_metrics_dict = {

    ''' 相对历史收益指标 '''
    'ReturnZScore_2m':
        ['当前平均收益相对近2个月收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近2个月平均收益) / 近2个月标准差',
         ['1d', '2d', '3d', '5d', '10d', '25d'],
         ],
    'ReturnZScore_6m':
        ['当前平均收益相对近半年收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近半年平均收益) / 近半年标准差',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d'],
         ],
    'ReturnZScore_1y':
        ['当前平均收益相对近1年收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近1年平均收益) / 近1年标准差',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m'],
         ],
    'ReturnZScore_2y':
        ['当前平均收益相对近2年收益的 Z 分数',
         '收益率Z分数 = (当前平均收益 - 近2年平均收益) / 近2年标准差',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m', '1y'],
         ],
    'ReturnPercentile_2m':
        ['当前收益率相对近2个月收益的分位数',
         '收益率分位数 = 当前收益率在近2个月的同类收益率中的分位数',
         ['1d', '2d', '3d', '5d', '10d', '25d'],
         ],
    'ReturnPercentile_6m':
        ['当前收益率相对近6个月收益的分位数',
         '收益率分位数 = 当前收益率在近2个月的同类收益率中的分位数',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d'],
         ],
    'ReturnPercentile_1y':
        ['当前收益率相对近1年收益的分位数',
         '收益率分位数 = 当前收益率在近1年的同类收益率中的分位数',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m'],
         ],
    'ReturnPercentile_2y':
        ['当前收益率相对近2年收益的分位数',
         '收益率分位数 = 当前收益率在近2年的同类收益率中的分位数',
         ['1d', '2d', '3d', '5d' '10d', '25d', '50d', '75d', '5m', '6m', '1y'],
         ],
    'AverageDeviationReturn_2m':
        ['相较近2月收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近2月收益均值) / 近2月收益均值',
         ['1d', '2d', '3d', '5d', '10d', '25d'],
         ],
    'AverageDeviationReturn_6m':
        ['相较近6月收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近6月收益均值) / 近6月收益均值',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d'],
         ],
    'AverageDeviationReturn_1y':
        ['相较近1年收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近1年收益均值) / 近1年收益均值',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m'],
         ],
    'AverageDeviationReturn_2y':
        ['相较近2年收益均线偏离率',
         '均线偏离率 = (当前收益率均值 - 近2年收益均值) / 近2年收益均值',
         ['1d', '2d', '3d', '5d' '10d', '25d', '50d', '75d', '5m', '6m', '1y'],
         ],

    ''' 相对历史风险指标 '''
    'VolatilityRollingRatio_2m':
        ['当前波动率相较近2个月波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近2个月波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['5d', '10d', '25d'],
         ],
    'VolatilityRollingRatio_6m':
        ['当前波动率相较近6个月波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近6个月波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['5d', '10d', '25d', '50d', '75d'],
         ],
    'VolatilityRollingRatio_1y':
        ['当前波动率相较近1年波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近1年波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m'],
         ],
    'VolatilityRollingRatio_2y':
        ['当前波动率相较近2年波动率的滚动比率',
         '波动率滚动比率 = 当前波动率 / 近2年波动率 (值 > 1 表示近期强于长期，< 1 则反之)',
         ['5d', '10d', '25d', '50d', '75d', '5m', '6m', '1y'],
         ],
    'VolatilityPercentile_2m':
        ['当前波动率相对近2个月波动率的分位数',
         '波动率分位数 = 当前收益率在近2个月的同类波动率中的分位数',
         ['1d', '2d', '3d', '5d', '10d', '25d'],
         ],
    'VolatilityPercentile_6m':
        ['当前波动率相对近6个月波动率的分位数',
         '波动率分位数 = 当前收益率在近2个月的同类波动率中的分位数',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d'],
         ],
    'VolatilityPercentile_1y':
        ['当前波动率相对近1年收波动率分位数',
         '波动率分位数 = 当前收益率在近1年的同类收波动中的分位数',
         ['1d', '2d', '3d', '5d', '10d', '25d', '50d', '75d', '5m', '6m'],
         ],
    'VolatilityPercentile_2y':
        ['当前波动率相对近2年收波动率分位数',
         '波动率分位数 = 当前收益率在近2年的同类收波动中的分位数',
         ['1d', '2d', '3d', '5d' '10d', '25d', '50d', '75d', '5m', '6m', '1y'],
         ],
}

# 滚动计算指标的滚动天数
rolling_days = [
    5, 10, 15, 30, 60, 120,
]

# 滚动计算指标
rolling_metrics = {
    'PriceSigma':
        ['过去N天收盘价的滚动标准差',
         ],
    'MA':
        ['N日移动平均收盘价',
         ],
    'BollUp':
        ['bolling 上轨',
         '布林带上轨 = N日移动平均收盘价 + 2 * N日移动标准差',
         ],
    'BollDo':
        ['bolling 下轨',
         '布林带下轨 = N日移动平均收盘价 - 2 * N日移动标准差',
         ],
    'L':
        ['过去N天的最低价 (KDJ相关指标)',
         ],
    'H':
        ['过去N天的最低价 (KDJ相关指标)',
         ],
    'RSV':
        ['RSV指标 (KDJ相关指标)',
         'RSV = (当前收盘价 - 最近N天最低价) / (最近N天最高价 - 最近N天最低价) * 100',
         ],
    'K':
        ['KDJ 指标的 K 值',
         'K = RSV 的指数移动平均值',
         ],
    'D':
        ['KDJ 指标的 D 值',
         'D = K 的指数移动平均值',
         ],
    'J':
        ['KDJ 指标的 J 值',
         'J = 3 * K - 2 * D',
         ],
    'EMA':
        ['N日指数移动平均收盘价 (MACD相关指标)',
         'EMA = ',
         ],
    'DIF':
        ['MACD 指标的 DIF 值 (快线-趋势变化)',
         'DIF = EMA12 - EMA26',
         ],
    'DEA':
        ['MACD 指标的 DEA 值 (慢线-趋势平滑)',
         'DEA = DIF 的指数移动平均值',
         ],
    'MACD':
        ['MACD 值 (快线-慢线)',
         'MACD = 2 (DIF - DEA)',
         ],
    'RSI':
        ['RSI 指标 (相对强弱指数)',
         'RSI = 过去N天收益之和 / (过去N天收益之和 + |过去N天亏损之和|)',
         ],
    'OBV':
        ['OBV 指标 (能量潮指标)',
         'OBV = ?',
         ],
    'DMA':
        ['DMA 指标 (平均线差)',
         'DMA = MAD短期 - MA长期',
         ],
    'MTM':
        ['MTM 指标 (动量指标)',
         'MTM = 当前收盘价 - N天前收盘价',
         ],
    'TRIX':
        ['TRIX 指标 (三重指数平滑移动平均)',
         'TRIX = EMA(EMA(EMA(收盘价))) / 收盘价 - 1',
         ],
    'BBI':
        ['BBI 指标 (多空指标)',
         'BBI = (MA3 + MA6 + MA12 + MA24) / 4',
         ],
    'PSY':
        ['PSY 指标 (心理线)',
         'PSY = 过去N天上涨天数 / 过去N天交易日数',
         ],
    'CCI':
        ['CCI 指标 (顺势指标)',
         'CCI = (当前收盘价 - N天移动平均价) / (0.015 * N天移动平均离差)',
         ],
    'CR':
        ['CR 指标 (能量指标)',
         'CR = (当日最高价 - 昨日收盘价) / (昨日收盘价 - 当日最低价)',
         ],

}


# 创建一个按周期分组的指标映射表
def create_period_metrics_map():
    """
    创建一个按周期分组的指标映射表。

    该函数遍历 `log_return_metrics_dict` 中的每个指标，根据指标的周期将其分组，
    最终返回一个字典，其中键为周期，值为该周期下的所有指标键列表。

    返回值:
        dict: 一个字典，键为周期，值为该周期下的所有指标键列表。
    """
    # 初始化一个字典，默认值为列表
    period_metrics_map = defaultdict(list)

    # 遍历 log_return_metrics_dict 中的每个指标
    for metric_key, (name, formula, periods) in log_return_metrics_dict.items():
        # 遍历该指标的所有周期，并将指标键添加到对应周期的列表中
        for period in periods:
            period_metrics_map[period].append(metric_key)

    # 转换为普通字典并返回
    return dict(period_metrics_map)


if __name__ == '__main__':
    import re
    s = '10d'
    match = re.match(r'^(\d+).*d$', s)
    if match:
        print(int(match.group(1)))
    else:
        print(None)

