# -*- encoding: utf-8 -*-
"""
@File: test.py
@Modify Time: 2025/4/20 09:27       
@Author: Kevin-Chen
@Descriptions: 
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from ReturnClassification.metrics_data_prepare import (
    get_fund_close_price, cal_future_log_return, get_fund_metrics_data, preprocess_data, get_fund_basic_data)

# 获取指标数据
metrics_data = get_fund_metrics_data('510050.SH',
                                     '../Data/Metrics',
                                     '../Data',
                                     True)
# 预处理指标数据
metrics_data = preprocess_data(metrics_data,
                               nan_method='drop'
                               )

print(metrics_data.head(5))