# -*- encoding: utf-8 -*-
"""
@File: train_and_test.py
@Modify Time: 2025/4/14 16:03       
@Author: Kevin-Chen
@Descriptions: 
"""
import os
import joblib
import warnings
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from ReturnClassification.metrics_data_prepare import main_data_prepare, dim_reduction_func
from ReturnClassification.select_import_features import find_important_features
from ReturnClassification.build_collaborative_features import get_cross_metrics_func

warnings.filterwarnings("ignore")
pd.set_option('display.width', 1000)  # 表格不分段显示
pd.set_option('display.max_columns', 1000)  # 显示字段的数量


# 自动化参数调优函数 (随机森林)
def auto_parameter_tuning(x_train, y_train, random_seed=42, n_iter=20, cv=5, return_threshold=0.0):
    """
    自动化参数调优函数。

    使用随机搜索和交叉验证的方法来自动调整随机森林分类器的超参数。

    参数:
    - x_train: 训练特征数据集。
    - y_train: 训练标签数据集。
    - random_seed: 随机种子，用于确保结果的可重复性。
    - n_iter: 随机搜索的迭代次数。
    - cv: 交叉验证的折数。
    - return_threshold: 标签生成方法，默认为 0, 表示使用未来收益率大于 0 的样本标记为 1，否则为 0;

    返回:
    - 最优超参数的字典。
    """
    print("[INFO] 开始参数调优...")
    # 定义随机搜索的超参数空间
    param_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
    }

    # 初始化随机森林分类器
    rf = RandomForestClassifier(
        random_state=random_seed,
        max_features='sqrt',
    )

    # 初始化随机搜索
    rs = RandomizedSearchCV(
        estimator=rf,  # 使用的机器学习模型
        param_distributions=param_dist,  # 超参数分布
        n_iter=n_iter,  # 随机搜索的迭代次数
        cv=cv,  # 交叉验证的折数
        scoring='f1' if return_threshold == 0.0 else 'f1_macro',  # 评估指标
        random_state=random_seed,  # 随机种子，确保结果可重复
        n_jobs=-1  # 使用所有可用的CPU核心进行并行计算
    )

    # 执行随机搜索
    rs.fit(x_train, y_train)

    # 输出最优参数
    print("[INFO] 最优参数:", rs.best_params_)
    return rs.best_params_


# 使用随机森林分类器进行训练和测试，并进行超参数优化和特征重要性评估
def train_and_test_random_forest(x_train, x_test, y_train, y_test, metrics_data,
                                 random_seed=42, n_iter=20, cv=5, best_dict=None,
                                 return_threshold=0.0, dim_reduction=False
                                 ):
    """
    使用随机森林分类器进行训练和测试，并进行超参数优化和特征重要性评估。

    参数:
    :param x_train: 训练集特征数据
    :param x_test: 测试集特征数据
    :param y_train: 训练集标签
    :param y_test: 测试集标签
    :param metrics_data: 指标结果dataframe
    :param random_seed: 随机种子，用于确保结果可重复
    :param n_iter: 随机搜索的迭代次数
    :param cv: 交叉验证的折数
    :param best_dict: 最优参数字典，如果为None，则自动调参
    :param dim_reduction: bool, 是否做了PCA

    :return: 训练好的随机森林模型
    """

    # 筛选特征列
    feature_cols = [
        col for col in metrics_data.columns if col not in ['ts_code', 'date']]

    ''' 自动超参数调优 '''
    if best_dict is None:
        best_dict = auto_parameter_tuning(x_train, y_train,
                                          random_seed=random_seed, n_iter=n_iter, cv=cv,
                                          return_threshold=return_threshold
                                          )

    # 从最优参数中提取每个超参数的值
    n_estimators = best_dict['n_estimators']
    max_depth = best_dict['max_depth']
    min_samples_split = best_dict['min_samples_split']
    min_samples_leaf = best_dict['min_samples_leaf']

    # 使用最优参数重新初始化随机森林分类器
    clf = RandomForestClassifier(
        n_estimators=n_estimators,  # 构建n棵决策树
        max_depth=max_depth,  # 决策树最大深度, 防止过拟合
        min_samples_split=min_samples_split,  # 子表中最小样本数,越大越保守，减少过拟合
        min_samples_leaf=min_samples_leaf,  # 叶子节点最小样本数, 增大可提升泛化能力
        max_features='sqrt',  # 每棵树随机选择的特征数, 'sqrt' 或 'log2' (sqrt（默认）适用于分类任务，log2/None 也可尝试)
        random_state=random_seed,  # 随机种子
    )
    # 训练模型
    clf.fit(x_train, y_train)

    # 预测测试集
    y_pred = clf.predict(x_test)
    # 输出分类报告
    print("[RESULT] 分类报告:")
    print(classification_report(y_test, y_pred))
    '''
    precision (精确率): 模型预测为某类的样本中，实际属于该类的比例。公式：TP / (TP + FP)
        举例: 类别 0 的精确率为 0.61 → 模型预测为 0 的样本中，61% 是真实的 0
        举例: 类别 1 的精确率 0.83 → 模型预测为 1 的样本中，83% 是真实的 1
    recall (召回率): 实际属于某类的样本中，被模型正确预测的比例。公式：TP / (TP + FN)
        举例: 类别 0 的召回率 0.95 → 真实为 0 的样本中，95% 被正确预测
        举例: 类别 1 的召回率 0.28 → 真实为 1 的样本中，仅有 28% 被正确预测
    f1-score (F1 值)：精确率和召回率的调和平均值，综合衡量模型对某类的预测能力。公式：2 * (precision * recall) / (precision + recall)
        举例: 类别 0 的 F1 值 0.74 → 综合表现较好。
        举例: 类别 1 的 F1 值 0.42 → 综合表现较差。
    support (支持数): 测试集中每个类别的样本数量。    
    accuracy (准确率): 所有样本中被正确预测的比例。公式：(TP + TN) / (TP + TN + FP + FN)
        举例: 整体准确率 0.64 → 模型正确预测了 64% 的样本
    macro avg (宏平均): 对各类别指标（precision/recall/f1）的简单算术平均。
    weighted avg (加权平均): 根据各类别样本量加权计算的指标平均值
    '''

    ''' 查看重要因子 '''
    if dim_reduction:
        # 如果做了PCA, 需要将x_test也做PCA
        print("[INFO] 数据做了PCA, 不再查看重要因子 ... ")
        return clf
    # 获取并输出特征重要性
    importance = pd.Series(clf.feature_importances_, index=feature_cols)
    # 排序并选择前10个重要特征
    importance = importance.sort_values(ascending=False).head(10)
    print("[RESULT] 前10大重要因子:")
    print(importance)

    # 返回训练好的模型
    return clf


def confidence_based_random_forest(x_train, x_test, y_train, y_test,
                                   random_seed=42, n_iter=20, cv=5, threshold=0.7,
                                   parameter_dict=None, return_threshold=0.0):
    """
    基于置信度的随机森林分类器函数。该函数旨在通过自动调整或使用给定的超参数来优化随机森林模型，
    并仅对模型预测置信度高于给定阈值的测试样本进行评估和报告，以提高预测结果的可信度。

    :param x_train: 训练集特征数据
    :param x_test: 测试集特征数据
    :param y_train: 训练集标签
    :param y_test: 测试集标签
    :param random_seed: 随机种子，用于确保结果的可重复性
    :param n_iter: 随机搜索的迭代次数
    :param cv: 交叉验证的折数
    :param threshold: 置信度阈值，仅预测置信度高于此值的样本
    :param parameter_dict: 必须要有: n_estimators, max_depth, min_samples_split, min_samples_leaf
    :param return_threshold: 标签生成方法，默认为 0, 表示使用未来收益率大于 0 的样本标记为 1，否则为 0;

    :return: 训练好的随机森林分类器模型
    """
    print("[INFO] 开始使用 随机森林模型 进行训练...")
    # 自动调参或使用给定的参数字典
    if parameter_dict is None:
        best_dict = auto_parameter_tuning(x_train, y_train,
                                          random_seed=random_seed, n_iter=n_iter, cv=cv,
                                          return_threshold=return_threshold)
    else:
        best_dict = parameter_dict

    # 从最优参数中提取每个超参数的值
    n_estimators = best_dict['n_estimators']
    max_depth = best_dict['max_depth']
    min_samples_split = best_dict['min_samples_split']
    min_samples_leaf = best_dict['min_samples_leaf']

    # 使用最优参数重新初始化随机森林分类器
    clf = RandomForestClassifier(
        n_estimators=n_estimators,  # 构建n棵决策树
        max_depth=max_depth,  # 决策树最大深度, 防止过拟合
        min_samples_split=min_samples_split,  # 子表中最小样本数,越大越保守，减少过拟合
        min_samples_leaf=min_samples_leaf,  # 叶子节点最小样本数, 增大可提升泛化能力
        max_features='sqrt',  # 每棵树随机选择的特征数, 'sqrt' 或 'log2' (sqrt（默认）适用于分类任务，log2/None 也可尝试)
        random_state=random_seed,  # 随机种子
    )
    # 训练模型
    clf.fit(x_train, y_train)

    # 预测和计算置信度
    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)
    confidence = np.max(y_proba, axis=1)

    # 筛选高置信度的预测
    mask = confidence >= threshold
    y_test_confident = y_test[mask]
    y_pred_confident = y_pred[mask]

    # 打印信息
    print(f"原测试集样本数量: {len(y_test)}")
    print(f"置信度 ≥ {threshold} 的样本数量: {mask.sum()}")
    print(f"占比: {mask.sum() / len(y_test):.2%}\n")

    # 打印分类评估结果
    if len(y_test) == 0:
        print("没有满足置信度要求的样本, 无法生成分类报告")
    else:
        print(classification_report(y_test_confident, y_pred_confident))
    return clf


# 未来收益分类主函数，用于执行基金数据预处理、模型训练等任务 (使用随机森林)
def predict_main_random_forest(the_fund_code='159919.SZ',
                               n_days=20,
                               folder_path='../Data',
                               metrics_folder='../Data/Metrics',
                               train_start=None,
                               train_end='2023-12-31',
                               test_start='2024-01-01',
                               test_end='2025-03-31',
                               nan_method='drop',
                               standardize_method='both',
                               period_metrics=True,
                               rolling_metrics=True,
                               random_seed=42, n_iter=20, cv=5,
                               threshold=None, basic_data_as_metric=False,
                               import_feature_only=False, top_n=200,
                               return_threshold=0.0, parameter_dict=None,
                               dim_reduction=False, dim_reduction_limit=0.9, n_components=None,
                               index_folder_path='../Data/Index', index_close_as_metric=True,
                               metrics_list=None, model_path='../Data/Models', cross_model_name='autofeat.joblib',
                               ):
    """
    使用随机森林模型预测基金走势。

    :param the_fund_code: 基金代码，默认为 '159919.SZ'。
    :param n_days: 用于预测的天数，默认为 20 天。
    :param folder_path: 数据文件夹路径，默认为 '../Data'。
    :param metrics_folder: 评价指标文件夹路径，默认为 '../Data/Metrics'。
    :param train_start: 训练数据开始日期，默认为 None 表示从数据的开始日期开始。
    :param train_end: 训练数据结束日期，默认为 '2023-12-31'。
    :param test_start: 测试数据开始日期，默认为 '2024-01-01'。
    :param test_end: 测试数据结束日期，默认为 '2025-03-31'。
    :param nan_method: 训练数据中 nan 的处理方法, 默认为 'drop', 可选 'median' 或 'mean'。
    :param standardize_method: 数据标准化/归一化的方法，默认为 'both'，可选['both', 'minmax', 'zscore', 'none']。
    :param period_metrics: bool, 是否使用区间指标作为训练参数，默认为 True
    :param rolling_metrics: bool, 是否使用滚动指标作为训练参数，默认为 True
    :param random_seed: 随机种子，默认为 42。
    :param n_iter: 随机搜索的迭代次数，默认为 20 次。
    :param cv: 交叉验证的折数，默认为 5 折。
    :param threshold: 置信度阈值，默认为 None。
    :param basic_data_as_metric: bool, 是否将基本数据(例如:开盘价/收盘价/交易量等等)作为指标数据，默认为 False
    :param import_feature_only: bool, 是否仅仅使用重要特征进行训练，默认为 False。
    :param top_n: int, 从每个模型中选择的顶级特征数量，默认为 200。
    :param return_threshold: float, 标签生成方法，默认为 0, 表示使用未来收益率大于 0 的样本标记为 1，否则为 0;
    :param parameter_dict: dict, 最优参数字典，如果为 None，则自动调参。
    :param dim_reduction: bool, 是否做PCA降维
    :param dim_reduction_limit: float, 做PCA降维时保留的解释方差比率，默认为 0.9。
    :param n_components: int, PCA维度
    :param index_folder_path: 指数数据文件夹路径，默认为 '../Data/Index'。
    :param index_close_as_metric: 是否使用指数收盘价作为指标数据，默认为 True。
    :param metrics_list: list, 需要交叉特征的指标列表，默认为 None。None 表示不使用交叉特征
    :param model_path: 模型保存路径，默认为 '../Data/Models'。
    :param cross_model_name: 模型文件名，默认为 'autofeat.joblib'。

    :return: 训练完成的随机森林模型。
    """
    ''' 数据准备 '''
    print(f"[开始] 预测 {the_fund_code}, 未来{d}天的收益率 .....")

    # 数据读取，拆分训练集和测试集，以及 原始指标数据
    x_train, y_train, x_test, y_test, metrics_data = main_data_prepare(
        the_fund_code=the_fund_code,  # 基金代码，指定需要处理的基金（如'510050.SH'）。
        n_days=n_days,  # 预测未来收益的天数，用于计算未来对数收益率和生成目标标签。
        folder_path=folder_path,  # 基金价格数据的文件夹路径，包含基金的日频价格数据。
        index_folder_path=index_folder_path,  # 基金价格数据的文件夹路径，包含基金的日频价格数据。
        metrics_folder=metrics_folder,  # 基金指标数据的文件夹路径，包含用于训练模型的特征数据。
        train_start=train_start,  # 训练集开始日期，如果为None，则从数据的最早日期开始。
        train_end=train_end,  # 训练集结束日期，默认为'2023-12-31'。
        test_start=test_start,  # 测试集开始日期，默认为'2024-01-01'。
        test_end=test_end,  # 测试集结束日期，默认为'2025-03-31'。
        nan_method=nan_method,  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'。
        standardize_method=standardize_method,  # 数据标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'。
        period_metrics=period_metrics,
        rolling_metrics=rolling_metrics,
        basic_data_as_metric=basic_data_as_metric,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为False。
        return_threshold=return_threshold,  # 标签生成方法
        dim_reduction=dim_reduction,  # 是否PCA
        dim_reduction_limit=dim_reduction_limit,  # PCA数据降维保留的解释方差比率
        n_components=n_components,  # PCA维度
        index_close_as_metric=index_close_as_metric,  # 是否将指数收盘价作为指标数据，默认为True。
    )

    ''' 训练交叉特征 '''
    x_train, x_test = get_cross_metrics_func(x_train, x_test, metrics_list,
                                             model_folder_path=model_path,
                                             joblib_file_name=cross_model_name)

    ''' PCA降维 '''
    if dim_reduction:
        x_train, x_test = dim_reduction_func(x_train, x_test,
                                             dim_reduction_limit=dim_reduction_limit,
                                             n_components=n_components)

    ''' 特征选择 '''
    if import_feature_only:  # 是否使用重要特征进行模型训练
        print("[INFO] 仅使用重要特征进行训练")
        im_path = os.path.join(folder_path, f'selected_features_{n_days}.parquet')
        # 判断 '../Data/selected_features_{n_days}.parquet' 是否存在
        if os.path.exists(im_path):
            print(f"[INFO] {im_path} 文件已经存在")
            # 读取重要特征
            selected_features = pd.read_parquet(im_path)
            # 将重要特征转换为列表
            selected_features = selected_features['feature_name'].tolist()
            # 根据重要特征筛选训练数据
            x_train = x_train[selected_features]
            # 根据重要特征筛选测试数据
            x_test = x_test[selected_features]

        else:
            print(f"[INFO] {im_path} 文件不存在, 开始重要性特征筛查")
            selected_features, x_train, x_test, y_train, y_test, metrics_data = (
                find_important_features(
                    x_train,  # 训练集特征数据。
                    y_train,  # 训练集标签数据。
                    x_test,  # 测试集特征数据。
                    y_test,  # 测试集标签数据。
                    metrics_data,
                    folder_path=folder_path,  # 保存输出文件的文件夹路径。
                    n_days=n_days,  # 用于训练和测试的天数，默认为10。
                    top_n=top_n,  # 从每个模型中选择的顶级特征数量，默认为200。
                    save_features_name=True,  # 是否保存选定特征的名称，默认为False。
                ))
        # 根据重要特征筛选指标数据
        metrics_data = metrics_data[['ts_code', 'date'] + selected_features]
        print(f"[INFO] 重要性特征筛选后, 训练集数据量为: {x_train.shape[1]}列, 测试集数据量为: {x_test.shape[1]}列")

    ''' 训练模型 '''
    if threshold:
        trained_model = confidence_based_random_forest(
            x_train, x_test, y_train, y_test,
            random_seed=random_seed, n_iter=n_iter, cv=cv, threshold=threshold,
            parameter_dict=parameter_dict, return_threshold=return_threshold)
    else:
        # 训练随机森林模型
        trained_model = train_and_test_random_forest(
            x_train, x_test, y_train, y_test, metrics_data,
            random_seed=random_seed, n_iter=n_iter, cv=cv, best_dict=parameter_dict,
            return_threshold=return_threshold, dim_reduction=dim_reduction)

    # 输出模型训练完成的信息
    print("[完成] 模型训练完成")
    # 返回训练完成的模型
    return trained_model


if __name__ == '__main__':
    metrics_list = [
        'low', 'high', 'amount', 'open', 'close', 'vol',
        'pct', 'change', 'log',
        'TotalReturn:5d', 'TotalReturn:10d', 'TotalReturn:15d', 'TotalReturn:25d',
        'Volatility:5d', 'Volatility:10d', 'Volatility:15d', 'Volatility:25d',
        'AvgLow:5d', 'AvgLow:10d', 'AvgLow:15d', 'AvgLow:25d',
        'AvgHigh:5d', 'AvgHigh:10d', 'AvgHigh:15d', 'AvgHigh:25d',
        'VolAvg:5d', 'VolAvg:10d', 'VolAvg:15d', 'VolAvg:25d',
        'PriceSigma:3', 'PriceSigma:5', 'PriceSigma:10', 'PriceSigma:15',
        'CloseMA:3', 'CloseMA:5', 'CloseMA:10', 'CloseMA:15',
        'VolMA:3', 'VolMA:5', 'VolMA:10', 'VolMA:15',
        'TRIX:3', 'TRIX:5', 'TRIX:10', 'TRIX:15',
        'PVT:0', 'OBV:0',
        'RSI:3', 'RSI:5', 'RSI:10', 'RSI:15',
        'EMA:3', 'EMA:5', 'EMA:10', 'EMA:15',
        'PSY:3', 'PSY:5', 'PSY:10', 'PSY:15',
        'CCI:3', 'CCI:5', 'CCI:10', 'CCI:15',
        'CR:3', 'CR:5', 'CR:10', 'CR:15',
        'VR:3', 'VR:5', 'VR:10', 'VR:15',
        'AR:3', 'AR:5', 'AR:10', 'AR:15',
        'BR:3', 'BR:5', 'BR:10', 'BR:15',
        'PDI:3', 'PDI:5', 'PDI:10', 'PDI:15',
        'MDI:3', 'MDI:5', 'MDI:10', 'MDI:15',
        'DKX:3', 'DKX:5', 'DKX:10', 'DKX:15',
        'BIAS:3', 'BIAS:5', 'BIAS:10', 'BIAS:15',
        'KDJ-K-3:3', 'KDJ-K-3:5', 'KDJ-K-3:10', 'KDJ-K-3:15',
        'KDJ-D-3:3', 'KDJ-D-3:5', 'KDJ-D-3:10', 'KDJ-D-3:15',
        'KDJ-J-3:3', 'KDJ-J-3:5', 'KDJ-J-3:10', 'KDJ-J-3:15',
        'BollDo-2:3', 'BollDo-2:5', 'BollDo-2:10', 'BollDo-2:15',
        'BollUp-2:3', 'BollUp-2:5', 'BollUp-2:10', 'BollUp-2:15',
    ]
    for d in [10]:
        # 调用预测主函数 predict_main_random_forest，用于执行基金数据预处理、模型训练和测试等任务
        final_model = predict_main_random_forest(
            the_fund_code='510050.SH',  # 指定基金代码，此处为 '510050.SH'
            n_days=d,  # 预测未来收益的天数，变量 d 在循环中定义，表示不同的预测周期
            folder_path='../Data',  # 基金价格数据的文件夹路径，默认为 '../Data'
            metrics_folder='../Data/Metrics',  # 基金指标数据的文件夹路径，默认为 '../Data/Metrics'
            train_start=None,  # 训练集开始日期，如果为 None，则从数据的最早日期开始
            train_end='2024-03-31',  # 训练集结束日期，指定为 '2024-11-30'
            test_start='2024-04-01',  # 测试集开始日期，指定为 '2024-12-01'
            test_end='2025-04-30',  # 测试集结束日期，指定为 '2025-04-30'
            nan_method='drop',  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'
            standardize_method='zscore',  # 指标标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'。
            period_metrics=True,  # 是否使用区间指标作为训练参数，默认为 True
            rolling_metrics=True,  # 是否使用滚动指标作为训练参数，默认为 True
            random_seed=42,  # 随机种子，确保结果可重复，默认为 42
            n_iter=20,  # 随机搜索的迭代次数，默认为 20 次
            cv=5,  # 交叉验证的折数，默认为 5 折
            threshold=None,  # 置信度阈值，如果为 None，则不使用置信度筛选
            basic_data_as_metric=True,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为 True
            import_feature_only=False,  # 是否仅使用重要特征进行训练，默认为 True
            return_threshold=0.0,  # 标签生成方法，未来收益率大于 0.01 的样本标记为 1，否则为 0
            parameter_dict=None,  # 最优参数字典，如果为 None，则自动调参
            dim_reduction=True,  # 是否PCA
            dim_reduction_limit=0.99,  # PCA降维的方差解释比例, 默认为0.9
            n_components=None,  # PCA维度, 写None表示自动选择, 保留90%方差解释比率
            index_folder_path='../Data/Index',  # 指数数据的文件夹路径，默认为 '../Data/Index'
            index_close_as_metric=True,  # 是否使用指数收盘价作为指标数据，默认为 True
            metrics_list=metrics_list,
            model_path='../Data/Models', cross_model_name='autofeat_model_2_10_PeriodAndRolling.joblib'
        )
        # 保存模型
        filename = f'random_forest_model_{d}.joblib'
        joblib.dump(final_model, filename)
