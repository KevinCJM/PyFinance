# -*- encoding: utf-8 -*-
"""
@File: auto_tuning.py
@Modify Time: 2025/4/20 16:26       
@Author: Kevin-Chen
@Descriptions: 
"""
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform

# 定义一个字典 param_spaces，用于存储不同分类器模型及其对应的超参数搜索空间
param_spaces = {
    # 随机森林分类器 (RandomForestClassifier)
    "RandomForestClassifier": {
        # 模型：随机森林分类器，random_state=42 确保结果可复现，max_features='sqrt' 表示每个分裂考虑的特征数为 sqrt(n_features)
        "model": RandomForestClassifier(random_state=42, max_features='sqrt'),
        # 参数空间：
        "params": {
            "n_estimators": randint(100, 1000),  # 决策树的数量，范围为 [100, 1000]
            "max_depth": randint(5, 20),         # 树的最大深度，范围为 [5, 20]
            "min_samples_split": randint(2, 10), # 分裂内部节点所需的最小样本数，范围为 [2, 10]
            "min_samples_leaf": randint(1, 5),   # 叶节点所需的最小样本数，范围为 [1, 5]
        },
    },
    # 极端随机树分类器 (ExtraTreesClassifier)
    "ExtraTreesClassifier": {
        "model": ExtraTreesClassifier(random_state=42),  # 模型：极端随机树分类器，random_state=42 确保结果可复现
        "params": {
            "n_estimators": randint(100, 1000),  # 决策树的数量，范围为 [100, 1000]
            "max_depth": randint(5, 20),         # 树的最大深度，范围为 [5, 20]
            "min_samples_split": randint(2, 10), # 分裂内部节点所需的最小样本数，范围为 [2, 10]
            "min_samples_leaf": randint(1, 5),   # 叶节点所需的最小样本数，范围为 [1, 5]
        },
    },
    # 梯度提升分类器 (GradientBoostingClassifier)
    "GradientBoostingClassifier": {
        "model": GradientBoostingClassifier(random_state=42),  # 模型：梯度提升分类器，random_state=42 确保结果可复现
        "params": {
            "n_estimators": randint(100, 1000),  # 弱学习器（通常是决策树）的数量，范围为 [100, 1000]
            "learning_rate": uniform(0.01, 0.3), # 学习率，控制每棵树对最终结果的影响程度，范围为 [0.01, 0.31]
            "max_depth": randint(3, 10),         # 树的最大深度，范围为 [3, 10]
            "min_samples_split": randint(2, 10), # 分裂内部节点所需的最小样本数，范围为 [2, 10]
            "min_samples_leaf": randint(1, 5),   # 叶节点所需的最小样本数，范围为 [1, 5]
        },
    },
    # 逻辑回归分类器 (LogisticRegression)
    "LogisticRegression": {
        "model": LogisticRegression(solver="saga", max_iter=10000, random_state=42),  # 模型：逻辑回归分类器，solver="saga" 支持 L1/L2 正则化，max_iter=10000 设置最大迭代次数
        "params": {
            "C": uniform(0.01, 10),  # 正则化强度的倒数，范围为 [0.01, 10.01]
            "penalty": ["l1", "l2"], # 正则化类型，可选值为 'l1' 或 'l2'
        },
    },
    # 线性支持向量机分类器 (LinearSVC)
    "LinearSVC": {
        "model": LinearSVC(dual=False, max_iter=10000, random_state=42),  # 模型：线性支持向量机分类器，dual=False 适用于样本数大于特征数的情况，max_iter=10000 设置最大迭代次数
        "params": {
            "C": uniform(0.01, 10),  # 正则化强度的倒数，范围为 [0.01, 10.01]
        },
    },
    # XGBoost 分类器 (XGBClassifier)
    "XGBClassifier": {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),  # 模型：XGBoost 分类器，use_label_encoder=False 禁用标签编码器，eval_metric="logloss" 使用对数损失作为评估指标
        "params": {
            "n_estimators": randint(100, 1000),  # 弱学习器的数量，范围为 [100, 1000]
            "learning_rate": uniform(0.01, 0.3), # 学习率，范围为 [0.01, 0.31]
            "max_depth": randint(3, 10),         # 树的最大深度，范围为 [3, 10]
            "subsample": uniform(0.6, 0.4),      # 训练每棵树时使用的样本比例，范围为 [0.6, 1.0]
            "colsample_bytree": uniform(0.6, 0.4), # 训练每棵树时使用的特征比例，范围为 [0.6, 1.0]
        },
    },
    # LightGBM 分类器 (LGBMClassifier)
    "LGBMClassifier": {
        "model": LGBMClassifier(random_state=42),  # 模型：LightGBM 分类器，random_state=42 确保结果可复现
        "params": {
            "n_estimators": randint(100, 1000),  # 弱学习器的数量，范围为 [100, 1000]
            "learning_rate": uniform(0.01, 0.3), # 学习率，范围为 [0.01, 0.31]
            "num_leaves": randint(20, 150),      # 每棵树的叶子数，范围为 [20, 150]
            "max_depth": randint(3, 10),         # 树的最大深度，范围为 [3, 10]
            "subsample": uniform(0.6, 0.4),      # 训练每棵树时使用的样本比例，范围为 [0.6, 1.0]
            "colsample_bytree": uniform(0.6, 0.4), # 训练每棵树时使用的特征比例，范围为 [0.6, 1.0]
        },
    },
    # CatBoost 分类器 (CatBoostClassifier)
    "CatBoostClassifier": {
        "model": CatBoostClassifier(verbose=0, random_state=42),  # 模型：CatBoost 分类器，verbose=0 禁用训练过程中的输出信息
        "params": {
            "iterations": randint(100, 1000),  # 迭代次数（弱学习器数量），范围为 [100, 1000]
            "learning_rate": uniform(0.01, 0.3), # 学习率，范围为 [0.01, 0.31]
            "depth": randint(3, 10),           # 树的最大深度，范围为 [3, 10]
        },
    }
}
