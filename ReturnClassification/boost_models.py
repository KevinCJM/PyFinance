# -*- encoding: utf-8 -*-
"""
@File: boost_models.py
@Modify Time: 2025/4/24 17:22       
@Author: Kevin-Chen
@Descriptions: 
"""
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import numpy as np


def optimize_threshold(y_true, proba):
    """根据验证集 PR 曲线寻找 F1 最优阈值"""
    p, r, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / (p + r + 1e-12)
    idx = np.nanargmax(f1)
    return thr[idx]


def kfold_objective(trial, model_name, X, y, n_splits=5):
    """Optuna 目标函数：返回 K-Fold 平均 F1"""
    # —— 1. 超参空间 —— #
    if model_name == "lgb":
        import lightgbm as lgb
        params = {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("leaves", 16, 256, log=True),
            "max_depth": trial.suggest_int("depth", 3, 12),
            "feature_fraction": trial.suggest_float("feat_frac", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bag_frac", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_leaf", 10, 100),
            "lambda_l1": trial.suggest_float("l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("l2", 0.0, 5.0),
            "objective": "binary",
            "verbose": -1,
        }
        clf_factory = lambda: lgb.LGBMClassifier(
            n_estimators=3000,
            **params,
        )

    elif model_name == "xgb":
        import xgboost as xgb
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("mcw", 1, 50, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "lambda": trial.suggest_float("reg_l2", 0.0, 5.0),
            "alpha": trial.suggest_float("reg_l1", 0.0, 5.0),
            "objective": "binary:logistic",
            "eval_metric": "auc",
        }
        clf_factory = lambda: xgb.XGBClassifier(
            n_estimators=3000,
            tree_method="hist",
            **params,
        )

    elif model_name == "cat":
        from catboost import CatBoostClassifier
        params = {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
            "border_count": trial.suggest_int("border", 32, 255),
        }
        clf_factory = lambda: CatBoostClassifier(
            iterations=3000,
            loss_function="Logloss",
            verbose=False,
            **params,
        )

    else:  # logistic
        from sklearn.linear_model import LogisticRegression
        params = {
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
        }
        clf_factory = lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="liblinear",
                    max_iter=1000,
                    class_weight="balanced",
                    **params
                )),
            ]
        )

    # —— 2. K-Fold 评估 —— #
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1_scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = clf_factory()

        if model_name in {"lgb", "xgb", "cat"}:
            # 类别不平衡
            ratio = (y_train == 0).sum() / (y_train == 1).sum()
            if model_name == "lgb":
                clf.set_params(is_unbalance=True)
            elif model_name == "xgb":
                clf.set_params(scale_pos_weight=ratio)
            # early stopping
            clf.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=200,
                verbose=False
            )
        else:
            clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else clf.predict_proba(X_val)
        thr = optimize_threshold(y_val, proba)
        y_pred = (proba >= thr).astype(int)
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))

    return np.mean(f1_scores)


def run_optuna(model_name, x_train, y_train, n_trials=50):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: kfold_objective(t, model_name, x_train, y_train), n_trials=n_trials, n_jobs=1)
    print("最佳 F1:", study.best_value, "最佳参数:", study.best_params)
    return study.best_params


# 使用给定的模型名称和最佳参数对模型进行最终的训练和评估
# 使用给定的模型名称和最佳参数对模型进行最终的训练和评估
def final_train_eval(model_name, x_tr, y_tr, x_te, y_te, n_trials):
    """
    使用给定的模型名称和最佳参数对模型进行最终的训练和评估。

    参数:
    - model_name: 模型名称，支持 "lgb", "xgb", "cat" 和 "logistic"。
    - x_tr: 训练集特征数据。
    - y_tr: 训练集标签数据。
    - x_te: 测试集特征数据。
    - y_te: 测试集标签数据。
    - n_trials: Optuna 超参数优化的试验次数。

    返回:
    - clf: 训练好的模型。
    - thr: 优化后的阈值。
    """

    # 使用 Optuna 进行超参数优化，并返回最佳参数
    best_params = run_optuna(model_name, x_train=x_tr, y_train=y_tr, n_trials=n_trials)

    # —— 构造模型 —— #
    if model_name == "lgb":
        # 如果模型名称是 "lgb"，则使用 LightGBM 构造模型
        import lightgbm as lgb

        clf = lgb.LGBMClassifier(
            n_estimators=3000,  # 决策树的数量，这里设置为3000棵
            objective="binary",  # 目标函数类型，"binary"表示二分类问题
            verbose=-1,  # 控制日志输出的详细程度，-1表示不输出任何信息
            **best_params  # 使用通过 Optuna 得到的最佳参数字典进行模型初始化
        )

        # 训练模型，并设置相关的训练参数
        clf.fit(
            x_tr,  # 训练集特征数据，用于训练模型
            y_tr,  # 训练集标签数据，与特征数据一一对应
            eval_set=[(x_te, y_te)],  # 验证集数据，包含特征和标签，用于监控模型在验证集上的表现
            early_stopping_rounds=200,  # 提前停止训练的轮数。如果验证集的表现连续200轮没有提升，则停止训练
            verbose=False  # 控制是否输出训练过程中的详细信息。False表示不输出任何训练日志
        )

    elif model_name == "xgb":
        # 如果模型名称是 "xgb"，则使用 XGBoost 构造模型
        import xgboost as xgb

        # 计算类别不平衡的比例，用于调整正负样本权重
        ratio = (y_tr == 0).sum() / (y_tr == 1).sum()

        clf = xgb.XGBClassifier(
            n_estimators=3000,  # 决策树的数量，这里设置为3000棵
            tree_method="hist",  # 使用直方图方法加速训练（适用于大规模数据）
            scale_pos_weight=ratio,  # 正负样本的权重比例，用于处理类别不平衡问题
            **best_params  # 使用通过 Optuna 得到的最佳参数字典进行模型初始化
        )

        # 训练模型，并设置相关的训练参数
        clf.fit(
            x_tr,  # 训练集特征数据，用于训练模型
            y_tr,  # 训练集标签数据，与特征数据一一对应
            eval_set=[(x_te, y_te)],  # 验证集数据，包含特征和标签，用于监控模型在验证集上的表现
            early_stopping_rounds=200,  # 提前停止训练的轮数。如果验证集的表现连续200轮没有提升，则停止训练
            verbose=False  # 控制是否输出训练过程中的详细信息。False表示不输出任何训练日志
        )

    elif model_name == "cat":
        # 如果模型名称是 "cat"，则使用 CatBoost 构造模型
        from catboost import CatBoostClassifier

        # 创建 CatBoost 分类器，并设置相关参数
        clf = CatBoostClassifier(
            iterations=3000,  # 决策树的最大迭代次数（即模型中树的数量），这里设置为3000棵
            loss_function="Logloss",  # 损失函数类型，"Logloss"表示逻辑损失函数，适用于二分类问题
            verbose=False,  # 控制是否输出训练过程中的日志信息。False表示不输出任何日志
            **best_params  # 使用通过 Optuna 得到的最佳参数字典进行模型初始化
        )

        # 训练模型，并设置相关的训练参数
        clf.fit(
            x_tr,  # 训练集特征数据，用于训练模型
            y_tr,  # 训练集标签数据，与特征数据一一对应
            eval_set=[(x_te, y_te)],  # 验证集数据，包含特征和标签，用于监控模型在验证集上的表现
            early_stopping_rounds=200  # 提前停止训练的轮数。如果验证集的表现连续200轮没有提升，则停止训练
        )

    else:  # logistic
        # 如果模型名称不是上述三种，则使用 Logistic 回归构造模型
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # 创建一个包含特征缩放和逻辑回归的管道模型
        clf = Pipeline([
            ("scaler", StandardScaler()),  # 第一步：对特征进行标准化处理，将数据缩放到均值为0，标准差为1
            ("clf", LogisticRegression(  # 第二步：使用逻辑回归模型进行分类
                max_iter=1000,  # 设置逻辑回归的最大迭代次数为1000，防止因迭代次数不足导致模型不收敛
                class_weight="balanced",  # 自动调整类别权重，用于处理类别不平衡问题。"balanced"会根据样本数量自动计算权重
                **best_params  # 使用通过 Optuna 得到的最佳参数字典，动态设置逻辑回归的其他参数
            ))
        ])

        # 训练模型
        clf.fit(x_tr, y_tr)

    # —— 阈值优化 —— #
    # 根据测试集的预测概率和真实标签优化阈值，并打印分类报告
    proba = clf.predict_proba(x_te)[:, 1]  # 获取测试集的正类预测概率
    thr = optimize_threshold(y_te, proba)  # 根据 PR 曲线寻找 F1 最优阈值
    y_pred = (proba >= thr).astype(int)  # 根据优化后的阈值生成预测标签

    # 打印分类报告
    print(f"\n{model_name.upper()} 在测试集的报告：\n", classification_report(y_te, y_pred, digits=3))

    return clf, thr  # 返回训练好的模型和优化后的阈值


if __name__ == '__main__':
    from ReturnClassification.metrics_data_prepare import main_data_prepare

    # 数据读取，拆分训练集和测试集，以及 原始指标数据
    x_train, y_train, x_test, y_test, metrics_data = main_data_prepare(
        the_fund_code='510050.SH',  # 指定基金代码，此处为 '510050.SH'
        n_days=5,  # 预测未来收益的天数，变量 d 在循环中定义，表示不同的预测周期
        folder_path='../Data',  # 基金价格数据的文件夹路径，默认为 '../Data'
        metrics_folder='../Data/Metrics',  # 基金指标数据的文件夹路径，默认为 '../Data/Metrics'
        train_start=None,  # 训练集开始日期，如果为 None，则从数据的最早日期开始
        train_end='2024-03-31',  # 训练集结束日期，指定为 '2024-11-30'
        test_start='2024-04-01',  # 测试集开始日期，指定为 '2024-12-01'
        test_end='2025-04-30',  # 测试集结束日期，指定为 '2025-04-30'
        nan_method='drop',  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'
        period_metrics=True,  # 是否使用区间指标，默认为 True
        rolling_metrics=False,  # 是否使用滚动指标，默认为 True
        standardize_method='zscore',  # 指标标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'。
        basic_data_as_metric=True,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为 True
        return_threshold=0.0,  # 标签生成方法，未来收益率大于 0.01 的样本标记为 1，否则为 0
        dim_reduction=True,  # 是否PCA
        dim_reduction_limit=0.9,
        index_folder_path='../Data/Index',  # 指数数据的文件夹路径，默认为 '../Data/Index'
        index_close_as_metric=False,  # 是否使用指数收盘价作为指标数据，默认为 True
    )
