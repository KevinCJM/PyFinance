from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import os


# 训练模型并返回前n个重要特征名
def get_top_n_features(model, x, y, n=150):
    """
    提取模型训练后最重要的n个特征。

    :param model: 训练模型，需支持特征重要性或系数属性。
    :param x: 特征数据，用于模型训练。
    :param y: 目标标签，与x对应。
    :param n: 需要提取的特征数量，默认为150。
    :return: 最重要的n个特征的名称列表。
    """
    # 训练模型
    model.fit(x, y)

    # 根据模型类型提取特征重要性或系数
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_).flatten()
    else:
        # 如果模型不支持特征重要性提取，抛出异常
        raise ValueError("模型不支持特征重要性提取")

    # 获取最重要的n个特征的索引
    top_idx = np.argsort(importance)[::-1][:n]

    # 返回最重要的n个特征的名称列表
    return x.columns[top_idx].tolist()


# 从多个模型中获取重要特征并取并集
def feature_union_from_multiple_models(x_train, y_train, x_test, models, top_n=150):
    """
    从多个模型中提取重要特征，并将这些特征取并集，用于后续的模型训练和预测。

    :param x_train: 训练数据集的特征数据
    :param y_train: 训练数据集的目标变量
    :param x_test: 测试数据集的特征数据
    :param models: 一个包含多个模型的列表，用于特征选择
    :param top_n: 从每个模型中选择的特征数量，默认值为150
    :return: 返回经过特征选择后的训练数据集和测试数据集，以及被选中的特征列表
    """
    # 初始化一个空的集合，用于存储所有模型选出的重要特征
    selected_features = set()
    # 遍历每个模型，提取其认为最重要的特征
    for model in models:
        # 调用get_top_n_features函数，获取当前模型认为最重要的n个特征
        top_feats = get_top_n_features(model, x_train, y_train, n=top_n)
        # 将当前模型选出的特征添加到集合中，自动去重
        selected_features.update(top_feats)

    # 将集合转换为列表，便于后续操作
    selected_features = list(selected_features)
    # 使用选中的特征，减少训练和测试数据集的维度
    x_train_reduced = x_train[selected_features]
    x_test_reduced = x_test[selected_features]
    # 返回经过特征选择后的训练数据集和测试数据集，以及被选中的特征列表
    return x_train_reduced, x_test_reduced, selected_features


# 从给定的训练和测试数据中识别重要特征
def find_important_features(x_train, y_train, x_test, y_test, metrics_data,
                            folder_path, n_days=10, top_n=200, save_features_name=False,
                            ):
    """
    从给定的训练和测试数据中识别重要特征，并可选地保存这些特征的名称。

    :param x_train: 训练集特征数据。
    :param y_train: 训练集标签数据。
    :param x_test: 测试集特征数据。
    :param y_test: 测试集标签数据。
    :param metrics_data: 用于特征选择的附加指标数据。
    :param folder_path: 保存输出文件的文件夹路径。
    :param n_days: 用于训练和测试的天数，默认为10。
    :param top_n: 从每个模型中选择的顶级特征数量，默认为200。
    :param save_features_name: 是否保存选定特征的名称，默认为False。
    :return: 返回选定的特征名称列表，以及训练和测试数据集的特征和标签。
    """
    # 模型列表，定义多个机器学习模型用于特征选择
    model_list = [
        RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1),
        ExtraTreesClassifier(
            n_estimators=300, random_state=42, n_jobs=-1),
        GradientBoostingClassifier(
            n_estimators=300, random_state=42),
        LogisticRegression(
            penalty='l1', solver='liblinear', max_iter=1000),
        LinearSVC(
            penalty='l1', dual=False, max_iter=1000)
    ]
    print("[INFO] 开始特征选择...")

    # 调用执行特征选择，从多个模型中提取重要特征并取并集
    x_train_reduced, x_test_reduced, selected_feature_names = feature_union_from_multiple_models(
        x_train,  # 训练集特征数据。
        y_train,  # 训练集标签数据。
        x_test,  # 测试集特征数据。
        models=model_list,  # 使用的模型列表，包含多个不同的机器学习模型。
        top_n=top_n  # 从每个模型中提取的重要特征数量，设置为200个。
    )

    # 打印最终选出的特征数量
    print(f"[INFO] 最终选出的特征数量为: {len(selected_feature_names)} 个")
    im_path = os.path.join(folder_path, f'selected_features_{n_days}.parquet')

    # 保存选出的特征名称
    if save_features_name:
        selected_feature_names_df = pd.DataFrame(
            selected_feature_names, columns=['feature_name'])
        selected_feature_names_df.to_parquet(im_path)
        print(f"[INFO] 特征名称保存在: {im_path}")

    # 返回选定的特征名称列表，以及训练和测试数据集的特征和标签
    return (selected_feature_names,  # 最终选出的特征名称列表
            x_train_reduced,  # 经过特征选择后的训练集特征数据
            x_test_reduced,  # 经过特征选择后的测试集特征数据
            y_train,  # 训练集的目标标签数据
            y_test,  # 测试集的目标标签数据
            metrics_data  # 原始指标数据
            )


if __name__ == "__main__":
    # # 数据读取，包括训练集和测试集，以及 原始指标数据
    # x_train, y_train, x_test, y_test, metrics_data = main_data_prepare(
    #     the_fund_code='510050.SH',
    #     n_days=10,
    #     folder_path='../Data',
    #     metrics_folder='../Data/Metrics',
    #     train_start=None,
    #     train_end='2024-11-30',
    #     test_start='2024-12-01',
    #     test_end='2025-04-30',
    #     nan_method='drop',
    #     basic_data_as_metric=True,
    # )
    #
    # selected_features, x_train_reduced, x_test_reduced, y_train, y_test, metrics_data = find_important_features(
    #     x_train,  # 训练集特征数据。
    #     y_train,  # 训练集标签数据。
    #     x_test,  # 测试集特征数据。
    #     y_test,  # 测试集标签数据。
    #     metrics_data,
    #     folder_path='../Data',  # 保存输出文件的文件夹路径。
    #     n_days=10,  # 用于训练和测试的天数，默认为10。
    #     top_n=200,  # 从每个模型中选择的顶级特征数量，默认为200。
    #     save_features_name=True,  # 是否保存选定特征的名称，默认为False。
    # )
    #
    # print(selected_features)
    pass
