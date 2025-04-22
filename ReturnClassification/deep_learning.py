# -*- encoding: utf-8 -*-
"""
@File: test.py.py
@Modify Time: 2025/4/21 19:35
@Author: Kevin-Chen
@Descriptions:
"""
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, Add, Multiply, Reshape
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation, Reshape, Lambda
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score
import tensorflow as tf


# 自定义FM层（使用Lambda封装tf操作）
def fm_cross_layer(embed):
    """
    实现FM（Factorization Machine）模型的交叉层计算。

    参数:
    embed: 输入的特征嵌入张量，形状为(batch_size, num_features, embedding_size)。

    返回:
    fm_output: 计算得到的FM交叉项输出张量，形状为(batch_size, embedding_size)。
    """
    # 计算所有特征嵌入向量的和
    summed_features_emb = tf.reduce_sum(embed, axis=1)  # ∑v_i
    # 计算上述和向量的平方
    summed_features_emb_square = tf.square(summed_features_emb)  # (∑v_i)^2

    # 计算每个特征嵌入向量的平方
    squared_features_emb = tf.square(embed)  # v_i^2
    # 计算上述平方向量的和
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, axis=1)  # ∑v_i^2

    # 根据FM模型的交叉项公式，计算输出
    fm_output = 0.5 * (summed_features_emb_square - squared_sum_features_emb)
    return fm_output


# 构建 DeepFM 模型
def build_deep_fm(input_dim,
                  embed_dim=8,
                  dnn_units=None,
                  dropout_rate=0.5,
                  activation='relu',
                  use_batch_norm=False,
                  learning_rate=0.001
                  ):
    """
    构建DeepFM模型。

    DeepFM是一种深度学习特征选择模型，它结合了因子分解机（FM）和深度神经网络（DNN），
    用于同时捕捉线性和非线性的特征交互。

    参数:
    - input_dim (int): 输入特征的维度。
    - embed_dim (int): 特征嵌入的维度。默认为8。
    - dnn_units (list of int): DNN部分每层的单元数。默认为[128, 64, 32]。
    - dropout_rate (float): Dropout层的比例。默认为0.5。
    - activation (str): DNN部分使用的激活函数。默认为'relu'。
    - use_batch_norm (bool): 是否在DNN部分使用批量归一化。默认为False。
    - learning_rate (float): 优化器的学习率。默认为0.001。

    返回:
    - tf.keras.Model: 构建的DeepFM模型。
    """
    # 输入层
    inputs = Input(shape=(input_dim,))

    # 如果未指定DNN层的单元数，使用默认值
    if dnn_units is None:
        dnn_units = [128, 64, 32]

    # Embedding层 👉 把每个离散特征转换为一个连续的低维向量表示
    embed = Dense(input_dim * embed_dim, activation='linear')(inputs)
    embed = Reshape((input_dim, embed_dim))(embed)

    # FM交叉项部分 👉 显式建模低阶特征交叉关系
    fm_output = Lambda(fm_cross_layer)(embed)

    # DNN部分 👉 学习高阶组合和非线性变换
    x = Flatten()(embed)
    for units in dnn_units:
        x = Dense(units)(x)
        # 如果使用批量归一化
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        # 如果Dropout率大于0
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # 合并FM和DNN部分的输出
    concat = Concatenate()([fm_output, x])
    # 输出层
    output = Dense(1, activation='sigmoid')(concat)

    # 构建模型
    if learning_rate is None:  # 如果没有提供学习率，则使用默认的动态学习率
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)
    else:  # 如果提供了学习率，则使用指定的学习率
        optimizer = Adam(learning_rate=learning_rate)

    # 定义模型并编译
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


# DeepFM 模型自动调参
def objective_deep_fm(trial,
                      validation_mode='kfold',
                      n_splits=5
                      ):
    """
    Optuna试验的目标函数，用于优化DeepFM模型的超参数。

    参数:
    - trial: Optuna的试验对象。
    - validation_mode: 'split' 或 'kfold'。
    - n_splits: KFold的折数，默认5折。

    返回:
    - 平均F1分数，作为评估指标。
    """

    # 超参数空间定义
    embed_dim = trial.suggest_categorical('embed_dim', [4, 8, 16, 32])
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    dnn_units = trial.suggest_categorical("dnn_units", [
        [128, 64],
        [128, 64, 32],
        [128, 64, 32, 16],
        [64, 32],
        [64, 32, 16],
        [64, 32, 16, 8],
        [32, 16],
        [32, 16, 8],
    ])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

    if validation_mode == 'split':
        x_subtrain, x_val, y_subtrain, y_val = train_test_split(
            x_train, y_train, test_size=0.1, shuffle=False
        )

        model = build_deep_fm(
            input_dim=x_train.shape[1],
            embed_dim=embed_dim,
            dnn_units=dnn_units,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm,
            learning_rate=learning_rate
        )

        model.fit(
            x_subtrain, y_subtrain,
            validation_data=(x_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )

        y_val_pred = (model.predict(x_val) > 0.5).astype(int)
        return f1_score(y_val, y_val_pred)

    elif validation_mode == 'kfold':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in kf.split(x_train):
            x_subtrain, x_val = x_train[train_idx], x_train[val_idx]
            y_subtrain, y_val = y_train[train_idx], y_train[val_idx]

            model = build_deep_fm(
                input_dim=x_train.shape[1],
                embed_dim=embed_dim,
                dnn_units=dnn_units,
                dropout_rate=dropout_rate,
                activation=activation,
                use_batch_norm=use_batch_norm,
                learning_rate=learning_rate
            )

            model.fit(
                x_subtrain, y_subtrain,
                validation_data=(x_val, y_val),
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[early_stop]
            )

            y_val_pred = (model.predict(x_val) > 0.5).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    else:
        raise ValueError("validation_mode must be either 'split' or 'kfold'")


# Wide&Deep 模型
def build_wide_deep(
        input_dim,
        deep_units=None,
        dropout_rate=0.5,
        use_batch_norm=False,
        wide_activation='linear',  # 可尝试 linear, relu, sigmoid, tanh
        activation='relu',  # 可尝试 'relu', 'tanh', 'selu'
        learning_rate=None
):
    """
    构建一个Wide&Deep模型。

    参数:
    - input_dim: 输入维度。
    - deep_units: Deep部分的单元数列表，每一项代表相应层的单元数。
    - dropout_rate: Dropout比率，用于防止过拟合。 0.3 ~ 0.5，小样本建议偏大
    - use_batch_norm: 是否使用批量归一化。
        深层网络（3层以上）	    ✅ 推荐
        特征分布差异大（如降维后）	✅ 推荐
        小样本+过拟合明显	        ❌ 可先不用，先用 Dropout
    - wide_activation: Wide部分的激活函数。
        'linear'（默认）	不加任何变换，输出 ∑wx+b     标准做法
        'relu'	        线性结果负数会变为 0	      强行非负，不推荐
        'sigmoid'	    将输出压缩到(0,1)区间        适用于概率表达，但通常在最终层用
        'tanh'	        压缩到(-1,1)               极少用于 Wide
    - activation: Deep部分的激活函数。
        'relu'	        默认首选        简单高效，收敛快
        'tanh'	        有中心化        适合对称问题	有梯度饱和风险
        'selu'	        自归一化        适合深网络	对输入敏感，需要搭配特定初始化
    - learning_rate: 学习率。
        小样本 + 小网络	    0.01 ~ 0.001
        中等样本 + 中深网络	0.001 ~ 0.0005（推荐）
        大样本 + 深模型	    0.0005 ~ 0.0001

    返回:
    编译后的Wide&Deep模型。
    """
    # 定义模型输入
    inputs = Input(shape=(input_dim,))
    # 默认Deep部分的结构为[128, 64, 32]
    if deep_units is None:
        deep_units = [128, 64, 32]

    # Wide部分 👉 建模“记忆性”特征和低阶关系
    wide_output = Dense(1, activation=wide_activation)(inputs)

    # Deep部分 👉 捕捉高阶特征组合的非线性关系 (多层前馈神经网络 MLP)
    x = inputs
    for units in deep_units:
        x = Dense(units)(x)  # 全连接层
        # 根据配置决定是否添加批量归一化层
        if use_batch_norm:
            x = BatchNormalization()(x)  # 将每层输出标准化为均值为0，方差为1
        # 添加激活函数层
        x = Activation(activation)(x)
        # 根据配置决定是否添加Dropout层
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)  # 随机“屏蔽”一部分神经元，防止过拟合
    # Deep部分的打分层,
    deep_output = Dense(1, activation='linear')(x)

    # 合并 Wide&Deep 的输出 👉 Wide与Deep是互补关系, Wide是低阶/线性/表达能力弱, Deep是高阶/非线性/容易过拟合
    combined = Add()([wide_output, deep_output])
    # 最终输出层，用于二分类问题
    output = Dense(1, activation='sigmoid')(combined)

    # 构建模型
    if learning_rate is None:  # 如果没有提供学习率，则使用默认的动态学习率
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)
    else:  # 如果提供了学习率，则使用指定的学习率
        optimizer = Adam(learning_rate=learning_rate)

    # 定义模型并编译
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


# Optuna试验的目标函数, 用于优化Wide&Deep模型的超参数
def objective_wide_deep(trial, n_splits=5):
    """
    使用 K 折交叉验证评估 Wide&Deep 模型的性能（f1-score）。
    """

    # 超参数搜索空间定义
    dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    wide_activation = trial.suggest_categorical("wide_activation", ['linear', 'sigmoid'])
    activation = trial.suggest_categorical("activation", ['relu', 'tanh'])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    deep_units = trial.suggest_categorical("deep_units", [
        [128, 64],
        [128, 64, 32],
        [128, 64, 32, 16],
        [64, 32],
        [64, 32, 16],
        [64, 32, 16, 8],
        [32, 16],
        [32, 16, 8],
    ])

    # K折交叉验证设置
    kf = KFold(
        n_splits=n_splits,  # K折交叉验证的折数
        shuffle=True,  # 打乱数据，避免偏倚
        random_state=42  # 返回平均 F1 分数作为评价指标
    )
    f1_scores = []

    for train_index, val_index in kf.split(x_train):
        x_subtrain, x_val = x_train[train_index], x_train[val_index]
        y_subtrain, y_val = y_train[train_index], y_train[val_index]

        # 构建模型
        model = build_wide_deep(
            input_dim=x_train.shape[1],
            deep_units=deep_units,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            wide_activation=wide_activation,
            activation=activation,
            learning_rate=learning_rate
        )

        # 早停回调
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=0
        )

        # 训练模型
        model.fit(
            x_subtrain, y_subtrain,
            validation_data=(x_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )

        # 验证集评估
        y_val_pred = (model.predict(x_val) > 0.5).astype(int)
        f1 = f1_score(y_val, y_val_pred)
        f1_scores.append(f1)

    # 返回平均 F1 作为目标优化指标
    return np.mean(f1_scores)


# DCN 模型 (深度交叉网络)
def build_dcn(input_dim,
              cross_layers=5,
              deep_units: list = None,
              dropout_rate: float = 0.5,
              learning_rate: float = None,
              use_batch_norm=False
              ):
    """
    构建深度交叉网络（Deep & Cross Network, DCN）模型。

    :param input_dim: 输入特征的维度。
    :param cross_layers: 交叉层的数量，默认为5层。
            小样本（几千）+ 高维（>200维）	    1 ~ 2 层（过多容易过拟合）
            中样本（几万）+ 中维（几十 ~ 百）	2 ~ 3 层（典型设置）
            大样本（百万）+ 低维（<50）	        3 ~ 4 层可尝试（提升高阶表达）
    :param deep_units: 深度部分每层的单元数列表，默认为[128, 64, 32]。
            小样本（几千）+ 高维（>200维）	    [64, 32] 或 [128, 64, 32]
            中样本（几万）	                [128, 64, 32] 或 [256, 128, 64]
            大样本（百万）	                [512, 256, 128] 或更深
    :param dropout_rate: Dropout比率，用于防止过拟合，默认为0.5。
            小样本（几千）     0.4 ~ 0.6
            中样本（几万）     0.3 ~ 0.5
            大样本（百万）     0.2 ~ 0.4
    :param learning_rate: 学习率，默认为 None 表示使用动态学习率。
            小样本 + 小网络	    0.01 ~ 0.001
            中等样本 + 中深网络	0.001 ~ 0.0005（推荐）
            大样本 + 深模型	    0.0005 ~ 0.0001
    :param use_batch_norm: 是否使用批量归一化（Batch Normalization），默认为False。
            深层网络（3层以上）	    ✅ 推荐
            特征分布差异大（如降维后）	✅ 推荐
            小样本+过拟合明显	        ❌ 可先不用，先用 Dropout
    :return: 构建好的DCN模型。
    """
    # 定义模型输入
    inputs = Input(shape=(input_dim,))
    x0 = inputs
    xl = inputs

    # 如果未提供deep_units，则使用默认值
    if deep_units is None:
        deep_units = [128, 64, 32]

    # Cross Part 👉 显式建模特征之间的交叉组合项
    for _ in range(cross_layers):
        xl_w = Dense(input_dim)(xl)
        cross = Multiply()([x0, xl_w])
        xl = Add()([cross, xl])

    # Deep Part 👉 捕捉高阶特征组合的非线性关系 (多层前馈神经网络 MLP)
    deep = inputs
    for units in deep_units:
        deep = Dense(units, activation='relu')(deep)
        # 根据use_batch_norm参数决定是否使用批量归一化
        if use_batch_norm:
            from tensorflow.keras.layers import BatchNormalization
            deep = BatchNormalization()(deep)
        # 根据dropout_rate参数决定是否应用dropout
        if dropout_rate > 0:
            from tensorflow.keras.layers import Dropout
            deep = Dropout(dropout_rate)(deep)

    # Final
    concat = Concatenate()([xl, deep])
    output = Dense(1, activation='sigmoid')(concat)

    # 构建模型
    if learning_rate is None:  # 如果没有提供学习率，则使用默认的动态学习率
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=10000,
            decay_rate=0.1,
            staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)
    else:  # 如果提供了学习率，则使用指定的学习率
        optimizer = Adam(learning_rate=learning_rate)

    # 定义模型并编译
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


# Optuna试验的目标函数, 用于优化DCN模型的超参数
def objective_dcn(trial, validation_mode='kfold', n_splits=5):
    """
    Optuna试验的目标函数，用于优化DCN模型的超参数。

    参数:
    - trial: Optuna的试验对象，用于选择最佳的超参数组合。
    - validation_mode: 选择 'split' 或 'kfold'，用于选择验证方式。

    返回:
    - 平均F1分数，用于评估模型性能。
    """

    # 超参数空间定义
    cross_layers = trial.suggest_int('cross_layers', 2, 6)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    deep_units = trial.suggest_categorical('deep_units', [
        [128, 64],
        [128, 64, 32],
        [128, 64, 32, 16],
        [64, 32],
        [64, 32, 16],
        [64, 32, 16, 8],
        [32, 16],
        [32, 16, 8],
    ])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

    # 模型训练与验证
    if validation_mode == 'split':
        # 固定划分验证集
        x_subtrain, x_val, y_subtrain, y_val = train_test_split(
            x_train, y_train, test_size=0.1, shuffle=False
        )

        model = build_dcn(
            input_dim=x_train.shape[1],
            cross_layers=cross_layers,
            deep_units=deep_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            use_batch_norm=use_batch_norm
        )

        model.fit(
            x_subtrain, y_subtrain,
            validation_data=(x_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[early_stop]
        )

        y_val_pred = (model.predict(x_val) > 0.5).astype(int)
        return f1_score(y_val, y_val_pred)

    elif validation_mode == 'kfold':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = []

        for train_idx, val_idx in kf.split(x_train):
            x_subtrain, x_val = x_train[train_idx], x_train[val_idx]
            y_subtrain, y_val = y_train[train_idx], y_train[val_idx]

            model = build_dcn(
                input_dim=x_train.shape[1],
                cross_layers=cross_layers,
                deep_units=deep_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                use_batch_norm=use_batch_norm
            )

            model.fit(
                x_subtrain, y_subtrain,
                validation_data=(x_val, y_val),
                epochs=30,
                batch_size=32,
                verbose=0,
                callbacks=[early_stop]
            )

            y_val_pred = (model.predict(x_val) > 0.5).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    else:
        raise ValueError("validation_mode must be either 'split' or 'kfold'")


# 训练和评估函数
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
    """
    训练并评估一个机器学习模型。

    参数:
    - model: 要训练和评估的模型。
    - x_train: 训练数据特征。
    - y_train: 训练数据标签。
    - x_test: 测试数据特征。
    - y_test: 测试数据标签。
    - epochs: 训练的轮数，默认为10。
    - batch_size: 每个训练批次的大小，默认为32。

    此函数没有返回值，但会在控制台输出模型的评估报告。
    """
    # 训练模型
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_split=0.1)

    # 使用模型进行预测并转换为二进制分类
    pred = (model.predict(x_test) > 0.5).astype(int)

    # 输出分类报告
    print(classification_report(y_test, pred))


if __name__ == '__main__':
    from ReturnClassification.train_and_test import main_data_prepare

    # 数据读取，包括训练集和测试集，以及 原始指标数据
    x_train, y_train, x_test, y_test, metrics_data = main_data_prepare(
        the_fund_code='510050.SH',  # 指定基金代码，此处为 '510050.SH'
        n_days=10,
        folder_path='../Data',  # 基金价格数据的文件夹路径，默认为 '../Data'
        metrics_folder='../Data/Metrics',  # 基金指标数据的文件夹路径，默认为 '../Data/Metrics'
        train_start=None,  # 训练集开始日期，如果为 None，则从数据的最早日期开始
        train_end='2024-11-30',  # 训练集结束日期，指定为 '2024-11-30'
        test_start='2024-12-01',  # 测试集开始日期，指定为 '2024-12-01'
        test_end='2025-04-30',  # 测试集结束日期，指定为 '2025-04-30'
        nan_method='drop',  # 处理缺失值的方法，默认为 'drop'（删除缺失值），可选 'median' 或 'mean'
        standardize_method='zscore',  # 指标标准化的方法,可选: 'minmax', 'zscore', 'both', 'none'。
        basic_data_as_metric=True,  # 是否将基本数据（如开盘价、收盘价、交易量等）作为特征数据，默认为 True
        return_threshold=0.0,  # 标签生成方法，未来收益率大于 0.01 的样本标记为 1，否则为 0
        dim_reduction=True,  # 是否进行特征降维，默认为 False
        n_components=None,  # PCA降维的目标维度，默认为 100
    )

    # 数据预处理
    # 数据预处理（注意统一scaler）
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    y_train_values = y_train.values
    y_test_values = y_test.values

    input_dim = x_train.shape[1]

    ''' 1) DeepFM 模型 '''
    print(" --- " * 20)
    print("Training DeepFM...")
    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_deep_fm(trial, validation_mode='kfold'),
        n_trials=15
    )
    print("Best trial:", study.best_trial.params)  # 输出最优超参数
    best_trial = study.best_trial.params
    # best_trial = {'embed_dim': 32, 'dropout_rate': 0.4152850750188673, 'learning_rate': 0.001492202547205302, 'activation': 'relu', 'use_batch_norm': False, 'dnn_units': [64, 32, 16, 8]}

    # 使用最优超参数训练最终模型
    deep_fm_model = build_deep_fm(
        input_dim,
        embed_dim=best_trial['embed_dim'],
        dnn_units=best_trial['dnn_units'],
        dropout_rate=best_trial['dropout_rate'],
        activation=best_trial['activation'],
        use_batch_norm=best_trial['use_batch_norm'],
        learning_rate=best_trial['learning_rate'],
    )

    # 训练并评估DCN模型
    train_and_evaluate(
        deep_fm_model,  # 要训练和评估的DCN模型，由 `build_dcn` 函数构建，使用了 Optuna 优化后的超参数。
        x_train_scaled,  # 训练数据特征，经过预处理（如归一化）后的特征矩阵，确保输入数据具有相同的尺度。
        y_train_values,  # 训练数据标签，对应于训练特征的真实值，用于监督学习过程。
        x_test_scaled,  # 测试数据特征，经过预处理后的特征矩阵，用于评估模型在未见数据上的表现。
        y_test_values,  # 测试数据标签，对应于测试特征的真实值，用于计算模型的预测性能指标。
        epochs=10,  # 指定模型训练的轮数（epoch），即模型在整个训练数据集上完整训练的次数。越大越容易过拟合, 越小越容易欠拟合。
        batch_size=32  # 指定每个批次（batch）的大小，即每次更新模型参数时使用的样本数量。越大越容易过拟合, 越小越容易欠拟合。
    )

    # ''' 2) Wide & Deep 模型 '''
    print(" --- " * 20)
    print("\nTraining Wide & Deep...")
    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_wide_deep,
                   n_trials=15  # 设置试验次数
                   )
    print("Best trial:", study.best_trial.params)  # 输出最优超参数
    best_trial = study.best_trial.params
    # best_trial = {'dropout_rate': 0.31110760370578694, 'use_batch_norm': True, 'wide_activation': 'linear', 'activation': 'relu', 'learning_rate': 0.0020187999563270574, 'deep_units': [64, 32]}

    # 使用最优超参数训练最终模型
    wd_model = build_wide_deep(
        input_dim=input_dim,  # 输入特征的维度，即模型输入的大小。
        deep_units=best_trial['deep_units'],
        dropout_rate=best_trial['dropout_rate'],
        use_batch_norm=best_trial['use_batch_norm'],
        wide_activation=best_trial['wide_activation'],
        activation=best_trial['activation'],
        learning_rate=best_trial['learning_rate'],
    )
    # 训练并评估Wide&Deep模型
    train_and_evaluate(wd_model,  # 要训练和评估的Wide&Deep模型，由 `build_wide_deep` 函数构建，使用了 Optuna 优化后的超参数。
                       x_train_scaled,  # 训练数据特征，经过预处理（如归一化）后的特征矩阵，确保输入数据具有相同的尺度。
                       y_train_values,  # 训练数据标签，对应于训练特征的真实值，用于监督学习过程。
                       x_test_scaled,  # 测试数据特征，经过预处理后的特征矩阵，用于评估模型在未见数据上的表现。
                       y_test_values,  # 测试数据标签，对应于测试特征的真实值，用于计算模型的预测性能指标。
                       epochs=10,  # 指定模型训练的轮数（epoch），即模型在整个训练数据集上完整训练的次数。越大越容易过拟合, 越小越容易欠拟合。
                       batch_size=32  # 指定每个批次（batch）的大小，即每次更新模型参数时使用的样本数量。越大越容易过拟合, 越小越容易欠拟合。
                       )

    ''' 3) NCD 模型 '''
    print(" --- " * 20)
    print("\nTraining DCN...")
    # 使用Optuna进行超参数优化
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_dcn(trial, validation_mode='kfold'),
        n_trials=15
    )
    print("Best trial:", study.best_trial.params)  # 输出最优超参数
    best_trial = study.best_trial.params
    # best_trial = {'cross_layers': 2, 'dropout_rate': 0.2043031648581299, 'learning_rate': 0.0009367655839776305, 'use_batch_norm': False, 'deep_units': [128, 64]}

    # 使用最优超参数训练最终模型
    dcn_model = build_dcn(
        input_dim=input_dim,  # 输入特征的维度，即模型输入的大小。
        cross_layers=best_trial['cross_layers'],  # 交叉层的数量，控制模型中显式建模特征交叉组合的层数。
        deep_units=best_trial['deep_units'],  # 深度部分每层的单元数列表，定义深度网络的结构。
        dropout_rate=best_trial['dropout_rate'],  # Dropout比率，用于防止过拟合，随机丢弃神经元的比例。
        learning_rate=best_trial['learning_rate'],  # 学习率，控制优化器更新模型参数的速度。
        use_batch_norm=best_trial['use_batch_norm'],  # 是否使用批量归一化（Batch Normalization），用于加速训练和提高模型稳定性。
    )

    # 训练并评估DCN模型
    train_and_evaluate(
        dcn_model,  # 要训练和评估的DCN模型，由 `build_dcn` 函数构建，使用了 Optuna 优化后的超参数。
        x_train_scaled,  # 训练数据特征，经过预处理（如归一化）后的特征矩阵，确保输入数据具有相同的尺度。
        y_train_values,  # 训练数据标签，对应于训练特征的真实值，用于监督学习过程。
        x_test_scaled,  # 测试数据特征，经过预处理后的特征矩阵，用于评估模型在未见数据上的表现。
        y_test_values,  # 测试数据标签，对应于测试特征的真实值，用于计算模型的预测性能指标。
        epochs=10,  # 指定模型训练的轮数（epoch），即模型在整个训练数据集上完整训练的次数。越大越容易过拟合, 越小越容易欠拟合。
        batch_size=32  # 指定每个批次（batch）的大小，即每次更新模型参数时使用的样本数量。越大越容易过拟合, 越小越容易欠拟合。
    )
