# -*- encoding: utf-8 -*-
"""
@File: score_estimate.py
@Modify Time: 2025/8/28 18:52       
@Author: Kevin-Chen
@Descriptions: 
"""
import traceback

import numpy as np
import cvxpy as cvx
from datetime import datetime
from calculate.base_functions import cal_disperse_a_b_c


def score_estimate(w, Mdl):
    """
    一个健壮的、带回退机制的优化问题求解函数。

    本函数接收一个已构建好的 cvxpy 模型 (Mdl)，并尝试使用一个预定义的求解器列表 (solvers)
    来对其进行求解。它会按顺序尝试列表中的每一个求解器，直到有一个成功返回结果（无论成功或失败）
    或所有求解器都因异常而失败为止。

    这种设计增强了求解的稳定性。例如，如果首选的 SCIP 求解器因特定问题或环境因素失败，
    代码会自动切换到备选的 CBC 求解器再次尝试，而不是直接崩溃。

    :param w: cvxpy.Variable, 优化问题中的决策变量（通常是权重向量），用于在求解成功后提取其数值。
    :param Mdl: cvxpy.Problem, 一个已完整定义了目标函数和约束的 cvxpy 问题实例。
    :return: tuple, 一个包含三个元素的元组:
             - status (str): 求解器返回的最终状态字符串 (例如, 'optimal', 'infeasible')。
             - w.value (np.ndarray or None): 优化成功时决策变量 `w` 的数值解，否则为 None。
             - info (list): 包含求解过程日志的列表。
    """
    # 定义可用的求解器列表和对应的初始状态字典
    solvers = [cvx.SCIP, cvx.CBC]  # cvx.CPLEX, cvx.GLPK_MI, cvx.ECOS_BB, cvx.SCIP, cvx.CBC]
    solver_status = dict(zip(solvers, [False] * len(solvers)))
    info = []
    status = '..'

    # 依次尝试每个求解器进行模型求解
    for i, solver in enumerate(solvers):
        print(f"\t尝试使用{solver}进行求解, 编号: {i + 1}")
        try:
            Mdl.solve(solver=solver)
            if Mdl.status == "optimal":
                info += [f"{i + 1}号求解器" + '执行成功：' + Mdl.status]
                print(f"\t{i + 1}号求解器{solver}执行成功")
            else:
                info += [f"{i + 1}号求解器" + '执行失败：' + Mdl.status]
                print(f"\t{i + 1}号求解器{solver}执行失败，状态: {Mdl.status}")

            status = Mdl.status
            break
        except Exception as e:
            info += [f"{i + 1}号求解器" + '执行抛异常']
            print(f"{i + 1}号求解器, 执行抛异常: {e}")
            solver_status[solver] = False
            continue

    return status, w.value, info


def estimate_all(product_info, target_asset_alloc, asset_alloc_bias, nums_dict,
                 covMatrix, disperse_exclude, isSellProduct=False):
    """
    在给定的约束下，对各项核心投资指标进行极值估算。

    本函数是主优化流程开始前的一个关键预计算步骤。它通过求解一系列相对简单的
    线性或二次规划问题，来探测在当前产品池和约束条件下，各项核心指标（收益、
    流动性、分散度）的理论最大和最小值范围。

    这个估算出的范围（例如，收益率范围为[2%, 15%]）至关重要，因为它将作为后续
    非线性评分函数进行线性近似的校准基准。如果不知道这个范围，评分函数的线性近似
    可能会严重失真，从而影响最终优化结果的质量。

    主要流程：
    1.  构建基础约束集：包含资产权重和为1、权重非负、大类资产比例等核心约束。
    2.  逐一求解极值：
        -   最大化/最小化 “预期收益率”，并记录下极值点对应的 “收益率” 和 “波动率”。
        -   最大化/最小化 “现金类占比”。
        -   最大化/最小化 “分散度” (使用二次函数近似)。
    3.  失败回退：如果在严格约束下求解失败，会放宽大类资产约束，只保留最基本的权重约束后重试。
        如果仍然失败，则使用一个等权重组合作为最后的估算依据。
    4.  范围修正：对计算出的每个指标范围，如果最大值与最小值过于接近，会人为地略微扩大范围，
        以避免后续计算中出现除以零等数值稳定性问题。

    :param product_info: pd.DataFrame, 包含所有可用产品信息的完整数据表。
    :param target_asset_alloc: dict, 目标大类资产配置比例。
    :param asset_alloc_bias: dict, 各大类资产允许的最大偏离度。
    :param nums_dict: dict, 每个大类资产下可用的产品总数。
    :param covMatrix: pd.DataFrame, 资产收益率的协方差矩阵。
    :param disperse_exclude: set, 在计算分散度时需要排除的产品三级分类码集合。
    :param isSellProduct: bool, 是否为“管销”场景。若是，则不施加大类资产约束。
    :return: tuple, 一个包含两元素的元组:
             - list: 总是返回一个空列表，可能用于预留的错误信息通道。
             - resDict (dict): 包含所有核心指标估算出的最大/最小值的字典。
               例如: {'最大收益率': 0.15, '最小收益率': 0.02, ...}
    """

    ''' 创建基本约束 '''
    # 获取产品数量
    nPrd = product_info.shape[0]
    # 创建优化变量：投资组合权重
    w = cvx.Variable(shape=(nPrd,), name="weight")
    # 构建约束条件列表
    Constraints = []
    # 添加权重和约束：总权重不超过100%
    Constraints.append(cvx.sum(w) <= 1.0)
    # 添加权重和约束：总权重至少为99.9%，确保权重和接近100%
    Constraints.append(cvx.sum(w) >= 0.999)
    # 添加非负约束：各产品权重不能为负
    Constraints.append(w >= 0)
    # 添加上界约束：各产品权重不能超过100%
    Constraints.append(w <= 1)

    # AssetInfo 是一个独热编码形式的 DataFrame: 列名为 "大类资产-XXX"，值为 0/1; 用于表示每个产品在不同大类资产中的分类归属
    AssetInfo = product_info.loc[:, product_info.columns.str.contains("大类资产-")]

    ''' 添加大类资产权重约束 '''
    # 只有在“非管销”场景下，才会施加大类资产配置的约束。在“管销”场景中，模型的目标可能是清仓某个产品，而不是要满足特定的资产配置比例。
    if (target_asset_alloc is not None) and (AssetInfo.shape[1] > 0) and (not isSellProduct):
        for iAsset, iTarget in target_asset_alloc.items():  # 循环遍历每一个需要配置的大类资产及其目标权重
            if "大类资产-" + iAsset not in AssetInfo:  # 检查产品信息表中是否包含该大类资产的暴露数据
                raise Exception(f"产品信息表中不存在大类资产-{iAsset}的暴露数据！")
            if not np.any(AssetInfo["大类资产-" + iAsset].values > 0):  # 检查是否有任何产品属于该大类资产
                continue
            # 使用矩阵点积计算投资组合在当前大类资产上的总暴露
            iPortfolioExpose = AssetInfo["大类资产-" + iAsset].values @ w

            # 如果某个大类资产，其可用产品数为0，那么就不可能满足大于0的下限要求。此时将该类资产的权重约束放宽为 [0, 上限]，避免无解。
            if nums_dict[iAsset] == 0:
                Constraints += [iPortfolioExpose >= 0,
                                iPortfolioExpose <= iTarget * (1 + asset_alloc_bias[iAsset])]
            # 标准情况下代码会添加两条约束，资产的总权重必须落在 [目标 * (1 - 偏离度), 目标 * (1 + 偏离度)] 这个范围之内
            else:
                Constraints += [iPortfolioExpose >= iTarget * (1 - asset_alloc_bias[iAsset]),
                                iPortfolioExpose <= iTarget * (1 + asset_alloc_bias[iAsset])]
    print(f"大类资产权重约束条件上下限配置完毕")
    # 计算协方差矩阵
    cov = covMatrix.loc[product_info["基准指数代码"].values, product_info["基准指数代码"].values].values
    # 初始化一个空字典, 用于存放后续所有估算任务（如最大收益率、最小收益率、最大现金比例等）的计算结果
    resDict = {}

    ''' 1) 估算 最大收益率&最大波动率 '''
    print("进行最大收益率估算...")
    # 定义优化问题
    Mdl = cvx.Problem(
        cvx.Maximize(product_info['基准指数收益率'].values @ w),  # 目标函数：最大化预期收益率
        constraints=Constraints,  # 约束条件
    )

    # 启动一个最多尝试10次的循环。可能是为了应对某些求解器偶然性的失败
    s = datetime.now()
    for i in range(10):
        # 调用score_estimate健壮求解函数，尝试求解当前的优化问题
        status, weight, info = score_estimate(w, Mdl)
        # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
        if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
            break

    # 如果优化失败, 进行第一次回退再处理
    if 'optimal' not in status:
        print("最大收益率求解失败，进行第一次回退处理, 放宽约束条件 ...")
        # 重建约束条件, 只保留最基本的权重约束
        Constraints = [cvx.sum(w) <= 1.0, cvx.sum(w) >= 0.99, w >= 0, w <= 1]
        # 重新定义优化问题
        Mdl = cvx.Problem(
            cvx.Maximize(product_info['基准指数收益率'].values @ w),
            constraints=Constraints,
        )
        # 启动一个最多尝试5次的循环
        s = datetime.now()
        for i in range(5):
            status, weight, info = score_estimate(w, Mdl)
            # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
            if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
                break

    # 如果优化再次失败, 进行第二次回退再处理
    if 'optimal' not in status:
        print("最大收益率求解仍然失败，进行第二次回退处理, 使用等权重进行求解 ...")
        # 使用等权重求解作为最后的回退方案
        weight = np.ones(product_info['基准指数收益率'].values.shape) / product_info['基准指数收益率'].values.shape[0]

    resDict['最大收益率'] = product_info['基准指数收益率'].values @ weight
    resDict['最大波动率'] = (np.dot(cov, weight) @ weight) ** 0.5  # 假设最大收益率对应的波动率最大
    print(f"估算完毕, 最大收益率: {resDict['最大收益率']:.4f}, 最大波动率: {resDict['最大波动率']:.4f}")

    ''' 2) 估算 最小收益率&最小波动率 '''
    print("开始计算最小收益率")
    # 定义优化问题
    Mdl = cvx.Problem(
        cvx.Minimize(product_info['基准指数收益率'].values @ w),  # 目标函数：最小化预期收益率
        constraints=Constraints,
    )
    # 启动一个最多尝试10次的循环
    s = datetime.now()
    for i in range(10):
        # 调用score_estimate健壮求解函数，尝试求解当前的优化问题
        status, weight, info = score_estimate(w, Mdl)
        # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
        if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
            break

    # 如果优化失败, 进行第一次回退再处理
    if 'optimal' not in status:
        print("最小收益率求解失败，进行第一次回退处理, 放宽约束条件 ...")
        # 重建约束条件, 只保留最基本的权重约束
        Constraints = [cvx.sum(w) <= 1.0, cvx.sum(w) >= 0.99, w >= 0, w <= 1]
        # 重新定义优化问题
        Mdl = cvx.Problem(
            cvx.Minimize(product_info['基准指数收益率'].values @ w),
            constraints=Constraints,
        )
        # 启动一个最多尝试5次的循环
        s = datetime.now()
        for i in range(5):
            status, weight, info = score_estimate(w, Mdl)
            # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
            if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
                break

    # 如果优化再次失败, 进行第二次回退再处理
    if 'optimal' not in status:
        print("最小收益率求解仍然失败，进行第二次回退处理, 使用等权重进行求解 ...")
        # 使用等权重求解作为最后的回退方案
        weight = np.ones(product_info['基准指数收益率'].values.shape) / product_info['基准指数收益率'].values.shape[0]

    resDict['最小收益率'] = product_info['基准指数收益率'].values @ weight
    resDict['最小波动率'] = (np.dot(cov, weight) @ weight) ** 0.5
    print(f"估算完毕, 最小收益率: {resDict['最小收益率']:.4f}, 最小波动率: {resDict['最小波动率']:.4f}")

    ''' 3) 估算 最小流动性 '''
    print("开始计算最小流动性")
    # 定义优化问题
    Mdl = cvx.Problem(cvx.Minimize(product_info["大类资产-现金类"].values @ w), constraints=Constraints)
    # 启动一个最多尝试10次的循环
    s = datetime.now()
    for i in range(10):
        # 调用score_estimate健壮求解函数，尝试求解当前的优化问题
        status, weight, info = score_estimate(w, Mdl)
        # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
        if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
            break

    # 如果优化失败, 进行第一次回退再处理
    if 'optimal' not in status:
        print("最小流动性求解失败，进行第一次回退处理, 放宽约束条件 ...")
        # 重建约束条件, 只保留最基本的权重约束
        Constraints = [cvx.sum(w) <= 1.0, cvx.sum(w) >= 0.99, w >= 0, w <= 1]
        # 重新定义优化问题
        Mdl = cvx.Problem(cvx.Minimize(product_info["大类资产-现金类"].values @ w), constraints=Constraints)
        # 启动一个最多尝试5次的循环
        s = datetime.now()
        for i in range(5):
            # 调用score_estimate健壮求解函数，尝试求解当前的优化问题
            status, weight, info = score_estimate(w, Mdl)
            # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
            if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
                break

    # 如果优化再次失败, 进行第二次回退再处理
    if 'optimal' not in status:
        print("最小流动性求解仍然失败，进行第二次回退处理, 使用等权重进行求解 ...")
        weight = np.ones(product_info["基准指数收益率"].values.shape) / product_info["基准指数收益率"].values.shape[0]

    resDict["最小现金类比例"] = product_info["大类资产-现金类"].values @ weight
    print(f"估算完毕, 最小现金类比例: {resDict['最小现金类比例']:.4f}")

    ''' 估算 最大流动性 '''
    print("开始计算最大流动性")
    # 定义优化问题
    Mdl = cvx.Problem(cvx.Maximize(product_info["大类资产-现金类"].values @ w), constraints=Constraints)
    # 启动一个最多尝试10次的循环
    s = datetime.now()
    for i in range(10):
        # 调用score_estimate健壮求解函数，尝试求解当前的优化问题
        status, weight, info = score_estimate(w, Mdl)
        # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
        if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
            break

    # 如果优化失败, 进行第一次回退再处理
    if 'optimal' not in status:
        print("最大流动性求解失败，进行第一次回退处理, 放宽约束条件 ...")
        # 重建约束条件, 只保留最基本的权重约束
        Constraints = [cvx.sum(w) <= 1.0, cvx.sum(w) >= 0.99, w >= 0, w <= 1]
        # 重新定义优化问题
        Mdl = cvx.Problem(cvx.Maximize(product_info["大类资产-现金类"].values @ w), constraints=Constraints)
        # 启动一个最多尝试5次的循环
        s = datetime.now()
        for i in range(5):
            status, weight, info = score_estimate(w, Mdl)
            # 循环的退出条件: 要么求解器返回'optimal'状态, 要么已经尝试超过1秒钟
            if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
                break

    # 如果优化再次失败, 进行第二次回退再处理
    if 'optimal' not in status:
        print("最大流动性求解仍然失败，进行第二次回退处理, 使用等权重进行求解 ...")
        # 使用等权重求解作为最后的回退方案
        weight = np.ones(product_info["基准指数收益率"].values.shape) / product_info["基准指数收益率"].values.shape[0]

    resDict["最大现金类比例"] = product_info["大类资产-现金类"].values @ weight
    print(f"估算完毕, 最大现金类比例: {resDict['最大现金类比例']:.4f}")

    '''在投资组合理论中，分散度的标准数学度量是“信息熵”，公式为 -sum(w * log(w))，其中 w 是各项资产的权重。
    这是一个对数函数，属于非线性、非二次函数，对于 cvxpy 这类凸优化求解器来说直接处理会非常困难和缓慢。
    为了能够在估算阶段高效地求解，代码采用了一种近似方法。它使用一个更简单的二次函数 `a*sum(w²) + b*sum(w) + c*n` 来模拟真实的分散度。
    这里的数值（-1.5, -1.5, 0）很可能是通过离线分析和拟合得出的，对真实分散度函数给出一个近似的“魔法数字”。
    '''
    # 变量 a, b, c 的值被硬编码为 -1.5, -1.5, 0
    a, b, c = cal_disperse_a_b_c()
    disperse_flag = product_info['三级分类码值'].apply(
        lambda x: True if x not in disperse_exclude else False
    ).values

    if np.sum(disperse_flag) > 0:
        Mdl = cvx.Problem(
            cvx.Maximize(
                a * cvx.sum_squares(w[disperse_flag])
                + b * cvx.sum(w[disperse_flag])
                + w[disperse_flag].shape[0] * c
            ),
            constraints=Constraints,
        )
        s = datetime.now()
        for i in range(10):
            status, weight, info = score_estimate(w, Mdl)
            if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
                break

        if 'optimal' not in status:
            Constraints = [cvx.sum(w) <= 1.0, cvx.sum(w) >= 0.99, w >= 0, w <= 1]
            Mdl = cvx.Problem(
                cvx.Maximize(
                    a * cvx.sum_squares(w[disperse_flag])
                    + b * cvx.sum(w[disperse_flag])
                    + w[disperse_flag].shape[0] * c
                ),
                constraints=Constraints,
            )

            s = datetime.now()
            for i in range(5):
                status, weight, info = score_estimate(w, Mdl)
                if ('optimal' in status) or ((datetime.now() - s).total_seconds() >= 1):
                    break

        if 'optimal' not in status:
            weight = np.ones(
                product_info['基准指数收益率'].values.shape) / product_info['基准指数收益率'].values.shape[0]
        weight += 1e-20
        resDict['最大分散度'] = np.sum(-1 * weight * np.log(weight))
        array = np.array(list(target_asset_alloc.values())) + 1e-20
        resDict['最小分散度'] = np.sum(-1 * array * np.log(array))
    else:
        resDict['最大分散度'] = 0.0
        resDict['最小分散度'] = 0.0

    for key in ['收益率', '波动率', '现金类比例', '分散度']:
        if resDict['最大' + key] - resDict['最小' + key] <= 1e-2:
            resDict['最大' + key] = resDict['最小' + key] + 1e-2

    return [], resDict
