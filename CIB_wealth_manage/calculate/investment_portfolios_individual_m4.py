# -*- encoding: utf-8 -*-
"""
@File: investment_portfolios_individual_m4.py
@Modify Time: 2025/8/26 19:16       
@Author: Kevin-Chen
@Descriptions: 
"""
import os
import time
import random
import traceback

import numpy as np
import pandas as pd
import cvxpy as cvx
from datetime import datetime

from utils.limit import time_limit
from calculate.product_filter import setting_by_sjdy
from config_py import ASSET_CODE_DICT, DEBUG, FUN_CLASS
from calculate.opti_functions import ConstraintsCheck, do_one

product_class = pd.read_csv(r'/Users/chenjunming/Desktop/CIB_wealth_manage/calculate/product_class.csv').dropna(
    subset=['一级分类码值', '二级分类码值', '三级分类码值', '自定义分类码值'])

product_class['一级分类码值'] = product_class['一级分类码值'].astype(str).apply(
    lambda x: str(int(float(x))).zfill(2))
product_class['二级分类码值'] = product_class['二级分类码值'].astype(str).apply(
    lambda x: str(int(float(x))).zfill(5))
product_class['三级分类码值'] = product_class['三级分类码值'].astype(str).apply(
    lambda x: str(int(float(x))).zfill(8))
product_class['自定义分类码值'] = product_class['自定义分类码值'].astype(str).apply(
    lambda x: str(int(float(x))).zfill(3))


def clazzCfgMaxAbsOiff(srcClazzCfg, clazzCfg):
    rs = 0
    for clazzName in ["现金类", "固定类", "权益类", "混合类", "另类"]:
        d = srcClazzCfg.get(clazzName, 0) - clazzCfg.get(clazzName, 0)
        if d < 0:
            d = -d
        if d > rs:
            rs = d
    return rs


# 过滤筛选符合条件的产品
def filterProduct(product_info, args={}):
    """
    对产品信息进行过滤和特征计算处理

    参数:
        product_info (pandas.DataFrame): 包含产品信息的数据框
        args (dict): 配置参数字典，可包含无风险利率等参数

    返回:
        pandas.DataFrame: 处理后的产品信息数据框，包含新增的业绩基准、波动率、夏普比率等特征列
    """

    # 计算业绩基准：上下限的平均值
    product_info["业绩基准"] = (product_info["业绩比较基准上限"].fillna(0) + product_info["业绩比较基准下限"].fillna(
        0)) / 2

    # 计算波动率：基于业绩基准上下限的差值与基准的比值
    product_info["波动率"] = (product_info["业绩比较基准上限"].fillna(0) - product_info["业绩比较基准下限"].fillna(
        0)) / (product_info["业绩基准"] + 1e-6)
    product_info["波动率"] = product_info["波动率"].apply(lambda x: abs(x) + 1e-6)

    # 计算夏普比率：基于业绩基准和波动率计算，并进行数值裁剪
    product_info["夏普比率"] = (product_info["业绩基准"] * 100 - args.get(
        "无风险利率", 0)) / product_info.get("波动率", 1)
    product_info["夏普比率"] = product_info["夏普比率"].apply(lambda x: x if x <= 10 else 10)
    product_info["夏普比率"] = product_info["夏普比率"].apply(lambda x: x if x >= -10 else -10)

    # 填充缺失值：产品周期、起购金额填充为0
    product_info["产品周期"] = product_info["产品周期"].fillna(0)
    product_info["起购金额"] = product_info["起购金额"].fillna(0)

    # 填充大类资产相关列的缺失值为0
    Cols = product_info.columns[product_info.columns.str.contains("大类资产-")]
    product_info.loc[:, Cols] = product_info.loc[:, Cols].fillna(0)

    return product_info


def filterProductByRisk(product_info, args={}):
    """
    根据风险等级过滤产品信息

    参数:
        product_info (pandas.DataFrame): 包含产品信息的数据框，需包含风险等级、大类资产-混合类、自定义分类等列
        args (dict): 过滤参数字典，需包含risk键表示风险等级阈值

    返回:
        pandas.DataFrame: 过滤后的产品信息数据框，只保留不符合过滤条件的行
    """
    columns = product_info.columns
    # 将风险等级字符串转换为数值，去除第一个字符后转换为整数
    product_info["风险等级-数值"] = product_info["风险等级"].apply(lambda x: int(x[1:]))
    # 构建过滤条件：大类资产为混合类、自定义分类为银行理财、且风险等级小于指定阈值
    filter_condition = (product_info["大类资产-混合类"] == 1) & (product_info["自定义分类"] == "银行理财") & (
            (args["risk"] - product_info["风险等级-数值"]) > 0)
    # 返回不满足过滤条件的产品信息
    return product_info[False == filter_condition][columns]


def calClazzCfg(prods, clazzMap):
    clazzCfg = {"clazz01": 0, "clazz02": 0, "clazz03": 0, "clazz04": 0, "clazz05": 0}
    for idx, item in prods.iterrows():
        clazz = "clazz" + clazzMap[idx]
        clazzCfg[clazz] = clazzCfg.get(clazz, 0.0) + item["持仓权重"]
    return {
        "现金类": clazzCfg["clazz01"],
        "固定类": clazzCfg["clazz02"],
        "权益类": clazzCfg["clazz04"],
        "混合类": clazzCfg["clazz03"],
        "另类": clazzCfg["clazz05"]
    }


def portfolio_duplicates(result_lst, indicators_lst, cur_ret):
    # 组合去重
    if cur_ret['data'] is None:
        return result_lst, indicators_lst
    b_is_individual = True
    for df in result_lst:
        df = pd.merge(df[['持仓权重']], cur_ret['data'][['持仓权重']],
                      left_index=True, right_index=True, how='outer')
        df = df.fillna(0.0)
        if np.max(np.abs(df.values[:, 0] - df.values[:, 1])) <= 0.02:
            b_is_individual = False
            break
    if b_is_individual:
        result_lst.append(cur_ret['data'])
        indicators_lst.append(cur_ret['indicators'])
    return result_lst, indicators_lst


def do_patch(clazzMap, product_info, amt, obj_alloc, args, target_asset_alloc, asset_alloc_rate, rate_limit,
             asset_alloc_bias, longest_idle_time, other_args, mdl_args, method, sampling_proportion,
             weight_sum_range=[0.99], random_nums=10, opti_time_limit=15):
    """
    执行单次优化尝试的核心封装函数，是连接策略层 (`lst_portfolio_generation`) 和执行层 (`do_one`) 的桥梁。

    该函数主要负责：
    1.  **数据预处理**: 调用 `filterProduct` 和 `filterProductByRisk` 对传入的 `product_info` 进行过滤，
        剔除不符合基本业务规则或风险要求的产品。
    2.  **首次优化尝试**: 将过滤后的产品和所有参数传递给 `do_one` 函数，进行第一次完整的优化求解。
    3.  **失败诊断**: 如果首次尝试失败（返回 "fail"），则调用 `ConstraintsCheck` 类来自动、系统地放宽约束，
    以诊断导致优化无解的具体原因。
    4.  **多样性生成 (当前禁用)**: 函数内部包含一套复杂的、基于产品分层（重点推荐、银行理财、普通产品）和
        随机抽样的循环逻辑。该逻辑的设计目的是在首次优化失败或需要更多样化结果时，通过多次运行不同的产品子集来寻找“次优解”或“特定场景解”。
        **注意**: 当前该循环被 `for i in range(0):` 禁用，因此这部分逻辑不会被执行。
    5.  **结果汇总**: 收集优化结果（成功或失败），并将其与诊断信息一同返回给上层调用者。

    :param clazzMap: dict, 产品ID到大类资产编码的映射字典。
    :param product_info: pd.DataFrame, 包含产品池和用户持仓的完整产品信息表。
    :param amt: float, 总投资金额。
    :param obj_alloc: dict, 优化目标权重字典。
    :param args: dict, 包含无风险利率等基础参数的字典。
    :param target_asset_alloc: dict, 目标大类资产配置比例。
    :param asset_alloc_rate: dict, 各大类资产的预期收益率范围。
    :param rate_limit: list, 组合整体收益率的范围限制 [min, max]。
    :param asset_alloc_bias: float, 各大类资产允许的最大偏离度。
    :param longest_idle_time: int, 组合允许的最大久期（天）。
    :param other_args: dict, 包含所有其他上下文和约束参数的“参数工具包”。
    :param mdl_args: dict, 求解器（如 SCIP, CPLEX）相关配置。
    :param method: int, 在目标函数中计算某些因子时使用的排名方法。
    :param sampling_proportion: float, 产品抽样比例（当前逻辑中未被使用）。
    :param weight_sum_range: list, 权重加总范围的列表，在 `do_one` 中使用。
    :param random_nums: int, (当前禁用) 在多样性生成循环中，随机抽样的产品数量。
    :param opti_time_limit: int, 本次 `do_patch` 执行的超时时间（秒）。
    :return: dict, 一个包含单次优化结果的字典，结构与 `lst_portfolio_generation` 的返回值类似。
    """
    start_time = datetime.now()

    ''' 对传入的原始产品数据 product_info 进行过滤和特征计算处理 '''
    # 调用 filterProduct 函数计算衍生指标: 函数内部会基于产品已有的字段计算出三个新的衍生指标: 业绩基准的平均值, 基于业绩基准上下限估算的一个简易波动率, 基于计算出的业绩基准和波动率估算出的夏普比率
    SelectedProductInfo = filterProduct(product_info, args=args)
    # 针对符合资产配置进行过滤 和 目标大类资产配置调整, 3个过滤条件：大类资产为混合类、自定义分类为银行理财、风险等级小于指定阈值
    SelectedProductInfo = filterProductByRisk(SelectedProductInfo, other_args)
    # 设置产品ID为索引
    SelectedProductInfo = SelectedProductInfo.set_index("产品ID")

    ''' 调用 do_one 执行器，进行首次优化尝试 '''
    print(f"调用 do_patch 执行器，进行首次优化尝试")
    result_lst = []
    indicators_lst = []
    cur_ret = do_one(
        clazzMap,  # 产品ID到大类资产编码的映射字典，用于确定每个产品的资产类别
        SelectedProductInfo,  # 经过筛选和处理后的产品信息DataFrame，包含产品特征数据
        amt,  # 投资总金额，用于计算各产品的持仓金额
        obj_alloc,  # 优化目标权重字典，定义各优化目标的重要性权重
        target_asset_alloc,  # 目标大类资产配置比例，定义各类资产的目标配置权重
        asset_alloc_rate,  # 各大类资产的预期收益率范围，用于设定收益率约束
        rate_limit,  # 组合整体收益率的范围限制[min, max]，控制组合整体收益水平
        asset_alloc_bias,  # 各大类资产允许的最大偏离度，控制资产配置偏离目标的程度
        longest_idle_time,  # 组合允许的最大久期(天)，限制组合的流动性风险
        other_args,  # 包含所有其他参数的字典，如约束条件、用户持仓等
        mdl_args,  # 求解器相关配置参数，如求解器类型、求解参数等
        method,  # 目标函数中使用的排名方法(1:pct=True, 2:dense, 3:log normalization)
        weight_sum_range  # 权重加总范围，控制组合权重和的约束范围
    )

    if cur_ret['status'] == 'fail' and cur_ret.get('has_check', False):
        return cur_ret
    elif cur_ret['status'] != 'success':
        # 进行约束检查，检查完毕，返回
        print('优化无解，进行参数校查')
        cc = ConstraintsCheck(clazzMap, SelectedProductInfo, amt, obj_alloc, target_asset_alloc, asset_alloc_rate,
                              rate_limit, asset_alloc_bias, longest_idle_time, other_args, mdl_args, method,
                              weight_sum_range)
        cur_ret['message'] += cc.check_all()
        cur_ret['has_check'] = True

    result_lst, indicators_lst = portfolio_duplicates(result_lst, indicators_lst, cur_ret)

    if len(result_lst) >= other_args['portfolio_num']:
        cur_ret['data'] = result_lst
        cur_ret['indicators'] = indicators_lst
        cur_ret['status'] = 'success'
        cur_ret['message'] = []
        return cur_ret
    if (datetime.now() - start_time).total_seconds() > opti_time_limit:
        cur_ret['data'] = result_lst
        cur_ret['indicators'] = indicators_lst
        cur_ret['status'] = 'success'
        cur_ret['info'] = ['超时结束']
        return cur_ret

    np.random.seed(1782238)
    try_set_lst = []
    privateVisible = other_args['privateVisible']
    eps = 0.001
    if privateVisible:
        v_vip_conditions = ((SelectedProductInfo['重点推荐'] == True) |
                            (SelectedProductInfo['是否面向合格投资者'] == privateVisible))
    else:
        v_vip_conditions = (SelectedProductInfo['重点推荐'] == True)
    v_vip_product_df = SelectedProductInfo[v_vip_conditions]

    if privateVisible:
        vip_conditions = ((SelectedProductInfo['自定义分类'] == '银行理财') |
                          (SelectedProductInfo['是否面向合格投资者'] == privateVisible))
    else:
        vip_conditions = SelectedProductInfo['自定义分类'] == '银行理财'

    vip_product_df = SelectedProductInfo[vip_conditions]
    vip_product_df = vip_product_df[False == vip_product_df.index.isin(list(v_vip_product_df.index))]

    SelectedProductInfo = SelectedProductInfo[(False == vip_conditions) & (False == v_vip_conditions)]
    message = cur_ret['message']

    for i in range(0):
        lst = []
        for k in target_asset_alloc:
            try:
                if v_vip_product_df[v_vip_product_df[f'大类资产-{k}'] == 1].shape[0] > 0:
                    alloc_pool_df = v_vip_product_df[v_vip_product_df[f'大类资产-{k}'] == 1].sample(n=1)
                else:
                    alloc_pool_df = pd.DataFrame(data=[], columns=v_vip_product_df.columns)
                lst += [alloc_pool_df]
            except:
                print(traceback.format_exc())

        for k in target_asset_alloc:
            try:
                if vip_product_df[vip_product_df[f'大类资产-{k}'] == 1].shape[0] > 0:
                    alloc_pool_df = vip_product_df[vip_product_df[f'大类资产-{k}'] == 1].sample(n=1)
                else:
                    alloc_pool_df = pd.DataFrame(data=[], columns=vip_product_df.columns)
                if alloc_pool_df.shape[0] > 0:
                    alloc_pool_df.index = alloc_pool_df.index.droplevel(0)
                lst += [alloc_pool_df]
            except:
                print(traceback.format_exc())

        for k in target_asset_alloc:
            try:
                has_df = pd.concat(lst).drop_duplicates()
                if len(has_df[has_df[f'大类资产-{k}'] == 1]) > 0 and random.random() > 0.5:
                    continue
                if SelectedProductInfo[SelectedProductInfo[f'大类资产-{k}'] == 1].shape[0] > 0:
                    alloc_pool_df = SelectedProductInfo[SelectedProductInfo[f'大类资产-{k}'] == 1].sample(n=1)
                else:
                    alloc_pool_df = pd.DataFrame(data=[], columns=SelectedProductInfo.columns)
                if alloc_pool_df.shape[0] > 0:
                    alloc_pool_df.index = alloc_pool_df.index.droplevel(0)
                lst += [alloc_pool_df]
            except:
                print(traceback.format_exc())

        sample_df = pd.concat(lst, sort=False)
        sample_df = sample_df.drop_duplicates()

        try_again = False
        for one in try_set_lst:
            if one == set(sample_df.index):
                try_again = True
                break
        if try_again:
            continue
        try_set_lst += [set(sample_df.index)]

        sample_df = sample_df.reset_index().rename(columns={'index': '产品ID'}).set_index('产品ID')

        cur_ret = do_one(clazzMap, sample_df, amt, obj_alloc, target_asset_alloc, asset_alloc_rate, rate_limit,
                         asset_alloc_bias, longest_idle_time, other_args, mdl_args, method, weight_sum_range)

        if (len(result_lst) >= other_args['portfolio_num']) or (
                (datetime.now() - start_time).total_seconds() > opti_time_limit):
            print('方案数量满足 或者 超时, 提前结束')
            break

        if 'success' != cur_ret['status']:
            continue

        result_lst, indicators_lst = portfolio_duplicates(result_lst, indicators_lst, cur_ret)

        message = ['success', '次优化解'] + message

        if len(result_lst) >= other_args['portfolio_num']:
            cur_ret['data'] = result_lst
            cur_ret['indicators'] = indicators_lst
            return cur_ret

    if len(result_lst) > 0:
        cur_ret['status'] = 'success'
        cur_ret['data'] = result_lst
        cur_ret['indicators'] = indicators_lst
        cur_ret['message'] = []
    else:
        cur_ret['status'] = 'fail'
        cur_ret['data'] = []
        cur_ret['indicators'] = []
        cur_ret['message'] = ['优化失败'] + list(dict.fromkeys(message))

    return cur_ret


def lst_portfolio_generation(clazzMap, product_info, amt, obj_alloc, args={},
                             target_asset_alloc=None, asset_alloc_rate=None,
                             rate_limit=None, asset_alloc_bias=0.05,
                             longest_idle_time=None, other_args={},
                             mdl_args={}, method=2,
                             sampling_proportion=0.3, opti_time_limit=15, **kwargs):
    """
    作为核心的组合生成器和调度器，负责调用底层优化函数来生成一个或多个投资组合方案。

    本函数是连接上层业务参数和底层优化执行的中间层。它管理着整个优化过程的迭代、
    结果的收集与去重，以及在无解时的初步诊断。

    主要逻辑流程：
    1.  **调用执行器**: 循环调用 `do_patch` 函数（`do_patch` 内部会调用 `do_one` 来实际执行 `cvxpy` 优化）。
        这个循环的设计允许多次尝试以寻找多样化的解，尽管在当前配置下通常只执行一次。
    2.  **结果去重**: 每当 `do_patch` 返回一个成功的解，会调用 `portfolio_duplicates` 函数，
        将新组合与已存储的结果进行比较。只有当持仓权重差异足够大（>2%）时，才被视为一个新方案加入结果列表。
    3.  **终止条件**: 在每次迭代后，检查是否已达到期望的组合数量 (`portfolio_num`) 或是否已超过总的
        时间限制 (`opti_time_limit`)。满足任一条件则提前终止并返回结果。
    4.  **失败诊断**: 如果所有尝试结束后仍未生成任何有效组合，则调用 `has_no_result_check` 进行
        初步诊断，检查是否存在某些大类下无产品可投的情况。

    :param clazzMap: dict, 产品ID到大类资产编码的映射字典。
    :param product_info: pd.DataFrame, 包含产品池和用户持仓的完整、已处理的产品信息表。
    :param amt: float, 总投资金额。
    :param obj_alloc: dict, 优化目标权重字典。
    :param args: dict, 包含无风险利率等基础参数的字典。
    :param target_asset_alloc: dict, 目标大类资产配置比例。
    :param asset_alloc_rate: dict, 各大类资产的预期收益率范围。
    :param rate_limit: list, 组合整体收益率的范围限制 [min, max]。
    :param asset_alloc_bias: float, 各大类资产允许的最大偏离度。
    :param longest_idle_time: int, 组合允许的最大久期（天）。
    :param other_args: dict, 包含所有其他上下文和约束参数的“参数工具包”。
    :param mdl_args: dict, 求解器（如 SCIP, CPLEX）相关配置。
    :param method: int, 在目标函数中计算某些因子时使用的排名方法 (1: pct=True, 2: dense, 3: log normalization)。
    :param sampling_proportion: float, 产品抽样比例（当前逻辑中未被充分利用）。
    :param opti_time_limit: int, 整个函数执行的超时时间（秒）。
    :param kwargs: dict, 预留的额外关键字参数。
    :return: dict, 一个包含优化结果的字典，结构如下：
             {
                 'status': str,  # 'success' 或 'fail'
                 'data': list,  # 包含一个或多个 portfolio DataFrame 的列表
                 'indicators': list,  # 与 data 对应的指标字典列表
                 'message': list,  # 错误或警告信息列表
                 'info': list  # 详细的日志信息列表
             }
    """
    start_dt = datetime.now()

    def has_no_result_check():
        """
        一个内部诊断函数，在优化完全失败后，检查是否存在因缺少产品导致无解的明显情况。

        该函数作为最后一道检查，用于定位最基本的配置错误。它的逻辑是：
        1. 分别统计“产品池”和“用户持仓”中，每个目标大类资产下的产品数量。
        2. 将两者相加，得到每个大类资产总共可用于配置的产品数量。
        3. 遍历所有大类资产，如果发现某个大类的总可用产品数为0，而其目标配置又不为0，
           则说明优化目标在逻辑上不可能实现。
        4. 在这种情况下，生成一条明确的错误信息并返回。

        :return: list, 一个包含错误信息的字符串列表。如果所有大类都有足够的产品，则返回空列表。
        """
        # 统计产品池中各类资产的产品数量
        pool_nums_dict = dict(
            [(iAsset, len(product_info[product_info[f"大类资产-{iAsset}"] == 1]))
             for iAsset, _ in target_asset_alloc.items()])

        # 统计用户持仓中各类资产的产品数量
        hold_nums_dict = dict(
            [(iAsset, len(other_args['userHolding'][other_args['userHolding'][f"大类资产-{iAsset}"] == 1]))
             for iAsset, _ in target_asset_alloc.items()])

        # 合并产品池和用户持仓中的产品数量
        nums_dict = dict([(key, value + hold_nums_dict.get(key, 0)) for key, value in pool_nums_dict.items()])

        # 检查是否存在产品数量为0的资产类别
        message = []
        for iAsset, num in nums_dict.items():
            if num <= 0:
                message += [f"大类资产{iAsset}的产品数量为0!"]
        return message

    ''' 根据可用产品数量的多少，动态地决定后续优化要采用的 “产品抽样策略” '''
    # 注意: 后续 do_patch 代码中未使用抽样功能, 没有使用 ratio/sampling_proportion 参数
    if product_info.shape[0] <= other_args['min_num']:
        # 可供选择的产品总数，是否小于或等于我们要求构建的最小组合数量？例如，产品池里只有8个产品，但要求最少配置10个
        s_ratio, e_ratio = 1.0, 1.01  # 后续循环只会产生一个值：1.0。后续调用 do_patch 函数时，传入的 ratio 参数将是 1.0
    else:  # 备选产品数量是充足的，超过要构建组合中的最小产品数量
        # sampling_proportion写死 0.3。后续调用 do_patch 函数时，传入的 ratio 参数将是 0.3。表示只在 30% 的产品中进行抽样优化
        s_ratio, e_ratio = sampling_proportion, sampling_proportion + 0.01

    ''' 调用 do_patch 执行器 '''
    result_lst = []
    # 实际上这个 for 循环, 只会被调用一次, 并且 ratio 参数在 do_patch 中也未被使用 ...
    for ratio in np.arange(s_ratio, e_ratio, 0.1):
        print(f"当前产品抽样比例: {ratio:.2f}，开始调用 do_patch 执行器...")
        # 调用 do_patch 执行器: do_patch 函数会封装所有参数，并最终调用 do_one 来构建并求解 cvxpy 优化模型
        res = do_patch(
            clazzMap,  # 产品ID到大类资产编码的映射字典
            product_info,  # 包含产品池和用户持仓的完整、已处理的产品信息表
            amt,  # 总投资金额
            obj_alloc,  # 优化目标权重字典
            args,  # 包含无风险利率等基础参数的字典
            target_asset_alloc,  # 目标大类资产配置比例
            asset_alloc_rate,  # 各大类资产的预期收益率范围
            [rate_limit[0] - 1.0,  # 极大地放宽了整体收益率约束的最小值(原最小值-1.0)
             rate_limit[1] + 1.0],  # 极大地放宽了整体收益率约束的最大值(原最大值+1.0)
            asset_alloc_bias,  # 各大类资产允许的最大偏离度
            longest_idle_time,  # 组合允许的最大久期（天）
            other_args,  # 包含所有其他上下文和约束参数的"参数工具包"
            mdl_args,  # 求解器（如 SCIP, CPLEX）相关配置
            method,  # 在目标函数中计算某些因子时使用的排名方法
            ratio,  # 产品抽样比例
            weight_sum_range=[0.99],  # 权重加总范围
            opti_time_limit=opti_time_limit  # 优化时间限制（秒）
        )

        ''' 结果提取 '''
        print(f"当前产品抽样比例: {ratio:.2f}，do_patch 执行器返回，状态: {res['status']}")
        # 如果优化失败，则直接进入下一次循环 (实际只执行一次)
        if 'fail' == res['status']:
            continue
        # 如果优化成功，则从返回结果 res 中提取出包含投资组合方案的列表 data，并赋值给 lst 变量以供后续处理
        lst = res['data']

        ''' 组合去重 
        作用是确保最终返回给用户的多个投资组合方案是有实质性差异的，而不是非常相似的重复方案。
        '''
        add_lst = []
        for cur_ret in lst:
            b_is_individual = True
            for df in result_lst:
                df_ = pd.merge(df[['持仓权重']], cur_ret[['持仓权重']],
                               left_index=True, right_index=True, how='outer')
                df_ = df_.fillna(0.0)
                if np.max(np.abs(df_.values[:, 0] - df_.values[:, 1])) <= 0.01:
                    b_is_individual = False
                    break
            if b_is_individual:
                add_lst.append(cur_ret)
        result_lst += add_lst

        ''' 终止条件判断与返回 '''
        if ((datetime.now() - start_dt).total_seconds() >= opti_time_limit or
                len(result_lst) >= other_args['portfolio_num']):
            res['data'] = result_lst
            return res

    print("所有尝试结束，准备返回...")
    res['data'] = result_lst
    if 'info' not in res:
        res['info'] = []

    if res['status'] != 'success':
        # 进行一些参数检测
        res['info'] = res.get('info', []) + has_no_result_check()

    return res


def build_portfolio(products, risk=3, rf=0.02, maxAllowedDev=None, cpkAmount=5e6, cpkAllowedCloseDay=0,
                    privateVisible=False, productsCountRangeInOneCpk=None, clazzCfg=None, ObjAlloc=None,
                    whitePlanType='', rateLimit=None, max_weight=None, clazzRate=None, userHoldingDict=None,
                    intereScoreDict=None, portfolio_num=1, opti_time_limit=60 * 10, **kwargs):
    """
    构建个人投资组合的核心入口函数。

    该函数作为整个优化流程的顶层封装，负责接收所有业务相关的输入参数，
    进行全面的数据预处理、参数校验、默认值设定和优化参数组装，
    然后调用底层的优化执行模块 `lst_portfolio_generation` 来生成投资组合。

    主要流程包括：
    1.  设置默认值：为可选参数（如 `productsCountRangeInOneCpk`, `clazzCfg`）提供默认配置。
    2.  组装优化目标：根据 `kwargs` 中的 `scoreWeight` 等参数计算并组装优化目标权重 `ObjAlloc`。
    3.  数据转换与预处理：将输入的 `products` 和 `userHoldingDict` 列表转换为内部使用的 Pandas DataFrame 格式。
    4.  数据清洗与校验：对用户持仓数据进行空值检查和填充，确保数据完整性。
    5.  参数打包：将所有处理过的、与优化相关的参数打包到 `OtherArgs` 和 `MdlArgs` 字典中。
    6.  调用核心优化：调用 `lst_portfolio_generation` 函数，启动优化过程。
    7.  返回结果：处理并返回优化结果。

    :param products: list, 可投资的产品池列表，每个元素是一个包含产品属性的字典。
    :param risk: int, 客户的风险等级。
    :param rf: float, 无风险利率，用于计算夏普比率等衍生指标。
    :param maxAllowedDev: dict, 各大类资产允许的最大偏离度。
    :param cpkAmount: float, 客户总投资金额。
    :param cpkAllowedCloseDay: int, 组合允许的最大加权平均久期（天）。
    :param privateVisible: bool, 是否为合格投资者，用于筛选产品。
    :param productsCountRangeInOneCpk: list, 组合中产品数量的范围 [min, max]。
    :param clazzCfg: dict, 目标大类资产配置比例。
    :param ObjAlloc: dict, (可选) 优化目标权重字典，若不提供则根据 `scoreWeight` 等自动计算。
    :param whitePlanType: str, 白名单计划类型，用于筛选特定场景的产品。
    :param rateLimit: list, 组合整体收益率的范围限制 [min, max]。
    :param max_weight: dict, 单一产品在其所属大类中的最大权重上限。
    :param clazzRate: dict, 各大类资产的预期收益率范围，可能用于估算或校验。
    :param userHoldingDict: list, 客户的现有持仓列表，用于调仓场景。
    :param intereScoreDict: dict, 产品兴趣分字典 (`三级分类码 -> 分数`)，可人为影响产品得分。
    :param portfolio_num: int, 希望生成的不同投资组合的数量。
    :param opti_time_limit: int, 优化流程的超时时间（秒）。
    :param kwargs: dict, 额外的关键字参数，必须包含 `scoreWeight`, `holdAfterYearStdWeight`, `prodCrWeight` 等用于计算优化目标的参数。
    :return: tuple, 一个包含四个元素的元组:
             - portfolios (list): 推荐的投资组合列表，每个组合是一个由产品字典组成的列表。
             - msg (list): 优化过程中产生的消息或错误信息列表。
             - indicators (list): 与每个投资组合对应的详细指标数据。
             - info (list): 详细的日志信息，用于调试。
    """

    def ComplanType(x, y):
        """
        计算两个字符串中逗号分隔元素的交集数量

        参数:
            x: 第一个字符串
            y: 第二个字符串

        返回值:
            交集元素的数量
        """
        if pd.isnull(x) or pd.isnull(y):
            return 0
        if (x.strip() == '') or (y.strip() == ''):
            return 0
        set1 = set(x.split(',')) - set([''])
        set2 = set(y.split(',')) - set([''])
        return len(set1 & set2)

    def ywlx(ywlxdm):
        """
        根据业务类型代码返回对应的业务类型名称

        参数:
            ywlxdm: 业务类型代码

        返回值:
            对应的业务类型名称
        """
        if 'lc' == ywlxdm.lower():
            return '银行理财'
        elif ywlxdm == '1':
            return '公募基金'
        elif ywlxdm == '2':
            return '基金专户'
        elif ywlxdm == '3':
            return '券商集合'
        elif ywlxdm == '4':
            return '信托计划'
        elif ywlxdm == '5':
            return '保险资管'
        elif ywlxdm == '6':
            return '其他产品'
        else:
            return '未知类型'

    def gen_default_product_info():
        """
        生成默认产品信息列表

        该函数创建一个包含默认产品信息的字典列表，用于初始化产品数据结构。
        产品信息包含产品基本信息、交易信息、风险信息、费率信息等字段。

        Returns:
            list: 包含一个产品信息字典的列表，字典包含以下字段：
                - 产品ID: 产品唯一标识符
                - 产品名称: 产品显示名称
                - 单位: 产品金额单位
                - 风险等级: 产品风险评级
                - 起购金额: 最低购买金额
                - 递增金额: 购买金额递增单位
                - 产品周期: 产品投资周期
                - 业绩比较基准上限: 业绩比较基准的上限值
                - 业绩比较基准下限: 业绩比较基准的下限值
                - 产品优先度: 产品展示优先级
                - 是否面向合格投资者: 投资者适当性要求
                - 自定义分类: 产品业务分类
                - 一级分类码值: 一级分类编码
                - 三级分类码值: 三级分类编码
                - 是否可买入: 产品申购状态
                - 是否可卖出: 产品赎回状态
                - 是否持仓: 当前持仓状态
                - 当前持仓金额: 持仓金额
                - 买入换手费率: 申购费率
                - 卖出换手费率: 赎回费率
                - 产品评级: 产品综合评级
                - 重点推荐: 是否为重点推荐产品
                - 持仓剩余期限: 持仓到期剩余时间
                - 基准指数代码: 业绩比较基准指数代码
                - 基准指数收益率: 基准指数预期收益率
                - 产品固设机构: 产品销售渠道标识
                - 最高持有金额: 单一客户最高持有金额限制
                - 大类资产相关字段: 各类资产配置比例
        """
        ProductInfo = [dict({
            '产品ID': 'xxxxxxxx',
            '产品名称': 'xxxxxxxx',
            '单位': '人民币',
            '风险等级': 'R0',
            '起购金额': 0.01,
            '递增金额': 0.01,
            '产品周期': 1,
            '业绩比较基准上限': 1 * 100,
            '业绩比较基准下限': 0 * 100,
            '产品优先度': 0,
            '是否面向合格投资者': False,
            '自定义分类': '银行理财',
            '一级分类码值': '01',
            '三级分类码值': '01001001',
            '是否可买入': False,
            '是否可卖出': False,
            '是否持仓': False,
            '当前持仓金额': 0.0,
            '买入换手费率': 0.0,
            '卖出换手费率': 1.0,
            '产品评级': -1,
            '重点推荐': False,
            '持仓剩余期限': 0,
            '基准指数代码': 'XYHQCX',
            '基准指数收益率': 0.0025,
            '产品固设机构': 0,  # 0 无  1 财富  2 私行
            '最高持有金额': np.nan,
            **{f'大类资产-{k}': (k == '现金类') * 1.0 for k, v in ASSET_CODE_DICT.items()}
        })]

        return ProductInfo

    # 对已有持仓字典进行兼容性处理
    '''检查输入的 userHoldingDict（用户持仓列表）中的每个产品，是否存在一个名为 'is_recomm' 的键。
    如果存在，它会将其值赋给一个新键 'isRecomm'，然后删除旧键。
    这确保了无论上游系统传入的是新字段名 (isRecomm) 还是旧字段名 (is_recomm)，程序内部都能统一处理，增强了代码的向后兼容性。
    '''
    for one_holding in userHoldingDict:
        if 'is_recomm' in one_holding:
            one_holding['isRecomm'] = one_holding['is_recomm']
            del one_holding['is_recomm']

    # 如果未传入产品数量范围，设置默认范围 [2, 5]，即最终生成的投资组合应包含 2 到 15 个产品。
    if not productsCountRangeInOneCpk:
        productsCountRangeInOneCpk = [2, 15]

    # 优化模型目标函数 (Objective Function) 的核心定义.
    ''' 该字典的键是优化要考虑的各个子目标（如流动性、收益率等），值是这些子目标在最终总分计算中所占的权重。
    优化器 (cvxpy) 的任务就是找到一个投资组合，使得 (组合的流动性分数 * 流动性权重) + (组合的收益率分数 * 收益率权重) + ... 
    优化目标是使得这个总和最大化。
    '''
    ObjAlloc = {
        '流动性': max(kwargs['scoreWeight'][0] / 100, 0.01),
        '收益率': max(kwargs['scoreWeight'][1] / 100, 0.01),
        '波动率': max(kwargs['scoreWeight'][2] * kwargs['holdAfterYearStdWeight'] / 100, 0.001),
        '分散度': max(kwargs['scoreWeight'][2] * kwargs['prodCrWeight'] * 10, 0.001),
        '产品数量': kwargs['scoreWeight'][2] * kwargs['prodCrWeight'] / 10.0,
        '产品评级': kwargs['scoreWeight'][2] * kwargs['prodCrWeight'] / 5,
        '换手费率': 0.5,
        '重点推荐': kwargs['scoreWeight'][2] * kwargs['prodCrWeight'] / 3,
        '兴趣评分': kwargs['scoreWeight'][2] * kwargs['prodCrWeight'] / 10,
        '久期': 0.01,
    }
    # 动态调整和限制“换手费率”权重
    ''' 首先，它会找到在 ObjAlloc 字典中当前所有权重里的最大值。
    然后，它用这个最大值和 0.5 进行比较，取其中的较小者作为“换手费率”的最终权重。
    这个操作的目的是让换手成本的重要性与当前最重要的优化目标保持在同一水平，但又不超过一个上限。
    '''
    ObjAlloc['换手费率'] = min(max(list(ObjAlloc.values())), 0.5)

    if not clazzCfg:
        clazzCfg = {
            "clazz01": 0.25,
            "clazz02": 0.20,
            "clazz03": 0.30,
            "clazz04": 0.10,
            "clazz05": 0.15
        }

    Amount = cpkAmount  # 购买金额 (元)

    # 产品筛选参数
    FilterArgs = {
        "风险等级": f'R{risk}',  # 最高风险等级
        "无风险利率": rf,
    }

    # 资金最长闲置时间 (天). 代表“客户可以接受的资金最长封闭期”
    LongestIdleTime = cpkAllowedCloseDay  # 例如: 657
    # TargetAssetAlloc 定义了目标的大类资产配置比例，格式为 {'另类': 0.1, '固收类': 0.1, '权益类': 0.6, ...}
    TargetAssetAlloc = {k: clazzCfg.get(f"clazz{v}") for k, v in ASSET_CODE_DICT.items()}  # 目标的大类资产配置比例
    # AssetAllocRate 定义了各大类资产的预期收益率范围，格式为 {'另类': [-0.16, 0.28], '固收类': [-0.01, 0.09], ...}
    AssetAllocRate = {k: clazzRate.get(f"clazz{v}") for k, v in ASSET_CODE_DICT.items()}  # 各大类资产的预期收益率范围

    # ASSET_CODE_DICT_INV 是一个反向映射，其格式为 {'clazz01': '现金类', 'clazz02': '固收类', 'clazz03': '混合类', ...}
    ASSET_CODE_DICT_INV = dict(zip(map(lambda x: "clazz" + x, list(ASSET_CODE_DICT.values())), ASSET_CODE_DICT.keys()))
    # AssetAllocBias 定义了各大类资产允许的最大偏离度，格式为 {'另类': 0.2, '固收类': 0.2, '权益类': 0.2, '混合类': 0.2, ...}
    AssetAllocBias = dict({ASSET_CODE_DICT_INV[key]: value for key, value in maxAllowedDev.items()})

    # 遍历各个产品, 生成一个列表, 列表内的每个元素是一个字典, 包含产品的各个属性
    if len(products) > 0:
        ProductInfo = [dict({
            '产品ID': item.get('productId'),
            '产品名称': item.get('productName', item.get('productId')),
            '单位': item.get('bz', '人民币'),
            '风险等级': f"R{item.get('risk')}",
            '起购金额': item.get('amount'),
            '递增金额': item.get('incAmount'),
            '产品周期': item.get('closeDays'),
            '业绩比较基准上限': item.get('bestRate') * 100,
            '业绩比较基准下限': item.get('worstRate') * 100,
            '产品优先度': item.get('prio', np.nan),
            '是否面向合格投资者': item.get('privateVisible', False),
            '自定义分类': ywlx(item.get('ywlxdm', '')),
            '一级分类码值': item.get('yjdy', np.nan),
            '三级分类码值': item.get('sjdy', np.nan),
            '是否可买入': True,
            '是否可卖出': np.nan,
            '是否持仓': False,
            '产品评级': item.get('level', np.nan),
            '重点推荐': item.get('isRecomm', False),
            '持仓剩余期限': 0,
            '基准指数代码': item.get('baseCode', np.nan),
            '基准指数收益率': item.get('baseRate', np.nan),
            '产品创设机构': float(item.get('cpcsjg', 0)),  # 0=无 1=财富 2=私行
            '专属场景类型': item.get('planType', ''),
            '最高持有金额': item.get('zgcyje', np.nan),
            **{f"大类资产-{k}": float(1) if item.get('clazz') == v else float(0)
               for k, v in ASSET_CODE_DICT.items()}
        }) for item in products]
    else:
        # 当没有可投资产品时，使用默认产品信息占位，防止后续代码报错
        ProductInfo = gen_default_product_info()

    # 转换为 DataFrame 方便处理
    ProductInfo = pd.DataFrame.from_dict(ProductInfo)

    # 对产品信息ProductInfo内的 "起购金额" 和 "递增金额" 进行清洗和填充
    ProductInfo['起购金额'] = ProductInfo['起购金额'].fillna(0.01)
    ProductInfo['起购金额'] = ProductInfo['起购金额'].apply(lambda x: 1e4 if x < 1e4 else x)
    ProductInfo['递增金额'] = ProductInfo['递增金额'].fillna(0.01)

    '''对产品信息ProductInfo内的 "产品优先度" 进行清洗和填充'''
    # 1) 提升所有自定义分类为“银行理财”产品的优先级 (乘以100)
    ProductInfo.loc[:, '产品优先度'] = ProductInfo.apply(
        lambda x: x['产品优先度'] * (100.0 if x['自定义分类'] == '银行理财' else 1.0), axis=1)
    # 2) 下面两个条件合并，完整的含义是：“当客户是合格投资者，并且当前产品是一个公募产品时, 优先级乘以10”
    ProductInfo.loc[:, '产品优先度'] = ProductInfo.apply(
        lambda x: x['产品优先度'] * (
            10.0 if (  # 为 True 时，表示当前正在为“合格投资者”（或称私行客户）进行计算
                            True == privateVisible
                    ) & (  # x['是否面向合格投资者'] 是产品自身的属性; 它判断的是，这个产品不是一个仅面向合格投资者的产品，即它是一个公募产品
                            True != x['是否面向合格投资者']
                    ) else 1.0),
        axis=1)
    # 3) 在风险允许的范围内，应尽量选择风险收益特征更极致的产品，以充分利用客户的“风险预算”，而不是一味地推荐最保守的产品
    ProductInfo.loc[:, '产品优先度'] = ProductInfo.apply(
        lambda x: x['产品优先度'] * (
            (6 - (  # 这里的 6 是一个基准数（可能因为最高风险等级是R5，所以用6作为基数）。这个计算的含义是：风险差距越小，计算结果越大。
                    risk - int(x['风险等级'][1:]))  # 计算客户风险上限与产品风险之间的“差距”。差距越大，说明产品越保守
             ) if risk > int(x['风险等级'][1:]) else 0.0  # 如果产品的风险等级大于客户的风险等级，则该产品的 '产品优先度' 乘以0
        ),
        axis=1)

    '''将入参中的 “兴趣评分 interest_score” 数据，整合到主产品信息表 `ProductInfo` 中'''
    # 将 兴趣评分 interest_score 字典转换成一个 Pandas DataFrame
    intereScore = pd.DataFrame(index=intereScoreDict.keys(), data=intereScoreDict.values(), columns=['interest_score'])
    # 将 兴趣评分(intereScore) 以 "三级分类码值" 为键，合并到主产品信息表 ProductInfo 中
    ProductInfo = pd.merge(ProductInfo, intereScore, left_on='三级分类码值', right_index=True, how='left')
    ProductInfo['interest_score'] = ProductInfo['interest_score'].fillna(0.0)  # 填充缺失值
    ProductInfo = ProductInfo.rename(columns={'interest_score': '兴趣评分'})  # 重命名列

    # 当前持仓金额
    ProductInfo['当前持仓金额'] = 0.0
    ProductInfo['买入换手费率'] = 0.0
    ProductInfo['卖出换手费率'] = 1.0

    ''' 用户持仓数据预处理 '''
    # 将外部传入的用户持仓数据（`userHoldingDict`）转换成一个 DataFrame
    if userHoldingDict:
        userHolding = pd.DataFrame(userHoldingDict)
    else:
        userHolding = pd.DataFrame()  # 处理无持仓数据时，创建一个空的 DataFrame 做占位
    # 确保 `userHolding` DataFrame 拥有所有必需的列，即使输入的原始数据中缺少这些字段
    for x in ['productId', 'productName', 'asset', 'sjdy', 'cpcsjg', 'privateVisible',
              'risk', 'ywlxdm', 'remain_term', 'interest_score', 'is_can_buy',
              'is_can_sell', 'amount', 'incAmount', 'level', 'isRecomm', 'closeDays',
              'baseRate', 'baseCode']:
        if x not in userHolding.columns:
            userHolding[x] = np.nan
    # 为用户的每一个持仓产品计算并填充其对应的“兴趣分”, 以"三级分类码值" 为键，从入参的 intereScoreDict 字典中获取
    userHolding['interest_score'] = userHolding['sjdy'].apply(lambda x: intereScoreDict.get(x, 0.0))

    ''' 对用户持仓数据进行清洗和填充 '''
    lst = []  # 创建一个列表，用于存储处理后的用户持仓数据
    # 获取所有产品ID的集合，用于后续检查持仓产品ID的有效性
    ProductIDSet = set(list(ProductInfo['产品ID'].apply(lambda x: x.split('$^')[0])))
    # 创建一个产品分类映射字典，用于将三级分类码值映射到对应的一级分类
    sjdy2yjdy_dict = product_class[['一级分类', '三级分类码值']].set_index('三级分类码值')['一级分类'].to_dict()
    count = 0
    # 遍历每一条用户持仓记录，进行逐条检查和填充
    for (productId, productName, asset, sjdy, cpcsjg, privateVisible, risk, ywlxdm, remain_term, interest_score,
         is_can_buy, is_can_sell, amount, incAmount, level, isRecomm, closeDays, baseCode, baseRate, zgcyje) in \
            userHolding[['productId', 'productName', 'asset', 'sjdy', 'cpcsjg', 'privateVisible',
                         'risk', 'ywlxdm', 'remain_term', 'interest_score', 'is_can_buy',
                         'is_can_sell', 'amount', 'incAmount', 'level', 'isRecomm', 'closeDays',
                         'baseCode', 'baseRate', 'zgcyje']].values:
        ''' 对每个字段进行检查和填充 '''
        count += 1
        if '' == productId or pd.isnull(productId):
            print(f"持仓产品ID={productId} ID为空填充为:wxwxwxwx_{count}")
            productId = f"wxwxwxwx_{int(count)}"
        if '' == productName or pd.isnull(productName):
            productName = ''
            print(f"持仓产品ID={productId}, 产品名称为空填充为:{productName}")
        if '' == asset or pd.isnull(asset):
            asset = 0.00
            print(f"持仓产品ID={productId}, 持仓金额为空填充为:{asset}")
        if '' == sjdy or pd.isnull(sjdy):
            sjdy = '07003001'  # 其他类型
            ywlxdm = '07'
            print(f"持仓产品ID={productId}, 二级分类为空填充为其他类型:{sjdy}")
        if '' == cpcsjg or pd.isnull(cpcsjg):
            cpcsjg = 0
            # print(f"产品ID={productId}, 二级分类为空填充为其他类型:{sjdy}")
        if '' == privateVisible or pd.isnull(privateVisible):
            privateVisible = False
            print(f"持仓产品ID={productId}, 是否合格投资者为空填充为:{privateVisible}")
        if '' == risk or pd.isnull(risk):
            risk = 1
            print(f"持仓产品ID={productId}, 风险等级为空填充为:{risk}")
        if '' == ywlxdm or pd.isnull(ywlxdm):
            ywlxdm = ''
            print(f"持仓产品ID={productId}, 业务类型为空填充为:{ywlxdm}")
        if '' == remain_term or pd.isnull(remain_term):
            remain_term = 0.0
            print(f"持仓产品ID={productId}, 剩余期限为空填充为:{remain_term}")
        if '' == is_can_buy or pd.isnull(is_can_buy):
            is_can_buy = False
            print(f"持仓产品ID={productId}, 是否可买入值为空填充为:{is_can_buy}")
        if '' == is_can_sell or pd.isnull(is_can_sell):
            is_can_sell = False
            print(f"持仓产品ID={productId}, 是否可卖出值为空填充为:{is_can_sell}")
        if '' == amount or pd.isnull(amount):
            amount = 0.01
            print(f"持仓产品ID={productId}, 起购金额值为空填充为:{amount}元")
        if '' == incAmount or pd.isnull(incAmount):
            incAmount = 0.01
            print(f"持仓产品ID={productId}, 递增金额值为空填充为:{incAmount}元")
        if '' == level or pd.isnull(level):
            level = 0.0
            print(f"持仓产品ID={productId}, 产品评级为空填充为:{level}")
        if '' == isRecomm or pd.isnull(isRecomm):
            isRecomm = False
            print(f"持仓产品ID={productId}, 是否重点推荐值为空填充为:{isRecomm}")
        if '' == closeDays or pd.isnull(closeDays):
            closeDays = 1
            print(f"持仓产品ID={productId}, 产品闭期为为空填充为:{closeDays}")
        if '' == baseCode or pd.isnull(baseCode):
            baseCode = ''
            print(f"持仓产品ID={productId}, 基准指数代码为空填充为:{baseCode}")
        if '' == baseRate or pd.isnull(baseRate):
            baseRate = 0.0
            print(f"持仓产品ID={productId}, 基准指数收益率为空填充为:{baseRate}")
        if '' == zgcyje or pd.isnull(zgcyje):
            zgcyje = np.nan
            print(f"持仓产品ID={productId}, 最高持有金额为空填充为:nan")

        yjdy = sjdy[:2]  # 从“三级分类码” (sjdy) 中提取前两位作为“一级分类码” (yjdy)
        remain_term = max(remain_term, 0)  # 确保“剩余期限” (remain_term) 的值不会是负数

        ''' 计算 “卖出换手费率” (sell_fee) 和 “买入换手费率” (buy_fee) '''
        # 线性插值法计算与产品流动性相关的“卖出成本系数”: 资产剩余期限越长其流动性越差，卖出的隐性成本就越高, 使得优化器更倾向于卖出短期限的产品
        sell_fee = min(1.0,  # 确保最终的成本系数不会超过 1.0
                       1.0 if pd.isnull(remain_term)  # 首先判断剩余期限是否为空值。如果为空，直接赋予最高的卖出成本 1.0
                       else max(0.7,  # 确保计算出的成本系数不会低于 0.7
                                # 下面定义了成本系数在两个端点之间的变化规则: 0 ~ 1825 天之间线性变化, 从 0.7 线性上升到 1.0
                                0.7 + (remain_term - 0) / (1825 - 0)
                                ) * (1 - 0.7)
                       )
        # 特殊业务规则的覆盖: 模拟现实世界中活期存款(01001001)几乎没有赎回成本、流动性极高的特性 (用极小值1e-3来表示该类产品的卖出成本)
        sell_fee = 1e-3 if sjdy == '01001001' else sell_fee
        # 为买入成本赋一个固定的值,
        buy_fee = 0.0

        ''' 检查持仓产品是否在产品池中, 并进行相应的填充 '''
        productIdPure = productId.split('$^')[0]
        if productIdPure in ProductIDSet:
            # 如果持有产品在产品池里面, 则用产品池数据进行填充, 部分字段以持仓数据为准
            dics = ProductInfo.loc[ProductInfo['产品ID'] == productIdPure, :].to_dict('records')
            for dic in dics:
                dic['产品ID'] = productId
                dic['当前持仓金额'] = asset
                dic['是否持仓'] = True
                dic['买入换手费率'] = buy_fee
                dic['卖出换手费率'] = sell_fee
                dic['持仓剩余期限'] = remain_term

                if False == pd.isnull(is_can_buy):
                    dic['是否可买入'] = is_can_buy
                if False == pd.isnull(is_can_sell):
                    dic['是否可卖出'] = is_can_sell
                if False == pd.isnull(level):
                    dic['产品评级'] = level
                if False == pd.isnull(isRecomm):
                    dic['重点推荐'] = isRecomm
                lst += [dic]
        else:
            # 如果持有产品不在产品池里面, 则用持仓数据进行填充
            lst += [{
                '一级分类码值': yjdy,
                '三级分类码值': sjdy,
                '专属场景类型': np.nan,
                '业绩比较基准上限': np.nan,
                '业绩比较基准下限': np.nan,
                '产品ID': productId,
                '产品优先度': 1.0,
                '产品创设机构': cpcsjg,
                '产品名称': productName,
                '产品周期': closeDays,
                '持仓剩余期限': remain_term,
                '单位': '人民币',
                '大类资产-另类': 1 if '另类投资' == sjdy2yjdy_dict.get(sjdy, '') else 0,
                '大类资产-固收类': 1 if '固定收益' == sjdy2yjdy_dict.get(sjdy, '') else 0,
                '大类资产-权益类': 1 if '权益投资' == sjdy2yjdy_dict.get(sjdy, '') else 0,
                '大类资产-混合类': 1 if '混合策略' == sjdy2yjdy_dict.get(sjdy, '') else 0,
                '大类资产-现金类': 1 if '现金管理' == sjdy2yjdy_dict.get(sjdy, '') else 0,
                '是否面向合格投资者': privateVisible if privateVisible else False,  # 如果值为空，默认认为面向合格投资者，只有合格投资者才能增配
                '自定义分类': ywlx(ywlxdm),
                '起购金额': amount,
                '递增金额': incAmount,
                '风险等级': f'R{int(float(risk))}',
                '兴趣评分': interest_score,
                '当前持仓金额': asset,
                '是否持仓': True,
                '是否可买入': is_can_buy,
                '是否可卖出': is_can_sell,
                '产品评级': level,
                '重点推荐': isRecomm,
                '基准指数代码': baseCode,
                '基准指数收益率': baseRate,
                '买入换手费率': buy_fee,
                '卖出换手费率': sell_fee,
                '最高持有金额': zgcyje
            }]

    ''' 整合持仓与产品池 DataFrame '''
    # 将前一个循环处理好的持仓数据（存储在列表 lst 中）最终转换为一个 DataFrame
    if len(lst) > 0:
        userHolding = pd.DataFrame(lst)
    else:
        userHolding = pd.DataFrame(columns=ProductInfo.columns, data=[])  # 处理无持仓数据时，创建一个空的 DataFrame 做占位
    # 从产品池中剔除已经持仓的产品: 这是一个数据去重步骤，目的是避免同一个产品在待优化列表中出现两次（一次作为持仓，一次作为备选产品）
    isholding = ProductInfo['产品ID'].apply(
        lambda x: x.split('$^')[0] in set([productId.split('$^')[0] for productId in list(userHolding['产品ID'])])
    )
    ProductInfo = ProductInfo.loc[False == isholding, :]
    # 如果产品池被剔除持仓产品后，变成了空的，则用默认的占位产品进行填充，防止后续代码报错
    if ProductInfo.shape[0] == 0:
        ProductInfo = pd.DataFrame.from_dict(gen_default_product_info())

    ''' 应用产品交易规则 '''
    # 对已经整合好的产品池 (`ProductInfo`) 和用户持仓 (`userHolding`)，统一应用来自 product_priority.csv 文件中定义的产品可交易性规则
    ProductInfo = setting_by_sjdy(ProductInfo)  # 三级分类 对应的 是否允许增持, 若为 否, 那么 '是否可买入' = False
    userHolding = setting_by_sjdy(userHolding)  # 三级分类 对应的 是否允许增持, 若为 否, 那么 '是否可买入' = False

    userHolding = userHolding.set_index('产品ID')

    ''' 动态调整“单一产品最大持仓权重”这一约束，以避免其与“大类资产配置目标”产生逻辑冲突，从而防止优化无解。
    如果模型要求权益类占60%，但池中只有一个权益类产品且其上限要求是15%，这是不可能实现的。
    这段代码通过将该产品的上限动态提升到60%，使得约束条件在逻辑上变得可行，从而让优化器能够找到解。。
    '''
    max_weight = dict(  # 覆盖原本的 max_weight 字典
        {key: min(1,  # 确保最终的上限不会超过1
                  (max(  # 在“放宽后的单品上限”和“大类总目标”之间取一个较大值
                      value * 1.2,  # 放宽产品配置上限 (提升20%)
                      TargetAssetAlloc[ASSET_CODE_DICT_INV[key]]  # 查找大类的总配置目标
                  )))
         for key, value in max_weight.items()
         }
    )

    ''' 优化模型的参数工具字典
    将 build_portfolio 函数接收到的、分散的、面向业务的众多参数，以及在函数内部经过预处理和转换后的数据，打包成一个统一的、结构化的字典。
    '''
    OtherArgs = {
        "min_amt": 1000,  # 最小持仓金额: 规定任何一只被选入组合的产品，其持仓金额都不能低于1000元。用于避免产生金额过小的零碎仓位。
        "max_amt": None,  # 最大持仓金额: None 表示不从此参数施加全局限制，但产品自身的 zgcyje（最高持有金额）字段仍然会在约束中生效。
        "min_weight": 0.005,  # 最小持仓权重: 规定任何一只被选入组合的产品，其权重不能低于整个投资组合的0.5%。避免组合中有过多微小仓位
        "max_weight": max_weight,  # 最大持仓权重: 动态计算后的字典, 定义了单一产品在其所属大类中的最大权重
        "min_num": productsCountRangeInOneCpk[0],  # 最小持仓数量, 从入参Json的 "productsCountRangeInOneCpk" 字段获取
        "max_num": productsCountRangeInOneCpk[1],  # 最大持仓数量, 从入参Json的 "productsCountRangeInOneCpk" 字段获取
        "weight_sum": 0.95,  # 权重加总范围: 约束投资组合中所有产品权重之和必须不低于95%，确保大部分资金都被有效投资，避免过多现金闲置。
        "weight_balance_weight": 10.0,  # 权重均衡权重
        "b_grid_search": False,  # 是否网格搜索, 主逻辑中未使用
        "privateVisible": privateVisible,  # 客户是否为合格投资者 (私行客户)
        'risk': risk,  # 客户的风险等级
        'indexCov': kwargs.get('indexCov', None),  # Json中没有该参数, 主逻辑中也未使用
        'indexReturn': kwargs.get('indexReturn', None),  # Json中没有该参数, 主逻辑中也未使用
        'userHolding': userHolding,  # 处理后的用户持仓 DataFrame
        'portfolio_num': portfolio_num,  # 需返回的投资组合数量, 从Json中输入
        'ASSET_CODE_DICT': ASSET_CODE_DICT,  # 大类资产编码字典
        'covMatrix': pd.DataFrame(index=kwargs['matrixBaseCodeArray'],  # 协方差矩阵, 将输入的原始 二维列表 转换为 DataFrame
                                  data=kwargs['matrix'],
                                  columns=kwargs['matrixBaseCodeArray']),
        "scoreFuncParas": {  # 评分函数参数: 将输入的 scoreFunction 和 scoreArgs 列表，重新组织成一个以指标名称为键的字典
            FUN_CLASS[i]: {"scoreFunction": kwargs["scoreFunction"][i],
                           "scoreArgs": kwargs["scoreArgs"][i]
                           }
            for i in range(len(kwargs["scoreFunction"]))
            # 示例: {'现金类占比评分': {'scoreFunction': 'atan', 'scoreArgs': [68.04, 60.65, 0.08, -5.764, 1.0, 0.0]}, ...}
        },
        'scoreConstraints': {},  # 分数约束, 预留的参数位置
        'debug': kwargs.get('debug', DEBUG),  # 调试模式开关
        'isSellProduct': '管销' in kwargs.get('index', ""),  # 是否管销场景, True时会触发 sellproduct_config_setting.py
        "clazzInterestOfFineTuning": [  # 客户偏好大类列表, 根据输入的 pref 参数生成
            ASSET_CODE_DICT_INV[f"clazz{i}"] for i in kwargs["pref"].split(",")] if kwargs.get("pref", None) else []
    }
    # 是否微调模式
    OtherArgs["isFineTuning"] = kwargs.get("isFineTuning", len(OtherArgs["clazzInterestOfFineTuning"]) > 0)

    ''' 分散度排除名单处理 '''
    # 读取分散度排除名单文件, 如果文件存在, 则读取其中的三级定义代码列, 并转换为一个集合, 赋值给 OtherArgs["disperse_exclude"]
    disperse_exclude_filename = r'/Users/chenjunming/Desktop/CIB_wealth_manage/calculate/config/disperse_exclude_conf.csv'
    if os.path.exists(disperse_exclude_filename):
        # 此处的代码加载的是那些不应被计入分散度计算的三级资产类别（如活期存款、国债等）
        OtherArgs["disperse_exclude"] = set(  # 从读取的数据中，只选择“三级定义代码”这一列;
            pd.read_csv(disperse_exclude_filename)["三级定义代码"].apply(lambda x: str(x).zfill(8))
        )
    else:
        OtherArgs["disperse_exclude"] = set()
        print(f"不存在配置品文件: {disperse_exclude_filename}")

    ''' 构建层级化的优化目标权重 '''
    # 将多个独立的权重参数，重新组织成一个具有层级关系的字典，存入 OtherArgs 中。
    OtherArgs["scoreWeightLevel1"] = dict(
        zip(["流动性", "收益", "风险"], kwargs["scoreWeight"])
    )
    OtherArgs["scoreWeightLevel2"] = {
        "波动率": kwargs["holdAfterYearStdWeight"],
        "分散度": kwargs["prodCrWeight"]
    }

    ''' 调试模式下的数据一致性校验 '''
    if OtherArgs.get("debug", DEBUG):
        class3to12_dic = product_class[["三级分类码值", "自定义分类"]].set_index("三级分类码值")["自定义分类"].to_dict()
        for item in products:
            if ywlx(item["ywlxdm"]) != class3to12_dic.get(item.get("sjdy"), "未知类型"):
                print(
                    f"产品业务分类校验不通过: "
                    f"{item.get('productName')}, {item.get('productId')}, "
                    f"{ywlx(item['ywlxdm'])}, {class3to12_dic.get(item.get('sjdy'), '未知类型')}"
                )

    ''' 创建产品到大类资产的映射 '''
    clazzMap = {p["productId"]: p["clazz"] for p in products}

    ''' 配置求解器参数字典 '''
    # GLPK_MI ：可用，非商用，已经安装
    # CPLEX   ：商用，免费版只能解决小规模问题，
    # GUROBI  ：商用，免费版只能解决小规模问题，
    # MOSEK   ：商用，免费版只能解决小规模问题，
    # CBC     ：可用，非商用，没有安装
    # SCIP    ：可用，非商用，已经安装
    MdlArgs = {
        "verbose": False if OtherArgs.get("debug", DEBUG) else False,  # 控制求解器是否输出详细信息
        "solver": cvx.SCIP,  # 指定使用的求解器
        "cplex_params": {  # 求解器的参数配置
            "timelimit": 3000,  # 时间限制, 原代码是30
            "simplex.tolerances.feasibility": 1e-5,  # 单纯形法可行性容忍度
            "barrier.convergetol": 1e-7,  # 内点法收敛容忍度
            "preprocessing.presolve": 0,  # 预处理设置为0(关闭)
            "mip.tolerances.absmipgap": 1e-07,  # MIP绝对间隙容忍度
            "emphasis.numerical": 1,  # 数值稳定性强调设置为1
        }
    }

    StartT = time.perf_counter()

    print("所有输入参数均已准备完毕，开始进行投资组合优化计算...")
    if OtherArgs.get('debug', DEBUG):
        print("使用调试模式进行计算，优化时间限制被强制设定为1000秒")
        opti_time_limit = 1000
        _timeout_limit = opti_time_limit * 2
        Portfolio_lst, t = lst_portfolio_generation(
            clazzMap, ProductInfo, Amount, ObjAlloc, FilterArgs,
            target_asset_alloc=TargetAssetAlloc, asset_alloc_rate=AssetAllocRate,
            asset_alloc_bias=AssetAllocBias, rate_limit=rateLimit,
            longest_idle_time=LongestIdleTime, mdl_args=MdlArgs,
            other_args=OtherArgs, opti_time_limit=opti_time_limit,
            timeout_limit=10
        ), ''
    else:
        print("使用正式模式进行计算，优化时间限制为..秒")
        _timeout_limit = opti_time_limit * 2
        # Portfolio_lst, t = time_limit(lst_portfolio_generation,
        #                               clazzMap, ProductInfo,
        #                               Amount, ObjAlloc, FilterArgs,
        #                               target_asset_alloc=TargetAssetAlloc,
        #                               asset_alloc_rate=AssetAllocRate,
        #                               asset_alloc_bias=AssetAllocBias, rate_limit=rateLimit,
        #                               longest_idle_time=LongestIdleTime, mdl_args=MdlArgs,
        #                               other_args=OtherArgs, opti_time_limit=opti_time_limit,
        #                               _default_result=[], _timeout_limit=_timeout_limit)
        Portfolio_lst = lst_portfolio_generation(
            clazzMap,  # 产品ID到大类资产编码的映射字典，用于确定每个产品的资产类别
            ProductInfo,  # 包含产品池和用户持仓的完整、已处理的产品信息表(DataFrame格式)
            Amount,  # 总投资金额(元)，用于计算各产品的持仓金额
            ObjAlloc,  # 优化目标权重字典，定义各优化目标的重要性权重
            FilterArgs,  # 产品筛选参数，包含风险等级和无风险利率等基础参数
            target_asset_alloc=TargetAssetAlloc,  # 目标大类资产配置比例，定义各类资产的目标配置权重
            asset_alloc_rate=AssetAllocRate,  # 各大类资产的预期收益率范围，用于设定收益率约束
            asset_alloc_bias=AssetAllocBias,  # 各大类资产允许的最大偏离度，控制资产配置偏离目标的程度
            rate_limit=rateLimit,  # 组合整体收益率的范围限制[min, max]，控制组合整体收益水平
            longest_idle_time=LongestIdleTime,  # 组合允许的最大久期(天)，限制组合的流动性风险
            mdl_args=MdlArgs,  # 求解器相关配置参数，如求解器类型、求解参数等
            other_args=OtherArgs,  # 包含所有其他参数的字典，如约束条件、用户持仓等
            opti_time_limit=opti_time_limit,  # 优化流程的超时时间(秒)，控制整个优化过程的执行时间
        )
        t = 0
    if t == -1:
        print('投资组合优化超时')
        Portfolio_lst = {
            'status': 'fail',
            'data': [],
            'message': ['投资组合优化超时'],
            'indicators': [],
            'info': []
        }

    if OtherArgs['isFineTuning']:
        Portfolio_lst['info'] += ['微调场景']

    return ([x.reset_index().to_dict(orient='records') for x in Portfolio_lst['data']],
            list(dict.fromkeys(Portfolio_lst['message'])),
            Portfolio_lst['indicators'],
            list(dict.fromkeys(Portfolio_lst['info'])))
