# -*- encoding: utf-8 -*-
"""
@File: opti_functions.py
@Modify Time: 2025/8/28 17:29       
@Author: Kevin-Chen
@Descriptions: 
"""
import copy
import time
import traceback

import pandas as pd
from itertools import product

from config_py import DEBUG
from utils.limit import time_limit
from calculate import indicator_calculate
from calculate.param_check_func import ParaAdj
from calculate.score_estimate import estimate_all
from calculate.object_functions import ObjectScoreCal
from calculate.portfolio_adjust import adjust_numerical_precision
from calculate.constructPortfolioIndividualM4 import constructPortfolio
from calculate.finetuning_config_setting import FineTuningConfigSetting
from calculate.sellproduct_config_setting import SellProductConfigSetting


class ConstraintsCheck():
    def __init__(self, clazzMap, product_df, amt, obj_alloc, target_asset_alloc,
                 asset_alloc_rate, rate_limit, asset_alloc_bias, longest_idle_time,
                 other_args, mdl_args, method, weight_sum_range=(0.99, 0.95)):
        self.clazzMap = clazzMap
        self.product_df = product_df
        self.amt = amt
        self.obj_alloc = obj_alloc
        self.target_asset_alloc = target_asset_alloc
        self.asset_alloc_rate = asset_alloc_rate
        self.rate_limit = rate_limit
        self.asset_alloc_bias = asset_alloc_bias
        self.longest_idle_time = longest_idle_time
        self.other_args = other_args
        self.mdl_args = mdl_args
        self.method = method
        self.weight_sum_range = weight_sum_range

        # 是否启用超时控制（默认 False，除非 debug 模式关闭）
        self.use_time_limit = other_args.get('debug', DEBUG) == False
        self.timeout_limit = 3  # 超时时间默认 3 秒

        # 主要参数字典
        self.params = {
            'clazzMap': clazzMap,
            'product_df': product_df,
            'amt': amt,
            'obj_alloc': obj_alloc,
            'target_asset_alloc': target_asset_alloc,
            'asset_alloc_rate': asset_alloc_rate,
            'rate_limit': rate_limit,
            'asset_alloc_bias': asset_alloc_bias,
            'longest_idle_time': longest_idle_time,
            'other_args': other_args,
            'mdl_args': mdl_args,
            'method': method,
            'weight_sum_range': weight_sum_range,
            'checking': True
        }

        # 参数备份，避免后续修改污染原始配置
        self.params_bak = copy.deepcopy(self.params)

    def prams_restore(self):
        """恢复初始参数"""
        self.params = copy.deepcopy(self.params_bak)

    def change_hold_sell(self):
        """把用户持仓里的 '是否可卖出' 改成 True"""
        self.params['other_args']['userHolding']['是否可卖出'] = True

    def change_alloc_bias(self):
        """修改资产配置偏差，把所有偏差设置为 1.0"""
        asset_alloc_bias = copy.deepcopy(self.params['asset_alloc_bias'])
        asset_alloc_bias_new = dict(
            zip(asset_alloc_bias.keys(), len(asset_alloc_bias) * [1.0])
        )
        self.params['asset_alloc_bias'] = asset_alloc_bias_new

    def change_amt(self):
        """把总金额 amt 增加 1e8"""
        self.params['amt'] += 1e8

    def change_alloc_bias_global(self):
        """修改全局资产配置偏差，把所有 target_asset_alloc 的上下限统一成 (0,1)"""
        self.params['other_args']['asset_alloc_bias_global'] = dict(
            zip(
                self.target_asset_alloc.keys(),
                [{'l_asset_alloc': 0, 'u_asset_alloc': 1}] * len(self.target_asset_alloc)
            )
        )

    def change_max_num(self):
        """修改最大产品数量"""
        self.params['other_args']['max_num'] = 1000

    # 起购金额检测
    def change_buy_min(self):
        """修改起购金额，统一设置为 0.01"""
        self.params['product_df']['起购金额'] = 0.01
        self.params['other_args']['userHolding']['起购金额'] = 0.01

    # 最大持仓权重限制
    def change_max_weight(self):
        self.params['other_args']['max_weight'] = dict(
            zip(
                list(self.params['other_args']['max_weight'].keys()),
                [1.0] * len(self.params['other_args']['max_weight'])
            )
        )

    def change_duration(self):
        """修改最长久期时间"""
        self.params['longest_idle_time'] = 1e6

    def change_hold_buy(self):
        """把持仓设为可买入"""
        self.params['other_args']['userHolding']['是否可买入'] = True

    def change_weight_sum(self):
        """修改权重和"""
        self.params['other_args']['weight_sum'] = 0.5

    # 递延金额检测
    def change_buy_inc(self):
        """修改递增金额，设置成极小值 0.01"""
        self.params['product_df']['递增金额'] = 0.01
        self.params['other_args']['userHolding']['递增金额'] = 0.01

    def chage_max_amt(self):
        """修改最大持有金额"""
        self.params['product_df']['最高持有金额'] = 10000e8

    def test_alloc(self):
        userHolding = self.params_bak['other_args']['userHolding']
        amt = self.params_bak['amt']
        target_asset_alloc = self.params_bak['target_asset_alloc']
        asset_alloc_bias = self.params_bak['asset_alloc_bias']

        # 筛选持仓中“不可卖出”的部分
        df = userHolding[
            (userHolding['是否持仓'] == True) &
            (userHolding['是否可卖出'] == False)
            ]

        msg = []

        # ===============================
        # 检测大类资产配置是否满足约束
        # ===============================
        for iAlloc, iTarget in target_asset_alloc.items():
            # 计算某大类的当前持仓占比
            ratio = df[df[f'大类资产-{iAlloc}'] == 1]['当前持仓金额'].sum() / amt

            if ratio < 1e-6:  # 忽略占比极小的类
                continue

            # 允许的上下偏差区间
            u = min(iTarget * (1 + asset_alloc_bias[iAlloc]), 1)
            l = max(iTarget * (1 - asset_alloc_bias[iAlloc]), 0)

            # 超过上限 → 报错
            if ratio > u:
                msg += [
                    f"无法减持的持仓中属于大类={iAlloc}的产品持仓占比 "
                    f"{round(ratio * 100, 2)}%，超过了大类={iAlloc}的目标区间："
                    f"[{round(l * 100, 2)}%，{round(u * 100, 2)}%]上限"
                ]

        return msg

    def test_max_weight(self):
        # 检测单一产品集中度
        userHolding = self.params_bak['other_args']['userHolding']
        amt = self.params_bak['amt']

        # 只看 “已持仓 且 不可卖出” 的部分
        df = userHolding[
            (userHolding['是否持仓'] == True) &
            (userHolding['是否可卖出'] == False)
            ]

        # 计算每个产品的持仓比例
        df.loc[:, '持仓比例'] = (
            0.0 if amt < 1e-6 else df['当前持仓金额'] / amt
        )

        # 资产类别映射字典
        ASSET_CODE_DICT = self.params_bak['other_args']['ASSET_CODE_DICT']
        ASSET_CODE_DICT_INV = dict(
            zip(
                map(lambda x: "clazz" + x, list(ASSET_CODE_DICT.values())),
                ASSET_CODE_DICT.keys()
            )
        )

        # 最大持仓权重限制
        max_weight = self.params_bak['other_args']['max_weight']

        msg = []

        # ===============================
        # 遍历每个大类的最大权重设置
        # ===============================
        for iAlloc, iValue in max_weight.items():
            # 找出超过单品上限的产品
            for product_id in df[
                (df[f'大类资产-{ASSET_CODE_DICT_INV[iAlloc]}'] == 1) &
                (df['持仓比例'] > iValue)
            ].index:
                name = df.loc[product_id, '产品名称']
                ratio = df.loc[product_id, '持仓比例']
                hold_amt = df.loc[product_id, '当前持仓金额']

                msg += [
                    f"由于无法减持导致单一产品集中度超限: 产品ID={product_id.split('$^')[0]}，"
                    f"产品名称：{name}，持仓比例={hold_amt}/{amt}={round(ratio * 100, 2)}%，"
                    f"超过了限制：{iValue * 100}%"
                ]

        return msg

    def test_duration(self):
        # =============== 产品久期检查 ===============
        userHolding = self.params_bak['other_args']['userHolding']
        amt = self.params_bak['amt']

        # 过滤条件：必须是持仓且不可卖出
        df = userHolding[
            (userHolding['是否持仓'] == True) &
            (userHolding['是否可卖出'] == False)
            ]
        msg = []

        # 计算久期：持仓金额占比 * 剩余期限
        duration = ((df['当前持仓金额'] / amt) * df['持仓剩余期限']).sum()

        # 超过阈值就报错
        if duration > self.params_bak['longest_idle_time']:
            msg += [f"无法减持产品的久期为:{round(duration, 2)}，大于阈值:{self.params_bak['longest_idle_time']}"]

        return msg

    def test_max_num(self):
        # =============== 产品数量检查 ===============
        userHolding = self.params_bak['other_args']['userHolding']

        df = userHolding[
            (userHolding['是否持仓'] == True) &
            (userHolding['是否可卖出'] == False)
            ]
        msg = []

        # 检查是否超过最大持仓数量
        if df.shape[0] > self.params_bak['other_args']['max_num']:
            msg += [f"无法减持的持仓产品数量:{df.shape[0]}，持仓数量上限:{self.params_bak['other_args']['max_num']}"]

        return msg

    def test_amt(self):
        msg = []
        # 如果要输出总资产额度，可以启用下面这一行
        # msg = [f"总资产额度为:{self.params_bak['amt']}"]
        return msg

    def test_cannot_sell_product_list(self):
        """
        提取无法减持的产品清单
        """
        userHolding = self.params_bak['other_args']['userHolding']

        # 找出不可卖出的产品
        df = userHolding[userHolding['是否可卖出'] == False][['产品名称']]

        id_lst = list(df.index)
        name_lst = list(df['产品名称'])

        msg = ["以下产品无法减持："]
        for id, name in zip(id_lst, name_lst):
            msg += [f"产品ID={id.split('$^')[0]}, 产品名称={name}"]

        # 如果确实存在无法卖出的产品，则返回清单
        if len(msg) > 1:
            return msg
        else:
            return []

    # 生成不能减持的检测信息
    def gen_msg_for_cannot_hold_sell(self):
        msg = []
        msg += self.test_alloc()
        msg += self.test_max_weight()
        msg += self.test_duration()
        msg += self.test_max_num()
        msg += self.test_cannot_sell_product_list()
        return msg

    def gen_msg_for_amt(self):
        msg = []

    def check_run(self):
        cur_ret, t = do_one(**self.params), 0

        if (t != -1) and (cur_ret['status'] == 'success'):
            return True
        else:
            return False

    def check_all(self):
        dicts = {
            self.change_hold_sell: (
                "持仓产品中存在无法减配的产品，导致资金不足 或者 数量超额 或者 久期超限",
                self.gen_msg_for_cannot_hold_sell),
            self.change_duration: (
                f"组合久期下限:{self.params_bak['longest_idle_time']}，上调组合参数久期值将会生成组合", None),
            self.change_amt: (
                "总资产额度太低", None),
            self.change_max_num: (
                f"组合产品数量:{self.params_bak['other_args']['max_num']}，约束太严格，建议上调数量上限", None),
            self.change_buy_min: (
                "产品池或者持仓中存在起购金额较高的产品，导致无法完成配置", None),
            self.change_max_weight: (
                "单一产品配置比例上限太低，需要更多产品进行配置, 建议上调比例门限或者增加产品池", None),
            self.change_hold_buy: (
                "持仓产品中存在无法增配的产品，且产品池无替代产品，导致无法调仓", None),
            self.change_weight_sum: (
                f"投资金额占总资产比例下限:{self.params_bak['other_args']['weight_sum']}，建议下调此比例", None),
            self.change_buy_inc: (
                "产品池或者持仓中存在递增金额较高的产品，导致无法完成配置", None),
            self.change_alloc_bias: (
                "无法按照大类比例进行配置，可能是由于持仓产品无法减持或者产品池数量较少等原因导致", None),
            self.change_alloc_bias_global: (
                "无法按照大类比例进行配置, 可能是由于持仓产品无法减持或者产品池数量较少等原因导致", None),
            self.chage_max_amt: (
                "产品持仓无法满足最高持有金额上限的要求", None)
        }

        message = []
        # 单参数尝试
        print("开始单参数尝试...")
        for change_params, (msg, gen_msg_fun) in dicts.items():
            print(f"尝试参数: {change_params.__name__}")
            change_params()
            if self.check_run():
                message += [msg]
                if gen_msg_fun is not None:
                    message += gen_msg_fun()
                self.prams_restore()
                return message
            else:
                self.prams_restore()
        print(message)

        # 两两组合尝试
        print("开始两两参数组合尝试...")
        # try:
        funs = list(dicts.keys())
        for i in range(len(funs)):
            for j in range(i + 1, len(funs)):
                funs[i]()
                funs[j]()
                print(f"尝试组合: {funs[i].__name__} + {funs[j].__name__}")
                if self.check_run():
                    if funs[i].__name__ == 'change_amt':
                        msg_i, gen_msg_fun_i = dicts[funs[i]]
                        message += [msg_i]
                        if gen_msg_fun_i is not None:
                            message += gen_msg_fun_i()
                    elif funs[j].__name__ == 'change_amt':
                        msg_j, gen_msg_fun_j = dicts[funs[j]]
                        message += [msg_j]
                        if gen_msg_fun_j is not None:
                            message += gen_msg_fun_j()
                    else:
                        msg_i, gen_msg_fun_i = dicts[funs[i]]
                        message += [msg_i]
                        if gen_msg_fun_i is not None:
                            message += gen_msg_fun_i()
                        msg_j, gen_msg_fun_j = dicts[funs[j]]
                        message += [msg_j]
                        if gen_msg_fun_j is not None:
                            message += gen_msg_fun_j()
                    self.prams_restore()
                    return message
                else:
                    self.prams_restore()
        # except:
        #     print(traceback.format_exc())
        print(message)
        return message


def alloc_check(nums_dict, obj_alloc, asset_alloc_bias_dic):
    # 大类资产配置约束条件过滤
    message = []
    status = 'success'
    for iAsset in obj_alloc:
        l_ = obj_alloc[iAsset] - asset_alloc_bias_dic[iAsset]['l_asset_alloc_bias']
        if l_ <= 0:
            continue
        if (l_ > 0) and (nums_dict[iAsset] > 0):
            continue
        status = 'fail'
        message += [f"大类-{iAsset}的产品数量为0, 且要求进行配置"]
    return status, message


def do_one(clazzMap, product_df, amt, obj_alloc, target_asset_alloc, asset_alloc_rate, rate_limit, asset_alloc_bias,
           longest_idle_time, other_args, mdl_args, method, weight_sum_range=[0.99], checking=False):
    """
    执行单次完整的投资组合优化计算。

    本函数是优化流程的核心执行单元，它接收所有预处理和配置好的参数，
    进行最终的动态调整，然后调用 `constructPortfolio` 来构建并求解优化问题，
    最后对结果进行后处理和指标计算。

    主要流程包括：
    1.  **数据整合**: 将产品池 (`product_df`) 和用户持仓 (`userHolding`) 合并成一个包含所有待优化资产的
        完整数据表 `df_product_hold`。
    2.  **参数动态调整**: 根据不同的业务场景（如是否管销、是否微调）或现有持仓的特性，对优化参数
        （特别是大类资产配置目标和久期）进行最终的、适应性的调整。
    3.  **指标范围估算**: 调用 `estimate_all` 函数，通过简化模型预估各项指标（收益、风险等）的取值范围，
        为后续评分函数的线性近似做准备。
    4.  **迭代求解**: 遍历 `weight_sum_range` 和 `min_num_lst` 等可变参数，多次调用 `constructPortfolio`
        进行求解，以寻找在不同约束下的最优解。
    5.  **结果后处理**: 如果优化成功，调用 `adjust_numerical_precision` 对结果进行数值精度调整，并调用
        `calIndicators` 计算调仓前后的各项详细绩效指标。

    :param clazzMap: dict, 产品ID到大类资产编码的映射字典。
    :param product_df: pd.DataFrame, 经过筛选的产品池信息。
    :param amt: float, 总投资金额。
    :param obj_alloc: dict, 优化目标权重字典。
    :param target_asset_alloc: dict, 目标大类资产配置比例。
    :param asset_alloc_rate: dict, 各大类资产的预期收益率范围。
    :param rate_limit: list, 组合整体收益率的范围限制 [min, max]。
    :param asset_alloc_bias: float, 各大类资产允许的最大偏离度。
    :param longest_idle_time: int, 组合允许的最大久期（天）。
    :param other_args: dict, 包含所有其他上下文和约束参数的“参数工具包”。
    :param mdl_args: dict, 求解器（如 SCIP, CPLEX）相关配置。
    :param method: int, 在目标函数中计算某些因子时使用的排名方法。
    :param weight_sum_range: list, 权重加总下限的尝试列表，函数会遍历此列表以寻找可行解。
    :param checking: bool, 是否处于“检查模式”。在检查模式下，部分参数调整逻辑会被跳过。
    :return: dict, 一个包含单次优化结果的字典，结构如下：
             {
                 'status': str,  # 'success' 或 'fail'
                 'data': pd.DataFrame or None,  # 优化成功的投资组合方案
                 'indicators': dict or None,  # 组合的详细指标
                 'message': list,  # 错误或警告信息列表
                 'info': list,  # 详细的日志信息列表
                 'has_check': bool # 是否经过了约束检查
             }
    """
    # 合并产品池与用户持仓
    df_product_hold = pd.concat([product_df, other_args['userHolding']], sort=True)

    ''' 识别与排序 “离散型产品” '''
    # 如果一个产品的“递增金额”大于1 (例如，只能按1000元、2000元的整数倍购买)，它就被标记为离散型产品。如果递增金额很小 (如0.01元)，则被视为可按任意金额购买的“连续型产品”
    df_product_hold['离散型产品'] = df_product_hold['递增金额'] > 1
    # 将所有连续型产品排在前面，离散型产品排在后面
    df_product_hold = df_product_hold.sort_values(by='离散型产品', ascending=True)

    ''' 处理产品ID并设定索引 
    在优化开始前，计算出用户当前的投资组合的大类资产配置比例
    '''
    df_product_hold = df_product_hold.reset_index()
    df_product_hold['原始ID'] = df_product_hold['产品ID'].apply(lambda x: x.split('$^')[0])
    df_product_hold = df_product_hold.set_index('产品ID')
    product_info = df_product_hold.sort_values(by='离散型产品', ascending=True)

    ''' 分析当前持仓的资产配置状况 '''
    userHolding = df_product_hold[df_product_hold['是否持仓'] == True]  # 只看持仓部分
    holdAmount = userHolding['当前持仓金额'].sum()  # 计算当前持仓总金额
    hold_alloc = dict(  # 计算各大类资产的当前占比
        [(iAsset, userHolding[userHolding[f'大类资产-{iAsset}'] == 1]['当前持仓金额'].sum() / holdAmount)
         if holdAmount > 1e-3 else (iAsset, 0.0) for iAsset in target_asset_alloc]
    )

    ''' 评估当前持仓的偏离度 '''
    bias_judge = dict([
        (
            iAsset,  # False 表示该类资产的持仓已 “偏离”，True 表示 “在轨”。(空仓情况下,全部都是False)
            (hold_alloc[iAsset] >= iTarget * (1 - asset_alloc_bias[iAsset]))
            & (hold_alloc[iAsset] <= iTarget * (1 + asset_alloc_bias[iAsset]))
        )
        for iAsset, iTarget in target_asset_alloc.items()
    ])  # 数据示例: {'另类': False, '固收类': False, '权益类': True, '混合类': False, '现金类': False}

    if not checking:  # checking 默认是 False, 只有在 ConstraintsCheck 类运行时会是 True
        # 判断 是否所有大类资产的持仓都处于合理区间内？-- sum(bias_judge.values()) 就是计算有多少个大类资产是“在轨”的
        if sum(bias_judge.values()) / len(bias_judge.values()) == 1.0:  # 所有的大类资产都“在轨”
            # 如果用户当前的资产配置已经很接近目标了，说明不需要进行大规模的调整。因此, 直接使用原始的约束参数，进行一次优化求解就足够了。
            asset_alloc_bias_ratio_range = [1.0]  # 该变量是一个“偏离乘数”的列表,优化循环会遍历这个列表去动态地调整约束的“严格程度”
            min_weight_range = [other_args['min_weight']]  # 最小权重范围，从other_args参数中获取min_weight
            weight_sum_range = weight_sum_range  # 权重总和范围，使用传入的weight_sum_range参数
        else:  # 至少有一个大类资产的持仓是“偏离轨道”的
            # 当用户持仓有偏离时，直接使用宽松的约束进行一次优化不一定能找到最好的解。因此，模型采用“三步走”策略
            asset_alloc_bias_ratio_range = [1e-8, 0.5, 1.0]  # 极度严格的偏离约束, 中等严格的偏离约束, 宽松的偏离约束
            min_weight_range = [other_args['min_weight']]
            weight_sum_range = weight_sum_range
    else:  # 在 ConstraintsCheck 类运行时会是 True, 才会执行下面逻辑
        asset_alloc_bias_ratio_range = [1.0]
        min_weight_range = [other_args['min_weight']]
        weight_sum_range = [min(weight_sum_range)]

    ''' 大类数量统计 '''
    # 统计每个大类资产分别有多少个产品, 例: {'另类': 1, '固收类': 4, '权益类': 2, '混合类': 2, '现金类': 1}
    pool_nums_dict = dict(
        [(iAsset, len(product_df[product_df[f"大类资产-{iAsset}"] == 1])) for iAsset, _ in target_asset_alloc.items()])
    if other_args['userHolding'].shape[0] > 0:
        # 统计在用户的现有持仓中每个大类资产分别有多少个产品, 例: {'另类': 0, '固收类': 0, '权益类': 0, '混合类': 0, '现金类': 1}
        hold_nums_dict = dict(
            [(iAsset,
              len(other_args['userHolding']
                  [(other_args['userHolding'][f"大类资产-{iAsset}"] == 1)
                   & (False == other_args['userHolding'].index.str.startswith('wxwxwxwx'))
                   ])) for iAsset, _ in target_asset_alloc.items()])
    else:
        hold_nums_dict = dict([(iAsset, 0) for iAsset, _ in target_asset_alloc.items()])
    # 合并计算总数, 将前两步的结果相加，得到每个大类资产总共可用的产品数量
    nums_dict = dict([(key, value + hold_nums_dict.get(key, 0)) for key, value in pool_nums_dict.items()])

    result = {'status': 'fail', 'data': None, 'message': [], 'info': [], 'indicators': []}

    # 确保所有资产的运行偏离值值为正数，避免后续计算中出现除零或负数问题
    for iAsset in asset_alloc_bias:
        if asset_alloc_bias[iAsset] <= 0:  # 将小于等于0的偏差值重置为0.001 (最小正值)
            asset_alloc_bias[iAsset] = 0.001

    ''' 遍历不同的参数组合进行优化求解 '''
    for asset_alloc_bias_ratio, min_weight, weight_sum in product(
            asset_alloc_bias_ratio_range, min_weight_range, weight_sum_range):
        print(f"尝试参数组合: "
              f"asset_alloc_bias_ratio={asset_alloc_bias_ratio}, min_weight={min_weight}, weight_sum={weight_sum}")

        ''' 动态调整各个大类资产配置约束 '''
        # 根据当前循环的 asset_alloc_bias_ratio 值，为每一个大类资产实时计算出本次优化所要遵守的、具体的上下限约束
        for iAsset in target_asset_alloc:
            # 初始化每个大类资产权重的偏离度约束细节
            if 'asset_alloc_bias_detail' not in other_args:
                other_args['asset_alloc_bias_detail'] = dict(
                    zip(target_asset_alloc.keys(),
                        [{'_asset_alloc_bias': None, 'u_asset_alloc_bias': None}] * len(target_asset_alloc)
                        ))

            if bias_judge[iAsset]:  # 如果该类资产的持仓是“在轨”的, 计算权重的上下限约束
                other_args['asset_alloc_bias_detail'][iAsset] = {
                    'l_asset_alloc_bias':
                        target_asset_alloc[iAsset]
                        * (1.0 if 0 == nums_dict[iAsset] else (1.0 - asset_alloc_bias[iAsset]))
                        + 1e-6,
                    'u_asset_alloc_bias':
                        target_asset_alloc[iAsset]
                        * (1.0 + asset_alloc_bias[iAsset])
                        + 1e-6
                }
            else:  # 如果该类资产的持仓是“偏离”的, 计算权重的上下限约束
                other_args['asset_alloc_bias_detail'][iAsset] = {
                    'l_asset_alloc_bias':
                        target_asset_alloc[iAsset]
                        * (1.0 if 0 == nums_dict[iAsset] else (asset_alloc_bias_ratio * asset_alloc_bias[iAsset]))
                        + 1e-6,
                    'u_asset_alloc_bias':
                        target_asset_alloc[iAsset]
                        * (target_asset_alloc[iAsset] + asset_alloc_bias[iAsset])
                        + 1e-6
                }
        print(f"本次优化的大类资产配置约束细节: {other_args['asset_alloc_bias_detail']}")

        ''' 可行性预检查 '''
        # 大类资产检测: 在进行优化计算之前，先进行一次快速的逻辑上的可行性检查
        status, message = alloc_check(nums_dict, target_asset_alloc,
                                      other_args['asset_alloc_bias_detail'])
        if 'fail' == status:
            print(message)
            print(f"大类资产检测 (可行性预检查) 失败, 跳过本次参数组合的优化计算")
            result = {'status': 'fail', 'data': None, 'message': message, 'info': [],
                      'has_check': True, 'indicators': []}
            continue
        print("大类资产检测 (可行性预检查) 通过")

        # 将本次循环的 min_weight 和 weight_sum 参数更新到 other_args 中
        other_args['weight_sum'] = weight_sum
        other_args['min_weight'] = min_weight

        # 通过一系列简化的线性规划，来预估在当前约束下，各项核心指标（如收益率、波动率、现金占比）理论上能达到的最大值和最小值
        _msg, other_args['indicatorEstimateResult'] = estimate_all(
            df_product_hold,
            target_asset_alloc,
            asset_alloc_bias,
            nums_dict,
            other_args['covMatrix'],
            other_args['disperse_exclude'],
            other_args['isSellProduct']
        )
        print(f"指标范围估算结果: {other_args['indicatorEstimateResult']}")

        if not other_args['indicatorEstimateResult']:
            result['message'] = result['message'] + _msg
            return result

        if other_args.get('min_num', None) is not None:
            min_num_lst = [
                min(other_args['min_num'], df_product_hold.shape[0]),
                min(max(other_args['min_num'], round(other_args['max_num'] * 0.8, 0)), df_product_hold.shape[0])
            ]
            min_num_lst = list(set(min_num_lst))
            min_num_lst.sort()
            min_num_lst.reverse()
        else:
            min_num_lst = [None]
        _other_args = copy.deepcopy(other_args)

        for min_num in min_num_lst:
            _other_args['min_num'] = min_num
            _df_product_hold, _obj_alloc, _target_asset_alloc = df_product_hold, obj_alloc, target_asset_alloc
            _amt, _longest_idle_time = amt, longest_idle_time
            if _other_args.get('isSellProduct', False):
                spcs = SellProductConfigSetting(df_product_hold,
                                                amt,
                                                obj_alloc,
                                                target_asset_alloc,
                                                longest_idle_time,
                                                asset_alloc_rate,
                                                rate_limit,
                                                asset_alloc_bias,
                                                _other_args

                                                )
                spcs.set_params()
                _df_product_hold = spcs.ProductInfo
                _other_args['asset_alloc_bias_detail'] = spcs.other_args['asset_alloc_bias_detail']
                _other_args['weight_sum'] = spcs.other_args['weight_sum']
                _obj_alloc, _longest_idle_time = spcs.obj_alloc, spcs.longest_idle_time
            elif _other_args.get('isFineTuning', False):
                # 微调场景的参数设置
                ftcs = FineTuningConfigSetting(_other_args['clazzInterestOfFineTuning'],
                                               df_product_hold,
                                               amt,
                                               obj_alloc,
                                               target_asset_alloc,
                                               asset_alloc_rate,
                                               rate_limit,
                                               asset_alloc_bias,
                                               _other_args
                                               )
                ftcs.set_params()
                _df_product_hold = ftcs.ProductInfo
                _other_args['asset_alloc_bias_detail'] = ftcs.other_args['asset_alloc_bias_detail']
                _other_args['weight_sum'] = ftcs.other_args['weight_sum']
                _obj_alloc = ftcs.obj_alloc
            else:
                # 由于无法持有的产品会导致大类无法平衡，需要进行处理
                pa = ParaAdj(target_asset_alloc,
                             asset_alloc_bias,
                             longest_idle_time,
                             amt,
                             _other_args
                             )
                _target_asset_alloc, _other_args['asset_alloc_bias_detail'] = pa.adj_alloc_by_holding()
                _longest_idle_time = pa.adj_duration_by_holding()
                pass
            if other_args.get('debug', DEBUG):
                df_product_hold.to_csv(r'df_product_hold.csv')
            res = constructPortfolio(_df_product_hold,
                                     _amt,
                                     _obj_alloc,
                                     target_asset_alloc=_target_asset_alloc,
                                     asset_alloc_rate=asset_alloc_rate,
                                     rate_limit=rate_limit,
                                     asset_alloc_bias=asset_alloc_bias,
                                     longest_idle_time=_longest_idle_time,
                                     other_args=_other_args,
                                     mdl_args=mdl_args,
                                     method=method)
            if res['data'] is not None:
                break

        if res['data'] is None:
            # 调整最大数量约束
            if other_args.get('max_num', None) is not None:
                max_num_lst = [other_args['max_num'], other_args['max_num'] + 5]
                max_num_lst = list(set(max_num_lst))
                max_num_lst.sort()
            else:
                max_num_lst = [None]
            _other_args = copy.deepcopy(other_args)
            for max_num in max_num_lst:
                _other_args['max_num'] = max_num
                _amt = amt
                _df_product_hold, _obj_alloc, _target_asset_alloc = df_product_hold, obj_alloc, target_asset_alloc

                if _other_args.get('isSellProduct', False):
                    spcs = SellProductConfigSetting(df_product_hold,
                                                    amt, obj_alloc, target_asset_alloc,
                                                    longest_idle_time, asset_alloc_rate,
                                                    rate_limit, asset_alloc_bias,
                                                    _other_args)
                    spcs.set_params()
                    _df_product_hold = spcs.ProductInfo
                    _other_args['asset_alloc_bias_detail'] = spcs.other_args['asset_alloc_bias_detail']
                    _other_args['weight_sum'] = spcs.other_args['weight_sum']
                    _obj_alloc, _longest_idle_time = spcs.obj_alloc, spcs.longest_idle_time
                elif _other_args.get('isFineTuning', False):
                    # 微调场景的参数配置
                    ftcs = FineTuningConfigSetting(_other_args['clazzInterestOfFineTuning'],
                                                   df_product_hold, amt, obj_alloc,
                                                   target_asset_alloc, asset_alloc_rate,
                                                   rate_limit, asset_alloc_bias,
                                                   _other_args)
                    ftcs.set_params()
                    _df_product_hold = ftcs.ProductInfo
                    _other_args['asset_alloc_bias_detail'] = ftcs.other_args['asset_alloc_bias_detail']
                    _other_args['weight_sum'] = ftcs.other_args['weight_sum']
                    _obj_alloc = ftcs.obj_alloc
                else:
                    # 由于无法持有的产品会导致大类无法平衡，需要进行处理
                    pa = ParaAdj(target_asset_alloc, asset_alloc_bias,
                                 longest_idle_time, amt, _other_args)
                    _target_asset_alloc, _other_args['asset_alloc_bias_detail'] = pa.adj_alloc_by_holding()
                    _longest_idle_time = pa.adj_duration_by_holding()
                    pass

                if other_args.get('debug', DEBUG):
                    df_product_hold.to_csv(r'df_product_hold.csv')
                res = constructPortfolio(_df_product_hold, amt, _obj_alloc,
                                         target_asset_alloc=_target_asset_alloc,
                                         asset_alloc_rate=asset_alloc_rate,
                                         rate_limit=rate_limit,
                                         asset_alloc_bias=asset_alloc_bias,
                                         longest_idle_time=_longest_idle_time,
                                         other_args=_other_args,
                                         mdl_args=mdl_args, method=method
                                         )
                if res['data'] is not None:
                    break

            if res['data'] is not None:
                # 对优化结果进行数值精度调整
                res['data'], info = adjust_numerical_precision(res['data'], df_product_hold, amt)
                res['info'] += info
                result['status'], result['data'], result['message'], result['info'], result['indicators'] = (
                    'success', res['data'], res['message'], res['info'],
                    calIndicators(
                        res['data'], df_product_hold, amt,
                        other_args['covMatrix'], other_args['disperse_exclude'],
                        other_args['scoreFuncParas'], other_args['indicatorEstimateResult'],
                        other_args['scoreWeightLevel1'], other_args['scoreWeightLevel2']
                    )
                )
                break
            else:
                result['message'] = result['message'] + res['message']
                result['info'] = result['info'] + res['info']

            return result


def calIndicators(portfolio, products, amt, cov_matrix, disperse_exclude, scoreFuncParas,
                  indicatorEstimateResult, scoreWeightLevel1, scoreWeightLevel2):
    osc = ObjectScoreCal(scoreFuncParas=scoreFuncParas,
                         indicatorEstimateResult=indicatorEstimateResult,
                         product_info=products)

    indicators = {}
    portfolio = pd.merge(portfolio,
                         products[['大类资产-现金类', '三级分类码值']],
                         left_index=True, right_index=True, how='left')
    portfolio['是否参与计算分散度'] = portfolio['三级分类码值'].apply(
        lambda x: True if x not in disperse_exclude else False
    )

    indicators['durationAfter'], indicators['durationBefore'] = indicator_calculate.calDuration(
        portfolio, products, amt)
    indicators['cashRatioAfter'], indicators['cashScoreAfter'] = indicator_calculate.calCashScore(
        portfolio, **scoreFuncParas['现金类占比评分'])
    indicators['returnAfter'], indicators['returnScoreAfter'] = indicator_calculate.calReturnScore(
        portfolio, products.loc[portfolio.index, '基准指数收益率'], **scoreFuncParas['预期收益率评分'])
    indicators['disperseAfter'], indicators['disperseScoreAfter'] = indicator_calculate.calDisperseScore(
        portfolio[portfolio['是否参与计算分散度'] == True], **scoreFuncParas['分散度评分'])

    df = pd.merge(portfolio, products[['基准指数代码']], left_index=True, right_index=True, how='left')
    indicators['volatilityAfter'], indicators['volatilityScoreAfter'] = indicator_calculate.calVolatilityScore(
        portfolio, cov_matrix.loc[df['基准指数代码'].values, df['基准指数代码'].values],
        **scoreFuncParas['预期波动率评分'])

    portfolio_before = products[products['是否持仓'] == True][['当前持仓金额', '大类资产-现金类', '三级分类码值']]
    portfolio_before['是否参与计算分散度'] = portfolio_before['三级分类码值'].apply(
        lambda x: True if x not in disperse_exclude else False
    )

    amt = portfolio_before['当前持仓金额'].sum()
    portfolio_before['持仓权重'] = portfolio_before['当前持仓金额'].apply(lambda x: x / amt if amt > 0.0 else 0.0)

    indicators['returnBefore'], indicators['returnScoreBefore'] = indicator_calculate.calReturnScore(
        portfolio_before, products.loc[portfolio_before.index, '基准指数收益率'],
        **scoreFuncParas['预期收益率评分'])
    indicators['cashRatioBefore'], indicators['cashScoreBefore'] = indicator_calculate.calCashScore(
        portfolio_before, **scoreFuncParas['现金类占比评分'])

    df = pd.merge(portfolio_before, products[['基准指数代码']],
                  left_index=True, right_index=True, how='left')
    indicators['volatilityBefore'], indicators['volatilityScoreBefore'] = indicator_calculate.calVolatilityScore(
        portfolio_before, cov_matrix.loc[df['基准指数代码'].values, df['基准指数代码'].values],
        **scoreFuncParas['预期波动率评分'])

    indicators['disperseBefore'], indicators['disperseScoreBefore'] = indicator_calculate.calDisperseScore(
        portfolio_before[portfolio_before['是否参与计算分散度'] == True], **scoreFuncParas['分散度评分'])

    indicators['riskScoreBefore'] = (scoreWeightLevel2['波动率'] * indicators['volatilityScoreBefore']
                                     + scoreWeightLevel2['分散度'] * indicators['disperseScoreBefore'])
    indicators['riskScoreAfter'] = (scoreWeightLevel2['波动率'] * indicators['volatilityScoreAfter']
                                    + scoreWeightLevel2['分散度'] * indicators['disperseScoreAfter'])

    indicators['finalScoreBefore'] = (scoreWeightLevel1['流动性'] * indicators['cashScoreBefore']
                                      + scoreWeightLevel1['收益'] * indicators['returnScoreBefore']
                                      + scoreWeightLevel1['风险'] * (
                                              scoreWeightLevel2['波动率'] * indicators['volatilityScoreBefore']
                                              + scoreWeightLevel2['分散度'] * indicators['disperseScoreBefore']))

    indicators['finalScoreAfter'] = (scoreWeightLevel1['流动性'] * indicators['cashScoreAfter']
                                     + scoreWeightLevel1['收益'] * indicators['returnScoreAfter']
                                     + scoreWeightLevel1['风险'] * (
                                             scoreWeightLevel2['波动率'] * indicators['volatilityScoreAfter']
                                             + scoreWeightLevel2['分散度'] * indicators['disperseScoreAfter']))

    # 计算目标损失函数数值 - 优化前
    indicators['cashLossValueBefore'] = osc.cal_cash_loss_value(
        portfolio_before['大类资产-现金类'].values, portfolio_before['持仓权重'].values)
    indicators['returnLossValueBefore'] = osc.cal_return_loss_value(
        products.loc[portfolio_before.index, '基准指数收益率'].values, portfolio_before['持仓权重'].values
    )

    df = pd.merge(portfolio_before, products[['基准指数代码']], left_index=True, right_index=True, how='left')
    indicators['volatilityLossValueBefore'] = osc.cal_volatility_loss_value_byquad(
        portfolio_before['持仓权重'].values, cov_matrix.loc[df['基准指数代码'].values, df['基准指数代码'].values]
    )

    # 计算目标损失函数数值 - 优化前
    indicators['disperseFitnessValueBefore'] = osc.cal_disperse_fitness_value(
        portfolio_before[portfolio_before['是否参与计算分散度'] == True]['持仓权重'].values,
        portfolio_before[portfolio_before['是否参与计算分散度'] == True]['持仓权重'].values > 0
    )
    # 计算目标损失函数数值 - 优化后
    indicators['disperseFitnessValueAfter'] = osc.cal_disperse_fitness_value(
        portfolio[portfolio['是否参与计算分散度'] == True]['持仓权重'].values,
        portfolio[portfolio['是否参与计算分散度'] == True]['持仓权重'].values > 0
    )

    # 计算目标损失函数数值 - 优化前
    indicators['disperseLossValueBefore'] = osc.cal_disperse_loss_value(
        portfolio_before[portfolio_before['是否参与计算分散度'] == True]['持仓权重'].values,
        portfolio_before[portfolio_before['是否参与计算分散度'] == True]['持仓权重'].values > 0
    )

    # 计算目标损失函数数值 - 优化后
    indicators['disperseLossValueAfter'] = osc.cal_disperse_loss_value(
        portfolio[portfolio['是否参与计算分散度'] == True]['持仓权重'].values,
        portfolio[portfolio['是否参与计算分散度'] == True]['持仓权重'].values > 0
    )

    indicators['cashLossValueAfter'] = osc.cal_cash_loss_value(
        portfolio['大类资产-现金类'].values, portfolio['持仓权重'].values)
    indicators['returnLossValueAfter'] = osc.cal_return_loss_value(
        products.loc[portfolio.index, '基准指数收益率'].values, portfolio['持仓权重'].values)

    df = pd.merge(portfolio, products[['基准指数代码']], left_index=True, right_index=True, how='left')
    indicators['volatilityLossValueAfter'] = osc.cal_volatility_loss_value_byquad(
        portfolio['持仓权重'].values, cov_matrix.loc[df['基准指数代码'].values, df['基准指数代码'].values])

    indicators['disperseLossValueAfter'] = osc.cal_disperse_loss_value(
        portfolio[portfolio['是否参与计算分散度'] == True]['持仓权重'].values,
        portfolio[portfolio['是否参与计算分散度'] == True]['持仓权重'].values > 0)

    indicators['riskLossValueBefore'] = (scoreWeightLevel2['波动率'] * indicators['volatilityLossValueBefore']
                                         + scoreWeightLevel2['分散度'] * indicators['disperseLossValueBefore'])

    indicators['riskLossValueAfter'] = (scoreWeightLevel2['波动率'] * indicators['volatilityLossValueAfter']
                                        + scoreWeightLevel2['分散度'] * indicators['disperseLossValueAfter'])

    indicators['finalLossValueBefore'] = (scoreWeightLevel1['流动性'] * indicators['cashLossValueBefore']
                                          + scoreWeightLevel1['收益'] * indicators['returnLossValueBefore']
                                          + scoreWeightLevel1['风险'] * (
                                                  scoreWeightLevel2['波动率'] * indicators['volatilityLossValueBefore']
                                                  + scoreWeightLevel2['分散度'] * indicators['disperseLossValueBefore']
                                          ))

    indicators['finalLossValueAfter'] = (scoreWeightLevel1['流动性'] * indicators['cashLossValueAfter']
                                         + scoreWeightLevel1['收益'] * indicators['returnLossValueAfter']
                                         + scoreWeightLevel1['风险'] * (
                                                 scoreWeightLevel2['波动率'] * indicators['volatilityLossValueAfter'] +
                                                 scoreWeightLevel2['分散度'] * indicators['disperseLossValueAfter']
                                         ))

    indicators['indicatorEstimateResult'] = indicatorEstimateResult
    indicators['scoreFuncParas'] = scoreFuncParas
    indicators['lossFunctions'] = osc.output()

    display = []
    display.append(f"duration:"
                   f"{round(indicators['durationBefore'], 4)}"
                   f"->{round(indicators['durationAfter'], 4)}")
    display.append(f"cashRatio:"
                   f"{round(indicators['cashRatioBefore'], 4)}"
                   f"->{round(indicators['cashRatioAfter'], 4)}")
    display.append(f"cashScore:"
                   f"{round(indicators['cashScoreBefore'], 4)}"
                   f"->{round(indicators['cashScoreAfter'], 4)}")
    display.append(f"cashLossValue:"
                   f"{round(indicators['cashLossValueBefore'], 4)}"
                   f"->{round(indicators['cashLossValueAfter'], 4)}")
    display.append(f"returnRatio:"
                   f"{round(indicators['returnBefore'], 4)}"
                   f"->{round(indicators['returnAfter'], 4)}")
    display.append(f"returnScore:"
                   f"{round(indicators['returnScoreBefore'], 4)}"
                   f"->{round(indicators['returnScoreAfter'], 4)}")
    display.append(f"returnLossValue:"
                   f"{round(indicators['returnLossValueBefore'], 4)}"
                   f"->{round(indicators['returnLossValueAfter'], 4)}")
    display.append(f"volatilityRatio:"
                   f"{round(indicators['volatilityBefore'], 4)}"
                   f"->{round(indicators['volatilityAfter'], 4)}")
    display.append(f"volatilityScore:"
                   f"{round(indicators['volatilityScoreBefore'], 4)}"
                   f"->{round(indicators['volatilityScoreAfter'], 4)}")
    display.append(f"volatilityLossValue:"
                   f"{round(indicators['volatilityLossValueBefore'], 4)}"
                   f"->{round(indicators['volatilityLossValueAfter'], 4)}")
    display.append(f"disperseRatio:"
                   f"{round(indicators['disperseBefore'], 4)}-"
                   f">{round(indicators['disperseAfter'], 4)}")
    display.append(f"disperseFitnessValue:"
                   f"{round(indicators['disperseFitnessValueBefore'], 4)}"
                   f"->{round(indicators['disperseFitnessValueAfter'], 4)}")
    display.append(f"disperseScore:"
                   f"{round(indicators['disperseScoreBefore'], 4)}"
                   f"->{round(indicators['disperseScoreAfter'], 4)}")
    display.append(f"disperseLossValue:"
                   f"{round(indicators['disperseLossValueBefore'], 4)}"
                   f"->{round(indicators['disperseLossValueAfter'], 4)}")
    display.append(f"riskScore:"
                   f"{round(indicators['riskScoreBefore'], 4)}"
                   f"->{round(indicators['riskScoreAfter'], 4)}")
    display.append(f"riskLossValue:"
                   f"{round(indicators['riskLossValueBefore'], 4)}"
                   f"->{round(indicators['riskLossValueAfter'], 4)}")
    display.append(f"finalScore:"
                   f"{round(indicators['finalScoreBefore'], 4)}"
                   f"->{round(indicators['finalScoreAfter'], 4)}")
    display.append(f"finalLossValue:"
                   f"{round(indicators['finalLossValueBefore'], 4)}"
                   f"->{round(indicators['finalLossValueAfter'], 4)}")

    indicators["display"] = display
    return indicators
