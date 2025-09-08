# -*- encoding: utf-8 -*-
"""
@File: constructPortfolioIndividualM4.py
@Modify Time: 2025/8/28 20:02       
@Author: Kevin-Chen
@Descriptions: 
"""
import platform
import traceback

import numpy as np
import pandas as pd
import cvxpy as cvx
from datetime import datetime

from calculate.constratints import ConstraintsClass
from calculate.object_functions import ObjectScoreCal

if 'windows' in platform.system().lower():
    SOLVERS = [cvx.CPLEX, cvx.SCIP]
else:
    SOLVERS = [cvx.SCIP]


def prdsCmp(A, B):
    m = {}
    for idx, v in A.iterrows():
        m[idx] = [v['持仓权重'], 0]
    for idx, v in B.iterrows():
        if idx in m:
            m[idx][1] = v['持仓权重']
        else:
            m[idx] = [0, v['持仓权重']]

    x2, y2, xy = 0, 0, 0
    for x, y in m.values():
        x2 += x * x
        y2 += y * y
        xy += x * y
    return xy / (x2 * y2) ** 0.5


def mean_std_normalization(x):
    if np.std(x) == 0:
        return np.array([0] * len(x))
    return (x - np.mean(x)) / np.std(x)


def max_min_normalization(x):
    if max(x) - min(x) == 0:
        return np.array([0] * len(x))
    return (x - min(x)) / (max(x) - min(x))


def algebraic_function(x):
    return x / np.sqrt(1 + x ** 2)


def denser_normalization(x):
    return mean_std_normalization(np.sign(x - np.mean(x)) * np.sqrt(
        np.abs(x - np.mean(x)) / np.std(x)
    ))


def constructPortfolio(product_info, amt, obj_alloc,
                       target_asset_alloc=None, asset_alloc_rate=None,
                       rate_limit=None, asset_alloc_bias=0.05, longest_idle_time=None,
                       other_args={}, mdl_args={}, method=2):
    t_set = set(product_info["基准指数代码"]) - set(other_args["covMatrix"].index)
    if len(t_set) > 0:
        return {'status': 'success', 'data': None,
                'message': [f"协方差矩阵异常, 缺少指数: {','.join(list(t_set))}"],
                'info': [f"协方差矩阵异常, 缺少指数: {','.join(list(t_set))}"]}

    covMatrix = other_args["covMatrix"].loc[product_info["基准指数代码"].values,
    product_info["基准指数代码"].values].values

    if np.sum(np.isnan(covMatrix)) > 0:
        return {'status': 'success', 'data': None,
                'message': ["协方差矩阵异常, 存在空值"], 'info': ["协方差矩阵异常, 存在空值"]}
    if not np.all(covMatrix == covMatrix.T):
        return {'status': 'success', 'data': None,
                'message': ["协方差矩阵不对称"], 'info': ["协方差矩阵不对称"]}

    # 判断金额是否小于1块钱，如果是，则判断产品是否都可以卖出
    if amt < 1:
        if np.sum(product_info["是否持仓"].values == True) - np.sum(product_info["是否可卖出"].values == True):
            return {'status': 'success', 'data': None,
                    'message': ["目标资金金额度太低"], 'info': ["目标资金金额度太低"]}
        else:
            return {'status': 'success', 'data': None,
                    'message': ["目标资金金额度太低"], 'info': ["目标资金金额度太低"]}

    # 产品数量
    nPrd = product_info.shape[0]
    IncAmt = product_info["递增金额"].values

    # 起购金额，进行处理：不能赎回的持仓产品，起购金额调整为当前持仓金额
    conditions = (product_info["是否持仓"].values == True) & (product_info["是否可卖出"].values == False)
    product_info["起购金额"] = np.where(conditions, product_info["当前持仓金额"].values,
                                        np.minimum(product_info["起购金额"].values,
                                                   product_info["当前持仓金额"].values))
    MinAmt = product_info["起购金额"].values

    Constraints = []

    # 定义决策变量
    z = cvx.Variable(shape=(nPrd,), name="if_in_portfolio", boolean=True)  # 是否纳入投资组合
    zs = cvx.Variable(shape=(nPrd,), name="if_sell", boolean=True)  # 是否卖出
    zb = cvx.Variable(shape=(nPrd,), name="if_buy", boolean=True)  # 是否买入
    zh = cvx.Variable(shape=(nPrd,), name="if_holding", boolean=True)  # 是否继续持有
    w_inc = cvx.Variable(shape=(nPrd,), name="weight_of_inc")  # 增持的比例
    w_dec = cvx.Variable(shape=(nPrd,), name="weight_of_dec")  # 减持的比例

    # 提取重复的产品ID
    series = product_info.groupby("原始ID")["原始ID"].count()
    series = series[series > 1]
    repeat_product_id = np.array(series.index)
    if repeat_product_id.shape[0] > 0:
        zs_r = cvx.Variable(shape=(repeat_product_id.shape[0],), name="repeatproduct_if_sell", boolean=True)
        zb_r = cvx.Variable(shape=(repeat_product_id.shape[0],), name="repeatproduct_if_buy", boolean=True)
        zh_r = cvx.Variable(shape=(repeat_product_id.shape[0],), name="repeatproduct_if_holding", boolean=True)
    else:
        zs_r = None
        zb_r = None
        zh_r = None

    # 离散型产品
    DMask = product_info["离散型产品"].values
    nDPrd, nCPrd = np.sum(DMask), nPrd - np.sum(DMask)
    if nDPrd > 0:
        a = cvx.Variable(shape=(nDPrd,), name="increment", integer=True)  # 递增倍数
        w_d = np.diag(IncAmt[DMask] / amt) @ a + np.diag(MinAmt[DMask] / amt) @ z[DMask]  # 持仓权重
        Constraints.append(a >= 0)
        Constraints.append(w_d >= 0)
        Constraints.append(np.diag(MinAmt[DMask]) @ z[DMask] + np.diag(IncAmt[DMask]) @ a <= amt * z[DMask])
    else:
        w_d = None
        a = None

    # 连续型产品
    if nCPrd > 0:
        w_c = cvx.Variable(shape=(nPrd - np.sum(DMask),), name="weight")  # 持仓权重
        Constraints.append(w_c >= 0)
        Constraints.append(w_c <= z[~DMask])
        Constraints.append(amt * w_c >= np.diag(MinAmt[~DMask]) @ z[~DMask])
    else:
        w_c = None

    if (nDPrd > 0) and (nCPrd > 0):
        w = cvx.hstack([w_c, w_d])
    elif nCPrd > 0:
        w = w_c
    elif nDPrd > 0:
        w = w_d

    min_amt = other_args.get("min_amt", None)
    max_amt = other_args.get("max_amt", None)

    osc = ObjectScoreCal(w=w, cov_matrix=covMatrix,
                         scoreFuncParas=other_args['scoreFuncParas'],
                         indicatorEstimateResult=other_args['indicatorEstimateResult'],
                         amt=amt, product_info=product_info)

    cons = ConstraintsClass(w, w_c, w_d, w_inc, w_dec, a, z, zb, zs, zh,
                            nCPrd, nDPrd, DMask, amt, min_amt, max_amt, covMatrix, product_info,
                            other_args['scoreFuncParas'], other_args['scoreWeightLevel2'],
                            other_args['indicatorEstimateResult'], other_args['ASSET_CODE_DICT'], osc,
                            repeat_product_id, zb_r, zs_r, zh_r)

    Constraints += cons.base_constraint(w, w_inc, w_dec, zb, zs, zh)

    # 边界约束 - 新版本
    Constraints += cons.budget_constraint(w, other_args.get('weight_sum', 0.99))

    # 持仓金额限制 - 新版本
    Constraints += cons.one_amount_constraint(
        other_args.get('min_amt', None), other_args.get('max_amt', None))

    # 持仓权重限制 - 新版本
    Constraints += cons.class_bound_constraint(
        other_args.get('min_weight', None), other_args.get('max_weight', None))

    # 持仓产品金额约束 - 新版本
    Constraints += cons.hold_amount_constraint(w=w, zb=zb, zs=zs, w_inc=w_inc, w_dec=w_dec)

    # 产品总数量限制 - 新版本
    Constraints += cons.portfolios_products_num_constraint(other_args.get('min_num', None),
                                                           other_args.get('max_num', None))

    # 久期约束 - 新版本
    Constraints += cons.portfolios_duration_constraint(longest_idle_time)

    # 大类资产配置约束 - 新版本
    Constraints += cons.asset_alloc_constraint(target_asset_alloc, other_args['asset_alloc_bias_detail'],
                                               other_args.get('asset_alloc_bias_global', {}))

    Constraints += cons.same_product_turnover_num_constraint(zb, zs)

    obj_return = cvx.minimum(osc.object_return_score(product_info['基准指数收益率'].values, w), 90)
    obj_cash = cvx.minimum(osc.object_cash_score(product_info['大类资产-现金类'].values, w), 90)

    obj_volatility = osc.object_volatility_score_byquad(w, covMatrix)
    if not cvx.Problem(cvx.Maximize(obj_volatility), constraints=Constraints).is_dcp():
        obj_volatility = cvx.minimum(osc.object_volatility_score_byline(w, product_info['基准指数收益率'].values),
                                     90)

    disperse_flag = product_info['三级分类码值'].apply(
        lambda x: True if x not in other_args['disperse_exclude'] else False).values
    obj_disperse = osc.object_disperse_score(w[disperse_flag], z[disperse_flag])

    Obj = 0
    for iFactor, iWeight in obj_alloc.items():
        if iWeight == 0:
            continue
        if '换手费率' == iFactor:
            Obj += iWeight * osc.object_turnover_score(product_info, w, amt)
        elif '流动性' == iFactor:
            Obj += iWeight * obj_cash
            pass
        elif '收益率' == iFactor:
            Obj += iWeight * obj_return
            pass
        elif '波动率' == iFactor:
            Obj += iWeight * obj_volatility
            pass
        elif '分散度' == iFactor:
            if np.sum(disperse_flag) > 0:
                Obj += iWeight * obj_disperse
        elif '产品数量' == iFactor:
            Obj += iWeight * osc.object_product_num_score(w, z)
        elif '久期' == iFactor:
            Obj += iWeight * cvx.minimum(osc.object_duration(w, w_inc, w_dec), 180.0)
            pass
        elif '调仓数量' == iFactor:
            Obj += iWeight * osc.object_turnover_num_score(zb, zs)
        else:
            if method == 1:
                iExpose = (product_info[iFactor].fillna(0) * 1.0).rank(pct=True).values
            elif method == 2:
                iExpose = (product_info[iFactor].fillna(0) * 1.0).rank(method="dense", pct=True).values
            else:
                iExpose = (product_info[iFactor].fillna(0) * 1.0).values
                if np.sum(iExpose) > 0:
                    iExpose = iExpose + np.abs(min(iExpose))
                iExpose = max_min_normalization(np.log(iExpose + 1))
            if iFactor == '兴趣评分':
                iExpose *= 0.01
            Obj += osc.object_common(iExpose, w)

    Mdl = cvx.Problem(cvx.Maximize(Obj), constraints=Constraints)
    solvers = SOLVERS
    solver_status = dict(zip(solvers, [False] * len(solvers)))
    message = []
    info = []
    check_ok = False
    for i, solver in enumerate(solvers):
        start = datetime.now()
        try:
            mdl_args['solver'] = solver
            mdl_args['verbose'] = False

            if cvx.SCIP == solver:
                Mdl.solve(solver=solver, verbose=mdl_args['verbose'], scip_params={'limits/time': 5})
            elif cvx.CPLEX == solver:
                Mdl.solve(solver=solver, verbose=mdl_args['verbose'], cplex_params={"timelimit": 5})
            else:
                Mdl.solve(solver=solver, verbose=mdl_args['verbose'])
            check_ok = True
            if (Mdl.status is not None) and "optimal" in Mdl.status:
                info += [f"{i + 1}号求解器：执行成功，" + Mdl.status + "耗时(s)：" + str(
                    (datetime.now() - start).total_seconds())]
                try:
                    constraint_check = cons.constraint_check(z=z.value,
                                                             zb=zb.value if zb is not None else zb,
                                                             zs=zs.value if zs is not None else zs,
                                                             zh=zh.value if zh is not None else zh,
                                                             zb_r=zb_r.value if zb_r is not None else zb_r,
                                                             zs_r=zs_r.value if zs_r is not None else zs_r,
                                                             zh_r=zh_r.value if zh_r is not None else zh_r,
                                                             a=a.value if a is not None else a,
                                                             w=w.value,
                                                             w_c=w_c.value if w_c is not None else w_c,
                                                             w_d=w_d.value if w_d is not None else w_d,
                                                             w_inc=w_inc.value, w_dec=w_dec.value,
                                                             target_asset_alloc=target_asset_alloc,
                                                             longest_idle_time=longest_idle_time,
                                                             other_args=other_args)
                    check_ok = np.sum(w.value) <= 1.01
                    info += ["组合约束检验状态：" + str(check_ok)]
                    info += ["买卖状态检验结果：" + str(np.all((zb.value + zs.value + zh.value) == 1))]
                    info += ["组合约束检验结果：" + str(constraint_check)]
                except:
                    info += ["组合约束检验：检验执行失败"]
                    print(traceback.format_exc())
                break
            else:
                info += [f"{i + 1}号求解器：执行失败，{str(Mdl.status)}"]
        except Exception as e:
            error_str = e.__str__()
            print(traceback.format_exc())
            print(error_str)
            if ('ibm' in error_str.lower()) or ('cplex' in error_str.lower()):
                info += [f"{i + 1}号求解器 执行抛异常"]
            else:
                info += [f"{i + 1}号求解器 执行抛异常, " + error_str]
            solver_status[solver] = False
            check_ok = False
            continue

    if (Mdl.status is not None) and "optimal" in Mdl.status and check_ok:
        Portfolio = pd.DataFrame({"是否购买": z.value}, index=product_info.index)
        Portfolio["递增倍率"] = np.full(shape=(nPrd,), fill_value=np.nan)
        if np.any(DMask):
            Portfolio["递增倍率"][DMask] = np.abs(a.value)
        Portfolio["持仓权重"] = np.full(shape=(nPrd,), fill_value=np.nan)
        if not np.all(DMask):
            Portfolio["持仓权重"][~DMask] = w_c.value
        Portfolio["持仓金额"] = (Portfolio["持仓权重"] * amt).where(
            pd.notnull(Portfolio["持仓权重"]),
            Portfolio["是否购买"] * product_info["起购金额"] +
            Portfolio["递增倍率"] * product_info["递增金额"])
        Portfolio["持仓权重"] = Portfolio["持仓金额"].where(
            pd.notnull(Portfolio["持仓权重"]),
            Portfolio["持仓金额"] / amt)

        # TODO 处理未知原因多出来的 TMP_XJ_FIXED 产品数置, 全部转为添利私享 (019B310006)
        # 已知的情况是当模型进入“调整数量限制”环节时，会出现 TMP_XJ_FIXED 多一万的情况
        TIANLI_PRODUCT_ID = "019B310006"
        CASH_PRODUCT_ID = "_TMP_XJ_FIXED"
        print(Portfolio.index)
        if CASH_PRODUCT_ID in Portfolio.index:
            if TIANLI_PRODUCT_ID in Portfolio.index:
                Portfolio.loc[
                    Portfolio.index == TIANLI_PRODUCT_ID, "是否购买"] = (
                    min(1.0,
                        Portfolio.loc[Portfolio.index.isin([TIANLI_PRODUCT_ID, CASH_PRODUCT_ID]), "是否购买"].sum()))
                Portfolio.loc[
                    Portfolio.index == TIANLI_PRODUCT_ID, "持仓权重"] = (
                    Portfolio.loc[Portfolio.index.isin([TIANLI_PRODUCT_ID, CASH_PRODUCT_ID]), "持仓权重"].sum())
                Portfolio.loc[
                    Portfolio.index == TIANLI_PRODUCT_ID, "持仓金额"] = (
                    Portfolio.loc[Portfolio.index.isin([TIANLI_PRODUCT_ID, CASH_PRODUCT_ID]), "持仓金额"].sum())
                Portfolio.loc[
                    Portfolio.index == CASH_PRODUCT_ID, "是否购买"] = 0.0
            else:
                Portfolio = Portfolio.rename(index={CASH_PRODUCT_ID: TIANLI_PRODUCT_ID})
        if np.sum(Portfolio["是否购买"] > 0.5):
            Portfolio = Portfolio[Portfolio["是否购买"] > 0.5]
            res_df = pd.merge(Portfolio, product_info,
                              how="left", left_index=True, right_index=True).loc[:, ["起购金额", "递增金额",
                                                                                     "递增倍率", "持仓权重",
                                                                                     "持仓金额"]]

            info += [
                f"disperseLossValue: "
                f"{round(osc.object_disperse_score(w.value[disperse_flag], z.value[disperse_flag], make_object=False), 4)}",
                f"riskLossValue: "
                f"{round(cons.cal_risk_loss_value(w.value[disperse_flag], covMatrix[disperse_flag][:, disperse_flag]), 4)}",
                f"turnoverLossValue: "
                f"{round(osc.cal_turnover_loss_value(product_info, w.value, amt), 4)}",
                f"durationLossValue: "
                f"{round(osc.cal_duration_loss_value(w.value, w_inc.value, w_dec.value), 4)}",
                f"productNumLossValue: "
                f"{round(osc.cal_product_num_loss_value(z.value), 4)}",
                f"基于协方差矩阵的波动率目标是否是DCP: "
                f"{osc.object_volatility_is_dcp(w, covMatrix)}",
                f"最终优化目标类型: "
                f"is_qp={Mdl.is_qp()}, is_dcp={Mdl.is_dcp()}, is_dgp={Mdl.is_dgp()}, is_dpp={Mdl.is_dpp()}"]
            return {"status": "success", "data": res_df, "message": [], "info": info}
    else:
        return {"status": "fail", "data": None, "message": message, "info": info}
