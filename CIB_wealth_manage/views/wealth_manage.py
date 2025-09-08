# -*- encoding: utf-8 -*-
"""
@File: wealth_manage.py
@Modify Time: 2025/8/25 14:34       
@Author: Kevin-Chen
@Descriptions: 
"""
from datetime import datetime
from server.base import app, route_dic
from views.base import JsonResBaseView
from config_py import DEBUG
from service import s
import pandas as pd
import numpy as np
import traceback
import os


class WealthManage(JsonResBaseView):

    def test(self, *args, **kwargs):
        return {"msg": "test ok"}

    def calculate_portfolio_performance(self, *args, **kwargs):
        return {"msg": "calculate_portfolio_performance"}

    def calculate_portfolio_performance_daily(self, *args, **kwargs):
        return {"msg": "calculate_portfolio_performance_daily"}

    def product_recom(self, *args, **kwargs):
        return {"msg": "product_recom"}

    def simulate_portfolio_return_daily(self, *args, **kwargs):
        return {"msg": "simulate_portfolio_return_daily"}

    def calculate_healthy_score(self, *args, **kwargs):
        return {"msg": "calculate_healthy_score"}

    def financial_plan_analysis_for_pension(self, *args, **kwargs):
        return {"msg": "financial_plan_analysis_for_pension"}

    def financial_plan_analysis_for_housepurchase(self, *args, **kwargs):
        return {"msg": "financial_plan_analysis_for_housepurchase"}

    def financial_plan_analysis_for_education(self, *args, **kwargs):
        return {"msg": "financial_plan_analysis_for_education"}

    def financial_plan_analysis_for_investment(self, *args, **kwargs):
        return {"msg": "financial_plan_analysis_for_investment"}

    def financial_plan_analysis_for_investment_user(self, *args, **kwargs):
        return {"msg": "financial_plan_analysis_for_investment_user"}

    def investment_portfolio_auto_establish(self, *args, **kwargs):
        return {"msg": "investment_portfolio_auto_establish"}

    def investment_portfolio_individual_auto_establish(self, *args, **kwargs):
        return {"msg": "investment_portfolio_individual_auto_establish"}

    def investment_portfolio_individual_m4_auto_establish_v2(self, *args, **kwargs):
        """
        修复之前的产品递增金额和起购金额参数
        """
        import copy
        # 深拷贝参数，避免修改原始 kwargs
        fix_kwargs = copy.deepcopy(kwargs)

        # 遍历 configs[0] -> products 列表
        for product_info in fix_kwargs.get("configs")[0].get("products", []):
            # 修正 incAmount
            product_info["incAmount"] = product_info.get("incAmt", 0.0)
            # 修正 amount
            product_info["amount"] = product_info.get("badPosAmt", 0.0)

        # 调用旧版的方法，传入修正后的参数
        return self.investment_portfolio_individual_m4_auto_establish(
            *args, **fix_kwargs
        )

    def investment_portfolio_individual_m4_auto_establish(self, *args, **kwargs):
        """
        自动构建投资组合（个人版 M4 模型）并进行相关日志记录与结果输出。

        该函数主要完成以下任务：
        1. 从传入的配置参数中提取产品信息和调试参数；
        2. 调用 [build_investment_portfolio_individual_m4](file:///Users/chenjunming/Desktop/CIB_wealth_manage/service.py#L89-L102) 构建投资组合；
        3. 对生成的投资组合进行去重处理（基于持仓权重差异）；
        4. 将结果封装为统一格式返回；
        5. 若开启调试模式，则将输入参数、优化结果等信息写入日志文件；
        6. 生成调仓前后对比的 CSV 文件用于分析。

        参数:
            *args: 可变位置参数（未使用）。
            **kwargs: 关键字参数，包含以下关键字段：
                configs (list): 配置参数列表，第一个元素为本次处理的配置字典。
                    其中应包括：
                        - products: 产品列表，每个元素为包含 productId 等信息的字典。
                        - userHoldingDict: 用户当前持仓信息。
                        - index: 当前任务索引标识。
                        - debug: 是否开启调试模式。
                        - logPath: 日志保存路径（若 debug 为 True）。

        返回值:
            list: 包含一个字典的结果列表，结构如下：
                {
                    "errorMsg": str,           # 错误信息
                    "index": any,              # 任务索引标识
                    "portfolios": list,        # 去重后的投资组合列表
                    "indicators": list,        # 指标计算结果列表
                    "msgList": list            # 消息列表
                }
        """
        # 初始化结果列表、计数器和投资组合列表
        result = []
        count = 0
        all_portfolios = []

        # 记录开始时间并获取配置参数
        start = datetime.now()
        x = kwargs.get("configs")[0]

        # 打印调试信息：配置参数摘要、产品数量及 ID 列表
        print('==============================')
        print([(key, x[key]) for key in x.keys() if key != 'products'])
        print(f"产品数量: {len(x['products'])}")
        print([v['productId'] for v in x['products']])
        print('时间:', datetime.now())
        print('==============================')
        print('时间:', datetime.now())
        if x.get('debug', DEBUG):
            print('参数:', x)

        count += 1
        print("调用 build_investment_portfolio_individual_m4 方法")
        # 调用build_investment_portfolio_individual_m4方法构建投资组合, 该方法返回投资组合数据、消息、指标和消息列表
        portfolios, msg, indicators, msgList = s.build_investment_portfolio_individual_m4(**x)
        # 初始化空列表用于存储有效的投资组合
        ok_portfolios = []

        # 对生成的投资组合进行去重处理，避免相似组合重复保留
        for cur_portfolio in portfolios:
            b_is_individual = True
            for one_portfolio in all_portfolios:
                df1 = pd.DataFrame(cur_portfolio).set_index('产品ID')
                df2 = pd.DataFrame(one_portfolio).set_index('产品ID')
                df_ = pd.merge(
                    df2[['持仓权重']], df1[['持仓权重']],
                    left_index=True, right_index=True, how='outer'
                )
                df_ = df_.fillna(0.0)
                if np.max(np.abs(df_.values[:, 0] - df_.values[:, 1])) <= 0.02:
                    b_is_individual = False
                    break
            if b_is_individual:
                all_portfolios.append(cur_portfolio)
                ok_portfolios.append(cur_portfolio)

        result.append({
            "errorMsg": msg,
            "index": x.get("index"),
            "portfolios": ok_portfolios,
            "indicators": indicators,
            "msgList": msgList
        })

        print('结果', result)
        print('优化耗时:', (datetime.now() - start).total_seconds())

        # 若开启调试模式且日志路径存在，则记录详细日志和输入输出数据
        if x.get('debug', DEBUG) and False == os.path.exists(x.get('logPath', '')):
            x['logPath'] = r"/Users/chenjunming/Desktop/CIB_wealth_manage/测试文件夹"
        if x.get('debug', DEBUG) and os.path.exists(x.get('logPath', '')):
            import json, pickle, time
            log_path = x.get('logPath', '')
            print(datetime.now())
            dt_str = datetime.now().strftime('%Y%m%d-%H%M%S-%f')

            with open(os.path.join(log_path, "优化日志.log"), "a") as f:
                if len(msg) > 0:
                    for one_msg in msg:
                        f.writelines(f"{x['index']}_{dt_str}" + "," + one_msg)
                        f.writelines('\n')
                else:
                    f.writelines("===================================")
                    f.writelines('\n')
                    f.writelines(f"{x['index']}_{dt_str}" + "," + "seccess")
                    f.writelines('\n')

                    indicators = result[0]['indicators'][0]
                    for one in indicators['display']:
                        f.writelines(str(one))
                        f.writelines('\n')
                    f.writelines(f"msgList: {result[0]['msgList'][0]}")
                    f.writelines('\n')
                    f.writelines(f"指标测算: " + str(indicators['indicatorEstimateResult']))
                    f.writelines('\n')
                    f.writelines(f"打分函数: " + str(indicators['scoreFuncParas']))
                    f.writelines('\n')
                    f.writelines(f"损失函数: " + str(indicators['lossFunctions']))
                    f.writelines('\n')

            try:
                with open(os.path.join(
                        log_path, 'detail', f"{x['index']}_{dt_str}-输入参数.json"), 'w'
                ) as fp:
                    json.dump(x, fp)
            except:
                print(traceback.format_exc())
                with open(os.path.join(
                        log_path, 'detail', f"{x['index']}_{dt_str}-输入参数.pkl"), 'wb'
                ) as fp:
                    pickle.dump(x, fp)
            time.sleep(0.1)

            try:
                import traceback
                product_class = pd.read_csv(
                    r'/Users/chenjunming/Desktop/CIB_wealth_manage/calculate/product_class.csv').dropna()
                product_class['一级分类码值'] = product_class['一级分类码值'].apply(lambda x: str(x).zfill(2))
                product_class['二级分类码值'] = product_class['二级分类码值'].apply(lambda x: str(x).zfill(5))
                product_class['三级分类码值'] = product_class['三级分类码值'].apply(lambda x: str(x).zfill(8))

                # 处理用户原始持仓数据
                if len(x['userHoldingDict']) <= 0 and not x['userHoldingDict']:
                    one_holding_before = pd.DataFrame(
                        data=[], columns=['productId', 'productName', 'asset', 'sjdy', 'weight', 'isRecomm']
                    )
                else:
                    one_holding_before = pd.DataFrame(x['userHoldingDict'])

                if 'isRecomm' not in one_holding_before.columns:
                    one_holding_before['isRecomm'] = False
                one_holding_before['count'] = np.linspace(1, one_holding_before.shape[0],
                                                          one_holding_before.shape[0])
                one_holding_before['productId'] = one_holding_before.apply(
                    lambda x: f"wxwxwxwx_{int(x['count'])}" if pd.isnull(x['productId']) else x['productId'],
                    axis=1
                )
                one_holding_before['weight'] = one_holding_before['asset'] / one_holding_before['asset'].sum()

                # 处理优化后持仓数据
                if len(result[0]['portfolios']) > 0:
                    one_holding_after = pd.DataFrame(result[0]['portfolios'][0])
                else:
                    one_holding_after = pd.DataFrame(data=[], columns=['产品ID', '产品名称', '持仓金额', '持仓权重'])

                # 构建产品 ID 到名称和三级定义的映射字典
                dict1 = dict(zip(list(one_holding_before['productId']), list(one_holding_before['productName'])))
                dict2 = dict(zip([one['productId'] for one in x['products']],
                                 [one['productName'] for one in x['products']]))
                id2name = dict(list(dict1.items()) + list(dict2.items()))

                dict1 = dict(zip(list(one_holding_before['productId']), list(one_holding_before['sjdy'])))
                dict2 = dict(zip([one['productId'] for one in x['products']],
                                 [one['sjdy'] for one in x['products']]))
                id2sjdy = dict(list(dict1.items()) + list(dict2.items()))

                one_holding_before = one_holding_before.set_index('productId')

                one_holding_after['产品名称'] = one_holding_after['产品ID'].apply(lambda x: id2name.get(x, ''))
                one_holding_after = one_holding_after.set_index('产品ID')

                df = pd.DataFrame(id2name.items())
                df.columns = ['产品ID', '产品名称']
                df = df.set_index('产品ID')
                df = pd.merge(df, one_holding_before[['asset', 'weight']],
                              left_index=True, right_index=True, how='left')
                df = pd.merge(df, one_holding_after[['持仓金额', '持仓权重']],
                              left_index=True, right_index=True, how='left')

                # 补齐产品名称
                df = df.reset_index()
                df = df.rename(columns={'index': '产品ID'})
                df['产品名称'] = df['产品ID'].apply(lambda x: id2name.get(x, ''))
                df = df.set_index('产品ID')

                # 补齐三级定义
                df = df.reset_index()
                df['三级分类码值'] = df['产品ID'].apply(lambda x: id2sjdy.get(x, ''))
                df = df.set_index('产品ID')

                # 补齐三级定义名称
                df = pd.merge(
                    df.reset_index(),
                    product_class[['三级分类码值', '一级分类', '二级分类', '三级分类']],
                    on='三级分类码值', how='left'
                )
                df = df.set_index('产品ID')
                df = df.rename(
                    columns={'asset': '原始持仓金额', 'weight': '原始持仓占比',
                             '持仓金额': '建议持仓金额', '持仓权重': '建议持仓占比'}
                )

                df = df.reset_index()
                vipProductId = set(one_holding_before[one_holding_before['isRecomm'] == True].index)

                df['持仓重点推荐'] = df['产品ID'].apply(lambda x: 1 if x in vipProductId else 0)
                df = df.set_index('产品ID')

                df['新增持仓额度'] = df['建议持仓金额'].fillna(0.0) - df['原始持仓金额'].fillna(0.0)
                df['新增持仓占比'] = df['建议持仓占比'].fillna(0.0) - df['原始持仓占比'].fillna(0.0)

                df = df[['产品名称', '原始持仓金额', '原始持仓占比',
                         '建议持仓金额', '建议持仓占比', '新增持仓额度', '新增持仓占比',
                         '三级分类码值', '一级分类', '二级分类', '三级分类', '持仓重点推荐']]

                df_product_hold = pd.read_csv(r"test/views/df_product_hold.csv")

                df = pd.merge(
                    df,
                    df_product_hold.set_index('产品ID').drop(['产品名称', '三级分类码值'], axis=1),
                    left_index=True, right_index=True, how='right'
                )

                df = df.sort_values(by=['建议持仓占比', '原始持仓占比'], ascending=[False, False])

                df.to_csv(os.path.join(
                    log_path, 'detail', f"{x['index']}_{dt_str}-调仓结果.csv"
                ))

            except Exception as e:
                print(traceback.format_exc())
                traceback.print_exc()
                pd.DataFrame().to_csv(os.path.join(
                    log_path, 'detail', f"{x['index']}_{dt_str}-调仓结果生成失败.csv"
                ))

            print('==============================')

        # 打印指标和消息信息
        if result[0]['indicators'] is not None and len(result[0]['indicators']) > 0:
            if len(result[0]['indicators']) > 0:
                indicators = result[0]['indicators'][0]
                for one in indicators['display']:
                    print(one)

            print(result[0]['errorMsg'])
            print(f"msgList: {result[0]['msgList'][0]}")
            print(f"指标测算: ", indicators['indicatorEstimateResult'])
            print(f"打分函数: ", indicators['scoreFuncParas'])
            print(f"损失函数: ", indicators['lossFunctions'])

        return result

    def score_function_fitness(self, *args, **kwargs):
        return s.score_function_fitness(*args, **kwargs)

    def kyc_cpk_select(self, *args, **kwargs):
        return {"msg": "kyc_cpk_select"}

    def backtest(self, *args, **kwargs):
        return {"msg": "backtest"}


if __name__ != '__main__':
    app.add_url_rule(
        route_dic['wealth_manage'],
        endpoint=None,
        view_func=WealthManage.as_view('wealth_manage')
    )

if __name__ == '__main__':
    pass
