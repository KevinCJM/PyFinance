import json
import os
import sys
from pprint import pprint

# --- Setup sys.path ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from views.wealth_manage import WealthManage
# 导入约束检查类
from calculate.opti_functions import ConstraintsCheck


def run_local_test_with_check():
    """
    本地测试函数，增加了失败后的约束检查逻辑
    """
    # 1. 准备输入数据 (与之前相同)
    sample_input_data = {
        "configs": [
            {"cpkAmount": 1000000.0,  # 总可用投资金额??????
             "matrixBaseCodeArray": [  # 9个可投资产品代码?????
                 "885000.WI", "885001.WI",
                 "XYDDQCK", "885007.WI",
                 "885008.WI", "XYHQCK",
                 "AU9999", "H11022", "H11025.CSI"
             ],
             "clazzCfg": {  # 五个战略层配置的目标权重
                 "clazz01": 0.05,
                 "clazz02": 0.10,
                 "clazz03": 0.15,
                 "clazz04": 0.60,
                 "clazz05": 0.10
             },
             "cpkAllowedCloseDay": 657,  # 组合允许的最大封闭期天数(所谓的久期)
             "isFineTuning": False,  # 是否进行精细调优
             "scoreWeight": [0.2, 0.5, 0.3],  # 多目标优化权重（风险、收益、稳健性权重）?????
             "matrix": [  # 9个可投资产品的协方差矩阵?????
                 [0.053844, 0.049901, 0.0, 0.009635, 0.001002, 0.0, 0.00239, 0.041255, -9.0E-6],
                 [0.049901, 0.046542, 0.0, 0.008971, 9.37E-4, 0.0, 0.002445, 0.038411, -9.0E-6],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.009635, 0.008971, 0.0, 0.002841, 5.58E-4, 0.0, 6.04E-4, 0.007424, 7.0E-6],
                 [0.001002, 9.37E-4, 0.0, 5.58E-4, 3.71E-4, 0.0, 1.68E-4, 7.92E-4, 6.0E-6],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.00239, 0.002445, 0.0, 6.04E-4, 1.68E-4, 0.0, 0.023622, 0.002354, 3.0E-6],
                 [0.041255, 0.038411, 0.0, 0.007424, 7.92E-4, 0.0, 0.002354, 0.032701, -6.0E-6],
                 [-9.0E-6, -9.0E-6, 0.0, 7.0E-6, 6.0E-6, 0.0, 3.0E-6, -6.0E-6, 3.0E-6]
             ],
             "products": [  # 9个可投资产品的详细信息
                 {
                     "incAmt": 1.00,
                     "amount": 0.00,
                     "productId": "019B310006",
                     "worstRate": 0.016315999999999997,
                     "level": 5.0,
                     "ywlxdm": "LC",
                     "bestRate": 0.027684,
                     "incAmount": 1.00,
                     "productName": "添利新私享",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 0.00,
                     "zgcyje": 10000000.00,
                     "rjdy": "01002",
                     "sjdy": "01002002",
                     "closeDays": 1.0,
                     "risk": "1",
                     "isRecomm": False,
                     "clazz": "01",
                     "yjdy": "01",
                     "fixProductId": "019B310006$^a002c30439c4bb3af4bc0823ee633b0",
                     "baseRate": 0.022,
                     "baseCode": "H11025.CSI"
                 },
                 {
                     "incAmt": 100.00,
                     "amount": 100.00,
                     "productId": "03000480",
                     "worstRate": -0.28704799999999997,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.453048,
                     "incAmount": 100.00,
                     "productName": "东方红新动力混合",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 100.00,
                     "zgcyje": None,
                     "rjdy": "03003",
                     "sjdy": "03003003",
                     "closeDays": 1.0,
                     "risk": "4",
                     "isRecomm": False,
                     "clazz": "03",
                     "yjdy": "03",
                     "fixProductId": "03000480$^0f867f120ab14686b25b520d01777bbc",
                     "baseRate": 0.083,
                     "baseCode": "H11022"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03000586",
                     "worstRate": -0.359264,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.5458639999999999,
                     "incAmount": 1000.00,
                     "productName": "景顺长城中小创精选股票",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "04002",
                     "sjdy": "04002001",
                     "closeDays": 1.0,
                     "risk": "4",
                     "isRecomm": True,
                     "clazz": "04",
                     "yjdy": "04",
                     "fixProductId": "03000586$^58f01677f51242e9aeaa6cf60256d42f",
                     "baseRate": 0.0933,
                     "baseCode": "885000.WI"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03000979",
                     "worstRate": -0.359264,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.5458639999999999,
                     "incAmount": 1000.00,
                     "productName": "景顺长城沪港深精选股票",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "04002",
                     "sjdy": "04002001",
                     "closeDays": 1.0,
                     "risk": "4",
                     "isRecomm": False,
                     "clazz": "04",
                     "yjdy": "04",
                     "fixProductId": "03000979$^724d0412bc7841d5bb3a8899bc21a594",
                     "baseRate": 0.0933,
                     "baseCode": "885000.WI"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03100058",
                     "worstRate": 0.011816000000000004,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.072184,
                     "incAmount": 1000.00,
                     "productName": "富国产业债债券",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "02004",
                     "sjdy": "02004001",
                     "closeDays": 1.0,
                     "risk": "3",
                     "isRecomm": False,
                     "clazz": "02",
                     "yjdy": "02",
                     "fixProductId": "03100058$^7ab36c0688124cb2abc207b75110b519",
                     "baseRate": 0.042,
                     "baseCode": "885008.WI"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 1.00,
                     "productId": "03110008",
                     "worstRate": -0.062016,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.15201599999999998,
                     "incAmount": 1.00,
                     "productName": "易方达稳健收益债券B",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 1.00,
                     "zgcyje": None,
                     "rjdy": "02004",
                     "sjdy": "02004004",
                     "closeDays": 1.0,
                     "risk": "3",
                     "isRecomm": False,
                     "clazz": "02",
                     "yjdy": "02",
                     "fixProductId": "03110008$^cbd2637949c144deb910fac602cad9ff",
                     "baseRate": 0.045,
                     "baseCode": "885007.WI"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03166009",
                     "worstRate": 0.026,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.026,
                     "incAmount": 1000.00,
                     "productName": "中欧新动力混合(LOF)A",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "02001",
                     "sjdy": "02001001",
                     "closeDays": 1.0,
                     "risk": "4",
                     "isRecomm": False,
                     "clazz": "02",
                     "yjdy": "02",
                     "fixProductId": "03166009$^cb07ca093ee741fd844c392874ddf4d4",
                     "baseRate": 0.026,
                     "baseCode": "XYDDQCK"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03260101",
                     "worstRate": -0.33320400000000006,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.509204,
                     "incAmount": 1000.00,
                     "productName": "景顺长城优选混合",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "03003",
                     "sjdy": "03003004",
                     "closeDays": 1.0,
                     "risk": "4",
                     "isRecomm": False,
                     "clazz": "03",
                     "yjdy": "03",
                     "fixProductId": "03260101$^7af6437776ef46cf2ae9462cf68831e90",
                     "baseRate": 0.088,
                     "baseCode": "885001.WI"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03320013",
                     "worstRate": -0.264776,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.364776,
                     "incAmount": 1000.00,
                     "productName": "诺安全球黄金（人民币份额）",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "05001",
                     "sjdy": "05001003",
                     "closeDays": 1.0,
                     "risk": "5",
                     "isRecomm": False,
                     "clazz": "05",
                     "yjdy": "05",
                     "fixProductId": "03320013$^654934a8e98c486b90ef09f0984aa94d7",
                     "baseRate": 0.05,
                     "baseCode": "AU9999"
                 },
                 {
                     "incAmt": 2000.00,
                     "amount": 5000.00,
                     "productId": "03485111",
                     "worstRate": -0.062016,
                     "level": 5.0,
                     "ywlxdm": "1",
                     "bestRate": 0.15201599999999998,
                     "incAmount": 1000.00,
                     "productName": "工银瑞信双利债券A",
                     "openEndDate": "20991231",
                     "privateVisible": 0,
                     "badPosAmt": 5000.00,
                     "zgcyje": None,
                     "rjdy": "02004",
                     "sjdy": "02004004",
                     "closeDays": 1.0,
                     "risk": "3",
                     "isRecomm": False,
                     "clazz": "02",
                     "yjdy": "02",
                     "fixProductId": "03485111$^042b26bed1c34b9da0e83d31ede951b1",
                     "baseRate": 0.045,
                     "baseCode": "885007.WI"
                 }
             ],
             "globalTurn": {  # 全局约束条件
                 "lowestScoreMaximize": 1,
                 "nearProductChange": 1,
                 "notesId": "000589",
                 "turnOverMinimize": 1,
                 "customInterestSumMaximize": 1,
                 "turnOverNumMinimize": 1,
                 "itemScoreThreshold": 80,
                 "levelSumMaximize": 1,
                 "totalScoreMaximize": 1,
                 "diffScoreThreshold": 0,
                 "recommSumMaximize": 1,
                 "totalScoreThreshold": 80
             },
             "scoreArgs": [  # 打分函数的参数矩阵???? (现金类占比评分', 预期收益率评分, 预期波动率评分, 分散度评分)
                 [68.046499053747, 60.655737094694686, 0.08495456570653678, -5.7670313329984397, 1.0, 0.0],
                 [100.0, 0.99, 1.4007403221237584E-6, 5.0, 1.0, 0.0],
                 [86.79615590870546, 1.259142491702407, -0.15330627312862127, 86.79679922948152, 1.0, 0.0],
                 [100.0, 0.0, 1.0, 2.75, 1.0, 0.0]
             ],
             "pref": None,
             "ObjAlloc": None,
             "scoreFunction": [  # 用于评分的函数集合 (自定义打分函数)
                 "atan", "morgan_mercer_flodin_customreturn", "tanh_neg", "morgan_mercer_flodin"],
             "holdAfterYearStdWeight": 0.7,
             "rateLimit": [0.085000000000001, 0.091],
             "maxAllowedDev": {  # 各大类资产的最大允许偏离度
                 "clazz01": 0.20,
                 "clazz02": 0.20,
                 "clazz03": 0.20,
                 "clazz04": 0.20,
                 "clazz05": 0.20
             },
             "clazzRate": {  # 各大类资产的实际配置比例?????
                 "clazz04": [-0.34806000000000004, 0.52806],
                 "clazz05": [-0.164472, 0.289072],
                 "clazz01": [0.016315999999999997, 0.027684],
                 "clazz02": [-0.016399999999999998, 0.0914],
                 "clazz03": [-0.28704799999999997, 0.453048]
             },
             "index": "0011_999000",
             "userHoldingDict": [{  # 用户当前已持有的资产配置，示例里只有一个「现金」资产
                 "amount": 0.01,
                 "baseCode": "XYHQCK",
                 "productId": "_TMP_XJ_FIXED",
                 "remain_term": 0,
                 "level": -1.0,
                 "ywlxdm": None,
                 "incAmount": 0.01,
                 "productName": "现金",
                 "privateVisible": 0,
                 "is_can_sell": True,
                 "cpcsjg": 1,
                 "baseRate": 0.0025,
                 "is_can_buy": True,
                 "zgcyje": 1.0E12,
                 "sjdy": "01001001",
                 "closeDays": None,
                 "risk": 1,
                 "isRecomm": 0,
                 "is_recomm": 0,
                 "asset": 0.0
             }],
             "whitePlanType": "free",
             "privateVisible": 1,
             "prodCrWeight": 0.3,  # 单个产品最大权重?????
             "max_weight": {  # 各大类资产的最大持仓比例限制
                 "clazz01": 0.35,
                 "clazz02": 0.35,
                 "clazz03": 0.35,
                 "clazz04": 0.35,
                 "clazz05": 0.35
             },
             "rf": 0.0175,
             "productsContRangeInOneCpk": [5, 12],
             "risk": 6,
             "portfolio_num": 1,
             "intereScoreDict": {}
             }
        ],
        "f": "investment_portfolio_individual_m4_auto_establish_v2"
    }

    # 2. 实例化 WealthManage 类
    wealth_manager = WealthManage()

    # 3. 调用目标方法
    print("---" + " 开始调用优化模型 " + "---")
    result = wealth_manager.investment_portfolio_individual_m4_auto_establish(**sample_input_data)
    print("---" + " 模型调用结束 " + "---\n")
    print(result)


if __name__ == "__main__":
    run_local_test_with_check()
