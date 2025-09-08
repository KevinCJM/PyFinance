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
    sample_input_data = {
        "f": "score_function_fitness",
        "x": [0.0, 0.05, 0.10, 0.20],
        "y": [0.0, 20.0, 60.0, 100.0],
        "func": "volatility_score_function_fitness"
    }

    # 2. 实例化 WealthManage 类
    wealth_manager = WealthManage()

    # 3. 调用目标方法
    result = wealth_manager.score_function_fitness(**sample_input_data)
    print(result)


if __name__ == "__main__":
    run_local_test_with_check()

'''
深入理解 
@/Users/chenjunming/Desktop/CIB_wealth_manage 下面的代码
以 @/Users/chenjunming/Desktop/CIB_wealth_manage/local_test_1.py 为入口的代码, 功能是什么? 
'''