# -*- encoding: utf-8 -*-
"""
@File: A03_Agents.py
@Modify Time: 2025/7/10 10:05       
@Author: Kevin-Chen
@Descriptions: 
"""

import os
import re
import sys
import json
import inspect
import requests
import traceback
from openai import OpenAI
import random  # 导入random模块用于模拟因子效果

# 确保A02_OperatorLibrary在Python路径中，以便inspect可以找到它
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 动态导入A02_OperatorLibrary以获取算子信息
import importlib.util


def _get_operator_descriptions():
    """
    动态读取A02_OperatorLibrary.py文件，提取所有算子函数的名称和文档字符串。
    
    Returns:
        str: 格式化的算子列表及其描述。
    """
    operator_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A02_OperatorLibrary.py")

    if not os.path.exists(operator_file_path):
        return "Error: A02_OperatorLibrary.py not found."

    spec = importlib.util.spec_from_file_location("A02_OperatorLibrary", operator_file_path)
    if spec is None:
        return "Error: Could not load A02_OperatorLibrary spec."

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(traceback.format_exc())
        return f"Error executing A02_OperatorLibrary.py: {e}"

    operators_info = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_") and obj.__module__ == module.__name__:  # 过滤私有函数和非本模块函数
            doc = inspect.getdoc(obj)
            if doc:
                operators_info.append(f"- {name}: {doc.strip()}")
            else:
                operators_info.append(f"- {name}: (No description available)")

    return "\n".join(operators_info)


def call_llm_api(sys_prompt: str, prompt: str, temperature: float = 0.7) -> str:
    """
    调用大模型API接口。

    Args:
        sys_prompt (str): System角色的提示。
        prompt (str): User角色的提示。
        temperature (float): 模型生成时的温度参数，控制随机性。

    Returns:
        str: 大模型返回的文本内容。

    Raises:
        FileNotFoundError: 如果config.json文件不存在。
        KeyError: 如果config.json中缺少必要的配置项。
        Exception: 如果API调用失败或返回非预期结果。
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    model = config.get("LLM_MODEL")
    api_key = config.get("API_KEY")
    base_url = config.get("LLM_URL")

    if not all([model, api_key, base_url]):
        raise KeyError("Missing one or more required keys (LLM_MODEL, API_KEY, LLM_URL) in config.json")

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    if 'lightcode-uis' in base_url:
        # 通过lightcode-ui接口调用大模型API
        print("Using lightcode-ui API...")
        headers = {'Accept': "*/*", 'Authorization': f"Bearer {api_key}", 'Content-Type': "application/json"}
        payload = {"model": model, "messages": messages, "stream": False}

        try:
            resp = requests.post(base_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()  # 检查HTTP响应状态码
            response_data = resp.json()
            # 解析lightcode的返回结果
            json_inner = response_data["choices"][0]["message"]["content"]
            return re.sub(r"^```json\n|```$", "", json_inner.strip(), flags=re.MULTILINE)
        except requests.exceptions.RequestException as e:
            raise Exception(f"lightcode-ui API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response format from lightcode-ui API: {e} Response: {resp.text}")

    else:
        # 通过OpenAI接口调用大模型API
        print("Using OpenAI API...")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            llm_response_content = resp.choices[0].message.content
            # 移除Markdown代码块标记
            if llm_response_content.startswith("```json") and llm_response_content.endswith("```"):
                llm_response_content = llm_response_content[len("```json"): -len("```")].strip()
            return llm_response_content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")


class FinancialMathematicianAgent:
    """金融数学家智能体：负责因子构思、逻辑生成和新算子需求提出。"""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature
        self.operator_descriptions = _get_operator_descriptions()
        self.history_factor_results = []  # 存储历史因子成果摘要

    def propose_factor_or_operator(self, current_evaluation_result: dict = None) -> dict:
        """
        构思新的因子计算逻辑或提出新算子需求。

        Args:
            current_evaluation_result (dict): 当前因子的评估结果。

        Returns:
            dict: 包含因子计算逻辑 (AST, LaTeX) 或新算子需求 (CreateNewCalFunc)。
        """
        sys_prompt = f"""你是一个顶级的金融数学家，精通量化投资和因子模型。你的任务是根据提供的算子库，构思新的、有效的金融因子计算逻辑。你必须严格按照以下JSON格式返回你的构思：

1.  **因子计算逻辑:**
    ```json
    {{
      "des": "对因子逻辑的详细解释，包括其经济学含义和预期效果。",
      "ast": "AST语法树，表示因子计算的步骤，使用算子库中的函数名。例如：{{\"func\": \"add\", \"args\": [{{\"var\": \"close_df\"}}, {{\"func\": \"log\", \"args\": [{{\"var\": \"vol_df\"}}]}}]}}",
      "latex": "AST对应的LaTex数学公式，清晰表达因子计算逻辑。"
    }}
    ```
    AST语法树的结构必须是嵌套的字典，其中包含 'func' (函数名) 或 'var' (变量名)，以及 'args' (参数列表)。参数可以是嵌套的AST结构或变量。
    
2.  **新算子需求:**
    如果你认为现有算子不足以表达你的因子构思，你可以提出一个新算子的需求。请严格按照以下JSON格式返回你的需求：
    ```json
    {{
      "action": "CreateNewCalFunc",
      "function_name": "新函数名",
      "description": "新函数的功能描述，包括输入、输出、数学逻辑等详细信息。",
      "example_usage": "一个使用新函数的代码示例，例如：new_func(data, param1=10)"
    }}
    ```

你可用的算子库如下：
{self.operator_descriptions}

请注意：
- 你的输出必须是有效的JSON格式，且只包含JSON内容，不要有任何额外文字。
- 优先使用现有算子构建因子。只有在现有算子确实无法表达你的构思时，才提出新算子需求。
- 构思的因子应尽可能简洁但有效，避免过度复杂化。
- 确保AST语法树中的函数名和变量名与算子库中提供的名称完全一致。
"""
        user_prompt = """请构思一个新的金融因子。"""

        if self.history_factor_results:
            user_prompt += "\n\n历史因子成果摘要：\n" + json.dumps(self.history_factor_results, indent=2)

        if current_evaluation_result:
            user_prompt += "\n\n当前因子的评估结果：\n" + json.dumps(current_evaluation_result, indent=2)
            user_prompt += "\n请根据这些结果，提出一个改进的因子或一个全新的因子。"

        print("\n--- 金融数学家智能体正在构思... ---")
        llm_response_str = None
        try:
            llm_response_str = call_llm_api(
                sys_prompt=sys_prompt,
                prompt=user_prompt,
                temperature=self.temperature
            )
            # 尝试解析JSON
            response_json = json.loads(llm_response_str)

            # 验证JSON结构
            if "action" in response_json and response_json["action"] == "CreateNewCalFunc":
                required_keys = ["function_name", "description", "example_usage"]
                if not all(key in response_json for key in required_keys):
                    raise ValueError("Invalid CreateNewCalFunc JSON format.")
                print("金融数学家智能体提出了新算子需求。")
                return response_json
            elif "des" in response_json and "ast" in response_json and "latex" in response_json:
                print("金融数学家智能体提出了新的因子计算逻辑。")
                # 可以在这里添加更复杂的AST结构验证
                return response_json
            else:
                raise ValueError("LLM response is not a valid factor or CreateNewCalFunc JSON.")

        except json.JSONDecodeError as e:
            print(f"Error: LLM response is not valid JSON. {e}\nResponse: {llm_response_str}")
            print(traceback.format_exc())
            return {"error": "Invalid JSON response from LLM", "raw_response": llm_response_str}
        except ValueError as e:
            print(f"Error: Invalid LLM response structure. {e}\nResponse: {llm_response_str}")
            print(traceback.format_exc())
            return {"error": "Invalid LLM response structure", "raw_response": llm_response_str}
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            print(f"Response: {llm_response_str}")
            print(traceback.format_exc())
            return {"error": f"LLM API call failed: {e}"}

    def add_history_factor_result(self, factor_result: dict):
        """
        添加历史因子成果摘要，用于LLM的迭代学习。
        """
        # 可以在这里对factor_result进行摘要处理，避免历史信息过长导致token超限
        self.history_factor_results.append(factor_result)


# 示例用法 (仅用于测试)
if __name__ == "__main__":
    # --- 测试 _get_operator_descriptions ---
    print("\n--- Testing _get_operator_descriptions ---")
    ops_desc = _get_operator_descriptions()
    print(ops_desc)

    # --- 金融数学家智能体三轮模拟 ---
    print("\n--- 金融数学家智能体三轮模拟 ---")
    fm_agent = FinancialMathematicianAgent(temperature=0.7)

    for i in range(1, 4):  # 模拟3轮
        print(f"\n--- 第 {i} 轮因子构思 ---")
        current_eval_result = None
        if i > 1:  # 从第二轮开始，模拟评估结果
            # 模拟因子评估结果
            rank_ic = round(random.uniform(0.01, 0.15), 4)  # 模拟Rank IC
            comment = "表现良好" if rank_ic > 0.05 else "表现一般"
            current_eval_result = {
                "factor_name": f"Factor_Round_{i - 1}",
                "rank_ic": rank_ic,
                "t_stat": round(random.uniform(1.5, 3.0), 2),
                "comment": comment
            }
            print(f"模拟评估结果: {current_eval_result}")

        proposed_output = fm_agent.propose_factor_or_operator(current_eval_result)
        print("金融数学家智能体输出:", json.dumps(proposed_output, indent=2, ensure_ascii=False))

        # 将本次构思的因子（如果不是新算子需求）添加到历史记录
        if "des" in proposed_output and "ast" in proposed_output:
            fm_agent.add_history_factor_result({
                "factor_name": f"Factor_Round_{i}",
                "rank_ic": current_eval_result["rank_ic"] if current_eval_result else None,  # 第一次没有评估结果
                "des": proposed_output["des"]
            })
