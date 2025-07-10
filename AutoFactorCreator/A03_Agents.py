# -*- encoding: utf-8 -*-
"""
@File: A03_Agents.py
@Modify Time: 2025/7/10 10:05       
@Author: Kevin-Chen
@Descriptions: 
"""

import json
import requests
from openai import OpenAI
import os


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
            # 假设lightcode-ui的响应结构与OpenAI类似
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"lightcode-ui API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected response format from lightcode-ui API: {e}\nResponse: {resp.text}")

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
            return resp.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")


# 示例用法 (仅用于测试)
if __name__ == "__main__":
    # 创建一个假的config.json文件用于测试
    test_config_openai = {
        "LLM_MODEL": "gpt-3.5-turbo",
        "API_KEY": "sk-YOUR_OPENAI_API_KEY",
        "LLM_URL": "https://api.openai.com/v1/chat/completions",
    }
    test_config_lightcode = {
        "LLM_MODEL": "DeepSeek-V3",
        "API_KEY": "eyJhbGciOiJIUzI1NiJ9",
        "LLM_URL": "http://lightcode-uis.hundsun.com:8080/uis/v1/chat/completions",
    }

    # 写入一个测试配置
    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(test_config_openai, f, indent=4)

    print("\n--- Testing OpenAI API (mock) ---")
    try:
        # 实际调用会失败，因为API Key是假的
        response = call_llm_api("You are a helpful assistant.", "Hello, world!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error during OpenAI test: {e}")

    # 写入另一个测试配置
    with open("config.json", 'w', encoding='utf-8') as f:
        json.dump(test_config_lightcode, f, indent=4)

    print("\n--- Testing lightcode-ui API (mock) ---")
    try:
        # 实际调用会失败，因为URL和API Key是假的
        response = call_llm_api("You are a helpful assistant.", "Tell me a joke.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error during lightcode-ui test: {e}")

    # 清理测试文件
    if os.path.exists("config.json"):
        os.remove("config.json")
