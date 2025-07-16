# -*- encoding: utf-8 -*-
"""
@File: B02_AgentTools.py
@Modify Time: 2025/7/16 10:00
@Author: Kevin-Chen
@Descriptions: 提供给智能体使用的公用工具, 包括: 大模型调用, 文件操作等。
"""
import os
import re
import json
import requests
from openai import OpenAI

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
    # 假设config.json位于项目根目录
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
        resp = None
        try:
            resp = requests.post(base_url, headers=headers, data=json.dumps(payload))
            resp.raise_for_status()
            response_data = resp.json()
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
            if llm_response_content.startswith("```json") and llm_response_content.endswith("```"):
                llm_response_content = llm_response_content[len("```json"): -len("```")].strip()
            return llm_response_content
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {e}")