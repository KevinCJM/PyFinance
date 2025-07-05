import json
import traceback
import os
import httpx
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from openai import OpenAI
from bs4 import BeautifulSoup
import time
import random

# --- 配置中心 ---
CONFIG = {
    "login_url": "https://dd.gildata.com/#/login",
    "table_to_search": [
        "HK_SecuMain", "SecuMain", "MF_FundArchives", "MF_FundType", "MF_Transformation",
        "MF_KeyStockPortfolio", "MF_QDIIPortfolioDetail", "MF_BondPortifolioDetail", "MF_QDIIPortfolioDetail",
        "MF_FundPortifolioDetail", "MF_QDIIPortfolioDetail", "MF_BalanceSheetNew", "MF_BondPortifolioStru",
        "MF_AssetAllocationNew", "MF_StockPortfolioDetail", "LC_DIndicesForValuation"
    ],
    "api_key": "sk-a221b0c62c8a460693fe00a627d4598e",
    "base_url": "https://api.deepseek.com",
    "model_name": "deepseek-chat",
    "window_size": (1920, 1080),
    "explicit_wait_timeout": 20,
    "final_observe_time": 5,
    "element_cache_file": "element_cache.json",
    "output_json_file": "table_definitions.json",
    "min_sleep_time": 0.5,
    "max_sleep_time": 2.5,
}


# --- 记忆/缓存/数据库 功能 ---
def load_json_file(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_json_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def random_sleep():
    sleep_time = random.uniform(CONFIG["min_sleep_time"], CONFIG["max_sleep_time"])
    print(f"😴 休眠 {sleep_time:.2f} 秒...")
    time.sleep(sleep_time)


# --- 浏览器与元素操作 ---
def launch_browser():
    options = webdriver.ChromeOptions()
    if CONFIG.get("window_size"):
        width, height = CONFIG["window_size"]
        options.add_argument(f"--window-size={width},{height}")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)


def find_element_safely(driver, by, value, timeout=CONFIG["explicit_wait_timeout"]):
    try:
        wait = WebDriverWait(driver, timeout)
        by_obj = getattr(By, by.upper())
        return wait.until(EC.presence_of_element_located((by_obj, value)))
    except Exception:
        print(traceback.format_exc())
        return None


# --- 数据提取与AI整理模块 ---
def simplify_html(html_source):
    soup = BeautifulSoup(html_source, 'html.parser')
    for tag in soup(['script', 'style', 'link', 'meta']):
        tag.decompose()
    body = soup.find('body')
    if body:
        return str(body.prettify())
    return ""


def get_locator_from_ai(html_source, element_description):
    print("🤖 正在精简HTML以适应模型上下文...")
    simplified_html = simplify_html(html_source)
    if len(simplified_html) > 200000:
        print(f"❌ 精简后的HTML仍然过长 ({len(simplified_html)} chars)，跳过API调用。")
        return None

    print("🤖 正式调用大模型API进行分析...")
    try:
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"], http_client=http_client)
        system_prompt = (
            "You are an expert web automation assistant. Your task is to analyze HTML source code "
            "and return a single, precise, and robust Selenium locator for a requested element. "
            "You must return the result as a JSON object with two keys: 'by' and 'value'. "
            "The 'by' key must be one of the following strings: 'ID', 'NAME', 'CLASS_NAME', 'TAG_NAME', 'LINK_TEXT', 'PARTIAL_LINK_TEXT', 'CSS_SELECTOR', 'XPATH'. "
            "The 'value' key is the corresponding locator string."
        )
        user_prompt = (
            f"Based on the following HTML, find the locator for the '{element_description}'.\n\n"
            f"HTML:\n```html\n{simplified_html}\n```\n\n"
            "Return only the JSON object."
        )
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False
        )
        response_text = response.choices[0].message.content
        print(f"🤖 大模型返回原始结果: {response_text}")
        json_part = response_text[response_text.find('{'):response_text.rfind('}') + 1]
        locator = json.loads(json_part)
        if isinstance(locator, dict) and 'by' in locator and 'value' in locator:
            return locator
        else:
            print("❌ 大模型返回的JSON格式不正确。")
            return None
    except Exception as e:
        print(f"❌ 调用大模型API时发生错误: {e}")
        print(traceback.format_exc())
        return None


def scrape_table_details(driver):
    """从表详情页HTML中抓取所有信息(v14.4 终极修正提取与格式)"""
    print("🔎 正在使用Selenium直接提取页面元素...")
    scraped_data = {"basic_info": {}, "columns_data": [], "notes_map": {}}

    # --- 终极修正 1: 使用更精确的XPath提取基本信息 ---
    # 尝试提取“表中文名”及其他基本信息
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # 查找包含中文表名的span标签
        chinese_name_span = soup.find('span', {'ng-bind-html': re.compile(r'table\.tableChiName')})
        if chinese_name_span:
            scraped_data["basic_info"]["tableChiName"] = chinese_name_span.get_text(strip=True)
        else:
            print("⚠️ 未能提取到'表中文名'。")

        # 提取 description
        description_span = soup.find('span', {'ng-bind-html': re.compile(r'table\.description')})
        if description_span:
            scraped_data["basic_info"]["description"] = description_span.get_text(strip=True)
        else:
            print("⚠️ 未能提取到'description'。")

        # 提取 tableUpdateTime
        table_update_time_span = soup.find('span', {'ng-bind': 'table.tableUpdateTime'})
        if table_update_time_span:
            scraped_data["basic_info"]["tableUpdateTime"] = table_update_time_span.get_text(strip=True)
        else:
            print("⚠️ 未能提取到'tableUpdateTime'。")

        # 提取 key (业务唯一性)
        key_span = soup.find('span', {'ng-bind': "index.columnName || '无'"})
        if key_span:
            scraped_data["basic_info"]["key"] = key_span.get_text(strip=True)
        else:
            print("⚠️ 未能提取到'key'。")

    except Exception as e:
        print(traceback.format_exc())
        print(f"⚠️ 提取基本信息时发生错误: {e}")

    # 2. 提取列信息 (逻辑不变)
    try:
        column_table_element = driver.find_element(By.CSS_SELECTOR, 'table.table-column.table-interval-bg')
        column_table_html = driver.execute_script("return arguments[0].outerHTML;", column_table_element)
        soup_column_table = BeautifulSoup(column_table_html, 'html.parser')

        headers = [th.text.strip() for th in soup_column_table.find('thead').find_all('th')]
        # 修正：ng-repeat可能在tbody上，而不是tr上，查找所有tr更稳妥
        rows = soup_column_table.find('tbody').find_all('tr')

        for row in rows:
            cells = row.find_all('td')
            if not cells:
                continue  # 跳过空行或表头内的行
            col_dict = {}
            for i, header in enumerate(headers):
                if i < len(cells):
                    col_dict[header] = cells[i].text.strip()
            if col_dict:  # 确保不是空字典
                scraped_data["columns_data"].append(col_dict)
    except Exception as e:
        print(traceback.format_exc())
        print(f"⚠️ 未能使用 Selenium 定位到列表格或提取列数据。错误: {e}")

    # --- 核心修正 2: 修正备注信息提取 ---
    try:
        remark_table_element = driver.find_element(By.CSS_SELECTOR, 'table.table-remark')
        remark_table_html = driver.execute_script("return arguments[0].outerHTML;", remark_table_element)
        soup_remark_table = BeautifulSoup(remark_table_html, 'html.parser')

        rows = soup_remark_table.find('tbody').find_all('tr')
        for row in rows:
            # 使用BeautifulSoup的find方法，而不是Selenium的find_element
            cells = row.find_all('td')
            if len(cells) == 2:
                # 假设第一列是key，第二列是value
                key = cells[0].text.strip().replace('[', '').replace(']', '').strip()
                value = cells[1].text.strip()
                if key:  # 确保key不为空
                    scraped_data["notes_map"][key] = value
    except Exception as e:
        print(traceback.format_exc())
        print(f"⚠️ 未能使用 Selenium 定位到备注表格或提取备注数据。错误: {e}")

    print(f"✅ 抓取完成：找到 {len(scraped_data['columns_data'])} 列数据，{len(scraped_data['notes_map'])} 条备注。")
    return scraped_data


def organize_data_locally(table_name, scraped_data):
    """
    使用纯Python代码将抓取的数据整理成最终的JSON结构。
    这是对 organize_data_with_ai 函数的直接替代。
    """
    print("🐍 正在使用本地代码进行数据整合与清洗...")

    # 1. 预处理列数据：将备注ID替换为实际备注内容
    processed_columns = []
    notes_map = scraped_data.get("notes_map", {})
    for col in scraped_data.get("columns_data", []):
        processed_col = col.copy()  # 创建副本以避免修改原始数据
        remark_key = processed_col.get("备注")
        if remark_key and remark_key in notes_map:
            processed_col["备注"] = notes_map[remark_key]
        processed_columns.append(processed_col)

    # 2. 构建最终的JSON对象
    final_table_definition = {
        "tableName": table_name,
        "tableChiName": scraped_data.get("basic_info", {}).get("tableChiName", ""),
        "description": scraped_data.get("basic_info", {}).get("description", ""),
        "tableUpdateTime": scraped_data.get("basic_info", {}).get("tableUpdateTime", ""),
        "key": scraped_data.get("basic_info", {}).get("key", ""),
        "columns": processed_columns
    }

    print("✅ 本地数据整理完成。")
    return final_table_definition


def organize_data_with_ai(table_name, scraped_data):
    print("🤖 正在调用大模型进行数据整合与清洗...")
    try:
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"], http_client=http_client)

        # --- 终极修正 2: 固化对AI的指令，强制其遵循JSON格式 ---
        system_prompt = (
            "You are a data structuring expert. Your job is to assemble raw, pre-scraped data parts into a final, clean JSON object representing a database table definition. "
            "You MUST strictly follow the output format specified in the user prompt."
        )

        # 预处理columns，替换备注
        processed_columns = []
        for col in scraped_data["columns_data"]:
            processed_col = col.copy()
            remark_key = processed_col.get("备注")
            if remark_key and remark_key in scraped_data["notes_map"]:
                processed_col["备注"] = scraped_data["notes_map"][remark_key]
            processed_columns.append(processed_col)

        user_prompt = (
                f"Generate a JSON object for the table '{table_name}' with the following structure and data.\n"
                f"Use the provided basic_info, columns_data, and notes_map to populate the fields.\n"
                f"Ensure that the '备注' field in each column object contains the full, replaced remark text from notes_map.\n\n"
                "```json\n"
                f"    \"{table_name}\": {{\n"
                "        \"tableName\": \"{table_name}\",\n"
                "        \"tableChiName\": \"" + scraped_data["basic_info"].get("tableChiName", "") + "\",\n"
                                                                                                      "        \"status\": \"" +
                scraped_data["basic_info"].get("status", "") + "\",\n"
                                                               f"        \"columns\": {json.dumps(processed_columns, indent=4, ensure_ascii=False).replace('\n', '\n').replace('"', '"')}\n"
                                                               "    }}\n"
                                                               "```\n"
                                                               "Return ONLY the final, assembled JSON object and nothing else. Do NOT add any extra text or explanations."
        )
        response = client.chat.completions.create(
            model=CONFIG["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False
        )
        response_text = response.choices[0].message.content
        json_part = response_text[response_text.find('{'):response_text.rfind('}') + 1]
        return json.loads(json_part)[table_name]
    except Exception as e:
        print(f"❌ 调用大模型进行数据整理时发生错误: {e}")
        print(traceback.format_exc())
        return None


# --- 核心业务逻辑 ---
def process_table_details_page(driver, table_name):
    """处理表详情页，提取、整理并保存数据"""
    print("--- 导航到详情页成功，开始数据提取流程 --- ")
    scraped_data = {"columns_data": []}  # 初始化为空，确保即使失败也有此键
    table_row_locator = (By.CSS_SELECTOR, "table.table-column.table-interval-bg tbody tr")

    for attempt in range(3):  # 最多尝试3次
        try:
            wait = WebDriverWait(driver, CONFIG["explicit_wait_timeout"])
            print(f"⏳ 尝试 {attempt + 1}/3: 正在等待元素出现: {table_row_locator}")
            wait.until(EC.presence_of_element_located(table_row_locator))
            random_sleep()  # 增加随机等待，确保页面完全渲染
            print("--- 列表格内容已加载，开始抓取数据 --- ")

            scraped_data = scrape_table_details(driver)
            if scraped_data and len(scraped_data.get("columns_data", [])) >= 2:
                print(f"✅ 尝试 {attempt + 1}/3: 成功抓取到 {len(scraped_data['columns_data'])} 条列信息。")
                break  # 成功抓取到足够数据，跳出循环
            else:
                print(
                    f"⚠️ 尝试 {attempt + 1}/3: 抓取到的列信息不足2条 ({len(scraped_data.get('columns_data', []))} 条)。")
                if attempt < 2:  # 如果不是最后一次尝试，则刷新页面重试
                    print("🔄 刷新页面并重试...")
                    driver.refresh()
                    random_sleep()  # 刷新后等待页面重新加载
                else:
                    print("❌ 达到最大重试次数，将使用当前抓取到的数据。")
        except Exception as e:
            print(f"❌ 尝试 {attempt + 1}/3: 在等待或抓取过程中发生错误: {e}")
            print(traceback.format_exc())
            if attempt < 2:
                print("🔄 刷新页面并重试...")
                driver.refresh()
                random_sleep()
            else:
                print("❌ 达到最大重试次数，无法继续。")
                # 调试步骤：保存当前页面HTML，以便分析
                debug_file = "debug_page_source.html"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                print(f"ℹ️ 为了便于调试，当前的页面HTML已保存到: {os.path.abspath(debug_file)}")
                return  # 彻底失败，返回

    if not scraped_data or not scraped_data.get("columns_data"):
        print("❌ 最终未能从页面抓取到任何列信息，无法继续。")
        return

    final_table_json = organize_data_locally(table_name, scraped_data)
    if final_table_json:
        print("💾 正在以增量/更新模式保存数据...")
        database = load_json_file(CONFIG["output_json_file"])
        database[table_name] = final_table_json
        save_json_file(database, CONFIG["output_json_file"])
        print(f"✅ 成功将表 '{table_name}' 的定义保存到 {CONFIG['output_json_file']}。")
    else:
        print("❌ AI未能成功整理数据，本次不保存。")


def login_and_search(driver):
    driver.get(CONFIG["login_url"])
    cache = load_json_file(CONFIG["element_cache_file"])
    input("请手动完成登录后，回到终端按回车继续...")

    search_box_locator = cache.get("search_box")
    search_box = None
    if search_box_locator:
        print("🧠 正在尝试从记忆中定位搜索框...")
        search_box = find_element_safely(driver, search_box_locator["by"], search_box_locator["value"], timeout=3)

    if not search_box:
        print("🤔 记忆失效或不存在，启动大模型智能发现模式...")
        html = driver.page_source
        search_box_locator = get_locator_from_ai(html, "the main search input box for table names")
        if search_box_locator:
            print(f"🤖 AI发现搜索框定位器: {search_box_locator}")
            cache["search_box"] = search_box_locator
            save_json_file(cache, CONFIG["element_cache_file"])
            search_box = find_element_safely(driver, search_box_locator["by"], search_box_locator["value"])
        else:
            print("❌ 智能发现失败，无法找到搜索框。")
            return

    if not search_box:
        print("❌ 最终未能定位到搜索框，程序终止。")
        return

    print("✅ 成功定位到搜索框。")

    for table_name in CONFIG["table_to_search"]:
        print(f"开始查找表：{table_name}")
        search_box.clear()
        search_box.send_keys(table_name)
        search_box.send_keys(Keys.ENTER)

        print(f"正在搜索 '{table_name}' 的链接...")
        try:
            wait = WebDriverWait(driver, CONFIG["explicit_wait_timeout"])
            # 定位到我们想要点击的那个精确的链接
            link_xpath = f"//a[contains(@href, '/tableShow/') and normalize-space(.)='{table_name}']"
            link_element = wait.until(EC.presence_of_element_located((By.XPATH, link_xpath)))

            # --- 直接提取href并导航 ---
            target_url = link_element.get_attribute('href')
            print(f"✅ 成功定位链接，提取到目标URL: {target_url}")

            if not target_url:
                print("❌ 无法从链接元素中提取到href属性，程序终止。")
                continue  # 继续处理下一个表

            print("🚀 正在直接导航到目标URL...")
            driver.get(target_url)

            # 直接导航后，我们仍然需要等待新视图的标志性元素出现
            print("⏳ 正在等待详情页加载...")
            details_page_identifier = (By.CSS_SELECTOR, "table.table-column.table-interval-bg")
            wait.until(EC.presence_of_element_located(details_page_identifier))
            print("✅ 详情页加载成功。")

            # 现在可以安全地调用详情页处理函数
            process_table_details_page(driver, table_name)
            print(f"--- 表 '{table_name}' 数据提取流程结束 ---")

        except Exception as e:
            print(f"⚠️ 在查找、导航或处理 '{table_name}' 时失败: {e}")
            print(traceback.format_exc())
            # 保存失败时的页面快照，以便调试
            debug_file = f"debug_page_source_final_{table_name}.html"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"ℹ️ 为了便于调试，最终失败时的页面HTML已保存到: {os.path.abspath(debug_file)}")
            print(traceback.format_exc())
        finally:
            # 每次处理完一个表后，返回到搜索页面，以便搜索下一个表
            driver.get(CONFIG["login_url"])
            # 重新定位搜索框，因为页面可能刷新了
            search_box = find_element_safely(driver, search_box_locator["by"], search_box_locator["value"])
            if not search_box:
                print("❌ 重新定位搜索框失败，无法继续处理后续表。")
                break  # 退出循环
            random_sleep()  # 在处理下一个表之前休眠随机时间


def main():
    driver = None
    try:
        driver = launch_browser()
        login_and_search(driver)
        print(f"操作完成，页面将保持打开{CONFIG['final_observe_time']} 秒... ")
        time.sleep(CONFIG['final_observe_time'])
    except Exception as e:
        print(f"程序发生意外错误: {e}")
        print(traceback.format_exc())
    finally:
        if driver:
            print("关闭浏览器。")
            driver.quit()


if __name__ == "__main__":
    main()
