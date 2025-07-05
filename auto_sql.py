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

# --- 配置中心 ---
CONFIG = {
    "login_url": "https://dd.gildata.com/#/login",
    "table_to_search": "SecuMain",
    "api_key": "xxxx",
    "base_url": "https://api.deepseek.com",
    "model_name": "deepseek-chat",
    "window_size": (1920, 1080),
    "explicit_wait_timeout": 20,
    "final_observe_time": 5,
    "element_cache_file": "element_cache.json",
    "output_json_file": "table_definitions.json",
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
    """从表详情页HTML中抓取所有信息(v14.0 Selenium直接提取版)"""
    print("🔎 正在使用Selenium直接提取页面元素...")
    scraped_data = {"basic_info": {}, "columns_data": [], "notes_map": {}}

    # 1. 提取基本信息 (使用Selenium直接提取)
    basic_info_labels = {
        '表中文名': 'tableChineseName',
        '状态': 'status'
    }
    for label_text, key_name in basic_info_labels.items():
        try:
            # 查找包含标签文本的元素，然后获取其兄弟节点
            label_element = driver.find_element(By.XPATH,
                                                f"//span[contains(text(), '{label_text}')] | //div[contains(text(), '{label_text}')] | //dt[contains(text(), '{label_text}')] ")
            value_element = label_element.find_element(By.XPATH, "following-sibling::*")
            scraped_data["basic_info"][key_name] = value_element.text.strip()
        except Exception:
            pass  # 忽略找不到的标签

    # 2. 提取列信息 (使用Selenium直接获取outerHTML)
    try:
        column_table_element = driver.find_element(By.CSS_SELECTOR, 'table.table-column.table-interval-bg')
        column_table_html = driver.execute_script("return arguments[0].outerHTML;", column_table_element)
        soup_column_table = BeautifulSoup(column_table_html, 'html.parser')

        headers = [th.text.strip() for th in soup_column_table.find('thead').find_all('th')]
        rows = soup_column_table.find('tbody').find_all('tr', attrs={'ng-repeat': re.compile(r'column in columns')})

        for row in rows:
            cells = row.find_all('td')
            col_dict = {}
            for i, header in enumerate(headers):
                if i < len(cells):
                    col_dict[header] = cells[i].text.strip()
            scraped_data["columns_data"].append(col_dict)
    except Exception as e:
        print(f"⚠️ 未能使用 Selenium 定位到列表格或提取列数据。错误: {e}")

    # 3. 提取备注信息 (使用Selenium直接获取outerHTML)
    try:
        remark_table_element = driver.find_element(By.CSS_SELECTOR, 'table.table-remark')
        remark_table_html = driver.execute_script("return arguments[0].outerHTML;", remark_table_element)
        soup_remark_table = BeautifulSoup(remark_table_html, 'html.parser')

        rows = soup_remark_table.find('tbody').find_all('tr')
        for row in rows:
            key_cell = None
            value_cell = None
            try:
                key_cell = row.find_element(By.CSS_SELECTOR, 'td.center')
                value_cell = row.find_element(By.CSS_SELECTOR, 'td:not(.center)')  # 查找非center类的td
            except Exception:
                pass  # 忽略找不到的单元格

            if key_cell and value_cell:
                key = key_cell.text.strip().replace('[', '').replace(']', '').strip()
                value = value_cell.text.strip()
                scraped_data["notes_map"][key] = value
    except Exception as e:
        print(f"⚠️ 未能使用 Selenium 定位到备注表格或提取备注数据。错误: {e}")

    print(f"✅ 抓取完成：找到 {len(scraped_data['columns_data'])} 列数据，{len(scraped_data['notes_map'])} 条备注。")
    return scraped_data


def organize_data_with_ai(table_name, scraped_data):
    print("🤖 正在调用大模型进行数据整合与清洗...")
    try:
        http_client = httpx.Client(verify=False)
        client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"], http_client=http_client)
        system_prompt = (
            "You are a data structuring expert. Your job is to assemble raw, pre-scraped data parts into a final, clean JSON object representing a database table definition."
        )
        user_prompt = (
            f"Please assemble the following data for the table named '{table_name}'.\n\n"
            f"1. Basic table information:\n"
            f"```json\n{json.dumps(scraped_data['basic_info'], indent=2, ensure_ascii=False)}\n```\n\n"
            f"2. A list of table columns:\n"
            f"```json\n{json.dumps(scraped_data['columns_data'], indent=2, ensure_ascii=False)}\n```\n\n"
            f"3. A mapping of note keys to their full explanatory text:\n"
            f"```json\n{json.dumps(scraped_data['notes_map'], indent=2, ensure_ascii=False)}\n```\n\n"
            "Your tasks:\n"
            "- Create a single JSON object.\n"
            "- The root object must have a 'tableName' key with the value '{table_name}'.\n"
            "- Merge the 'basic_info' into the root object.\n"
            "- Add a 'columns' key to the root, containing the list of columns.\n"
            "- For each column in the 'columns' array, if its '备注' value (e.g., '注1') is a key in the notes mapping, replace that value with the full text from the mapping.\n"
            "- Return ONLY the final, assembled JSON object and nothing else."
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
        return json.loads(json_part)
    except Exception as e:
        print(f"❌ 调用大模型进行数据整理时发生错误: {e}")
        return None


# --- 核心业务逻辑 ---
def process_table_details_page(driver, table_name):
    """处理表详情页，提取、整理并保存数据"""
    print("--- 导航到详情页成功，开始数据提取流程 --- ")
    # 增加一个短暂的固定等待，给JS初始化留出时间
    time.sleep(3)

    try:
        wait = WebDriverWait(driver, CONFIG["explicit_wait_timeout"])
        # 等待表格内部至少一个tr出现即可，放宽对ng-repeat的严格要求
        table_row_locator = (By.CSS_SELECTOR, "table.table-column.table-interval-bg tbody tr")
        print(f"⏳ 正在等待元素出现: {table_row_locator}")
        wait.until(EC.presence_of_element_located(table_row_locator))
        print("--- 列表格内容已加载，开始抓取数据 --- ")
    except Exception as e:
        print(f"❌ 在等待时间内未能找到列表格内容 ({table_row_locator})。页面可能加载失败或结构已改变。错误: {e}")
        # 调试步骤：保存当前页面HTML，以便分析
        debug_file = "debug_page_source.html"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"ℹ️ 为了便于调试，当前的页面HTML已保存到: {os.path.abspath(debug_file)}")
        return

    # 直接将driver传递给scrape_table_details，由其直接从DOM中提取数据
    scraped_data = scrape_table_details(driver)
    if not scraped_data or not scraped_data["columns_data"]:
        print("❌ 未能从页面抓取到列信息，无法继续。")
        return

    final_table_json = organize_data_with_ai(table_name, scraped_data)
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
    table_name = CONFIG["table_to_search"]
    print(f"登录完成，开始查找表：{table_name}")

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
    search_box.clear()
    search_box.send_keys(table_name)
    search_box.send_keys(Keys.ENTER)

    print(f"正在搜索 '{table_name}' 的链接...")
    try:
        wait = WebDriverWait(driver, CONFIG["explicit_wait_timeout"])
        # 定位到我们想要点击的那个精确的链接
        link_xpath = f"//a[contains(@href, '/tableShow/') and normalize-space(.)='{table_name}']"
        link_element = wait.until(EC.presence_of_element_located((By.XPATH, link_xpath)))

        # --- 终极解决方案：放弃模拟点击，直接提取href并导航 ---
        target_url = link_element.get_attribute('href')
        print(f"✅ 成功定位链接，提取到目标URL: {target_url}")

        if not target_url:
            print("❌ 无法从链接元素中提取到href属性，程序终止。")
            return

        print("🚀 正在直接导航到目标URL...")
        driver.get(target_url)

        # 直接导航后，我们仍然需要等待新视图的标志性元素出现
        print("⏳ 正在等待详情页加载...")
        details_page_identifier = (By.CSS_SELECTOR, "table.table-column.table-interval-bg")
        wait.until(EC.presence_of_element_located(details_page_identifier))
        print("✅ 详情页加载成功。")

        # 现在可以安全地调用详情页处理函数
        process_table_details_page(driver, table_name)
        print("--- 数据提取流程结束 ---")

    except Exception as e:
        print(f"⚠️ 在查找、导航或处理 '{table_name}' 时失败: {e}")
        # 保存失败时的页面快照，以便调试
        debug_file = "debug_page_source_final.html"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        print(f"ℹ️ 为了便于调试，最终失败时的页面HTML已保存到: {os.path.abspath(debug_file)}")
        print(traceback.format_exc())


def main():
    driver = None
    try:
        driver = launch_browser()
        login_and_search(driver)
        print(f"\n操作完成，页面将保持打开 {CONFIG['final_observe_time']} 秒...")
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
