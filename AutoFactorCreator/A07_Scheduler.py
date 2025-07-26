# -*- encoding: utf-8 -*-
"""
@File: A07_Scheduler.py
@Modify Time: 2025/7/16 12:54
@Author: Kevin-Chen
@Descriptions: 调度器, 协调各个智能体之间的任务安排, 并且分配计算任务
"""
import os
import json
import logging
import traceback

# 导入项目中的核心模块
from AutoFactorCreator.A03_FinancialAgent import FinancialMathematicianAgent
from AutoFactorCreator.A04_AnalysisAgent import AnalysisAgent
from AutoFactorCreator.A06_CalFactors import generate_factor_report

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Scheduler:
    """
    自动化因子挖掘流程的调度器。
    负责协调金融数学家、计算引擎和分析师智能体，形成一个完整的、自动化的因子发现与迭代闭环。
    """

    def __init__(self, max_iterations: int, performance_thresholds: dict):
        """
        初始化调度器。

        Args:
            max_iterations (int): 循环的最大迭代次数。
            performance_thresholds (dict): 因子性能阈值，用于提前终止循环。
                                           例如: {"rank_icir": 0.1, "sharpe_ratio": 1.0}
        """
        self.max_iterations = max_iterations
        self.performance_thresholds = performance_thresholds
        self.fm_agent = FinancialMathematicianAgent(temperature=0.7)
        self.analyst_agent = AnalysisAgent(temperature=0.5)

        # 定义所有处理好的数据文件的绝对路径
        # 假设此脚本位于AutoFactorCreator目录下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        processed_data_dir = os.path.join(script_dir, '..', 'Data', 'Processed_ETF_Data')
        self.data_paths = {
            'amount': os.path.join(processed_data_dir, 'processed_amount_df.parquet'),
            'close': os.path.join(processed_data_dir, 'processed_close_df.parquet'),
            'high': os.path.join(processed_data_dir, 'processed_high_df.parquet'),
            'log_return': os.path.join(processed_data_dir, 'processed_log_df.parquet'),
            'low': os.path.join(processed_data_dir, 'processed_low_df.parquet'),
            'open': os.path.join(processed_data_dir, 'processed_open_df.parquet'),
            'vol': os.path.join(processed_data_dir, 'processed_vol_df.parquet'),
            'benchmark_ew': os.path.join(processed_data_dir, 'benchmark_ew_log_returns.parquet'),
            'benchmark_min_var': os.path.join(processed_data_dir, 'benchmark_min_var_log_returns.parquet'),
            'benchmark_erc': os.path.join(processed_data_dir, 'benchmark_erc_log_returns.parquet')
        }
        # 检查所有数据路径是否存在
        for name, path in self.data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据文件未找到: {path}。请先运行 A01_DataPrepare.py。")

    def run_pipeline(self, initial_research_topic: str = None):
        """
        启动并执行完整的因子挖掘自动化流程。

        Args:
            initial_research_topic (str, optional): 一个初始的研究方向，用于指导第一轮的因子构思。
                                                    如果为None，则由智能体自由发挥。
        """
        logging.info("===== 自动化因子挖掘流程启动 =====")
        if initial_research_topic:
            logging.info(f"初始研究方向: {initial_research_topic}")

        # --- 步骤 0: 获取初始因子 ---
        logging.info("调用金融数学家智能体，构思初始因子...")
        current_ast = self.fm_agent.propose_factor_or_operator(initial_idea=initial_research_topic)
        if "ast" not in current_ast:
            logging.error(f"无法获取有效的初始因子，流程终止。响应: {current_ast}")
            return

        for i in range(1, self.max_iterations + 1):
            logging.info(f"\n{'=' * 20} 第 {i}/{self.max_iterations} 轮迭代开始 {'=' * 20}")

            # --- 步骤 1: 计算与评估 ---
            logging.info(f"正在评估因子: {current_ast.get('des', 'N/A')}")
            logging.info(f"因子 AST: {json.dumps(current_ast.get('ast', {}), indent=2)}")
            try:
                report_json_str = generate_factor_report(
                    user_ast=current_ast['ast'],
                    data_paths=self.data_paths,
                    close_path_key='close',
                    return_type='simple'
                )
                evaluation_report = json.loads(report_json_str)
                logging.info(f"因子评估完成。报告摘要: \n{json.dumps(evaluation_report, indent=2)}")
            except Exception as e:
                logging.error(f"在第 {i} 轮的因子计算或评估中发生严重错误: {e}")
                logging.error("此轮迭代终止，将尝试构思一个全新的因子。")
                # 在历史记录中记下这次失败
                error_report = {"因子性能分析": f"计算失败: {e}",
                                "因子升级建议": "检查AST结构，可能是算子使用错误或数据问题。"}
                logging.error(traceback.format_exc())
                self.fm_agent.add_history_factor_result(current_ast, error_report)
                # 构思一个全新的因子并继续
                current_ast = self.fm_agent.propose_factor_or_operator()
                continue

            # --- 步骤 2: 检查终止条件 ---
            if self._check_termination_conditions(evaluation_report):
                logging.info(f"因子性能已达到预设目标，流程终止。")
                break

            # --- 步骤 3: 分析师进行深度分析 ---
            logging.info("调用分析师智能体进行深度分析...")
            analysis_report = self.analyst_agent.analyze_factor_performance(
                factor_ast=current_ast['ast'],
                evaluation_report=evaluation_report
            )
            logging.info(f"分析师报告: \n{json.dumps(analysis_report, indent=2, ensure_ascii=False)}")

            # --- 步骤 4: 将分析结果反馈给数学家 ---
            self.fm_agent.add_history_factor_result(current_ast, analysis_report)
            logging.info("分析报告已添加至金融数学家的历史记录中。")

            # --- 步骤 5: 数学家基于反馈构思新因子 ---
            logging.info("调用金融数学家智能体，基于分析报告构思新因子...")
            new_proposal = self.fm_agent.propose_factor_or_operator()

            if "ast" in new_proposal:
                current_ast = new_proposal
                logging.info("金融数学家提出了新的因子计算逻辑。")
            elif "action" in new_proposal and new_proposal["action"] == "CreateNewCalFunc":
                logging.warning("金融数学家请求创建一个新算子。此功能尚未实现，流程终止。")
                logging.warning(f"新算子需求: {json.dumps(new_proposal, indent=2, ensure_ascii=False)}")
                break
            else:
                logging.error(f"金融数学家返回了未知格式的响应，流程终止: {new_proposal}")
                break

        logging.info(f"\n{'=' * 20} 自动化因子挖掘流程结束 {'=' * 20}")
        if i == self.max_iterations:
            logging.info(f"已达到最大迭代次数 ({self.max_iterations})。")

    def _check_termination_conditions(self, report: dict) -> bool:
        """
        检查是否满足任一性能终止条件。

        Args:
            report (dict): 因子评估报告。

        Returns:
            bool: 如果满足任一条件，则返回 True。
        """
        rank_icir = report.get("rank_ic_analysis", {}).get("rank_icir", 0)
        sharpe = report.get("long_short_portfolio_analysis", {}).get("sharpe_ratio", 0)

        if rank_icir >= self.performance_thresholds.get("rank_icir", float('inf')):
            logging.info(f"性能达标: Rank ICIR ({rank_icir:.4f}) >= 阈值 ({self.performance_thresholds['rank_icir']})")
            return True
        if sharpe >= self.performance_thresholds.get("sharpe_ratio", float('inf')):
            logging.info(
                f"性能达标: Sharpe Ratio ({sharpe:.4f}) >= 阈值 ({self.performance_thresholds['sharpe_ratio']})")
            return True
        return False


if __name__ == '__main__':
    # 定义运行参数
    MAX_ITERATIONS = 5  # 最多迭代5轮
    PERFORMANCE_THRESHOLDS = {
        "rank_icir": 0.15,  # 如果Rank ICIR达到0.15，则停止
        "sharpe_ratio": 1.5  # 如果多空组合夏普比率达到1.5，则停止
    }

    # 定义一个初始研究方向 (可选, 如果设为None, 则由AI自由发挥)
    INITIAL_TOPIC = "构建一个用于预测短期未来收益率的因子, 并与波动率预测相结合。"

    # 实例化并运行调度器
    scheduler = Scheduler(
        max_iterations=MAX_ITERATIONS,
        performance_thresholds=PERFORMANCE_THRESHOLDS
    )
    scheduler.run_pipeline(initial_research_topic=INITIAL_TOPIC)
