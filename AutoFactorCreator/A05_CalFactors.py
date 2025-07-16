# -*- encoding: utf-8 -*-
"""
@File: A05_CalFactors.py
@Modify Time: 2025/7/11 10:03       
@Author: Kevin-Chen
@Descriptions: 该模块负责基于抽象语法树(AST)计算因子值，并评估其有效性。
"""
import json
import warnings
import inspect
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.warnings import SmallSampleWarning
# 引入我们自定义的算子库，其中包含了所有因子计算需要的数学函数
from AutoFactorCreator import A02_OperatorLibrary as op_lib

# 忽略在分组收益分析中可能出现的 RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=SmallSampleWarning)


class FactorCalculator:
    """此类是实现因子计算与评估的核心引擎。

    它被设计用于处理以抽象语法树（AST）形式定义的复杂因子表达式。
    通过接收标准化的市场数据，它可以高效地计算出因子值，并运行一套完整的多维度评估流程。
    """

    def __init__(self, data_dfs: dict):
        """初始化因子计算器。

        Args:
            data_dfs (dict): 一个包含所有市场数据的字典。
                         键(key)是变量名（如 'open', 'close', 'vol'），
                         值(value)是对应的Pandas DataFrame。
        """
        self.data = data_dfs

    def calculate_factor(self, factor_ast: dict) -> pd.DataFrame:
        """公开方法，用于从AST表示中计算因子。

        这个版本使用有向无环图（DAG）来高效处理公共子表达式，避免重复计算。

        Args:
            factor_ast (dict): 以字典形式表示的因子计算逻辑（AST）。

        Returns:
            pd.DataFrame: 一个包含最终计算出的因子值的DataFrame。
        """
        # 1. 构建DAG并获取拓扑排序后的执行计划
        execution_plan, dag_representation = self._build_dag_and_plan(factor_ast)
        # 2. 根据执行计划，带缓存地执行计算
        final_result = self._execute_dag_plan(execution_plan, dag_representation)
        return final_result

    def _get_node_id(self, node: dict, known_nodes: dict) -> str:
        """为任何AST节点生成一个唯一的、确定性的ID。

        这是实现“公共子表达式消除”这一核心优化的关键。
        只要两个节点的函数和输入完全相同，它们就会得到相同的ID。
        """
        node_type = node.get('type')
        if node_type == 'literal':  # 如果是常量节点
            return f"literal::{node['value']}"
        elif node_type == 'variable':  # 如果是变量节点
            return f"variable::{node['name']}"
        elif node_type == 'operator':  # 如果是操作符节点
            # 递归地获取所有子节点的ID
            child_ids = [self._get_node_id(child, known_nodes) for child in node['children']]
            # 节点的ID由其函数名和所有子节点的ID共同决定
            node_id = f"operator::{node['func']}({', '.join(child_ids)})"
            return node_id
        else:
            raise ValueError(f"不支持的AST节点类型: {node_type}")

    def _build_dag_and_plan(self, factor_ast: dict) -> (list, dict):
        """从AST构建有向无环图(DAG)，并返回一个拓扑排序后的执行计划。
        """
        dag = {}  # 用于存储DAG, key是节点ID, value是节点信息

        # 递归函数，用于遍历AST并构建DAG
        def build_recursive(node):
            node_id = self._get_node_id(node, dag)
            if node_id in dag:  # 如果此节点已处理过，直接返回其ID
                return node_id

            node_type = node.get('type')
            if node_type in ['literal', 'variable']:
                dag[node_id] = {'type': node_type, 'node': node, 'deps': []}
                return node_id

            if node_type == 'operator':
                # 递归处理所有子节点，并记录依赖关系
                dependencies = [build_recursive(child) for child in node['children']]
                dag[node_id] = {'type': node_type, 'node': node, 'deps': dependencies}
                return node_id

            raise ValueError(f"不支持的AST节点类型: {node_type}")

        build_recursive(factor_ast)

        # 对DAG进行拓扑排序，以获得正确的计算顺序
        plan = []
        visited = set()

        def topological_sort_util(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            # 先递归处理所有依赖项
            for dep_id in dag[node_id]['deps']:
                topological_sort_util(dep_id)
            # 所有依赖项都处理完后，再将当前节点加入计划
            plan.append(node_id)

        for node_id in dag:
            topological_sort_util(node_id)

        return plan, dag

    def _execute_dag_plan(self, plan: list, dag: dict) -> pd.DataFrame:
        """使用缓存执行拓扑排序后的计划，计算最终结果。
        """
        cache = {}  # 用于缓存中间计算结果
        for node_id in plan:
            if node_id in cache:  # 如果已计算过，则跳过
                continue

            node_info = dag[node_id]
            node_type = node_info['type']

            if node_type == 'literal':
                result = node_info['node']['value']
            elif node_type == 'variable':
                result = self.data[node_info['node']['name']].copy()
            elif node_type == 'operator':
                # 从缓存中获取所有参数（子节点的结果）
                args = [cache[dep_id] for dep_id in node_info['deps']]
                # 从算子库中获取对应的计算函数
                func = getattr(op_lib, node_info['node']['func'])
                # 执行计算
                result = func(*args)
            else:
                raise ValueError(f"执行计划中存在不支持的节点类型: {node_type}")

            cache[node_id] = result  # 将计算结果存入缓存

        return cache[plan[-1]]  # 最后一个节点的结果即为最终因子值

    def evaluate_factor(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame):
        """对计算出的因子进行一整套评估。
        """
        # 在评估前，确保因子和收益率数据在时间上是对齐的，避免因索引不匹配导致错误
        aligned_factor, aligned_returns = factor_df.align(forward_returns_df, join='inner', axis=0)

        results = {
            'ic_analysis': self._calculate_ic(aligned_factor, aligned_returns),
            'rank_ic_analysis': self._calculate_rank_ic(aligned_factor, aligned_returns),
            'turnover_analysis': self._calculate_turnover(aligned_factor),
            'group_return_analysis': self._calculate_group_returns(aligned_factor, aligned_returns),
            'long_short_portfolio_analysis': self._calculate_long_short_returns(aligned_factor, aligned_returns)
        }
        return results

    def _calculate_ic(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> dict:
        """计算信息系数(IC)及相关指标(ICIR, t-test)。
        
        IC即因子值与未来收益率的皮尔逊(Pearson)相关系数，衡量因子的线性预测能力。
        """
        ic_dict = {}
        for date in factor_df.index:
            try:
                factor_slice = factor_df.loc[date].dropna()
                return_slice = forward_returns_df.loc[date].dropna()
                common_index = factor_slice.index.intersection(return_slice.index)
                if len(common_index) < 2:  # 数据点少于2个，无法计算相关性
                    ic_dict[date] = np.nan
                    continue
                ic = factor_slice.loc[common_index].corr(return_slice.loc[common_index], method='pearson')
                ic_dict[date] = ic
            except (ValueError, IndexError, KeyError) as e:
                print(f"计算IC时出错，日期: {date}: {e}")
                ic_dict[date] = np.nan

        ic_series = pd.Series(ic_dict, name='ic').reindex(factor_df.index)
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        icir = ic_mean / ic_std if ic_std != 0 and not np.isnan(ic_std) else np.nan
        t_stat, p_value = stats.ttest_1samp(ic_series.dropna(), 0, nan_policy='omit')
        return {"ic_series": ic_series, "ic_mean": ic_mean, "ic_std": ic_std, "icir": icir, "t_statistic": t_stat,
                "p_value": p_value}

    def _calculate_rank_ic(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame) -> dict:
        """计算等级信息系数(Rank IC)及其信息比率(IR)。
        
        Rank IC即因子值与未来收益率的斯皮尔曼(Spearman)等级相关系数，对异常值不敏感，衡量因子的单调预测能力。
        """
        rank_ic_dict = {}
        for date in factor_df.index:
            try:
                factor_slice = factor_df.loc[date].dropna()
                return_slice = forward_returns_df.loc[date].dropna()
                common_index = factor_slice.index.intersection(return_slice.index)
                if len(common_index) < 2:
                    rank_ic_dict[date] = np.nan
                    continue
                rank_ic = factor_slice.loc[common_index].corr(return_slice.loc[common_index], method='spearman')
                rank_ic_dict[date] = rank_ic
            except (ValueError, IndexError, KeyError) as e:
                print(f"计算Rank IC时出错，日期: {date}: {e}")
                rank_ic_dict[date] = np.nan

        rank_ic_series = pd.Series(rank_ic_dict, name='rank_ic').reindex(factor_df.index)
        rank_ic_mean = rank_ic_series.mean()
        rank_ic_std = rank_ic_series.std()
        rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std != 0 and not np.isnan(rank_ic_std) else np.nan
        return {"rank_ic_series": rank_ic_series, "rank_ic_mean": rank_ic_mean, "rank_ic_std": rank_ic_std,
                "rank_icir": rank_icir}

    def _calculate_turnover(self, factor_df: pd.DataFrame) -> dict:
        """计算因子的换手率。
        
        换手率衡量了基于该因子构建的投资组合每日的调仓幅度。换手率越低，交易成本越低。
        """
        # 如果因子值有负数，先将其转换为百分比排名，确保权重为正
        if (factor_df < 0).any().any():
            factor_positive = factor_df.rank(axis=1, pct=True)
        else:
            factor_positive = factor_df
        # 根据因子值计算每日的资产权重（每行归一化）
        weights = factor_positive.div(factor_positive.sum(axis=1), axis=0).fillna(0)
        # 计算每日权重变化的总和，除以2得到换手率（买入+卖出）
        turnover_series = weights.diff().abs().sum(axis=1) / 2
        mean_turnover = turnover_series.mean()
        return {"mean_turnover": mean_turnover, "turnover_series": turnover_series}

    def _calculate_group_returns(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame,
                                 num_quantiles=5) -> dict:
        """按因子值分组计算收益，并检验单调性。
        
        将资产按因子值高低分成N组，一个好的因子应该表现出明显的收益单调性（即因子值越高，收益越高/低）。
        """

        def get_quantile(row):
            return pd.qcut(row, num_quantiles, labels=False, duplicates='drop')

        factor_quantiles = factor_df.rank(axis=1, pct=True).apply(get_quantile, axis=1)
        aligned_quantiles = factor_quantiles.stack().rename('quantile')
        aligned_returns = forward_returns_df.stack().rename('returns')
        aligned_data = pd.concat([aligned_quantiles, aligned_returns], axis=1).dropna()
        mean_group_returns = aligned_data.groupby('quantile')['returns'].mean().rename_axis(f'Factor_Quantile')
        monotonicity = mean_group_returns.is_monotonic_increasing or mean_group_returns.is_monotonic_decreasing
        return {"mean_group_returns": mean_group_returns, "monotonicity": monotonicity}

    def _calculate_long_short_returns(self, factor_df: pd.DataFrame, forward_returns_df: pd.DataFrame,
                                      num_quantiles=5) -> dict:
        """计算多空组合的收益。
        
        模拟一个投资策略：做多因子值最高的一组，做空因子值最低的一组，并计算其累计收益和夏普比率。
        夏普比率是衡量风险调整后收益的核心指标。
        """

        def get_quantile(row):
            return pd.qcut(row, num_quantiles, labels=False, duplicates='drop')

        factor_quantiles = factor_df.rank(axis=1, pct=True).apply(get_quantile, axis=1)
        aligned_quantiles = factor_quantiles.stack().rename('quantile')
        aligned_returns = forward_returns_df.stack().rename('returns')
        aligned_data = pd.concat([aligned_quantiles, aligned_returns], axis=1).dropna()
        top_quantile_returns = aligned_data[aligned_data['quantile'] == num_quantiles - 1].groupby(level=0)[
            'returns'].mean()
        bottom_quantile_returns = aligned_data[aligned_data['quantile'] == 0].groupby(level=0)['returns'].mean()
        long_short_returns = top_quantile_returns.subtract(bottom_quantile_returns, fill_value=0)
        long_short_returns = long_short_returns.reindex(factor_df.index).fillna(0)
        cumulative_returns = (1 + long_short_returns).cumprod()
        sharpe_ratio = long_short_returns.mean() / long_short_returns.std() * np.sqrt(
            252) if long_short_returns.std() != 0 else 0
        return {"long_short_returns": long_short_returns, "cumulative_returns": cumulative_returns,
                "sharpe_ratio": sharpe_ratio}


def prepare_data_and_calculator(data_paths: dict, close_path_key='close', return_type='log'):
    """加载所有数据，准备因子计算器实例和远期收益率。
    
    这是整个流程的第一步，核心任务是确保所有输入数据都有一个统一的、格式正确的DatetimeIndex。

    Args:
        data_paths (dict): 包含数据文件路径的字典。
        close_path_key (str): `data_paths`中对应收盘价数据的键名。
        return_type (str): 指定远期收益率的计算方式。
                             'simple' (默认): 普通百分比收益率 (P_t+1 / P_t - 1)。
                             'log': 对数收益率 ln(P_t+1 / P_t)。
    """
    print("--- 步骤1: 加载和准备数据 ---")

    def load_and_prepare_df(path, name):
        if path.endswith('.parquet'):  # 如果是parquet格式
            df = pd.read_parquet(path)
        elif path.endswith('.csv'):  # 如果是csv格式
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        elif path.endswith('.xlsx'):  # 如果是Excel格式
            df = pd.read_excel(path, index_col=0, parse_dates=True)
        elif path.endswith('.pkl'):  # 如果是pickle格式
            df = pd.read_pickle(path)
        else:
            raise ValueError(f"不支持的文件格式: {path}。请使用 .parquet, .csv, .xlsx, .pkl")
        # 尝试将索引标准化为DatetimeIndex
        date_columns = ['date', 'Date', 'enddate', 'EnDate']
        for col in date_columns:
            if col in df.columns:
                df = df.set_index(col)
                break  # 用于在找到第一个匹配的日期列（如 'date'、'Date' 等）并将其设置为索引后，跳出循环，避免重复操作。
        # 确保数据框的索引为DatetimeIndex类型。如果当前索引不是时间类型，则将其转换为pd.DatetimeIndex，以便后续时间序列操作能正常进行。
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        print(f"已加载 {name}, 索引类型: {type(df.index)}, 形状: {df.shape}")
        return df

    # 加载数据
    loaded_data = {name: load_and_prepare_df(path, name) for name, path in data_paths.items()}

    if close_path_key not in loaded_data:
        raise ValueError(f"未找到收盘价数据 (键: '{close_path_key}')，无法计算远期收益率。")

    print(f"正在计算远期收益率 (类型: {return_type})...")
    close_prices = loaded_data[close_path_key]

    if return_type == 'simple':
        # 普通收益率 = (T+1日价格 / T日价格) - 1
        forward_returns = close_prices.pct_change(1).shift(-1)
    elif return_type == 'log':
        # 对数收益率 = ln(T+1日价格 / T日价格)
        forward_returns = np.log(close_prices / close_prices.shift(1)).shift(-1)
    else:
        raise ValueError(f"不支持的收益率类型: '{return_type}'。请选择 'simple' 或 'log'。")

    print("--- 初始化因子计算器 ---")
    calculator = FactorCalculator(data_dfs={n: d.values for n, d in loaded_data.items()})

    return calculator, forward_returns


def calculate_factor_values(factor_ast: dict, calculator: FactorCalculator) -> pd.DataFrame:
    """
    使用提供的因子计算器，基于抽象语法树（AST）表示的因子逻辑计算因子值。

    Args:
        factor_ast (dict): 表示因子计算逻辑的抽象语法树 (AST)。 该字典结构应包含 'func' 和 'args' 键来描述操作符及其参数。
        calculator (FactorCalculator): 一个 FactorCalculator 实例，包含用于执行因子计算的方法。

    Returns:
        pd.DataFrame: 包含最终计算出的因子值的 DataFrame。
    """
    print("\n--- 步骤2: 从AST计算因子 ---")
    # 调用 FactorCalculator 的 calculate_factor 方法，传入 AST 格式的因子公式进行计算
    factor_values = calculator.calculate_factor(factor_ast)
    print("因子计算完成。")
    # 返回因子值的 DataFrame
    return factor_values


def evaluate_factor_performance(factor_values: pd.DataFrame, calculator: FactorCalculator,
                                forward_returns: pd.DataFrame) -> dict:
    """
    使用提供的因子计算器对计算出的因子值进行多维度评估。
    该函数会调用 FactorCalculator 的评估方法，对因子在预测能力、单调性、换手率等方面进行全面评估，并返回包含各项指标的评估结果字典。

    Args:
        factor_values (pd.DataFrame): 包含计算出的因子值的 DataFrame。每行代表一个时间点，每列代表一个资产。
        calculator (FactorCalculator): 用于执行因子评估的 FactorCalculator 实例。
        forward_returns (pd.DataFrame): 前向收益率数据，用于与因子值进行相关性分析等评估操作。

    Returns:
        dict: 包含因子评估结果的字典，包括 IC、Rank IC、换手率、分组收益、多空组合表现等关键指标。
    """
    print("\n--- 步骤4: 评估因子表现 ---")
    # 调用 FactorCalculator 的 evaluate_factor 方法，传入因子值和前向收益率进行评估
    evaluation_results = calculator.evaluate_factor(factor_values, forward_returns)
    print("因子评估完成。")
    # 返回评估结果字典，包含 IC、Rank IC、换手率、分组收益、多空组合等多个维度的评估结果
    return evaluation_results


def convert_ast_format(node):
    """
    一个工具函数，用于将用户提供的AST格式转换为内部处理的格式。
    输入的AST可能是一个变量引用、操作符调用或常量值。该函数递归地将其转换为统一的内部节点格式，以便后续构建DAG和执行计算时使用。

    Args:
        node (dict or any): 用户定义的AST节点。可以是：
                            - 常量（如数字）
                            - 变量引用 {'var': 'variable_name'}
                            - 操作符调用 {'func': 'func_name', 'args': {...}}

    Returns:
        dict: 转换后的节点，格式如下：
              - 常量节点：{'type': 'literal', 'value': value}
              - 变量节点：{'type': 'variable', 'name': 'variable_name'}
              - 操作符节点：{'type': 'operator', 'func': 'func_name', 'children': [...]}
    """

    # 如果输入节点不是字典，则认为它是一个常量值（如数字），返回字面量节点
    if not isinstance(node, dict):
        return {'type': 'literal', 'value': node}

    # 如果节点包含 'var' 键，表示这是一个变量引用，返回变量节点
    if 'var' in node:
        return {'type': 'variable', 'name': node['var']}

    # 如果节点包含 'func' 键，表示这是一个操作符调用
    if 'func' in node:
        # 创建一个新的操作符节点
        new_node = {'type': 'operator', 'func': node['func']}
        args = node.get('args', {})  # 获取参数字典

        # 通过反射从op_lib模块获取函数对象
        func_obj = getattr(op_lib, node['func'])
        # 使用inspect动态获取函数的参数签名
        sig = inspect.signature(func_obj)
        arg_order = list(sig.parameters.keys())

        children = []  # 用于存储子节点

        # 按照函数实际的参数顺序遍历参数
        for key in arg_order:
            if key in args:
                # 对每个参数进行递归转换，并添加到 children 列表中
                children.append(convert_ast_format(args[key]))

        # 将子节点列表赋值给新节点的 'children' 属性
        new_node['children'] = children

        # 返回构建好的操作符节点
        return new_node

    # 默认情况下，将输入视为字面量值（如直接传入的数字等）
    return {'type': 'literal', 'value': node}


def display_evaluation_report(results: dict):
    """格式化并打印因子评估报告。"""
    print("\n--- 最终步骤: 因子评估报告 ---")
    print("=" * 40)
    ic_res = results['ic_analysis']
    print(f"IC均值: {ic_res['ic_mean']:.4f}")
    print(f"IC标准差: {ic_res['ic_std']:.4f}")
    print(f"信息比率(ICIR): {ic_res['icir']:.4f}")
    print(f"T检验统计量: {ic_res['t_statistic']:.4f}, P值: {ic_res['p_value']:.4f}")
    print("-" * 40)
    rank_ic_res = results['rank_ic_analysis']
    print(f"Rank IC均值: {rank_ic_res['rank_ic_mean']:.4f}")
    print(f"Rank IC标准差: {rank_ic_res['rank_ic_std']:.4f}")
    print(f"Rank ICIR: {rank_ic_res['rank_icir']:.4f}")
    print("-" * 40)
    turnover_res = results['turnover_analysis']
    print(f"平均每日换手率: {turnover_res['mean_turnover']:.4f}")
    print("-" * 40)
    group_res = results['group_return_analysis']
    print("按因子分位数的平均收益:")
    print(group_res['mean_group_returns'].to_string())
    print(f"收益单调性: {group_res['monotonicity']}")
    print("-" * 40)
    ls_res = results['long_short_portfolio_analysis']
    print("多空组合表现:")
    print(f"年化夏普比率: {ls_res['sharpe_ratio']:.4f}")
    print("=" * 40)
    print("分析结束。")


if __name__ == '__main__':
    # 定义所有需要用到的数据的路径
    data_paths = {
        'amount': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_amount_df.parquet',
        'close': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_close_df.parquet',
        'high': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_high_df.parquet',
        'log_return': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_log_df.parquet',
        'low': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_low_df.parquet',
        'open': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_open_df.parquet',
        'vol': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/processed_vol_df.parquet',
        'benchmark_ew': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/benchmark_ew_log_returns.parquet',
        'benchmark_min_var': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/benchmark_min_var_log_returns.parquet',
        'benchmark_erc': '/Users/chenjunming/Desktop/KevinGit/PyFinance/Data/Processed_ETF_Data/benchmark_erc_log_returns.parquet'
    }

    # 以字典形式定义因子计算公式 (AST)
    # 示例因子: vol / std_dev(moving_average(vol, 20))
    user_ast = {
        "func": "subtract",
        "args": {
            "a": {
                "func": "cumulative_sum",
                "args": {
                    "data": {
                        "func": "excess_return",
                        "args": {
                            "data": {
                                "var": "log_return"
                            },
                            "benchmark_data": {
                                "var": "benchmark_ew"
                            },
                            "axis": 0
                        }
                    },
                    "axis": 0
                }
            },
            "b": {
                "func": "rolling_sum",
                "args": {
                    "data": {
                        "func": "subtract",
                        "args": {
                            "a": {
                                "func": "exponential_moving_average",
                                "args": {
                                    "data": {
                                        "var": "log_return"
                                    },
                                    "span": 20,
                                    "axis": 0
                                }
                            },
                            "b": {
                                "func": "ts_std",
                                "args": {
                                    "data": {
                                        "var": "log_return"
                                    },
                                    "window": 20,
                                    "axis": 0
                                }
                            }
                        }
                    },
                    "window": 5,
                    "axis": 0
                }
            }
        }
    }

    # 1. 将用户定义的AST转换为内部格式
    print("--- 转换AST格式 ---")
    internal_ast = convert_ast_format(user_ast)
    print("转换后的AST:", internal_ast)

    # 2. 准备数据和计算器
    calculator, forward_returns = prepare_data_and_calculator(
        data_paths,
        close_path_key='close',
        return_type='simple'  # 在此更改收益率类型: 'simple' (普通收益率) 或 'log' (对数收益率)
    )

    # 3. 计算因子值
    factor_values = calculate_factor_values(internal_ast, calculator)

    # 4. 调试步骤: 检查计算出的因子DataFrame
    print("\n--- 步骤3: 检查计算出的因子DataFrame (调试信息) ---")
    print("因子DataFrame信息:")
    factor_values_df = pd.DataFrame(factor_values)
    print(factor_values_df.info())
    print("\n因子DataFrame头部数据:")
    print(factor_values_df.head())
    print("\n因子DataFrame尾部数据:")
    print(factor_values_df.tail())
    print("--------------------------------------------")

    # 5. 评估因子表现
    final_results = evaluate_factor_performance(factor_values_df, calculator, forward_returns)

    # 6. 展示最终报告
    display_evaluation_report(final_results)
