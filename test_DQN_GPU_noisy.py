import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.cuda.amp import GradScaler, autocast
import random
import collections
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from enum import Enum
import traceback

warnings.filterwarnings('ignore')

# 禁用 torch.compile 相关的错误，以避免在某些环境中缺少Triton依赖导致的问题
import torch._dynamo

torch._dynamo.config.suppress_errors = True


class Actions(Enum):
    """为智能体的动作定义枚举，提高代码的可读性。"""
    HOLD = 0  # 持有
    BUY = 1  # 买入
    SELL = 2  # 卖出


class NoisyLinear(nn.Module):
    """
    带有因子分解高斯噪声的Noisy Net全连接层。
    """

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # 可学习的均值参数 (mu)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # 可学习的标准差参数 (sigma)
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # 注册非可学习的、用于生成噪声的buffer
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """初始化可学习的均值和标准差参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        """实现因子分解高斯噪声"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """生成新的噪声样本"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            # 如果是训练模式，则使用带噪声的权重
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # 如果是评估模式，则不加噪声，使用均值作为确定性权重
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)


# --- 1. 配置中心, 将所有超参数和配置集中管理，方便修改和实验 (CONFIG) ---
CONFIG = {
    # --- 数据相关配置 ---
    "data": {
        "fund_code": '510050.SH',  # 交易标的：例如 '510050.SH' 代表上证50ETF
        # 基础行情数据文件路径 (开盘价, 最高价, 最低价, 收盘价, 成交量, 成交额, 对数收益率)
        "basic_info_files": {
            "log": "./Data/wide_log_return_df.parquet",
            "high": "./Data/wide_high_df.parquet",
            "low": "./Data/wide_low_df.parquet",
            "vol": "./Data/wide_vol_df.parquet",
            "amount": "./Data/wide_amount_df.parquet",
            "close": "./Data/wide_close_df.parquet",
            "open": "./Data/wide_open_df.parquet",
        },
        # 各种时间跨度的技术指标数据文件路径
        "metrics_info_files": [
            "./Data/Metrics/2d.parquet",
            "./Data/Metrics/5d.parquet",
            "./Data/Metrics/10d.parquet",
            "./Data/Metrics/15d.parquet",
            "./Data/Metrics/25d.parquet",
            "./Data/Metrics/50d.parquet",
            "./Data/Metrics/75d.parquet",
            "./Data/Metrics/5m.parquet",
            "./Data/Metrics/6m.parquet",
            "./Data/Metrics/rolling_metrics.parquet",
        ]
    },
    # --- 基础设施和硬件配置 ---
    "infra": {
        "seed": 42,  # 随机种子，用于保证实验结果的可复现性
        "device": "auto",  # 计算设备, 'auto' 表示自动选择可用的GPU，否则使用CPU
        "use_mixed_precision": True,  # 是否在GPU上使用混合精度训练 (可大幅加速，需要Tensor Cores支持)
        "pin_memory": True,  # 是否使用锁页内存，可加速CPU到GPU的数据传输
        "non_blocking": True,  # 数据传输时是否使用非阻塞模式，配合pin_memory使用
        "compile_model": False,  # 是否使用 `torch.compile` 编译模型 (PyTorch 2.0+ 特性，可加速)
        "auto_select_best_gpu": True,  # 在有多张GPU时，是否自动选择显存最大的GPU
        "gpu_memory_fraction": 0.8,  # 限制单个进程的GPU显存使用比例
        "enable_cudnn_benchmark": True,  # 是否启用cuDNN的自动调优功能，可加速固定尺寸输入的卷积运算
    },
    # --- 数据预处理配置 ---
    "preprocessing": {
        "train_split_ratio": 0.85,  # 训练集占总数据的比例
        "volatility_window": 25,  # 计算滚动波动率的窗口大小，用于奖励函数
        # --- PCA降维配置 ---
        "pca": {
            "enabled": True,  # 是否启用PCA进行特征降维
            "mode": "variance_ratio",  # PCA模式: "n_components" (指定降维后的维度数量) 或 "variance_ratio" (指定希望保留的方差比例)
            "n_components": 100,  # 如果 mode 为 "n_components"，指定降维后的维度数量
            "variance_ratio": 0.9,  # 如果 mode 为 "variance_ratio"，指定希望保留的方差比例
        }
    },
    # --- 模拟交易环境配置 ---
    "environment": {
        "initial_capital": 100000,  # 初始资金
        "transaction_cost_pct": 0.005,  # 单边交易成本（手续费）的百分比
        "trade_penalty": 0.15,  # 对每次交易行为在奖励函数中施加的额外惩罚项，以抑制过于频繁的交易
        "hard_stop_loss_pct": -0.05,  # 硬止损百分比，当浮动亏损达到此比例时，环境会强制执行卖出
        "buy_price_mode": "close_slippage",
        "sell_price_mode": "close_slippage",
        # 交易执行价格模式: "extreme"(用当日最高/最低价), "open_slippage"(开盘价+滑点), "close_slippage"(收盘价+滑点)
        "slippage_pct": 0.005,  # 滑点百分比，模拟实际成交价与理想价之间的偏差
    },
    # --- DQN智能体配置 ---
    "agent": {
        # --- 新的奖励函数配置 ---
        "reward_config": {
            "strategy": "trade_cycle_shaping",
            # 可选: "trend_based", "pnl_based", "downside_risk_adjusted", "trade_cycle_shaping"
            "strategy_descriptions": {
                "trend_based": "奖励基于市场的对数收益率，旨在捕捉趋势。持仓时奖励与市场同向，空仓时与市场反向。有固定的交易惩罚。",
                "pnl_based": "奖励直接基于每日的投资组合净值变化百分比。更直接地反映盈利目标，但可能导致对短期波动的过度反应。",
                "downside_risk_adjusted": "该策略结合了每日PnL奖励和卖出时的风险调整收益奖励，以平衡学习效率和长期目标。",
                "trade_cycle_shaping": "在交易结束时，根据Pnl结合整个交易周期的夏普比率和最大回撤计算一个综合性奖励，旨在塑造高质量的交易行为。",
            },
            # --- trend_based 策略的参数 ---
            "trend_based_params": {
                "holding_reward_factor": 1.0,
                "missing_trend_penalty_factor": 1.0,
                "fixed_trade_penalty": 0.05
            },
            # --- pnl_based 策略的参数 ---
            "pnl_based_params": {
                "pnl_scaling_factor": 100.0,  # 将PnL变化放大，使其成为更有效的奖励信号
                "fixed_trade_penalty": 0.05
            },
            # --- downside_risk_adjusted 策略的参数 ---
            "downside_risk_adjusted_params": {
                "daily_pnl_factor": 0.1,  # 每日PnL奖励的缩放因子
                "downside_risk_bonus_factor": 10.0,  # 卖出时下行风险调整收益的奖励因子
                "fixed_trade_penalty": 0.05  # 固定的交易惩罚值
            },
            # --- trade_cycle_shaping 策略的参数 ---
            "trade_cycle_params": {
                "daily_pnl_factor": 0.05,  # 每日PnL奖励的缩放因子 (保持一个较小值)
                "sharpe_ratio_bonus_factor": 2.0,  # 夏普比率奖励的乘数
                "max_drawdown_penalty_factor": 2.0  # 交易内最大回撤的惩罚乘数
            }
        },
        # --- N-Step Learning 配置 ---
        "n_step_learning": {
            "enabled": True,  # 是否启用 N-step learning
            "n_steps": 5  # N的值，即向前看多少步
        },
        "exploration_method": "noisy",  # 探索策略: "epsilon_greedy" 或 "noisy"
        "noisy_std_init": 0.5,  # NoisyLinear层的初始标准差
        "network_type": "cnn",  # 神经网络类型: "feed_forward" (全连接网络) 或 "cnn" (卷积神经网络)
        "action_dim": 3,  # 动作空间的维度 (通常是 买入/卖出/持有，共3个)
        "memory_size": 50000,  # 经验回放池的大小
        "batch_size": 256,  # 每次从经验池中采样的批量大小
        "gamma": 0.99,  # 折扣因子，决定了未来奖励在当前决策中的重要性
        "learning_rate": 0.001,  # 优化器的学习率
        "target_update_freq": 100,  # 目标网络(Target Network)的更新频率 (每多少个学习步更新一次)
        "epsilon_start": 1.0,  # ε-贪心策略的初始探索率 (仅在 exploration_method 为 "epsilon_greedy" 时使用)
        "epsilon_min": 0.01,  # ε-贪心策略的最小探索率 (仅在 exploration_method 为 "epsilon_greedy" 时使用)
        "epsilon_linear_decay_steps": 500000,  # 探索率从初始值线性衰减到最小值所需要的总步数
        "gradient_clip": 1.0,  # 梯度裁剪的阈值，用于防止梯度爆炸
        "weight_decay": 1e-5,  # 权重衰减 (L2正则化)，用于防止模型过拟合
        "dropout_rate": 0.5,  # 在网络层中使用的Dropout比率，用于防止模型过拟合

        # --- 辅助任务配置 (Multi-task Learning) ---
        "auxiliary_tasks": {
            "enabled": True,  # 是否启用辅助任务
            "definitions": [  # 辅助任务定义列表
                {
                    "name": "future_log_return_5d",
                    "operator": "log_return",  # 算子：对数收益率
                    "horizon": 5,
                    "source_column": "close",  # 计算依据的原始列
                    "params": {},  # 算子参数
                    "loss_weight": 0.1
                },
                {
                    "name": "future_trend_10d",
                    "operator": "trend_direction",  # 算子：趋势方向
                    "horizon": 10,
                    "source_column": "close",
                    "params": {"trend_threshold": 0.001},  # 趋势判断阈值
                    "loss_weight": 0.05
                },
                {
                    "name": "future_volatility_20d",
                    "operator": "volatility",  # 算子：波动率
                    "horizon": 20,
                    "source_column": "log",  # 基于对数收益率计算波动率
                    "params": {},
                    "loss_weight": 0.02
                }
            ]
        },

        # --- 全连接网络特定配置 ---
        "feed_forward_config": {
            "dqn_hidden_layers": [512, 256, 128],  # 定义隐藏层的神经元数量列表
        },
        # --- CNN网络特定配置 ---
        "cnn_config": {
            "date_length": 20,  # CNN输入使用过去多少天的数据作为一个“图像”
            "conv_kernel_size": 3,  # CNN 卷积核的大小
            "cnn_out_channels": [32, 64],  # 定义CNN卷积层的输出通道数
            "use_attention": True,  # 是否在CNN层后使用注意力机制 (Transformer Encoder)
            "num_heads": 16,  # 注意力机制的头数，必须能被d_model整除
            "dim_feedforward": 256,  # Transformer Encoder 内部前馈网络的维度
            "num_attention_layers": 2,  # 堆叠的Transformer Encoder层数
            "dqn_hidden_layers": [256, 128],  # 经过CNN/Attention特征提取后，连接的全连接隐藏层
        },
        "state_dim": None,  # 状态空间的维度 (将由环境动态计算并填充)
        "num_market_features": None,  # 市场特征的数量 (将由环境动态计算并填充)
    },
    # --- 训练过程配置 ---
    "training": {
        "num_episodes": 2,  # 训练的总回合数 (一个回合指从头到尾完整地跑完一次数据集)
        "update_frequency": 4,  # 智能体在环境中每行动多少步(step)就执行一次学习(replay)
    }
}


class DeviceManager:
    """设备管理器，自动选择和配置最佳的计算设备（CPU或GPU）。"""

    def __init__(self, config):
        self.config = config
        self.device = self._select_device()
        self.use_amp = config["infra"]["use_mixed_precision"] and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self._setup_gpu_optimizations()

    def _select_device(self):
        """根据配置自动选择设备。"""
        device_config = self.config["infra"]["device"]
        if device_config == "auto":
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"可用GPU数量: {gpu_count}")
                if gpu_count > 1 and self.config["infra"]["auto_select_best_gpu"]:
                    max_memory = 0
                    best_gpu = 0
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        memory = props.total_memory
                        print(f"GPU {i}: {props.name}, 显存: {memory / 1e9:.1f} GB")
                        if memory > max_memory:
                            max_memory = memory
                            best_gpu = i
                    device = torch.device(f"cuda:{best_gpu}")
                else:
                    device = torch.device("cuda:0")
                props = torch.cuda.get_device_properties(device)
                print(f"PyTorch CUDA 版本: {torch.version.cuda}")
                print(f"已选择设备: {props.name} ({device})")
            else:
                device = torch.device("cpu")
                print("CUDA不可用，使用CPU")
        else:
            device = torch.device(device_config)
            print(f"使用指定设备: {device}")
        return device

    def _setup_gpu_optimizations(self):
        """配置GPU相关的优化选项。"""
        if self.device.type == "cuda":
            try:
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(
                        self.config["infra"]["gpu_memory_fraction"],
                        device=self.device
                    )
                    print(f"已设置GPU内存使用上限为: {self.config['infra']['gpu_memory_fraction']}")
                if self.config["infra"]["enable_cudnn_benchmark"]:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False
                    print("CuDNN benchmark 已启用")
                else:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    print("CuDNN 确定性模式 已启用")
            except Exception as e:
                print(f"警告: GPU优化设置失败: {e}")
                print(traceback.format_exc())

    def to_device(self, tensor, non_blocking=None):
        """将张量移动到选定的设备。"""
        if non_blocking is None:
            non_blocking = self.config["infra"]["non_blocking"] and self.device.type == "cuda"
        try:
            return tensor.to(self.device, non_blocking=non_blocking)
        except Exception as e:
            print(f"警告: 非阻塞模式移动张量失败 (non_blocking={non_blocking}), 尝试阻塞模式。错误: {e}")
            print(traceback.format_exc())
            return tensor.to(self.device, non_blocking=False)

    def print_memory_usage(self):
        """打印当前GPU的显存使用情况。"""
        if self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                cached = torch.cuda.memory_reserved(self.device) / 1e9
                total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                print(f"GPU显存 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB, 总共: {total:.2f}GB")
                print(f"GPU显存使用率: {(allocated / total) * 100:.1f}%")
            except Exception as e:
                print(f"警告: 无法获取GPU显存信息: {e}")
                print(traceback.format_exc())


def set_seeds(seed):
    """设置所有相关的随机种子，以确保实验结果的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --- 2. 数据加载与预处理 ---
def get_data(config):
    """根据配置加载基础行情数据和指标数据。"""
    fund_code = config['data']['fund_code']
    basic_info_files = config['data']['basic_info_files']
    metrics_info_files = config['data']['metrics_info_files']
    print("开始加载基础行情数据...")
    the_final_df = None
    for file_type, file_path in basic_info_files.items():
        try:
            df = pd.read_parquet(file_path)
            fund_data = df[[fund_code]].copy()
            fund_data.columns = [file_type]
            if the_final_df is None:
                the_final_df = fund_data
            else:
                the_final_df = the_final_df.join(fund_data, how='outer')
            print(f"已加载 {file_type} 数据: {fund_data.shape}")
        except Exception as e:
            print(f"警告: 加载 {file_type} 数据失败，路径: {file_path}: {e}")
            print(traceback.format_exc())

    print("开始加载指标数据...")
    for i, file_path in enumerate(metrics_info_files):
        try:
            df = pd.read_parquet(file_path)
            df = df[df['ts_code'] == fund_code].reset_index(drop=True)
            df = df.drop(columns=['ts_code'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index(['date']).sort_index()
            the_final_df = the_final_df.join(df, how='outer')
            print(f"已加载指标文件 {i + 1}/{len(metrics_info_files)}: {df.shape}")
        except Exception as e:
            print(f"警告: 加载指标数据失败，路径: {file_path}: {e}")
            print(traceback.format_exc())
    print(f"最终合并数据维度: {the_final_df.shape}")
    return the_final_df


def preprocess_and_split_data(df, config):
    """
    数据预处理主函数。
    1. 处理无穷值和缺失值。
    2. 计算用于奖励函数的滚动波动率。
    3. 划分训练集和测试集。
    4. 对特征进行标准化和PCA降维，同时保留环境所需的原始数据。
    """
    print(f"DEBUG: preprocess_and_split_data - Initial df date range: {df.index.min()} to {df.index.max()}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    first_valid_indices = [df[col].first_valid_index() for col in df.columns]
    start_date = max(dt for dt in first_valid_indices if dt is not None)
    print(f"DEBUG: preprocess_and_split_data - Max first valid index (start_date): {start_date}")
    df = df.loc[start_date:].copy()
    print(
        f"DEBUG: preprocess_and_split_data - After start_date filter df date range: {df.index.min()} to {df.index.max()}")

    vol_window = config['preprocessing']['volatility_window']
    df['reward_volatility'] = df['log'].rolling(window=vol_window, min_periods=vol_window).std()
    df.fillna(method='ffill', inplace=True)
    print(f"DEBUG: preprocess_and_split_data - After ffill df date range: {df.index.min()} to {df.index.max()}")
    df.dropna(inplace=True)
    print(f"DEBUG: preprocess_and_split_data - After dropna df date range: {df.index.min()} to {df.index.max()}")

    # === 计算未来辅助任务的目标值 ===
    aux_task_definitions = config['agent']['auxiliary_tasks'].get('definitions', [])
    max_horizon = 0

    # 动态计算辅助任务目标值
    def _calculate_aux_target(df_col, operator, horizon, params=None):
        if params is None: params = {}
        if operator == "log_return":
            return np.log(df_col.shift(-horizon) / df_col)
        elif operator == "trend_direction":
            threshold = params.get("trend_threshold", 0.0)
            return ((df_col.shift(-horizon) / df_col - 1) > threshold).astype(int)
        elif operator == "volatility":
            # 计算未来N天的滚动标准差
            return df_col.rolling(window=horizon).std().shift(-horizon + 1)
        else:
            raise ValueError(f"不支持的辅助任务算子: {operator}")

    # 动态构建 aux_task_configs
    dynamic_aux_task_configs = {}
    if aux_task_definitions:
        for task_def in aux_task_definitions:
            task_name = task_def['name']
            operator = task_def['operator']
            horizon = task_def['horizon']
            source_column = task_def['source_column']
            params = task_def.get('params', {})
            loss_weight = task_def['loss_weight']

            max_horizon = max(max_horizon, horizon)

            # 计算目标值并添加到df
            df[task_name] = _calculate_aux_target(df[source_column], operator, horizon, params)

            # 根据算子推断类型和激活函数
            task_type = "regression"
            output_dim = 1
            activation = None
            if operator == "trend_direction":
                task_type = "classification"
                activation = "sigmoid"

            dynamic_aux_task_configs[task_name] = {
                "type": task_type,
                "horizon": horizon,
                "loss_weight": loss_weight,
                "output_dim": output_dim,
                "activation": activation
            }

    # 更新CONFIG中的 aux_task_configs
    config['agent']['auxiliary_tasks']['configs'] = dynamic_aux_task_configs

    # 由于计算未来奖励会导致最后几行数据无效，必须在这里丢弃它们
    # 以确保所有数据都是对齐且有效的
    if max_horizon > 0:
        df.dropna(inplace=True)
        print(f"DEBUG: 计算并丢弃无效辅助任务目标后, df日期范围: {df.index.min()} to {df.index.max()}")
    # === 新增代码结束 ===

    split_index = int(len(df) * config['preprocessing']['train_split_ratio'])
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # 定义环境所需的原始数据列
    env_cols = ['close', 'high', 'low', 'open', 'reward_volatility', 'log']
    # 动态添加辅助任务的列到 env_cols
    if config['agent']['auxiliary_tasks']['enabled'] and config['agent']['auxiliary_tasks']['configs']:
        for task_name in config['agent']['auxiliary_tasks']['configs'].keys():
            env_cols.append(task_name)

    # 定义需要输入给模型进行学习的特征列 (这里我们使用所有列作为特征)
    feature_cols = df.columns.tolist()

    # 初始化scaler和pca
    scaler = StandardScaler()
    pca_config = config['preprocessing']['pca']
    pca = None
    if pca_config['enabled']:
        print("开始进行PCA降维...")
        pca_n_components = pca_config.get('n_components',
                                          0.9) if pca_config['mode'] == 'variance_ratio' else pca_config.get(
            'n_components')
        pca = PCA(n_components=pca_n_components)

    # --- 处理训练数据 ---
    # 1. 复制环境所需原始数据
    train_env_data = train_df[env_cols].copy()
    # 2. 标准化所有特征
    train_scaled_features = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), index=train_df.index,
                                         columns=feature_cols)
    # 3. PCA降维（如果启用）
    if pca:
        train_transformed_features = pca.fit_transform(train_scaled_features)
        train_features_df = pd.DataFrame(train_transformed_features, index=train_df.index)
        print(f"PCA降维完成。原始特征维度: {len(feature_cols)}，降维后维度: {pca.n_components_}")
        print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        train_features_df = train_scaled_features
    # 4. 合并处理后的模型特征和原始环境数据
    train_final = pd.concat([train_features_df, train_env_data], axis=1)
    train_final.dropna(inplace=True)

    # --- 处理测试数据 ---
    # 1. 复制环境所需原始数据
    test_env_data = test_df[env_cols].copy()
    # 2. 标准化所有特征 (使用训练集拟合的scaler)
    test_scaled_features = pd.DataFrame(scaler.transform(test_df[feature_cols]),
                                        index=test_df.index,
                                        columns=feature_cols)
    # 3. PCA降维（如果启用，使用训练集拟合的pca）
    if pca:
        test_transformed_features = pca.transform(test_scaled_features)
        test_features_df = pd.DataFrame(test_transformed_features, index=test_df.index)
    else:
        test_features_df = test_scaled_features
    # 4. 合并
    test_final = pd.concat([test_features_df, test_env_data], axis=1)
    test_final.dropna(inplace=True)

    print(f"最终训练数据维度: {train_final.shape}, 日期范围: {train_final.index.min()} to {train_final.index.max()}")
    print(f"最终测试数据维度: {test_final.shape}, 日期范围: {test_final.index.min()} to {test_final.index.max()}")

    return train_final, test_final


# --- 3. 优化的DQN网络 ---
class OptimizedDQN(nn.Module):
    """GPU优化的DQN网络结构，支持2D CNN和传统全连接网络，以及Noisy Nets。"""

    def __init__(self, network_type, input_dim, output_dim, hidden_layers, dropout_rate=0.1, date_length=None,
                 num_market_features=None, conv_kernel_size=None, cnn_out_channels=None,
                 use_attention=False, num_heads=None, dim_feedforward=None, num_attention_layers=None,
                 use_noisy_net=False, noisy_std_init=0.5, enable_aux_tasks=False, aux_task_configs=None):
        super(OptimizedDQN, self).__init__()
        self.network_type = network_type
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_noisy_net = use_noisy_net
        self.enable_aux_tasks = enable_aux_tasks
        self.aux_task_configs = aux_task_configs if aux_task_configs is not None else {}

        LinearLayer = NoisyLinear if self.use_noisy_net else nn.Linear

        # 共享特征提取器 (Shared Feature Extractor)
        # 根据网络类型构建共享主干
        if self.network_type == "feed_forward":
            print("使用全连接前馈神经网络作为共享主干")
            shared_layers = []
            prev_dim = input_dim
            for i, hidden_dim in enumerate(hidden_layers):
                shared_layers.append(
                    LinearLayer(prev_dim, hidden_dim, std_init=noisy_std_init) if use_noisy_net else LinearLayer(
                        prev_dim, hidden_dim))
                shared_layers.append(nn.LayerNorm(hidden_dim))
                shared_layers.append(nn.ReLU(inplace=True))
                if self.dropout_rate > 0:
                    shared_layers.append(nn.Dropout(self.dropout_rate))
                prev_dim = hidden_dim
            self.shared_feature_extractor = nn.Sequential(*shared_layers)
            self.shared_feature_dim = prev_dim

        elif self.network_type == "cnn":
            print("使用卷积神经网络 (CNN) 作为共享主干")
            if date_length is None or num_market_features is None or conv_kernel_size is None or cnn_out_channels is None:
                raise ValueError(
                    "For CNN network, date_length, num_market_features, conv_kernel_size and cnn_out_channels must be provided.")
            self.date_length = date_length
            self.num_market_features = num_market_features
            self.conv_kernel_size = conv_kernel_size
            self.cnn_out_channels = cnn_out_channels
            self.use_attention = use_attention
            self.num_heads = num_heads
            self.dim_feedforward = dim_feedforward
            self.num_attention_layers = num_attention_layers
            self.channels = 1  # Assuming single channel for market features

            cnn_layers = [
                nn.Conv2d(self.channels, self.cnn_out_channels[0],
                          kernel_size=(self.conv_kernel_size, num_market_features),
                          stride=1,
                          padding=(self.conv_kernel_size // 2, 0)),
                nn.BatchNorm2d(self.cnn_out_channels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                nn.Conv2d(self.cnn_out_channels[0], self.cnn_out_channels[1],
                          kernel_size=(self.conv_kernel_size, 1),
                          stride=1,
                          padding=(self.conv_kernel_size // 2, 0)),
                nn.BatchNorm2d(self.cnn_out_channels[1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ]

            if not self.use_attention:
                cnn_layers.append(nn.Flatten())
            self.cnn_extractor = nn.Sequential(*cnn_layers)

            cnn_output_shape = self._get_cnn_output_size()

            if self.use_attention:
                d_model = cnn_output_shape[1] * cnn_output_shape[3]
                sequence_length = cnn_output_shape[2]
                if d_model % self.num_heads != 0:
                    raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({self.num_heads}).")

                encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                           nhead=self.num_heads,
                                                           dim_feedforward=self.dim_feedforward,
                                                           batch_first=True)
                self.attention_encoder = nn.TransformerEncoder(encoder_layer,
                                                               num_layers=self.num_attention_layers)
                self.shared_feature_dim = d_model * sequence_length + 2  # 加上position_status和unrealized_pnl_pct
            else:
                self.shared_feature_dim = cnn_output_shape[1] * cnn_output_shape[2] * cnn_output_shape[3] + 2

            # CNN主干后连接的MLP层，作为共享特征提取器的最后部分
            shared_mlp_layers = []
            prev_dim = self.shared_feature_dim
            for i, hidden_dim in enumerate(hidden_layers):
                shared_mlp_layers.append(
                    LinearLayer(prev_dim, hidden_dim, std_init=noisy_std_init) if use_noisy_net else LinearLayer(
                        prev_dim, hidden_dim))
                shared_mlp_layers.append(nn.LayerNorm(hidden_dim))
                shared_mlp_layers.append(nn.ReLU(inplace=True))
                if self.dropout_rate > 0:
                    shared_mlp_layers.append(nn.Dropout(self.dropout_rate))
                prev_dim = hidden_dim
            self.shared_feature_extractor = nn.Sequential(*shared_mlp_layers)
            self.shared_feature_dim = prev_dim

        else:
            raise ValueError(f"Unsupported network_type: {network_type}")

        # Q-Value Head (主任务)
        self.q_value_head = LinearLayer(self.shared_feature_dim, output_dim,
                                        std_init=noisy_std_init) if use_noisy_net else nn.Linear(
            self.shared_feature_dim, output_dim)

        # Auxiliary Task Heads (辅助任务)
        self.aux_heads = nn.ModuleDict()
        if self.enable_aux_tasks and self.aux_task_configs:
            for task_name, config in self.aux_task_configs.items():
                output_dim = config.get('output_dim', 1)
                self.aux_heads[task_name] = LinearLayer(self.shared_feature_dim, output_dim,
                                                        std_init=noisy_std_init) if use_noisy_net else nn.Linear(
                    self.shared_feature_dim, output_dim)
        print(f"DEBUG: OptimizedDQN __init__ - self.aux_heads (after build): {self.aux_heads}")

        self._initialize_weights()

    def _get_cnn_output_size(self):
        if self.network_type != "cnn":
            return 0
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.channels, self.date_length, self.num_market_features)
            # 确保 cnn_extractor 存在
            if self.cnn_extractor is None:
                # 临时构建一个用于形状推断的 cnn_extractor
                cnn_layers = [
                    nn.Conv2d(self.channels, self.cnn_out_channels[0],
                              kernel_size=(self.conv_kernel_size, self.num_market_features),
                              stride=1,
                              padding=(self.conv_kernel_size // 2, 0)),
                    nn.BatchNorm2d(self.cnn_out_channels[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                    nn.Conv2d(self.cnn_out_channels[0], self.cnn_out_channels[1],
                              kernel_size=(self.conv_kernel_size, 1),
                              stride=1,
                              padding=(self.conv_kernel_size // 2, 0)),
                    nn.BatchNorm2d(self.cnn_out_channels[1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                ]
                if not self.use_attention:
                    cnn_layers.append(nn.Flatten())
                temp_cnn_extractor = nn.Sequential(*cnn_layers)
                output_shape = temp_cnn_extractor(dummy_input).shape
            else:
                output_shape = self.cnn_extractor(dummy_input).shape
            return output_shape

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, NoisyLinear):
                m.reset_parameters()

    def reset_noise(self):
        """Resets the noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x):
        # 共享特征提取
        if self.network_type == "feed_forward":
            shared_features = self.shared_feature_extractor(x)
        elif self.network_type == "cnn":
            x_market = x[:, :-2].reshape(-1, self.channels, self.date_length, self.num_market_features)
            x_additional = x[:, -2:]

            cnn_output = self.cnn_extractor(x_market)

            if self.use_attention:
                batch_size = cnn_output.shape[0]
                sequence_length = cnn_output.shape[2]
                d_model = cnn_output.shape[1] * cnn_output.shape[3]
                cnn_output = cnn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, sequence_length, d_model)
                attention_output = self.attention_encoder(cnn_output)
                processed_features_cnn = attention_output.view(batch_size, -1)
            else:
                processed_features_cnn = cnn_output.view(cnn_output.size(0), -1)  # Flatten

            # 将CNN/Attention的输出与额外特征拼接，然后送入共享MLP
            combined_input = torch.cat((processed_features_cnn, x_additional), dim=1)
            shared_features = self.shared_feature_extractor(combined_input)

        # Q-Value Head
        q_values = self.q_value_head(shared_features)

        # Auxiliary Task Heads
        aux_predictions = {}
        if self.enable_aux_tasks:
            if self.aux_heads:  # 只有当 aux_heads 被正确初始化时才遍历
                for task_name, head in self.aux_heads.items():
                    pred = head(shared_features)
                    # 应用激活函数（如果配置了）
                    # BCEWithLogitsLoss 内部处理 sigmoid，所以这里不需要显式应用
                    # if self.aux_task_configs[task_name].get('activation') == 'sigmoid':
                    #     pred = torch.sigmoid(pred)
                    aux_predictions[task_name] = pred

        return q_values, aux_predictions


# --- 4. 交易环境 ---
class StockTradingEnv(gym.Env):
    """
    股票交易环境，遵循OpenAI Gym接口。
    这个环境经过优化，使用NumPy数组进行快速数据访问。
    """

    def __init__(self, df, config, device_manager=None):
        super(StockTradingEnv, self).__init__()
        self.df = df

        # 分离模型特征和环境所需的原始数据
        self.env_cols = ['close', 'high', 'low', 'open', 'reward_volatility', 'log']
        self.feature_df = df.drop(columns=[col for col in self.env_cols if col in df.columns])

        # 获取环境所需的原始数据序列
        self.price_series = df['close']
        self.high_series = df['high']
        self.low_series = df['low']
        self.open_series = df['open']
        self.volatility_series = df['reward_volatility']
        self.log_return_series = df['log']

        self.device_manager = device_manager
        self.config = config
        self.env_config = config['environment']
        self.reward_config = config['agent']['reward_config']

        # --- 奖励策略分发器设置 ---
        self.reward_strategy = self.reward_config['strategy']
        # 初始化奖励函数调度器，用于根据不同的策略计算奖励
        self._reward_function_dispatcher = {
            'trend_based': self._calculate_reward_trend_based,
            'pnl_based': self._calculate_reward_pnl_based,
            'downside_risk_adjusted': self._calculate_reward_downside_risk_adjusted,
            'trade_cycle_shaping': self._calculate_reward_trade_cycle
        }
        print(f"使用奖励策略: {self.reward_strategy} - "
              f"{self.reward_config['strategy_descriptions'].get(self.reward_strategy, '无描述')}")

        self.hard_stop_loss_pct = self.env_config['hard_stop_loss_pct']
        self.initial_capital = self.env_config['initial_capital']
        self.transaction_cost_pct = self.env_config['transaction_cost_pct']
        self.buy_price_mode = self.env_config['buy_price_mode']
        self.sell_price_mode = self.env_config['sell_price_mode']
        self.slippage_pct = self.env_config['slippage_pct']
        self.action_space = spaces.Discrete(len(Actions))

        self.network_type = config['agent']['network_type']
        self.num_market_features = len(self.feature_df.columns)

        if self.network_type == "cnn":
            self.date_length = config['agent']['cnn_config']['date_length']
            self.total_state_dim = (self.date_length * self.num_market_features) + 2
        elif self.network_type == "feed_forward":
            self.date_length = 1
            self.total_state_dim = self.num_market_features + 2
        else:
            raise ValueError(f"Unsupported network_type: {self.network_type}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_state_dim,),
            dtype=np.float32
        )
        self._precompute_data_arrays()
        self.reset()

    def _precompute_data_arrays(self):
        """将Pandas Series转换为NumPy数组，以加速在step中的访问。"""
        self.features_array = self.feature_df.values.astype(np.float32)
        self.prices_array = self.price_series.values.astype(np.float32)
        self.highs_array = self.high_series.values.astype(np.float32)
        self.lows_array = self.low_series.values.astype(np.float32)
        self.opens_array = self.open_series.values.astype(np.float32)
        self.volatility_array = self.volatility_series.values.astype(np.float32)
        self.log_return_array = self.log_return_series.values.astype(np.float32)  # 新增：预计算对数收益率数组
        print(f"已预计算特征和环境数据数组，维度: {self.features_array.shape}")

    def reset(self):
        """重置环境到初始状态。"""
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.history = []
        self.entry_price = 0
        self.current_holding_log_returns = []  # 用于存储当前持仓期间的每日对数收益率
        self.pending_action = None  # 新增：存储前一个时间步的决策，用于T+1执行

        # --- 新增：交易周期跟踪变量 ---
        self.current_trade_portfolio_values = []  # 存储当前交易周期内的每日组合净值
        self.current_trade_peak_value = 0  # 当前交易周期内的峰值净值，用于计算最大回撤
        self.current_trade_entry_step = 0  # 当前交易周期的起始步数
        self.current_trade_max_drawdown = 0  # 当前交易周期内，实时记录的最大回撤值
        return self._get_observation()

    def _get_observation(self):
        """获取当前时间步的观察状态，包含历史窗口和当前持仓信息。"""
        # Ensure current_step does not exceed data range
        if self.current_step >= len(self.features_array):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        position_status = 1.0 if self.shares > 0 else 0.0
        current_price_for_pnl = self.prices_array[self.current_step]
        unrealized_pnl_pct = (
                (current_price_for_pnl / self.entry_price) - 1.0) if self.shares > 0 and self.entry_price > 0 else 0.0

        if self.network_type == "cnn":
            # Determine the start index for the window
            start_idx = max(0, self.current_step - self.date_length + 1)

            # 为窗口提取市场观察, 如果窗口在开始时没有满，则用零填充
            market_obs_window = np.zeros((self.date_length, self.num_market_features), dtype=np.float32)
            actual_window_len = self.current_step - start_idx + 1
            market_obs_window[-actual_window_len:] = self.features_array[start_idx:self.current_step + 1]

            # 平铺市场观察窗口
            flattened_market_obs = market_obs_window.flatten()

            # 将扁平化的市场观察与当前头寸/PnL联系起来
            return np.concatenate([flattened_market_obs, [position_status, unrealized_pnl_pct]]).astype(np.float32)
        elif self.network_type == "feed_forward":
            market_obs = self.features_array[self.current_step]
            return np.concatenate([market_obs, [position_status, unrealized_pnl_pct]]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported network_type: {self.network_type}")

    def step(self, action_from_agent):
        """
        执行一个时间步的动作。
        action_from_agent 是智能体在 self.current_step 日（观察日）做出的决策。
        该决策将在 self.current_step + 1 日（执行日）执行。
        """
        # 1. 存储智能体在当前观察日（self.current_step）做出的决策
        self.pending_action = action_from_agent

        # 2. 推进到下一个时间步，这个时间步将是动作的执行日
        self.current_step += 1

        # 检查是否到达数据末尾
        done = self.current_step >= len(self.df)
        if done:
            # 如果数据结束，返回最终状态和信息
            final_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            final_info = {
                'date': self.df.index[-1],  # 使用最后一个有效日期
                'portfolio_value': self.portfolio_value,
                'action': Actions.HOLD.value,  # 结束时默认为HOLD
                'price': self.prices_array[-1],
                'reward': 0.0
            }
            return final_obs, 0.0, True, final_info

        # 3. 获取执行日（新的 self.current_step）的价格
        execution_day_index = self.current_step
        price_for_execution_day = self.prices_array[execution_day_index]

        # 4. 硬止损检查：在执行动作前检查是否触发止损; 如果触发，则强制执行卖出动作
        unrealized_pnl = 0
        if self.shares > 0 and self.entry_price > 0:
            unrealized_pnl = (price_for_execution_day / self.entry_price) - 1
        if self.shares > 0 and unrealized_pnl < self.hard_stop_loss_pct:
            self.pending_action = Actions.SELL.value  # 强制卖出

        # 5. 执行前一个时间步（self.current_step - 1）决策的动作，使用当前时间步（执行日）的价格
        was_holding_before_action = self.shares > 0  # 关键修复：在执行动作前记录持仓状态
        prev_portfolio_value = self.portfolio_value  # 记录执行动作前的净值
        self._take_action(self.pending_action, execution_day_index)

        # 6. 计算执行日结束时的投资组合净值
        self.portfolio_value = self.cash + self.shares * price_for_execution_day

        # --- 新增：更新交易周期跟踪变量 ---
        if self.shares > 0:  # 如果正在持仓
            self.current_trade_portfolio_values.append(self.portfolio_value)
            # 更新峰值
            self.current_trade_peak_value = max(self.current_trade_peak_value, self.portfolio_value)
            # 实时计算并更新当前交易周期的最大回撤
            if self.current_trade_peak_value > 0:
                drawdown = (self.current_trade_peak_value - self.portfolio_value) / self.current_trade_peak_value
                self.current_trade_max_drawdown = max(self.current_trade_max_drawdown, drawdown)

        # 跟踪持仓期间的对数收益率
        if self.shares > 0 and execution_day_index < len(self.log_return_array):
            self.current_holding_log_returns.append(self.log_return_array[execution_day_index])
        elif self.shares == 0 and was_holding_before_action:  # 如果卖出了，清空持仓记录
            self.current_holding_log_returns = []

        # 7. 计算执行日的奖励
        reward = self._calculate_reward(
            was_holding_stock=was_holding_before_action,
            action=self.pending_action,
            execution_day_index=execution_day_index,
            prev_portfolio_value=prev_portfolio_value,
            current_portfolio_value=self.portfolio_value
        )

        # 8. 记录执行日的信息
        info = {
            'date': self.df.index[execution_day_index],
            'portfolio_value': self.portfolio_value,
            'action': self.pending_action,  # 记录实际执行的动作
            'price': price_for_execution_day,
            'reward': reward
        }
        self.history.append(info)

        # 9. 清除待执行动作，为下一个决策做准备
        self.pending_action = None

        # 10. 获取下一个观察状态（即当前执行日的状态，用于智能体为下一天做决策）
        obs = self._get_observation()

        return obs, reward, done, info

    def _calculate_reward(self, **kwargs):
        """
        奖励计算分发器。
        根据配置的策略，调用相应的奖励函数。
        """
        # 根据当前的奖励策略获取对应的奖励函数
        reward_function = self._reward_function_dispatcher.get(self.reward_strategy)
        if reward_function:
            # 如果找到了对应的奖励函数，则调用该函数并返回计算结果
            return reward_function(**kwargs)
        else:
            # 如果没有找到对应的奖励函数，则抛出异常提示未知的奖励策略
            raise ValueError(f"未知的奖励策略: {self.reward_strategy}")

    def _calculate_reward_trend_based(self, was_holding_stock, action, execution_day_index, **kwargs):
        """
        策略1: 基于市场趋势的奖励。
        奖励直接基于市场的对数收益率，移除了风险调整项。
        """
        params = self.reward_config['trend_based_params']
        market_log_return = self.log_return_array[execution_day_index]
        reward = 0.0

        if was_holding_stock:
            reward += market_log_return * params['holding_reward_factor']
        else:
            reward -= market_log_return * params['missing_trend_penalty_factor']

        if action == Actions.BUY.value or action == Actions.SELL.value:
            reward -= params['fixed_trade_penalty']

        return reward

    def _calculate_reward_pnl_based(self, action, prev_portfolio_value, current_portfolio_value, **kwargs):
        """
        策略2: 基于投资组合净值变化的奖励。
        """
        params = self.reward_config['pnl_based_params']
        # 计算净值变化的百分比
        pnl_pct = (current_portfolio_value / prev_portfolio_value) - 1.0 if prev_portfolio_value > 0 else 0.0

        # 核心奖励是经过缩放的净值变化
        reward = pnl_pct * params['pnl_scaling_factor']

        # 对交易行为施加固定惩罚
        if action == Actions.BUY.value or action == Actions.SELL.value:
            reward -= params['fixed_trade_penalty']

        return reward

    def _calculate_reward_downside_risk_adjusted(self, was_holding_stock, action, execution_day_index,
                                                 prev_portfolio_value, current_portfolio_value, **kwargs):
        """
        策略3: 基于下行风险调整后的收益率的奖励。
        混合奖励：每日基于PnL变化，卖出时额外给予持仓期间的风险调整收益奖励。
        """
        params = self.reward_config['downside_risk_adjusted_params']
        reward = 0.0

        # 1. 每日奖励：基于投资组合净值变化
        pnl_pct = (current_portfolio_value / prev_portfolio_value) - 1.0 if prev_portfolio_value > 0 else 0.0
        reward += pnl_pct * params['daily_pnl_factor']

        # 2. 卖出时的终端奖励：基于持仓期间的下行风险调整收益
        if action == Actions.SELL.value and was_holding_stock:  # 确保是平仓操作
            if self.entry_price > 0 and len(self.current_holding_log_returns) > 0:
                # 计算持仓期间总收益率
                # 注意：self.portfolio_value 已经是卖出后的净值，self.cash 是卖出后的现金
                # 这里的总收益率应该基于买入价和卖出价
                sell_price_for_calc = self.prices_array[execution_day_index]  # 使用收盘价作为卖出价
                holding_period_return = (sell_price_for_calc / self.entry_price) - 1.0

                # 计算持仓期间的下行波动率
                negative_returns = [r for r in self.current_holding_log_returns if r < 0]
                downside_volatility = np.std(negative_returns) if len(negative_returns) > 1 else 0.0

                # 避免除以零
                if downside_volatility > 1e-9:
                    risk_adjusted_return = holding_period_return / downside_volatility
                    reward += risk_adjusted_return * params['downside_risk_bonus_factor']
                else:
                    # 如果没有下行波动，且有正收益，给予奖励；否则不额外奖励
                    if holding_period_return > 0:
                        reward += holding_period_return * params['downside_risk_bonus_factor']  # 简单奖励正收益

            # 清空持仓记录，为下一次买入做准备
            self.current_holding_log_returns = []
            self.entry_price = 0

        # 3. 交易惩罚
        if action == Actions.BUY.value or action == Actions.SELL.value:
            reward -= params['fixed_trade_penalty']

        return reward

    def _calculate_reward_trade_cycle(self, was_holding_stock, action, prev_portfolio_value, current_portfolio_value,
                                      **kwargs):
        """
        策略4: 基于完整交易周期的塑造奖励。
        """
        params = self.reward_config['trade_cycle_params']
        reward = 0.0

        # 1. 每日奖励：基于投资组合净值变化的微小奖励，以提供即时反馈
        pnl_pct = (current_portfolio_value / prev_portfolio_value) - 1.0 if prev_portfolio_value > 0 else 0.0
        reward += pnl_pct * params['daily_pnl_factor']

        # 2. 交易结束时的终局奖励：在卖出时计算
        if action == Actions.SELL.value and was_holding_stock:
            # a. 计算夏普比率
            trade_returns = pd.Series(self.current_trade_portfolio_values).pct_change().dropna()
            if len(trade_returns) > 1 and trade_returns.std() > 1e-9:
                # 年化夏普比率 (假设每日数据)
                sharpe_ratio = trade_returns.mean() / trade_returns.std() * np.sqrt(252)
                reward += sharpe_ratio * params['sharpe_ratio_bonus_factor']

            # b. 使用实时计算的最大回撤进行惩罚
            reward -= self.current_trade_max_drawdown * params['max_drawdown_penalty_factor']

            # c. 重置交易周期变量
            self.current_trade_portfolio_values = []
            self.current_trade_peak_value = 0
            self.current_trade_entry_step = 0
            self.current_trade_max_drawdown = 0

        return reward

    def _take_action(self, action, execution_day_index):
        """
        根据动作执行交易。
        action: 实际要执行的动作。
        execution_day_index: 动作执行的日期索引。
        """
        buy_price = 0
        sell_price = 0

        # 根据动作和对应的价格模式获取执行价格
        if action == Actions.BUY.value:
            if self.buy_price_mode == "extreme":
                buy_price = self.highs_array[execution_day_index]
            elif self.buy_price_mode == "open_slippage":
                buy_price = self.opens_array[execution_day_index] * (1 + self.slippage_pct)
            elif self.buy_price_mode == "close_slippage":
                buy_price = self.prices_array[execution_day_index] * (1 + self.slippage_pct)
            else:
                buy_price = self.prices_array[execution_day_index]
        elif action == Actions.SELL.value:
            if self.sell_price_mode == "extreme":
                sell_price = self.lows_array[execution_day_index]
            elif self.sell_price_mode == "open_slippage":
                sell_price = self.opens_array[execution_day_index] * (1 - self.slippage_pct)
            elif self.sell_price_mode == "close_slippage":
                sell_price = self.prices_array[execution_day_index] * (1 - self.slippage_pct)
            else:
                sell_price = self.prices_array[execution_day_index]

        if action == Actions.BUY.value and self.cash > 0:
            cost = self.cash * self.transaction_cost_pct
            buy_capital = self.cash - cost
            if buy_price > 0:
                self.shares = buy_capital / buy_price
                self.cash = 0
                self.entry_price = buy_price
                # --- 新增：开始新的交易周期跟踪 (已修正) ---
                # 在更新持仓和现金后，记录精确的交易起始净值 (即扣除手续费后的资本)
                self.current_trade_entry_step = execution_day_index
                self.current_trade_portfolio_values = [buy_capital]
                self.current_trade_peak_value = buy_capital
                self.current_trade_max_drawdown = 0  # 重置最大回撤
        elif action == Actions.SELL.value and self.shares > 0:
            proceeds = self.shares * sell_price
            cost = proceeds * self.transaction_cost_pct
            self.cash = proceeds - cost
            self.shares = 0
            self.entry_price = 0


# --- 5. DQN智能体 ---
class OptimizedDQNAgent:
    """优化的DQN智能体，支持epsilon-greedy和Noisy Networks探索策略，以及N-step learning。"""

    def __init__(self, config, device_manager):
        self.config = config['agent']
        self.device_manager = device_manager
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']

        # --- N-Step Learning 配置 ---
        n_step_config = self.config.get("n_step_learning", {})
        self.n_step_enabled = n_step_config.get("enabled", False)
        self.n_steps = n_step_config.get("n_steps", 1)
        if self.n_step_enabled:
            print(f"启用 N-Step Learning, N = {self.n_steps}")

        # --- 探索策略选择 ---
        self.exploration_method = self.config.get("exploration_method", "epsilon_greedy")
        self.use_noisy_net = self.exploration_method == "noisy"

        if self.use_noisy_net:
            print("使用 Noisy Networks 进行探索")
            self.epsilon = 0.0  # Epsilon不用于Noisy Nets
        else:
            print("使用 epsilon-greedy 进行探索")
            self.epsilon = self.config['epsilon_start']

        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay_step = (self.config['epsilon_start'] - self.config['epsilon_min']) / self.config[
            'epsilon_linear_decay_steps']

        # 将需要频繁访问的参数设为对象的直接属性
        self.memory_size = self.config['memory_size']
        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.learning_rate = self.config['learning_rate']
        self.target_update_freq = self.config['target_update_freq']
        self.gradient_clip = self.config['gradient_clip']
        self.weight_decay = self.config['weight_decay']
        self.dropout_rate = self.config['dropout_rate']
        self.network_type = self.config['network_type']

        # --- 辅助任务配置 ---
        self.enable_aux_tasks = self.config['auxiliary_tasks']['enabled']
        # 从 definitions 构建 configs 字典
        self.aux_task_configs = {}
        if self.enable_aux_tasks:
            for task_def in self.config['auxiliary_tasks'].get('definitions', []):
                task_name = task_def['name']
                task_type = "regression"
                activation = None
                if task_def['operator'] == "trend_direction":
                    task_type = "classification"
                    activation = "sigmoid"
                self.aux_task_configs[task_name] = {
                    "type": task_type,
                    "horizon": task_def['horizon'],
                    "loss_weight": task_def['loss_weight'],
                    "output_dim": 1,  # 默认输出维度为1
                    "activation": activation
                }
        self.aux_loss_fns = nn.ModuleDict()  # 使用ModuleDict来存储损失函数，以便它们能被正确地移动到设备
        if self.enable_aux_tasks:
            for task_name, task_config in self.aux_task_configs.items():
                if task_config['type'] == 'regression':
                    self.aux_loss_fns[task_name] = nn.MSELoss()
                elif task_config['type'] == 'classification':
                    self.aux_loss_fns[task_name] = nn.BCEWithLogitsLoss()  # 二分类交叉熵，对混合精度更安全
                else:
                    raise ValueError(f"不支持的辅助任务类型: {task_config['type']}")
            self.aux_loss_fns = device_manager.to_device(self.aux_loss_fns)

        self.memory = collections.deque(maxlen=self.memory_size)

        print(f"初始化DQN网络，输入维度={self.state_dim}, 输出维度={self.action_dim}, 网络类型={self.network_type}")

        # --- 网络初始化 ---
        noisy_std_init = self.config.get('noisy_std_init', 0.5)
        common_net_args = {
            "network_type": self.network_type,
            "input_dim": self.state_dim,
            "output_dim": self.action_dim,
            "dropout_rate": self.dropout_rate,
            "use_noisy_net": self.use_noisy_net,
            "noisy_std_init": noisy_std_init,
            "enable_aux_tasks": self.enable_aux_tasks,
            "aux_task_configs": self.aux_task_configs
        }

        if self.network_type == "cnn":
            cnn_config = self.config['cnn_config']
            net_args = {
                **common_net_args,
                "hidden_layers": cnn_config['dqn_hidden_layers'],
                "date_length": self.config['cnn_config']['date_length'],
                "num_market_features": self.config['num_market_features'],
                "conv_kernel_size": cnn_config['conv_kernel_size'],
                "cnn_out_channels": cnn_config['cnn_out_channels'],
                "use_attention": cnn_config['use_attention'],
                "num_heads": cnn_config['num_heads'],
                "dim_feedforward": cnn_config['dim_feedforward'],
                "num_attention_layers": cnn_config['num_attention_layers']
            }
        elif self.network_type == "feed_forward":
            net_args = {
                **common_net_args,
                "hidden_layers": self.config['feed_forward_config']['dqn_hidden_layers']
            }
        else:
            raise ValueError(f"Unsupported network_type: {self.network_type}")

        self.policy_net = OptimizedDQN(**net_args)
        self.target_net = OptimizedDQN(**net_args)

        self.policy_net = device_manager.to_device(self.policy_net)
        self.target_net = device_manager.to_device(self.target_net)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        print(f"网络已移动到设备: {device_manager.device}")

        self.optimizer = optim.AdamW(self.policy_net.parameters(),
                                     lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='max', patience=1000, factor=0.8)
        self.loss_fn = nn.SmoothL1Loss()
        self.learn_step_counter = 0
        print("DQN智能体初始化成功")

    def remember(self, state, action, reward, next_state, done, aux_targets=None):
        """将经验存入回放缓冲区。"""
        self.memory.append((state, action, reward, next_state, done, aux_targets))

    def reset_noise(self):
        """重置策略网络和目标网络中的噪声。"""
        if self.use_noisy_net:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

    def act(self, state):
        """根据当前状态和探索策略选择一个动作。"""
        position_status = state[-2]
        valid_actions = [Actions.HOLD.value, Actions.BUY.value] if position_status == 0 else [Actions.HOLD.value,
                                                                                              Actions.SELL.value]

        if not self.use_noisy_net and random.random() <= self.epsilon:
            return random.choice(valid_actions)

        if self.use_noisy_net:
            self.policy_net.reset_noise()

        with torch.no_grad():
            state_tensor = self.device_manager.to_device(torch.FloatTensor(state).unsqueeze(0))
            q_values, _ = self.policy_net(state_tensor)
            # 确保 q_values 是 (1, action_dim) 的形状
            if q_values.shape != (1, self.action_dim):
                # 如果形状不匹配，尝试进行修正。这通常意味着网络输出有问题。
                # 警告：如果这里频繁触发，说明网络结构或初始化存在深层问题。
                print(f"警告: q_values 形状异常: {q_values.shape}。期望 (1, {self.action_dim})。尝试修正...")
                try:
                    q_values = q_values.view(1, self.action_dim)  # 强制转换为期望形状
                except RuntimeError as e:
                    print(traceback.format_exc())
                    raise RuntimeError(f"无法将 q_values 从 {q_values.shape} 修正为 (1, {self.action_dim}): {e}")

            # 创建一个布尔掩码，初始全部为True
            invalid_actions_mask = torch.ones_like(q_values, dtype=torch.bool)
            # 将有效动作对应的位置设置为False
            for i in valid_actions:
                invalid_actions_mask[0][i] = False

            # 使用 masked_fill_ 将无效动作的Q值设置为负无穷
            q_values.masked_fill_(invalid_actions_mask, -float('inf'))

            return torch.argmax(q_values[0]).item()

    def replay(self, global_step):
        """从经验回放缓冲区中采样并训练网络，支持1-step和N-step DQN。"""
        if self.n_step_enabled:
            return self._replay_n_step(global_step)
        else:
            return self._replay_1_step(global_step)

    def _replay_1_step(self, global_step):
        """传统的1-step Double DQN学习。"""
        if len(self.memory) < self.batch_size:
            return None

        if self.use_noisy_net:
            self.reset_noise()

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, aux_targets_list = zip(*minibatch)

        states = self.device_manager.to_device(torch.FloatTensor(np.array(states)))
        actions = self.device_manager.to_device(torch.LongTensor(actions).unsqueeze(1))
        rewards = self.device_manager.to_device(torch.FloatTensor(rewards).unsqueeze(1))
        next_states = self.device_manager.to_device(torch.FloatTensor(np.array(next_states)))
        dones = self.device_manager.to_device(torch.BoolTensor(dones).unsqueeze(1))

        # 辅助任务目标值转换为张量
        aux_targets_tensors = {}
        if self.enable_aux_tasks:
            for task_name, task_config in self.aux_task_configs.items():
                # 确保数据类型正确，特别是分类任务
                if task_config['type'] == 'classification':
                    aux_targets_tensors[task_name] = self.device_manager.to_device(
                        torch.FloatTensor([d[task_name] for d in aux_targets_list]).unsqueeze(1))
                else:
                    aux_targets_tensors[task_name] = self.device_manager.to_device(
                        torch.FloatTensor([d[task_name] for d in aux_targets_list]).unsqueeze(1))

        with autocast(enabled=self.device_manager.use_amp):
            # 网络现在返回Q值和辅助预测
            current_q_values_all, aux_predictions = self.policy_net(states)
            current_q_values = current_q_values_all.gather(1, actions)

            with torch.no_grad():
                best_next_actions = self.policy_net(next_states)[0].argmax(1).unsqueeze(1)  # [0] for q_values
                next_q_values_for_best_actions = self.target_net(next_states)[0].gather(1, best_next_actions)
            next_q_values_for_best_actions[dones] = 0.0
            target_q_values = rewards + (self.gamma * next_q_values_for_best_actions)
            rl_loss = self.loss_fn(current_q_values, target_q_values)

            # 计算辅助任务损失
            total_aux_loss = 0.0
            if self.enable_aux_tasks:
                for task_name, task_config in self.aux_task_configs.items():
                    aux_pred = aux_predictions[task_name]
                    aux_target = aux_targets_tensors[task_name]
                    aux_loss = self.aux_loss_fns[task_name](aux_pred, aux_target)
                    total_aux_loss += aux_loss * task_config['loss_weight']

            total_loss = rl_loss + total_aux_loss

        self._optimize_model(total_loss)
        self._update_epsilon(global_step)
        self._update_target_net()
        return total_loss.item()

    def _replay_n_step(self, global_step):
        """N-step Double DQN学习。"""
        if len(self.memory) < self.batch_size + self.n_steps:
            return None

        if self.use_noisy_net:
            self.reset_noise()

        # 采样起始索引
        indices = random.sample(range(len(self.memory) - self.n_steps), self.batch_size)

        # 准备N-step经验
        states, actions, n_step_rewards, n_step_next_states, n_step_dones, aux_targets_list = [], [], [], [], [], []

        for idx in indices:
            # 获取初始状态和动作
            state, action, _, _, _, aux_target_dict = self.memory[idx]
            states.append(state)
            actions.append(action)
            aux_targets_list.append(aux_target_dict)

            # 计算N-step累计奖励和最终状态
            G = 0.0
            gamma_power = 1.0
            final_done = True
            for i in range(self.n_steps):
                _, _, reward, next_s, done, _ = self.memory[idx + i]
                G += gamma_power * reward
                gamma_power *= self.gamma
                if done:
                    final_next_state = next_s  # 即使终止，也需要一个占位符
                    final_done = True
                    break
                final_next_state = next_s
                final_done = False

            n_step_rewards.append(G)
            n_step_next_states.append(final_next_state)
            n_step_dones.append(final_done)

        # 转换为张量
        states = self.device_manager.to_device(torch.FloatTensor(np.array(states)))
        actions = self.device_manager.to_device(torch.LongTensor(actions).unsqueeze(1))
        n_step_rewards = self.device_manager.to_device(torch.FloatTensor(n_step_rewards).unsqueeze(1))
        n_step_next_states = self.device_manager.to_device(torch.FloatTensor(np.array(n_step_next_states)))
        n_step_dones = self.device_manager.to_device(torch.BoolTensor(n_step_dones).unsqueeze(1))

        # 辅助任务目标值转换为张量
        aux_targets_tensors = {}
        if self.enable_aux_tasks:
            for task_name, task_config in self.aux_task_configs.items():
                if task_config['type'] == 'classification':
                    aux_targets_tensors[task_name] = self.device_manager.to_device(
                        torch.FloatTensor([d[task_name] for d in aux_targets_list]).unsqueeze(1))
                else:
                    aux_targets_tensors[task_name] = self.device_manager.to_device(
                        torch.FloatTensor([d[task_name] for d in aux_targets_list]).unsqueeze(1))

        with autocast(enabled=self.device_manager.use_amp):
            # 网络现在返回Q值和辅助预测
            current_q_values_all, aux_predictions = self.policy_net(states)
            current_q_values = current_q_values_all.gather(1, actions)

            with torch.no_grad():
                best_next_actions = self.policy_net(n_step_next_states)[0].argmax(1).unsqueeze(1)
                next_q_values = self.target_net(n_step_next_states)[0].gather(1, best_next_actions)

            next_q_values[n_step_dones] = 0.0
            target_q_values = n_step_rewards + (self.gamma ** self.n_steps) * next_q_values
            rl_loss = self.loss_fn(current_q_values, target_q_values)

            # 计算辅助任务损失
            total_aux_loss = 0.0
            if self.enable_aux_tasks:
                for task_name, task_config in self.aux_task_configs.items():
                    aux_pred = aux_predictions[task_name]
                    aux_target = aux_targets_tensors[task_name]
                    aux_loss = self.aux_loss_fns[task_name](aux_pred, aux_target)
                    total_aux_loss += aux_loss * task_config['loss_weight']

            total_loss = rl_loss + total_aux_loss

        self._optimize_model(total_loss)
        self._update_epsilon(global_step)
        self._update_target_net()
        return total_loss.item()

    def _optimize_model(self, loss):
        """执行一次模型优化步骤。"""
        if self.device_manager.use_amp:
            self.optimizer.zero_grad()
            self.device_manager.scaler.scale(loss).backward()
            self.device_manager.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.device_manager.scaler.step(self.optimizer)
            self.device_manager.scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.optimizer.step()

    def _update_epsilon(self, global_step):
        """更新epsilon值。"""
        if not self.use_noisy_net:
            self.epsilon = max(self.epsilon_min, self.config['epsilon_start'] - self.epsilon_decay_step * global_step)

    def _update_target_net(self):
        """更新目标网络。"""
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# --- 6. 训练与评估 ---
def calculate_episode_stats(history, initial_capital):
    """根据历史记录计算回合的性能统计数据。"""
    if not history:
        return {"final_value": initial_capital, "total_return": 0.0, "annual_volatility": 0.0, "total_rewards": 0.0}
    history_df = pd.DataFrame(history)
    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    daily_returns = history_df['portfolio_value'].pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        annual_volatility = 0.0
    else:
        annual_volatility = daily_returns.std() * np.sqrt(252)
    total_rewards = history_df['reward'].sum()
    return {"final_value": final_value, "total_return": total_return, "annual_volatility": annual_volatility,
            "total_rewards": total_rewards}


def train_dqn(config, train_df, device_manager):
    """DQN模型的主训练函数。"""
    print("初始化训练环境...")
    env = StockTradingEnv(train_df, config, device_manager)

    # Update config with actual state dimensions from the environment
    config['agent']['state_dim'] = env.observation_space.shape[0]
    config['agent']['num_market_features'] = env.num_market_features  # Pass num_market_features to agent config

    print("初始化DQN智能体...")
    agent = OptimizedDQNAgent(config, device_manager)
    initial_capital = config['environment']['initial_capital']
    num_episodes = config['training']['num_episodes']
    update_frequency = config['training']['update_frequency']
    global_step_counter = 0
    print(f"--- 开始训练，共 {num_episodes} 回合 ---")
    print(f"设备: {device_manager.device}, 混合精度: {device_manager.use_amp}")
    print(f"探索策略: {agent.exploration_method}")
    print(f"状态维度: {config['agent']['state_dim']}, 经验池大小: {agent.memory_size}, 批大小: {agent.batch_size}")
    training_losses = []
    episode_rewards = []
    try:
        for e in tqdm(range(num_episodes), desc="训练回合"):
            state = env.reset()
            done = False
            episode_loss, episode_reward, step_count = 0, 0, 0
            while not done:
                global_step_counter += 1
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)

                # 获取当前时间步的辅助任务目标值
                aux_targets = {}
                if agent.enable_aux_tasks:
                    current_date_idx = env.current_step - 1  # env.current_step 已经推进到下一天，所以要减1
                    for task_name in agent.aux_task_configs.keys():
                        # 确保索引在有效范围内
                        if current_date_idx < len(env.df):
                            aux_targets[task_name] = env.df[task_name].iloc[current_date_idx]
                        else:
                            # 如果超出范围，则使用默认值或跳过
                            aux_targets[task_name] = 0.0  # 或者根据任务类型设置合适默认值

                agent.remember(state, action, reward, next_state, done, aux_targets)
                state = next_state
                episode_reward += reward
                step_count += 1
                if len(agent.memory) > agent.batch_size and global_step_counter % update_frequency == 0:
                    try:
                        loss = agent.replay(global_step_counter)
                        if loss is not None:
                            episode_loss += loss
                    except Exception as replay_error:
                        print(f"警告: Replay失败于步骤 {global_step_counter}: {replay_error}")
                        import traceback
                        print(traceback.format_exc())
            training_losses.append(episode_loss / max(step_count / update_frequency, 1))
            episode_rewards.append(episode_reward)
            if (e + 1) % 10 == 0 or e == num_episodes - 1:
                stats = calculate_episode_stats(env.history, initial_capital)
                avg_loss = np.mean(training_losses[-10:]) if training_losses else 0
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print("")
                print(f"--- 回合 {e + 1}/{num_episodes} ---")
                print("  平均损失 (近10回合): {:.6f}".format(avg_loss))
                print("  平均奖励 (近10回合): {:.4f}".format(avg_reward))
                print("  本回合总奖励: {:.4f}".format(stats['total_rewards']))
                print("  最终组合净值: {:.2f}".format(stats['final_value']))
                print("  总回报率: {:.2%}".format(stats['total_return']))
                print("  年化波动率: {:.2%}".format(stats['annual_volatility']))
                if not agent.use_noisy_net:
                    print("  Epsilon: {:.4f}".format(agent.epsilon))
                print("  学习率: {:.6f}".format(agent.optimizer.param_groups[0]['lr']))
                print(f"  经验池大小: {len(agent.memory)}")
                if device_manager.device.type == "cuda":
                    device_manager.print_memory_usage()
                    if (e + 1) % 20 == 0:
                        torch.cuda.empty_cache()
                        print("GPU缓存已清理")
            if episode_rewards:
                agent.scheduler.step(episode_rewards[-1])
    except KeyboardInterrupt:
        print("")
        print("训练被用户中断")
    except Exception as train_error:
        print(f"训练过程中发生错误: {train_error}")
        import traceback
        traceback.print_exc()
    print("")
    print("--- 训练结束 ---")
    if device_manager.device.type == "cuda":
        torch.cuda.empty_cache()
        print("最终GPU缓存清理完成")
    return agent, training_losses, episode_rewards


def evaluate_agent(agent, eval_df, config, device_manager):
    """在评估数据集上评估智能体的性能。"""
    print("开始评估智能体...")
    agent.policy_net.eval()  # 切换到评估模式 (对于NoisyNet，这意味着不加噪声)
    env = StockTradingEnv(eval_df, config, device_manager)
    state = env.reset()
    done = False
    step_count = 0
    try:
        with torch.no_grad():
            while not done:
                action = agent.act(state)
                state, _, done, _ = env.step(action)
                step_count += 1
                if step_count % 100 == 0:
                    print(f"评估进度: {step_count / len(eval_df) * 100:.1f}%")
    except Exception as eval_error:
        print(f"评估过程中发生错误: {eval_error}")
    agent.policy_net.train()  # 评估结束后切回训练模式
    print(f"评估完成，共执行 {step_count} 步")
    return env.history


# --- 7. 可视化 ---
def plot_results(history, config, benchmark_df, title, training_losses=None, episode_rewards=None):
    """Enhanced visualization function to show portfolio value against a normalized ETF price benchmark."""
    if training_losses is not None and episode_rewards is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12), sharex='col')
    else:
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    history_df = pd.DataFrame(history).set_index('date')
    initial_capital = config['environment']['initial_capital']

    # 确保 history_df 和 benchmark_df 的索引类型一致，并进行对齐
    # 这对于确保绘图的日期轴正确对齐至关重要
    history_df.index = pd.to_datetime(history_df.index)
    benchmark_df.index = pd.to_datetime(benchmark_df.index)

    # 找到两个DataFrame共同的日期范围，以确保绘图范围一致
    common_start_date = max(history_df.index.min(), benchmark_df.index.min())
    common_end_date = min(history_df.index.max(), benchmark_df.index.max())

    history_df_aligned = history_df.loc[common_start_date:common_end_date]
    benchmark_df_aligned = benchmark_df.loc[common_start_date:common_end_date]

    # Plot 1: Portfolio Value vs. Normalized ETF Price
    ax1.set_title(title, fontsize=16)

    # Plot DQN Agent Portfolio Value (starts at 1)
    # 使用对齐后的数据进行归一化
    normalized_portfolio_value = history_df_aligned['portfolio_value'] / initial_capital
    ax1.plot(history_df_aligned.index, normalized_portfolio_value, label='DQN Agent Portfolio', color='blue',
             linewidth=2,
             zorder=2)

    # Plot Normalized ETF Price (starts at 1 on the same day as the agent)
    # 使用对齐后的数据进行归一化
    normalized_benchmark = benchmark_df_aligned['close'] / benchmark_df_aligned['close'].iloc[0]
    ax1.plot(benchmark_df_aligned.index, normalized_benchmark, label='Normalized ETF Price', color='grey',
             linestyle='--',
             linewidth=2, zorder=1)

    ax1.set_ylabel('Normalized Net Value (Start=1)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Plots 2 & 4: Training Loss and Rewards
    if training_losses is not None and episode_rewards is not None:
        ax2.set_title('Training Loss Over Episodes')
        ax2.plot(training_losses, color='red', alpha=0.7)
        if len(training_losses) > 10:
            ma_losses = pd.Series(training_losses).rolling(window=10).mean()
            ax2.plot(ma_losses, color='darkred', linewidth=2, label='10-Episode MA')
            ax2.legend()
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        ax4.set_title('Episode Rewards Over Time')
        ax4.plot(episode_rewards, color='green', alpha=0.7)
        if len(episode_rewards) > 10:
            ma_rewards = pd.Series(episode_rewards).rolling(window=10).mean()
            ax4.plot(ma_rewards, color='darkgreen', linewidth=2, label='10-Episode MA')
            ax4.legend()
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Total Reward')
        ax4.grid(True, alpha=0.3)

    # Plot 3: Agent Position Status
    ax3.set_title('Agent Position Status')

    position_active = False
    last_buy_time = None
    legend_added = False

    # 遍历 history_df 的每一行，根据动作绘制持仓区域
    for i in range(len(history_df_aligned.index)):
        current_date = history_df_aligned.index[i]
        action_on_current_date = history_df_aligned['action'].iloc[i]

        if action_on_current_date == Actions.BUY.value:
            if not position_active:  # 只有在没有持仓时才开始新的持仓
                position_active = True
                last_buy_time = current_date
                print(f"DEBUG: BUY action on {current_date}. last_buy_time set to {last_buy_time}")
        elif action_on_current_date == Actions.SELL.value:
            if position_active and last_buy_time:  # 只有在有持仓时才进行卖出
                # 如果在 current_date 卖出，则持仓到 current_date 的前一天结束。
                # 确保有前一天，避免索引错误
                if i > 0:
                    end_date_for_span = history_df_aligned.index[i - 1]
                else:
                    # 如果在第一个交易日就卖出，则没有有效的持仓区间
                    end_date_for_span = None

                if end_date_for_span is not None and last_buy_time <= end_date_for_span:
                    label = 'Long Position' if not legend_added else ""
                    ax3.axvspan(last_buy_time, end_date_for_span, color='green', alpha=0.3, label=label)
                    legend_added = True
                    print(
                        f"DEBUG: SELL action on {current_date}. Plotting span from {last_buy_time} to {end_date_for_span}")
                else:
                    print(
                        f"DEBUG: SELL action on {current_date}. No span plotted (last_buy_time={last_buy_time}, end_date_for_span={end_date_for_span})")

                position_active = False
                last_buy_time = None  # 卖出后重置买入时间

    # 处理在评估期结束时仍持有的仓位
    if position_active and last_buy_time:
        label = 'Long Position' if not legend_added else ""
        ax3.axvspan(last_buy_time, history_df_aligned.index[-1], color='green', alpha=0.3, label=label)
        legend_added = True
        print(f"DEBUG: Final open position. Plotting span from {last_buy_time} to {history_df_aligned.index[-1]}")

    ax3.set_yticks([])
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    if legend_added:
        ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def print_performance_comparison(train_history, test_history, train_df, test_df, config):
    """Prints a detailed performance comparison report."""
    initial_capital = config['environment']['initial_capital']
    train_stats = calculate_episode_stats(train_history, initial_capital)
    test_stats = calculate_episode_stats(test_history, initial_capital)
    train_benchmark_return = (train_df['close'].iloc[-1] / train_df['close'].iloc[0]) - 1
    test_benchmark_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0]) - 1

    def calculate_sharpe(history, initial_capital):
        if not history: return 0.0
        df = pd.DataFrame(history)
        returns = df['portfolio_value'].pct_change().dropna()
        return (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() != 0 else 0.0

    train_sharpe = calculate_sharpe(train_history, initial_capital)
    test_sharpe = calculate_sharpe(test_history, initial_capital)

    print("")
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    print("")
    header = "{:<20} {:<15} {:<15} {:<15} {:<15}".format('Strategy', 'Train Return', 'Test Return', 'Train Sharpe',
                                                         'Test Sharpe')
    print(header)
    print("-" * 80)
    dqn_stats_str = "{:<20} {:<15.2%} {:<15.2%} {:<15.3f} {:<15.3f}".format('DQN Agent', train_stats['total_return'],
                                                                            test_stats['total_return'], train_sharpe,
                                                                            test_sharpe)
    print(dqn_stats_str)
    b_and_h_stats_str = "{:<20} {:<15.2%} {:<15.2%} {:<15} {:<15}".format('Buy & Hold', train_benchmark_return,
                                                                          test_benchmark_return, 'N/A', 'N/A')
    print(b_and_h_stats_str)
    print("")
    print("DQN vs. Benchmark Excess Return:")
    print("  Training: {:.2%}".format(train_stats['total_return'] - train_benchmark_return))
    print("  Testing:  {:.2%}".format(test_stats['total_return'] - test_benchmark_return))
    print("")
    print("Detailed Statistics:")
    print(f"  Training Period: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Testing Period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")

    if train_history:
        train_df_hist = pd.DataFrame(train_history)
        print(
            f"  Training - Buy Actions: {(train_df_hist['action'] == Actions.BUY.value).sum()}, Sell Actions: {(train_df_hist['action'] == Actions.SELL.value).sum()}")
    if test_history:
        test_df_hist = pd.DataFrame(test_history)
        print(
            f"  Testing - Buy Actions: {(test_df_hist['action'] == Actions.BUY.value).sum()}, Sell Actions: {(test_df_hist['action'] == Actions.SELL.value).sum()}")


def safe_model_save(agent, config, device_manager):
    """Saves the model safely, ensuring it is saved on the CPU for better compatibility."""
    try:
        fund_code = config['data']['fund_code']
        model_path = f"dqn_model_{fund_code}_{device_manager.device.type}.pth"
        policy_state = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
        target_state = {k: v.cpu() for k, v in agent.target_net.state_dict().items()}
        torch.save({
            'policy_net_state_dict': policy_state,
            'target_net_state_dict': target_state,
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'config': config,
            'model_info': {
                'state_dim': agent.state_dim,
                'action_dim': agent.action_dim,
                'network_type': agent.network_type,
                'hidden_layers': agent.config['feed_forward_config'][
                    'dqn_hidden_layers'] if agent.network_type == "feed_forward" else agent.config['cnn_config'][
                    'dqn_hidden_layers'],
                'dropout_rate': agent.dropout_rate,
                'date_length': agent.config.get('cnn_config', {}).get('date_length'),
                'num_market_features': agent.config.get('num_market_features'),
                'conv_kernel_size': agent.config.get('cnn_config', {}).get('conv_kernel_size'),
                'cnn_out_channels': agent.config.get('cnn_config', {}).get('cnn_out_channels'),
                'use_attention': agent.config.get('cnn_config', {}).get('use_attention'),
                'num_heads': agent.config.get('cnn_config', {}).get('num_heads'),
                'dim_feedforward': agent.config.get('cnn_config', {}).get('dim_feedforward'),
                'num_attention_layers': agent.config.get('cnn_config', {}).get('num_attention_layers'),
                'use_noisy_net': agent.use_noisy_net,
                'noisy_std_init': agent.config.get('noisy_std_init')
            }
        }, model_path)
        print(f"Model saved successfully to: {model_path}")
        return model_path
    except Exception as e:
        print(f"Warning: Failed to save model: {e}")
        return None


# --- 8. Main Execution Block ---
if __name__ == '__main__':
    print("=" * 80)
    print("GPU-Optimized DQN Trading System")
    print("=" * 80)
    s_t = time.time()
    try:
        print("")
        print("1. Initializing Device Manager...")
        device_manager = DeviceManager(CONFIG)

        print("")
        print("2. Setting Random Seeds...")
        set_seeds(CONFIG["infra"]["seed"])

        print("")
        print("System Configuration:")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  Device: {device_manager.device}")
        print(f"  Mixed Precision: {device_manager.use_amp}")

        print("")
        print("3. Loading Data...")
        full_df = get_data(CONFIG)
        print(f"Raw data shape: {full_df.shape}")
        print(f"Date range: {full_df.index[0]} to {full_df.index[-1]}")

        print("")
        print("4. Preprocessing Data...")
        train_df, test_df = preprocess_and_split_data(full_df, CONFIG)
        print(f"Training data: {train_df.shape} ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"Testing data: {test_df.shape} ({test_df.index[0]} to {test_df.index[-1]})")

        print("")
        print("5. Training DQN Agent...")
        trained_agent, training_losses, episode_rewards = train_dqn(CONFIG, train_df, device_manager)

        print("")
        print("6. Evaluating Agent on Training Data...")
        train_history = evaluate_agent(trained_agent, train_df, CONFIG, device_manager)

        print("")
        print("7. Evaluating Agent on Test Data...")
        test_history = evaluate_agent(trained_agent, test_df, CONFIG, device_manager)

        print("")
        print("8. Generating Performance Analysis...")
        print_performance_comparison(train_history, test_history, train_df, test_df, CONFIG)

        print("")
        print("9. Generating Visualizations...")
        fund_code = CONFIG['data']['fund_code']
        plot_results(train_history, CONFIG, train_df, f"DQN Performance (Training Set) - {fund_code}", training_losses,
                     episode_rewards)
        plot_results(test_history, CONFIG, test_df, f"DQN Performance (Test Set) - {fund_code}")

        print("")
        print("10. Saving Model...")
        model_path = safe_model_save(trained_agent, CONFIG, device_manager)

        print("")
        print("=" * 80)
        print("Execution Completed Successfully")
        print("=" * 80)

    except KeyboardInterrupt:
        print("")
        print("")
        print("Execution interrupted by user.")
    except Exception as e:
        print("")
        print(f"An uncaught error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("")
        print("11. Final Cleanup...")
        if 'device_manager' in locals() and device_manager.device.type == "cuda":
            torch.cuda.empty_cache()
            device_manager.print_memory_usage()
            print("GPU memory cleared.")
        print("Program finished.")

    print("")
    print(f"Total execution time: {time.time() - s_t:.2f} seconds")
