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


# --- 1. 配置中心 (CONFIG) ---
# 将所有超参数和配置集中管理，方便修改和实验
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
            "holding_reward_factor": 1.0,  # 持仓时（无论盈亏），奖励/惩罚的基准乘数
            "missing_trend_penalty_factor": 1.0,  # 空仓时，对错过上涨的惩罚和躲过下跌的奖励的基准乘数
            "dynamic_trade_penalty_factor": 0.1,  # 动态交易惩罚乘数（乘以波动率）
            "fixed_trade_penalty": 0.05  # 固定的交易惩罚值
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
            "num_heads": 8,  # 注意力机制的头数，必须能被d_model整除
            "dim_feedforward": 256,  # Transformer Encoder 内部前馈网络的维度
            "num_attention_layers": 2,  # 堆叠的Transformer Encoder层数
            "dqn_hidden_layers": [256, 128],  # 经过CNN/Attention特征提取后，连接的全连接隐藏层
        },
        "state_dim": None,  # 状态空间的维度 (将由环境动态计算并填充)
        "num_market_features": None,  # 市场特征的数量 (将由环境动态计算并填充)
    },
    # --- 训练过程配置 ---
    "training": {
        "num_episodes": 5,  # 训练的总回合数 (一个回合指从头到尾完整地跑完一次数据集)
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

    def to_device(self, tensor, non_blocking=None):
        """将张量移动到选定的设备。"""
        if non_blocking is None:
            non_blocking = self.config["infra"]["non_blocking"] and self.device.type == "cuda"
        try:
            return tensor.to(self.device, non_blocking=non_blocking)
        except Exception as e:
            print(f"警告: 非阻塞模式移动张量失败 (non_blocking={non_blocking}), 尝试阻塞模式。错误: {e}")
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

    split_index = int(len(df) * config['preprocessing']['train_split_ratio'])
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # 定义环境所需的原始数据列
    env_cols = ['close', 'high', 'low', 'open', 'reward_volatility', 'log']
    # 定义需要输入给模型进行学习的特征列 (这里我们使用所有列作为特征)
    feature_cols = df.columns.tolist()

    # 初始化scaler和pca
    scaler = StandardScaler()
    pca_config = config['preprocessing']['pca']
    pca = None
    if pca_config['enabled']:
        print("开始进行PCA降维...")
        pca_n_components = pca_config.get('n_components', 0.9) if pca_config['mode'] == 'variance_ratio' else pca_config.get('n_components')
        pca = PCA(n_components=pca_n_components)

    # --- 处理训练数据 ---
    # 1. 复制环境所需原始数据
    train_env_data = train_df[env_cols].copy()
    # 2. 标准化所有特征
    train_scaled_features = pd.DataFrame(scaler.fit_transform(train_df[feature_cols]), index=train_df.index, columns=feature_cols)
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
    test_scaled_features = pd.DataFrame(scaler.transform(test_df[feature_cols]), index=test_df.index, columns=feature_cols)
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
                 use_noisy_net=False, noisy_std_init=0.5):
        super(OptimizedDQN, self).__init__()
        self.network_type = network_type
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_noisy_net = use_noisy_net

        LinearLayer = NoisyLinear if self.use_noisy_net else nn.Linear

        if self.network_type == "feed_forward":
            print("使用全连接前馈神经网络")
            layers = []
            prev_dim = input_dim
            for i, hidden_dim in enumerate(hidden_layers):
                layers.append(
                    LinearLayer(prev_dim, hidden_dim, std_init=noisy_std_init) if use_noisy_net else LinearLayer(
                        prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                prev_dim = hidden_dim
            layers.append(
                LinearLayer(prev_dim, output_dim, std_init=noisy_std_init) if use_noisy_net else LinearLayer(prev_dim,
                                                                                                             output_dim))
            self.decision_mlp = nn.Sequential(*layers)
        elif self.network_type == "cnn":
            print("使用卷积神经网络 (CNN)")
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

            self.cnn_extractor = None
            self.attention_encoder = None
            self.decision_mlp = None

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
                fc_input_dim = d_model * sequence_length + 2
            else:
                fc_input_dim = cnn_output_shape[1] * cnn_output_shape[2] * cnn_output_shape[3] + 2

            layers = []
            prev_dim = fc_input_dim
            for i, hidden_dim in enumerate(hidden_layers):
                layers.append(
                    LinearLayer(prev_dim, hidden_dim, std_init=noisy_std_init) if use_noisy_net else LinearLayer(
                        prev_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
                prev_dim = hidden_dim
            layers.append(
                LinearLayer(prev_dim, output_dim, std_init=noisy_std_init) if use_noisy_net else LinearLayer(prev_dim,
                                                                                                             output_dim))
            self.decision_mlp = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported network_type: {network_type}")

        self._initialize_weights()

    def _get_cnn_output_size(self):
        if self.network_type != "cnn":
            return 0
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.channels, self.date_length, self.num_market_features)
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
        if self.network_type == "feed_forward":
            return self.decision_mlp(x)
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
                processed_features = attention_output.view(batch_size, -1)
            else:
                processed_features = cnn_output

            combined_input = torch.cat((processed_features, x_additional), dim=1)
            return self.decision_mlp(combined_input)


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
        self.pending_action = None  # 新增：存储前一个时间步的决策，用于T+1执行
        return self._get_observation()

    def _get_observation(self):
        """获取当前时间步的观察状态，包含历史窗口和当前持仓信息。"""
        # Ensure current_step does not exceed data range
        if self.current_step >= len(self.features_array):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        position_status = 1.0 if self.shares > 0 else 0.0
        current_price_for_pnl = self.prices_array[self.current_step]
        unrealized_pnl_pct = ((
                                          current_price_for_pnl / self.entry_price) - 1.0) if self.shares > 0 and self.entry_price > 0 else 0.0

        if self.network_type == "cnn":
            # Determine the start index for the window
            start_idx = max(0, self.current_step - self.date_length + 1)

            # Extract market observations for the window
            # Pad with zeros if the window is not full at the beginning
            market_obs_window = np.zeros((self.date_length, self.num_market_features), dtype=np.float32)
            actual_window_len = self.current_step - start_idx + 1
            market_obs_window[-actual_window_len:] = self.features_array[start_idx:self.current_step + 1]

            # Flatten the market observation window
            flattened_market_obs = market_obs_window.flatten()

            # Concatenate flattened market observations with current position/PnL
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

        # 4. 硬止损检查：在执行动作前检查是否触发止损
        #    如果触发，则强制执行卖出动作
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

        # 7. 计算执行日的奖励
        reward = self._calculate_reward(was_holding_before_action, self.pending_action,
                                        execution_day_index)

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

    def _calculate_reward(self, was_holding_stock, action, execution_day_index):
        """
        根据新的趋势交易理念计算奖励。
        """
        # 1. 获取基础数据
        market_log_return = self.log_return_array[execution_day_index]
        market_volatility = self.volatility_array[execution_day_index]

        # 2. 标准化市场收益率（统一量纲）
        # 使用波动率对收益率进行标准化，得到风险调整后的趋势强度
        # 添加一个极小值 epsilon 来防止除以零
        normalized_market_return = market_log_return / (market_volatility + 1e-9)

        reward = 0.0

        # 3. 根据持仓状态计算核心奖励/惩罚
        if was_holding_stock:
            # 持仓时：抓住趋势则奖励，承受回撤则惩罚
            reward += normalized_market_return * self.reward_config['holding_reward_factor']
        else:  # 空仓时
            # 空仓时：躲过下跌则奖励，错过上涨则惩罚
            reward -= normalized_market_return * self.reward_config['missing_trend_penalty_factor']

        # 4. 计算交易惩罚
        if action == Actions.BUY.value or action == Actions.SELL.value:
            # 固定惩罚 + 动态惩罚（与波动率正相关）
            trade_penalty = (
                    self.reward_config['fixed_trade_penalty'] +
                    self.reward_config['dynamic_trade_penalty_factor'] * market_volatility
            )
            reward -= trade_penalty

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
            "noisy_std_init": noisy_std_init
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

    def remember(self, state, action, reward, next_state, done):
        """将经验存入回放缓冲区。"""
        self.memory.append((state, action, reward, next_state, done))

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
            q_values = self.policy_net(state_tensor)
            for i in range(self.action_dim):
                if i not in valid_actions:
                    q_values[0][i] = -float('inf')
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
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = self.device_manager.to_device(torch.FloatTensor(np.array(states)))
        actions = self.device_manager.to_device(torch.LongTensor(actions).unsqueeze(1))
        rewards = self.device_manager.to_device(torch.FloatTensor(rewards).unsqueeze(1))
        next_states = self.device_manager.to_device(torch.FloatTensor(np.array(next_states)))
        dones = self.device_manager.to_device(torch.BoolTensor(dones).unsqueeze(1))

        with autocast(enabled=self.device_manager.use_amp):
            current_q_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                best_next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values_for_best_actions = self.target_net(next_states).gather(1, best_next_actions)
            next_q_values_for_best_actions[dones] = 0.0
            target_q_values = rewards + (self.gamma * next_q_values_for_best_actions)
            loss = self.loss_fn(current_q_values, target_q_values)

        self._optimize_model(loss)
        self._update_epsilon(global_step)
        self._update_target_net()
        return loss.item()

    def _replay_n_step(self, global_step):
        """N-step Double DQN学习。"""
        if len(self.memory) < self.batch_size + self.n_steps:
            return None

        if self.use_noisy_net:
            self.reset_noise()

        # 采样起始索引
        indices = random.sample(range(len(self.memory) - self.n_steps), self.batch_size)

        # 准备N-step经验
        states, actions, n_step_rewards, n_step_next_states, n_step_dones = [], [], [], [], []

        for idx in indices:
            # 获取初始状态和动作
            state, action, _, _, _ = self.memory[idx]
            states.append(state)
            actions.append(action)

            # 计算N-step累计奖励和最终状态
            G = 0.0
            gamma_power = 1.0
            final_done = True
            for i in range(self.n_steps):
                _, _, reward, next_s, done = self.memory[idx + i]
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

        with autocast(enabled=self.device_manager.use_amp):
            current_q_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                best_next_actions = self.policy_net(n_step_next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(n_step_next_states).gather(1, best_next_actions)

            next_q_values[n_step_dones] = 0.0
            target_q_values = n_step_rewards + (self.gamma ** self.n_steps) * next_q_values
            loss = self.loss_fn(current_q_values, target_q_values)

        self._optimize_model(loss)
        self._update_epsilon(global_step)
        self._update_target_net()
        return loss.item()

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
                agent.remember(state, action, reward, next_state, done)
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
