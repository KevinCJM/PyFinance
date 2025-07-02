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
import warnings

warnings.filterwarnings('ignore')

# 禁用 torch.compile 相关的错误
import torch._dynamo

torch._dynamo.config.suppress_errors = True

the_fund_code = '510050.SH'  # 上证50ETF
the_basic_info_file_path_dict = {
    "log": "./Data/wide_log_return_df.parquet",  # 对数收益率
    "high": "./Data/wide_high_df.parquet",  # 最高价
    "low": "./Data/wide_low_df.parquet",  # 最低价
    "vol": "./Data/wide_vol_df.parquet",  # 成交量
    "amount": "./Data/wide_amount_df.parquet",  # 成交额
    "close": "./Data/wide_close_df.parquet",  # 收盘价
    "open": "./Data/wide_open_df.parquet",  # 开盘价
}
the_metrics_info_file_path_dict = [
    "./Data/Metrics/2d.parquet",  # 2天区间指标数据
    "./Data/Metrics/5d.parquet",
    "./Data/Metrics/10d.parquet",
    "./Data/Metrics/15d.parquet",
    "./Data/Metrics/25d.parquet",
    "./Data/Metrics/50d.parquet",
    "./Data/Metrics/75d.parquet",
    "./Data/Metrics/5m.parquet",
    "./Data/Metrics/6m.parquet",
    "./Data/Metrics/rolling_metrics.parquet",  # 滚动技术指标数据
]

# --- 1. 配置中心 (CONFIG) - 修复兼容性问题 ---
CONFIG = {
    "infra": {
        "seed": 42,
        # GPU相关配置
        "device": "auto",  # "auto", "cpu", "cuda", "cuda:0", etc.
        "use_mixed_precision": True,  # 使用混合精度训练
        "pin_memory": True,  # 固定内存，加速GPU传输
        "non_blocking": True,  # 非阻塞传输
        "compile_model": False,  # 禁用PyTorch 2.0 模型编译（避免Triton依赖）
        "auto_select_best_gpu": True,  # 是否选择最佳单个GPU
        "gpu_memory_fraction": 0.8,  # GPU内存使用比例
        "enable_cudnn_benchmark": True,  # 启用CuDNN benchmark
    },
    "preprocessing": {
        "train_split_ratio": 0.8,
        "use_gpu_preprocessing": True,  # GPU加速预处理
        "volatility_window": 21,  # 新增：计算波动率的滚动窗口
    },
    "environment": {
        "initial_capital": 100000,
        "transaction_cost_pct": 0.001,  # 手续费依然保留
        "trade_penalty": 0.1,  # 固定的交易行为惩罚
        "hard_stop_loss_pct": -0.15,  # 当未实现亏损达到-15%时，强制平仓
    },
    "agent": {
        "state_dim": None,
        "action_dim": 3,
        "dqn_hidden_layers": [512, 256, 128],  # 增大网络容量以利用GPU
        "memory_size": 50000,  # 增大经验回放缓冲区
        "batch_size": 256,  # 增大批处理大小以提高GPU利用率
        "gamma": 0.99,
        "learning_rate": 0.001,
        "target_update_freq": 100,  # 降低更新频率
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_linear_decay_steps": 300000,
        # GPU优化参数
        "gradient_clip": 1.0,  # 梯度裁剪
        "weight_decay": 1e-5,  # L2正则化
        "dropout_rate": 0.1,  # Dropout防止过拟合
    },
    "training": {
        "num_episodes": 100,  # 增加训练轮数
        "update_frequency": 4,  # 每N步更新一次
        # "prefetch_factor": 4,  # 数据预取因子
        # "num_workers": 4,  # 数据加载线程数
    }
}


class DeviceManager:
    """设备管理器，自动选择和配置最佳设备"""

    def __init__(self, config):
        self.config = config
        self.device = self._select_device()
        self.use_amp = config["infra"]["use_mixed_precision"] and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        self._setup_gpu_optimizations()

    def _select_device(self):
        device_config = self.config["infra"]["device"]

        if device_config == "auto":
            if torch.cuda.is_available():
                # 检查CUDA版本兼容性
                cuda_version = torch.version.cuda
                print(f"CUDA Version: {cuda_version}")

                # 选择显存最大的GPU
                gpu_count = torch.cuda.device_count()
                print(f"Available GPUs: {gpu_count}")

                if gpu_count > 1:
                    max_memory = 0
                    best_gpu = 0
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        memory = props.total_memory
                        print(f"GPU {i}: {props.name}, Memory: {memory / 1e9:.1f} GB")
                        if memory > max_memory:
                            max_memory = memory
                            best_gpu = i
                    device = torch.device(f"cuda:{best_gpu}")
                else:
                    device = torch.device("cuda:0")
                    props = torch.cuda.get_device_properties(0)
                    print(f"GPU 0: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")

                print(f"Selected device: {device}")
            else:
                device = torch.device("cpu")
                print("CUDA not available, using CPU")
        else:
            device = torch.device(device_config)
            print(f"Using specified device: {device}")

        return device

    def _setup_gpu_optimizations(self):
        if self.device.type == "cuda":
            try:
                # 设置GPU内存分配策略
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(
                        self.config["infra"]["gpu_memory_fraction"],
                        device=self.device
                    )
                    print(f"Set GPU memory fraction to {self.config['infra']['gpu_memory_fraction']}")

                # 启用CuDNN benchmark模式（如果配置启用）
                if self.config["infra"]["enable_cudnn_benchmark"]:
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.deterministic = False  # 牺牲确定性换取性能
                    print("CuDNN benchmark enabled")
                else:
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    print("CuDNN deterministic mode enabled")

            except Exception as e:
                print(f"Warning: GPU optimization setup failed: {e}")

    def to_device(self, tensor, non_blocking=None):
        """将张量移动到设备"""
        if non_blocking is None:
            non_blocking = self.config["infra"]["non_blocking"] and self.device.type == "cuda"

        try:
            return tensor.to(self.device, non_blocking=non_blocking)
        except Exception as e:
            print(f"Warning: Failed to move tensor to device with non_blocking={non_blocking}, trying blocking mode")
            return tensor.to(self.device, non_blocking=False)

    def print_memory_usage(self):
        """打印GPU内存使用情况"""
        if self.device.type == "cuda":
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                cached = torch.cuda.memory_reserved(self.device) / 1e9
                total = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")
                print(f"GPU Memory Usage: {(allocated / total) * 100:.1f}%")
            except Exception as e:
                print(f"Warning: Could not get GPU memory info: {e}")


def set_seeds(seed):
    """设置所有随机种子以保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# --- 2. 数据加载与预处理 - GPU优化版本 ---
def get_data(fund_code, basic_info_file_path_dict, metrics_info_file_path_dict):
    print("Loading basic market data...")
    the_final_df = None
    for file_type, file_path in basic_info_file_path_dict.items():
        try:
            df = pd.read_parquet(file_path)
            fund_data = df[[fund_code]].copy()
            fund_data.columns = [file_type]

            if the_final_df is None:
                the_final_df = fund_data
            else:
                the_final_df = the_final_df.join(fund_data, how='outer')
            print(f"Loaded {file_type} data: {fund_data.shape}")
        except Exception as e:
            print(f"Warning: Failed to load {file_type} from {file_path}: {e}")

    print("Loading metrics data...")
    for i, file_path in enumerate(metrics_info_file_path_dict):
        try:
            df = pd.read_parquet(file_path)
            df = df[df['ts_code'] == fund_code].reset_index(drop=True)
            df = df.drop(columns=['ts_code'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index(['date']).sort_index()
            the_final_df = the_final_df.join(df, how='outer')
            print(f"Loaded metrics file {i + 1}/{len(metrics_info_file_path_dict)}: {df.shape}")
        except Exception as e:
            print(f"Warning: Failed to load metrics from {file_path}: {e}")

    print(f"Final combined data shape: {the_final_df.shape}")
    return the_final_df


def preprocess_and_split_data(df, config):
    """
    预处理函数，增加滚动波动率作为新特征。
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    first_valid_indices = [df[col].first_valid_index() for col in df.columns]
    start_date = max(dt for dt in first_valid_indices if dt is not None)
    df = df.loc[start_date:].copy()

    # --- 新增：预计算用于奖励函数的滚动波动率 ---
    vol_window = config['preprocessing']['volatility_window']
    # 我们用对数收益率的滚动标准差来衡量波动率
    df['reward_volatility'] = df['log'].rolling(window=vol_window, min_periods=vol_window).std()

    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # 分割数据
    split_index = int(len(df) * config['preprocessing']['train_split_ratio'])
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # 归一化
    # 注意：'close', 'high', 'low'和'reward_volatility'不参与归一化，仅供环境内部使用
    feature_cols = df.columns.drop(['close', 'high', 'low', 'reward_volatility'])

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols]), index=train_df.index, columns=feature_cols)
    train_scaled[['close', 'high', 'low', 'reward_volatility']] = train_df[
        ['close', 'high', 'low', 'reward_volatility']]

    test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]), index=test_df.index, columns=feature_cols)
    test_scaled[['close', 'high', 'low', 'reward_volatility']] = test_df[['close', 'high', 'low', 'reward_volatility']]

    train_scaled.dropna(inplace=True)
    test_scaled.dropna(inplace=True)

    return train_scaled, test_scaled


# --- 3. 优化的DQN网络 ---
class OptimizedDQN(nn.Module):
    """GPU优化的DQN网络"""

    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate=0.1):
        super(OptimizedDQN, self).__init__()

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # 使用LayerNorm替代BatchNorm
            layers.append(nn.ReLU(inplace=True))  # inplace操作节省内存
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

        # Xavier初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


# --- 4. GPU优化的环境 (奖励逻辑完全重构) ---
class StockTradingEnv(gym.Env):
    def __init__(self, df, config, device_manager=None):
        super(StockTradingEnv, self).__init__()
        self.df = df
        # 分离特征和用于计算的原始数据
        self.feature_df = df.drop(columns=['close', 'high', 'low', 'reward_volatility'], errors='ignore')
        self.price_series = df['close']
        self.high_series = df['high']
        self.low_series = df['low']
        self.volatility_series = df['reward_volatility']

        self.device_manager = device_manager
        self.config = config
        self.env_config = config['environment']

        self.hard_stop_loss_pct = self.env_config['hard_stop_loss_pct']
        self.initial_capital = self.env_config['initial_capital']
        self.transaction_cost_pct = self.env_config['transaction_cost_pct']
        self.trade_penalty = self.env_config['trade_penalty']

        self.action_space = spaces.Discrete(config['agent']['action_dim'])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_df.columns) + 2,), dtype=np.float32
        )

        self._precompute_features()
        self.reset()

    def _precompute_features(self):
        self.features_array = self.feature_df.values.astype(np.float32)
        self.prices_array = self.price_series.values.astype(np.float32)
        print(f"Precomputed features: {self.features_array.shape}")

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.history = []
        self.entry_price = 0
        return self._get_observation()

    def _get_observation(self):
        market_obs = self.features_array[self.current_step]
        position_status = 1.0 if self.shares > 0 else 0.0
        if self.shares > 0:
            current_price = self.prices_array[self.current_step]
            unrealized_pnl_pct = (current_price / self.entry_price) - 1.0 if self.entry_price > 0 else 0.0
        else:
            unrealized_pnl_pct = 0.0
        return np.concatenate([market_obs, [position_status, unrealized_pnl_pct]]).astype(np.float32)

    def step(self, action):
        prev_portfolio_value = self.portfolio_value

        # --- 1. 实现强制止损机制 ---
        unrealized_pnl = 0
        if self.shares > 0 and self.entry_price > 0:
            unrealized_pnl = (self.price_series.iloc[self.current_step] / self.entry_price) - 1

        # 如果触发了强制止损，当前步的动作被覆盖为“卖出”
        if self.shares > 0 and unrealized_pnl < self.hard_stop_loss_pct:
            action = 2  # 覆盖动作为卖出

        # --- 2. 执行交易 (使用最差价格) ---
        self._take_action(action)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # 计算新的资产组合价值
        current_price = self.price_series.iloc[self.current_step if not done else -1]
        self.portfolio_value = self.cash + self.shares * current_price

        # --- 3. 全新的、基于原则的奖励函数 ---
        reward = 0.0

        # a. 计算核心奖励：风险调整后的超额收益 (Alpha)
        if prev_portfolio_value > 0:
            portfolio_log_return = np.log(self.portfolio_value / prev_portfolio_value)

            # 基准收益：买入并持有策略的单步对数收益
            benchmark_log_return = np.log(
                self.price_series.iloc[self.current_step if not done else -1] / self.price_series.iloc[
                    self.current_step - 1]
            )

            # 获取预先计算的市场波动率
            market_volatility = self.volatility_series.iloc[self.current_step if not done else -1]

            # 计算超额收益
            excess_return = portfolio_log_return - benchmark_log_return

            # 如果波动率很小或为0，则不进行风险调整，避免除以零
            if market_volatility > 1e-9:
                reward = excess_return / market_volatility
            else:
                reward = excess_return  # 如果市场无波动，奖励就是纯粹的超额收益

        # b. 对交易行为本身进行惩罚
        if action == 1 or action == 2:  # 只要执行了买或卖
            reward -= self.trade_penalty

        # --- 奖励计算结束 ---

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            'date': self.df.index[self.current_step - 1],
            'portfolio_value': self.portfolio_value,
            'action': action,
            'price': self.price_series.iloc[self.current_step - 1],
            'reward': reward
        }
        self.history.append(info)

        return obs, reward, done, info

    def _take_action(self, action):
        """
        重构交易执行逻辑：在最差价格成交，并包含手续费。
        """
        current_step_index = self.current_step

        # 动作1: 买入 (全仓买入)
        if action == 1 and self.cash > 0:
            # --- 在当日最高价买入 ---
            buy_price = self.high_series.iloc[current_step_index]

            # 计算并扣除手续费
            cost = self.cash * self.transaction_cost_pct
            buy_capital = self.cash - cost
            if buy_price > 0:
                self.shares = buy_capital / buy_price
                self.cash = 0
                self.entry_price = buy_price  # 记录入场价格

        # 动作2: 卖出 (全仓卖出)
        elif action == 2 and self.shares > 0:
            # --- 在当日最低价卖出 ---
            sell_price = self.low_series.iloc[current_step_index]

            # 计算卖出所得
            proceeds = self.shares * sell_price
            # 计算并扣除手续费
            cost = proceeds * self.transaction_cost_pct
            self.cash = proceeds - cost
            self.shares = 0
            self.entry_price = 0  # 清空入场价格


# --- 5. GPU优化的DQN智能体 (Double DQN) ---
class OptimizedDQNAgent:
    def __init__(self, config, device_manager):
        self.config = config['agent']
        self.device_manager = device_manager
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']

        # Epsilon衰减
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay_step = (
                                          self.config['epsilon_start'] - self.config['epsilon_min']
                                  ) / self.config['epsilon_linear_decay_steps']

        self.memory_size = self.config['memory_size']
        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.learning_rate = self.config['learning_rate']
        self.target_update_freq = self.config['target_update_freq']
        self.gradient_clip = self.config['gradient_clip']

        # 经验回放缓冲区
        self.memory = collections.deque(maxlen=self.memory_size)

        print(f"Initializing DQN networks with input_dim={self.state_dim}, output_dim={self.action_dim}")

        # 网络初始化
        self.policy_net = OptimizedDQN(
            self.state_dim,
            self.action_dim,
            self.config['dqn_hidden_layers'],
            self.config['dropout_rate']
        )
        self.target_net = OptimizedDQN(
            self.state_dim,
            self.action_dim,
            self.config['dqn_hidden_layers'],
            self.config['dropout_rate']
        )

        # 移动到设备
        self.policy_net = device_manager.to_device(self.policy_net)
        self.target_net = device_manager.to_device(self.target_net)

        # 同步目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        print(f"Networks moved to device: {device_manager.device}")

        # 优化器
        self.optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=1000, factor=0.8
        )

        self.loss_fn = nn.SmoothL1Loss()  # 使用Huber Loss
        self.learn_step_counter = 0

        print("DQN Agent initialized successfully")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        position_status = state[-2]
        valid_actions = [0, 1] if position_status == 0 else [0, 2]

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = self.device_manager.to_device(
                torch.FloatTensor(state).unsqueeze(0)
            )
            q_values = self.policy_net(state_tensor)

            for i in range(self.action_dim):
                if i not in valid_actions:
                    q_values[0][i] = -float('inf')

            return torch.argmax(q_values[0]).item()

    def replay(self, global_step):
        """
        优化的经验回放函数，已实现Double DQN逻辑。
        """
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = self.device_manager.to_device(torch.FloatTensor(np.array(states)))
        actions = self.device_manager.to_device(torch.LongTensor(actions).unsqueeze(1))
        rewards = self.device_manager.to_device(torch.FloatTensor(rewards).unsqueeze(1))
        next_states = self.device_manager.to_device(torch.FloatTensor(np.array(next_states)))
        dones = self.device_manager.to_device(torch.BoolTensor(dones).unsqueeze(1))

        # 使用混合精度训练
        use_amp = self.device_manager.use_amp
        scaler = self.device_manager.scaler

        with autocast(enabled=use_amp):
            # 1. 计算当前Q值: Q_policy(s, a)
            current_q_values = self.policy_net(states).gather(1, actions)

            # --- Double DQN 核心修改 ---
            with torch.no_grad():
                # 步骤1: 使用 policy_net 选择下一状态的最佳动作 a*
                best_next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)

                # 步骤2: 使用 target_net 评估动作 a* 的价值
                next_q_values_for_best_actions = self.target_net(next_states).gather(1, best_next_actions)
            # --- Double DQN 修改结束 ---

            # 如果是done状态，未来预期奖励为0
            next_q_values_for_best_actions[dones] = 0.0

            # 计算最终的目标Q值
            target_q_values = rewards + (self.gamma * next_q_values_for_best_actions)

            # 计算损失
            loss = self.loss_fn(current_q_values, target_q_values)

        # 梯度更新
        if use_amp:
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
            self.optimizer.step()

        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.config['epsilon_start'] - self.epsilon_decay_step * global_step)

        # 更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


# --- 6. 训练与评估函数 ---
def calculate_episode_stats(history, initial_capital):
    if not history:
        return {
            "final_value": initial_capital,
            "total_return": 0.0,
            "annual_volatility": 0.0,
            "total_rewards": 0.0
        }

    history_df = pd.DataFrame(history)
    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1

    daily_returns = history_df['portfolio_value'].pct_change().dropna()
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        annual_volatility = 0.0
    else:
        annual_volatility = daily_returns.std() * np.sqrt(252)

    total_rewards = history_df['reward'].sum()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_volatility": annual_volatility,
        "total_rewards": total_rewards
    }


def train_dqn(config, train_df, device_manager):
    """GPU优化的训练函数"""
    print("Initializing training environment...")
    env = StockTradingEnv(train_df, config, device_manager)
    config['agent']['state_dim'] = env.observation_space.shape[0]

    print("Initializing DQN agent...")
    agent = OptimizedDQNAgent(config, device_manager)

    initial_capital = config['environment']['initial_capital']
    num_episodes = config['training']['num_episodes']
    update_frequency = config['training']['update_frequency']
    global_step_counter = 0

    print(f"--- Starting Training for {num_episodes} episodes ---")
    print(f"Device: {device_manager.device}")
    print(f"Mixed Precision: {device_manager.use_amp}")
    print(f"State dimension: {config['agent']['state_dim']}")
    print(f"Memory size: {agent.memory_size}")
    print(f"Batch size: {agent.batch_size}")

    training_losses = []
    episode_rewards = []

    try:
        for e in tqdm(range(num_episodes), desc="Training Episodes"):
            state = env.reset()
            done = False
            episode_loss = 0
            episode_reward = 0
            step_count = 0

            while not done:
                global_step_counter += 1
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                step_count += 1

                # 定期更新网络
                if (len(agent.memory) > agent.batch_size and
                        global_step_counter % update_frequency == 0):
                    try:
                        loss = agent.replay(global_step_counter)
                        if loss is not None:
                            episode_loss += loss
                    except Exception as replay_error:
                        print(f"Warning: Replay failed at step {global_step_counter}: {replay_error}")

            training_losses.append(episode_loss / max(step_count / update_frequency, 1))
            episode_rewards.append(episode_reward)

            # 定期打印训练日志和清理GPU内存
            if (e + 1) % 10 == 0 or e == num_episodes - 1:
                stats = calculate_episode_stats(env.history, initial_capital)
                avg_loss = np.mean(training_losses[-10:]) if training_losses else 0
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0

                print(f"\n--- Episode {e + 1}/{num_episodes} ---")
                print(f"  Average Loss (last 10): {avg_loss:.6f}")
                print(f"  Average Reward (last 10): {avg_reward:.4f}")
                print(f"  Total Rewards: {stats['total_rewards']:.4f}")
                print(f"  Final Portfolio Value: {stats['final_value']:.2f}")
                print(f"  Total Return: {stats['total_return']:.2%}")
                print(f"  Annual Volatility: {stats['annual_volatility']:.2%}")
                print(f"  Epsilon: {agent.epsilon:.4f}")
                print(f"  Learning Rate: {agent.optimizer.param_groups[0]['lr']:.6f}")
                print(f"  Memory Size: {len(agent.memory)}")

                # 打印GPU内存使用情况
                if device_manager.device.type == "cuda":
                    device_manager.print_memory_usage()
                    # 定期清理GPU缓存
                    if (e + 1) % 20 == 0:
                        torch.cuda.empty_cache()
                        print("GPU cache cleared")

            # 学习率调度
            if episode_rewards:
                agent.scheduler.step(episode_rewards[-1])

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as train_error:
        print(f"Training error: {train_error}")
        import traceback
        traceback.print_exc()

    print("\n--- Training Finished ---")

    # 最终清理GPU缓存
    if device_manager.device.type == "cuda":
        torch.cuda.empty_cache()
        print("Final GPU cache cleanup completed")

    return agent, training_losses, episode_rewards


def evaluate_agent(agent, eval_df, config, device_manager):
    """GPU优化的评估函数"""
    print("Starting agent evaluation...")
    agent.epsilon = 0  # 关闭探索
    agent.policy_net.eval()  # 设置为评估模式

    env = StockTradingEnv(eval_df, config, device_manager)
    state = env.reset()
    done = False

    step_count = 0
    try:
        with torch.no_grad():  # 评估时不需要梯度
            while not done:
                action = agent.act(state)
                state, _, done, _ = env.step(action)
                step_count += 1

                # 定期显示进度
                if step_count % 100 == 0:
                    progress = step_count / len(eval_df) * 100
                    print(f"Evaluation progress: {progress:.1f}%")

    except Exception as eval_error:
        print(f"Evaluation error: {eval_error}")

    agent.policy_net.train()  # 恢复训练模式
    print(f"Evaluation completed: {step_count} steps")
    return env.history


# --- 7. 增强的可视化函数 ---
def plot_results(history, config, benchmark_df, title, training_losses=None, episode_rewards=None):
    """增强的可视化函数，包含训练指标"""
    if training_losses and episode_rewards:
        # 创建4个子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    else:
        # 创建2个子图
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

    history_df = pd.DataFrame(history).set_index('date')
    initial_capital = config['environment']['initial_capital']

    # --- 图1: 净值与价格对比 ---
    ax1.set_title(title, fontsize=16)

    normalized_portfolio_value = history_df['portfolio_value'] / initial_capital
    ax1.plot(history_df.index, normalized_portfolio_value,
             label='DQN Agent Portfolio (Normalized)', color='blue', linewidth=2, zorder=2)

    normalized_benchmark = benchmark_df['close'] / benchmark_df['close'].iloc[0]
    ax1.plot(benchmark_df.index, normalized_benchmark,
             label='Buy and Hold Benchmark (Normalized)', color='grey',
             linestyle='--', linewidth=2, zorder=1)

    ax1.set_ylabel('Normalized Net Value (Start=1)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)

    # 第二个Y轴显示原始价格
    ax1_twin = ax1.twinx()
    ax1_twin.plot(benchmark_df.index, benchmark_df['close'],
                  label='ETF Price (Raw)', color='orange', alpha=0.4, zorder=0)
    ax1_twin.set_ylabel('ETF Price', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # --- 训练指标图 ---
    if training_losses and episode_rewards:
        # 图2: 训练损失
        ax2.set_title('Training Loss Over Episodes')
        if training_losses:
            ax2.plot(training_losses, color='red', alpha=0.7)
            # 添加移动平均线
            if len(training_losses) > 10:
                window = min(10, len(training_losses) // 4)
                ma_losses = pd.Series(training_losses).rolling(window=window).mean()
                ax2.plot(ma_losses, color='darkred', linewidth=2, label=f'MA({window})')
                ax2.legend()
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        # 图4: 奖励变化
        ax4.set_title('Episode Rewards Over Time')
        if episode_rewards:
            ax4.plot(episode_rewards, color='green', alpha=0.7)
            # 添加移动平均线
            if len(episode_rewards) > 10:
                window = min(10, len(episode_rewards) // 4)
                ma_rewards = pd.Series(episode_rewards).rolling(window=window).mean()
                ax4.plot(ma_rewards, color='darkgreen', linewidth=2, label=f'MA({window})')
                ax4.legend()
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Total Reward')
        ax4.grid(True, alpha=0.3)

    # --- 持仓状态时间轴 ---
    ax3.set_title('Agent Position Status')

    actions = history_df['action']
    buy_times = actions[actions == 1].index
    sell_times = actions[actions == 2].index

    position_active = False
    last_buy_time = None
    legend_added = False

    for date in history_df.index:
        if date in buy_times:
            position_active = True
            last_buy_time = date
        if date in sell_times:
            if position_active:
                label = 'Long Position' if not legend_added else ""
                ax3.axvspan(last_buy_time, date, color='green', alpha=0.3, label=label)
                legend_added = True
            position_active = False

    if position_active and last_buy_time is not None:
        label = 'Long Position' if not legend_added else ""
        ax3.axvspan(last_buy_time, history_df.index[-1], color='green', alpha=0.3, label=label)

    ax3.set_yticks([])
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')

    handles, labels = ax3.get_legend_handles_labels()
    if handles:
        ax3.legend(handles, labels, loc='upper left')

    plt.tight_layout()
    plt.show()


def print_performance_comparison(train_history, test_history, train_df, test_df, config):
    """打印详细的性能对比"""
    initial_capital = config['environment']['initial_capital']

    # 计算DQN性能
    train_stats = calculate_episode_stats(train_history, initial_capital)
    test_stats = calculate_episode_stats(test_history, initial_capital)

    # 计算基准性能 (买入持有)
    train_benchmark_return = (train_df['close'].iloc[-1] / train_df['close'].iloc[0]) - 1
    test_benchmark_return = (test_df['close'].iloc[-1] / test_df['close'].iloc[0]) - 1

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Strategy':<20} {'Train Return':<15} {'Test Return':<15} {'Train Sharpe':<15} {'Test Sharpe':<15}")
    print("-" * 80)

    # 计算夏普比率
    def calculate_sharpe(history, initial_capital):
        if not history:
            return 0.0
        df = pd.DataFrame(history)
        returns = df['portfolio_value'].pct_change().dropna()
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    train_sharpe = calculate_sharpe(train_history, initial_capital)
    test_sharpe = calculate_sharpe(test_history, initial_capital)

    print(f"{'DQN Agent':<20} {train_stats['total_return']:<15.2%} {test_stats['total_return']:<15.2%} "
          f"{train_sharpe:<15.3f} {test_sharpe:<15.3f}")
    print(f"{'Buy & Hold':<20} {train_benchmark_return:<15.2%} {test_benchmark_return:<15.2%} {'N/A':<15} {'N/A':<15}")

    print(f"\nDQN vs Benchmark Excess Return:")
    print(f"  Training: {(train_stats['total_return'] - train_benchmark_return):.2%}")
    print(f"  Testing:  {(test_stats['total_return'] - test_benchmark_return):.2%}")

    # 额外的统计信息
    print(f"\nDetailed Statistics:")
    print(f"  Training Period: {train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Testing Period: {test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')}")

    if train_history:
        train_df_hist = pd.DataFrame(train_history)
        buy_actions = (train_df_hist['action'] == 1).sum()
        sell_actions = (train_df_hist['action'] == 2).sum()
        print(f"  Training - Buy Actions: {buy_actions}, Sell Actions: {sell_actions}")

    if test_history:
        test_df_hist = pd.DataFrame(test_history)
        buy_actions = (test_df_hist['action'] == 1).sum()
        sell_actions = (test_df_hist['action'] == 2).sum()
        print(f"  Testing - Buy Actions: {buy_actions}, Sell Actions: {sell_actions}")


def safe_model_save(agent, config, fund_code, device_manager):
    """安全地保存模型"""
    try:
        model_path = f"dqn_model_{fund_code}_{device_manager.device.type}.pth"

        # 确保模型在CPU上进行保存
        policy_state = agent.policy_net.state_dict()
        target_state = agent.target_net.state_dict()

        # 将状态字典移动到CPU
        if device_manager.device.type == "cuda":
            policy_state = {k: v.cpu() for k, v in policy_state.items()}
            target_state = {k: v.cpu() for k, v in target_state.items()}

        torch.save({
            'policy_net_state_dict': policy_state,
            'target_net_state_dict': target_state,
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'config': config,
            'device_type': device_manager.device.type,
            'model_info': {
                'state_dim': agent.state_dim,
                'action_dim': agent.action_dim,
                'hidden_layers': agent.config['dqn_hidden_layers'],
                'dropout_rate': agent.config['dropout_rate']
            }
        }, model_path)
        print(f"Model saved successfully to {model_path}")
        return model_path
    except Exception as e:
        print(f"Warning: Failed to save model: {e}")
        return None


# --- 8. 主程序 ---
if __name__ == '__main__':
    print("=" * 80)
    print("GPU-OPTIMIZED DQN TRADING SYSTEM")
    print("=" * 80)
    s_t = time.time()

    try:
        # 初始化设备管理器
        print("\n1. Initializing Device Manager...")
        device_manager = DeviceManager(CONFIG)

        # 设置随机种子
        print("\n2. Setting Random Seeds...")
        set_seeds(CONFIG["infra"]["seed"])

        print(f"\nSystem Configuration:")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  Device: {device_manager.device}")
        print(f"  Mixed Precision: {device_manager.use_amp}")

        if device_manager.device.type == "cuda":
            print(f"  CUDA Version: {torch.version.cuda}")
            device_manager.print_memory_usage()

        # 1. 加载数据
        print("\n3. Loading Data...")
        full_df = get_data(the_fund_code, the_basic_info_file_path_dict, the_metrics_info_file_path_dict)
        print(f"Raw data shape: {full_df.shape}")
        print(f"Date range: {full_df.index[0]} to {full_df.index[-1]}")

        # 2. 预处理和分割数据
        print("\n4. Preprocessing Data...")
        train_df, test_df = preprocess_and_split_data(full_df, CONFIG)
        print(f"Training data: {train_df.shape} ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"Testing data: {test_df.shape} ({test_df.index[0]} to {test_df.index[-1]})")

        # 3. 训练DQN智能体
        print("\n5. Training DQN Agent...")
        trained_agent, training_losses, episode_rewards = train_dqn(CONFIG, train_df, device_manager)

        # 4. 在训练集上进行评估
        print("\n6. Evaluating Agent on Training Data...")
        train_history = evaluate_agent(trained_agent, train_df, CONFIG, device_manager)
        print(f"Training evaluation completed: {len(train_history)} steps")

        # 5. 在测试集上进行评估
        print("\n7. Evaluating Agent on Test Data...")
        test_history = evaluate_agent(trained_agent, test_df, CONFIG, device_manager)
        print(f"Testing evaluation completed: {len(test_history)} steps")

        # 6. 性能对比
        print("\n8. Performance Analysis...")
        print_performance_comparison(train_history, test_history, train_df, test_df, CONFIG)

        # 7. 可视化结果
        print("\n9. Generating Visualizations...")
        plot_results(train_history, CONFIG, train_df,
                     f"DQN Performance (Training Set) - {the_fund_code}",
                     training_losses, episode_rewards)

        plot_results(test_history, CONFIG, test_df,
                     f"DQN Performance (Test Set) - {the_fund_code}")

        # 8. 保存模型
        print("\n10. Saving Model...")
        model_path = safe_model_save(trained_agent, CONFIG, the_fund_code, device_manager)

        print("\n" + "=" * 80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 最终清理
        print("\n11. Final Cleanup...")
        if 'device_manager' in locals() and device_manager.device.type == "cuda":
            torch.cuda.empty_cache()
            device_manager.print_memory_usage()
            print("GPU memory cleared.")
        print("Program finished.")

    print(f"\nTotal execution time: {time.time() - s_t:.2f} seconds")
