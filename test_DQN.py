import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import gym
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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

# --- 1. 配置中心 (CONFIG) ---
# 将所有超参数和配置集中在此处
CONFIG = {
    "infra": {
        "seed": 42,  # 新增：随机种子
    },
    "preprocessing": {
        "train_split_ratio": 0.8,
    },
    "environment": {
        "initial_capital": 100000,
        "transaction_cost_pct": 0.001,  # 交易成本百分比 (0.1%)
        "trade_penalty": 0.05,  # 固定交易惩罚 (奖励单位)
        "slippage_pct": 0.0005,  # 新增：0.05% 的滑点
        "reward_scaling": {
            # 每日持仓奖励的缩放系数
            "holding_pnl_factor": 1.0,
            # 卖出结算时，(回报率 * log(持仓天数)) 的缩放系数
            "settlement_return_factor": 1.0,
        }
    },
    "agent": {
        "state_dim": None,
        "action_dim": 3,
        "dqn_hidden_layers": [256, 128],
        "memory_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.0005,
        "target_update_freq": 20,
        # --- 线性Epsilon衰减配置 ---
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_linear_decay_steps": 250000,  # 在n步内完成衰减
    },
    "training": {
        "num_episodes": 100,  # 增加轮数以获得更好效果
    }
}


def set_seeds(seed):
    """ 设置所有随机种子以保证结果可复现 """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 2. 数据加载与预处理---
def get_data(fund_code, basic_info_file_path_dict, metrics_info_file_path_dict):
    the_final_df = None
    for file_type, file_path in basic_info_file_path_dict.items():
        # 读取parquet文件
        df = pd.read_parquet(file_path)

        # 提取指定基金的数据列
        fund_data = df[[fund_code]].copy()
        fund_data.columns = [file_type]  # 重命名列名为字典的键

        # 合并到final_df
        if the_final_df is None:
            the_final_df = fund_data
        else:
            the_final_df = the_final_df.join(fund_data, how='outer')

    for file_path in metrics_info_file_path_dict:
        # 读取parquet文件
        df = pd.read_parquet(file_path)
        # 提取指定基金的数据列
        df = df[df['ts_code'] == fund_code].reset_index(drop=True)  # 过滤出指定基金的数据
        df = df.drop(columns=['ts_code'])  # 删除ts_code列
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date']).sort_index()  # 将交易日期设置为索引
        the_final_df = the_final_df.join(df, how='outer')  # 合并指标数据
    return the_final_df


def preprocess_and_split_data(df, config):
    """
    重构后的预处理函数，解决了数据泄露和数据起点问题。
    """
    # 替换inf为nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 修正#2: 确定所有指标都可用的统一开始日期
    first_valid_indices = [df[col].first_valid_index() for col in df.columns]
    start_date = max(dt for dt in first_valid_indices if dt is not None)
    df = df.loc[start_date:].copy()

    # 修正#3: 只使用ffill后删除NaN，避免bfill的数据泄露
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    # 分割训练集和测试集
    split_index = int(len(df) * config['preprocessing']['train_split_ratio'])
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # --- 修正#1: 不再将原始'close'价格暴露给scaler ---
    feature_cols = df.columns.drop('close')
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = pd.DataFrame(scaler.transform(train_df[feature_cols]), index=train_df.index, columns=feature_cols)
    train_scaled['close'] = train_df['close']

    test_scaled = pd.DataFrame(scaler.transform(test_df[feature_cols]), index=test_df.index, columns=feature_cols)
    test_scaled['close'] = test_df['close']

    return train_scaled, test_scaled


# --- 3. Gym环境与DQN智能体 (奖励与状态已重构) ---
class StockTradingEnv(gym.Env):
    def __init__(self, df, config):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.feature_df = df.drop(columns=['close'])
        self.price_series = df['close']

        self.config = config
        self.env_config = config['environment']
        self.reward_config = self.env_config['reward_scaling']
        self.initial_capital = self.env_config['initial_capital']
        self.transaction_cost_pct = self.env_config['transaction_cost_pct']
        self.trade_penalty = self.env_config['trade_penalty']
        self.slippage_pct = self.env_config['slippage_pct']  # 新增：读取滑点配置

        self.action_space = spaces.Discrete(config['agent']['action_dim'])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.feature_df.columns) + 2,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.history = []
        self.entry_price = 0
        self.position_steps = 0
        return self._get_observation()

    def _get_observation(self):
        # --- 观测向量只使用归一化后的特征 ---
        market_obs = self.feature_df.iloc[self.current_step].values
        position_status = 1.0 if self.shares > 0 else 0.0
        if self.shares > 0:
            current_price = self.price_series.iloc[self.current_step]
            unrealized_pnl_pct = (current_price / self.entry_price) - 1.0
        else:
            unrealized_pnl_pct = 0.0
        return np.concatenate([market_obs, [position_status, unrealized_pnl_pct]]).astype(np.float32)

    def step(self, action):
        current_price = self.price_series.iloc[self.current_step]
        prev_portfolio_value = self.portfolio_value

        # --- 全新的混合式奖励计算 ---
        reward = 0.0

        # 1. 每日持仓奖励 (解决奖励稀疏问题)
        if self.shares > 0:
            # 计算当日的资产组合价值日收益率
            daily_return = (self.cash + self.shares * current_price - prev_portfolio_value) / prev_portfolio_value

            # 使用 sign(pnl) * sqrt(|pnl|) 对奖励进行塑造
            shaped_holding_reward = np.sign(daily_return) * np.sqrt(np.abs(daily_return))
            reward += shaped_holding_reward * self.reward_config['holding_pnl_factor']

        # 2. 交易终结奖励/惩罚
        if action == 2 and self.shares > 0:  # 仅在卖出时结算
            trade_return = (current_price / self.entry_price) - 1

            # 鼓励长趋势，惩罚长持亏损
            duration_factor = np.log(self.position_steps + 1)

            # 综合奖励: (回报率 * 持有周期因子) - 固定交易惩罚
            settlement_reward = (trade_return * duration_factor) - self.trade_penalty
            reward += settlement_reward * self.reward_config['settlement_return_factor']

        # --- 奖励计算结束 ---

        self._take_action(action, current_price)
        self.current_step += 1

        if self.shares > 0:
            self.position_steps += 1

        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)

        # 更新资产组合价值（用于下一次计算）
        self.portfolio_value = self.cash + self.shares * current_price

        info = {'date': self.df.index[self.current_step - 1], 'portfolio_value': self.portfolio_value, 'action': action,
                'price': current_price, 'reward': reward}
        self.history.append(info)

        return obs, reward, done, info

    def _take_action(self, action, current_price):
        """
        这个方法现在包含了滑点计算。
        """
        # Action 1: Buy
        if action == 1 and self.shares == 0:
            # --- 新增：计算滑点后的实际买入价 ---
            buy_price = current_price * (1 + self.slippage_pct)

            # 用实际买入价计算交易成本和股数
            cost = self.cash * self.transaction_cost_pct
            buy_capital = self.cash - cost
            self.shares = buy_capital / buy_price
            self.cash = 0

            # 记录的入场价也必须是滑点后的真实价格
            self.entry_price = buy_price
            self.position_steps = 1
            self.max_price_in_position = buy_price  # 持仓期最高价的起点也是买入价

        # Action 2: Sell
        elif action == 2 and self.shares > 0:
            # --- 新增：计算滑点后的实际卖出价 ---
            sell_price = current_price * (1 - self.slippage_pct)

            # 用实际卖出价计算总收入和成本
            proceeds = self.shares * sell_price
            cost = proceeds * self.transaction_cost_pct
            self.cash = proceeds - cost
            self.shares = 0

            # 重置交易状态变量
            self.entry_price = 0
            self.position_steps = 0
            self.max_price_in_position = 0


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(DQN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, config):
        self.config = config['agent']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']

        # Epsilon衰减逻辑
        self.epsilon = self.config['epsilon_start']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay_step = (self.config['epsilon_start'] - self.config['epsilon_min']) / self.config[
            'epsilon_linear_decay_steps']

        self.memory_size = self.config['memory_size']
        self.batch_size = self.config['batch_size']
        self.gamma = self.config['gamma']
        self.learning_rate = self.config['learning_rate']
        self.target_update_freq = self.config['target_update_freq']

        self.memory = collections.deque(maxlen=self.memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(self.state_dim, self.action_dim, self.config['dqn_hidden_layers']).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, self.config['dqn_hidden_layers']).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.learn_step_counter = 0

    # ... (remember, act, replay 函数与之前版本相同)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        position_status = state[-2]
        valid_actions = [0, 1] if position_status == 0 else [0, 2]
        if random.random() <= self.epsilon:
            return random.choice(valid_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        for i in range(self.action_dim):
            if i not in valid_actions:
                q_values[0][i] = -float('inf')
        return torch.argmax(q_values[0]).item()

    def replay(self, global_step):
        """
        执行DQN算法的训练步骤。从经验回放缓冲区中采样并更新策略网络参数，
        定期同步目标网络参数。

        流程说明：
        1. 检查经验回放缓冲区是否有足够样本
        2. 随机采样小批量经验
        3. 转换数据为PyTorch张量并移至指定设备
        4. 计算当前Q值和目标Q值
        5. 计算损失并更新策略网络
        6. 定期同步目标网络参数
        """
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放缓冲区随机采样小批量数据
        minibatch = random.sample(self.memory, self.batch_size)

        # 解压小批量数据为独立数组
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 将数据转换为PyTorch张量并移至指定设备（GPU/CPU）
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).unsqueeze(1).to(self.device)

        # 计算当前状态-动作对的Q值
        current_q_values = self.policy_net(states).gather(1, actions)

        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.config['epsilon_start'] - self.epsilon_decay_step * global_step)

        # 计算下一状态的max Q值（目标网络评估，不计算梯度）
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        # 终止状态的下一Q值设为0
        next_q_values[dones] = 0.0
        # 计算目标Q值（贝尔曼方程）
        target_q_values = rewards + (self.gamma * next_q_values)

        # 计算当前Q值和目标Q值之间的损失
        loss = self.loss_fn(current_q_values, target_q_values)

        # 梯度下降更新策略网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新学习步数计数器
        self.learn_step_counter += 1
        # 定期同步目标网络参数
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


# --- 4. 训练与评估函数 ---
def calculate_episode_stats(history, initial_capital):
    """根据一轮的历史记录计算性能指标"""
    if not history:
        return {
            "final_value": initial_capital,
            "total_return": 0.0,
            "annual_volatility": 0.0,
            "total_rewards": 0.0
        }

    history_df = pd.DataFrame(history)

    # 总收益
    final_value = history_df['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1

    # 波动率
    daily_returns = history_df['portfolio_value'].pct_change().dropna()
    # 如果交易次数太少或没有收益变化，波动率为0
    if len(daily_returns) < 2 or daily_returns.std() == 0:
        annual_volatility = 0.0
    else:
        # 年化波动率
        annual_volatility = daily_returns.std() * np.sqrt(252)  # 假设数据是日频

    # 总奖励
    total_rewards = history_df['reward'].sum()

    return {
        "final_value": final_value,
        "total_return": total_return,
        "annual_volatility": annual_volatility,
        "total_rewards": total_rewards
    }


def train_dqn(config, train_df):
    """
    重构后的训练函数，Epsilon在每轮结束后衰减。
    """
    env = StockTradingEnv(train_df, config)
    config['agent']['state_dim'] = env.observation_space.shape[0]
    agent = DQNAgent(config)

    initial_capital = config['environment']['initial_capital']
    num_episodes = config['training']['num_episodes']
    global_step_counter = 0  # 声明全局步数计数器

    print(f"--- Starting Training for {config['training']['num_episodes']} episodes ---")
    for e in range(config['training']['num_episodes']):
        state = env.reset()
        done = False
        while not done:
            global_step_counter += 1  # 每次与环境交互，计数器+1
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > agent.batch_size:
                agent.replay(global_step_counter)  # 将全局步数传入

        # 每10轮打印一次训练日志
        if (e + 1) % 10 == 0 or e == num_episodes - 1:
            stats = calculate_episode_stats(env.history, initial_capital)
            print(f"--- Episode {e + 1}/{num_episodes} ---")
            print(f"  Total Rewards: {stats['total_rewards']:.4f}")
            print(f"  Final Portfolio Value: {stats['final_value']:.2f}")
            print(f"  Total Return: {stats['total_return']:.2%}")
            print(f"  Annual Volatility: {stats['annual_volatility']:.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}\n")

    print("--- Training Finished ---")
    return agent


def evaluate_agent(agent, eval_df, config):
    agent.epsilon = 0  # 关闭探索进行评估
    env = StockTradingEnv(eval_df, config)
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done, _ = env.step(action)
    return env.history


# --- 5. 可视化函数 ---
def plot_results(history, config, benchmark_df, title):
    """
    重构后的可视化函数，使用独立的子图来显示持仓状态。
    """
    history_df = pd.DataFrame(history).set_index('date')
    initial_capital = config['environment']['initial_capital']

    # 创建3个子图，共享X轴
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # --- 图1: 净值与价格对比 ---
    ax1.set_title(title, fontsize=16)

    # 归一化净值曲线
    normalized_portfolio_value = history_df['portfolio_value'] / initial_capital
    ax1.plot(history_df.index, normalized_portfolio_value, label='DQN Agent Portfolio (Normalized)', color='blue',
             zorder=2)

    normalized_benchmark = benchmark_df['close'] / benchmark_df['close'].iloc[0]
    ax1.plot(benchmark_df.index, normalized_benchmark, label='Buy and Hold Benchmark (Normalized)', color='grey',
             linestyle='--', zorder=1)

    ax1.set_ylabel('Normalized Net Value (Start=1)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)

    # 创建第二个Y轴显示原始价格
    ax2 = ax1.twinx()
    ax2.plot(benchmark_df.index, benchmark_df['close'], label='ETF Price (Raw)', color='orange', alpha=0.3, zorder=0)
    ax2.set_ylabel('ETF Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- 图2: 持仓状态时间轴 ---
    ax3.set_title('Agent Position Status')

    # 找出持仓(action=1)和平仓(action=2)的时间点
    actions = history_df['action']
    buy_times = actions[actions == 1].index
    sell_times = actions[actions == 2].index

    # 填充持仓期间的背景
    position_active = False
    last_buy_time = None
    for date in history_df.index:
        if date in buy_times:
            position_active = True
            last_buy_time = date
        if date in sell_times:
            if position_active:
                ax3.axvspan(last_buy_time, date, color='green', alpha=0.3, label='Long Position')
            position_active = False
    # 如果最后仍然持仓
    if position_active and last_buy_time is not None:
        ax3.axvspan(last_buy_time, history_df.index[-1], color='green', alpha=0.3)

    ax3.set_yticks([])  # 不需要Y轴刻度
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')

    # 处理图例重复问题
    handles, labels = ax3.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3.legend(by_label.values(), by_label.keys(), loc='upper left')

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# --- 主程序 ---
if __name__ == '__main__':
    set_seeds(CONFIG["infra"]["seed"])

    # 1. 加载数据
    full_df = get_data(the_fund_code, the_basic_info_file_path_dict, the_metrics_info_file_path_dict)

    # 2. 预处理和分割数据
    train_df, test_df = preprocess_and_split_data(full_df, CONFIG)
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")

    # 3. 训练DQN智能体
    print("\n--- Starting DQN Agent Training ---")
    trained_agent = train_dqn(CONFIG, train_df)

    # 4. 在训练集上进行评估 (查看拟合效果)
    print("\n--- Evaluating Agent on Training Data ---")
    train_history = evaluate_agent(trained_agent, train_df, CONFIG)
    plot_results(train_history, CONFIG, train_df, f"DQN Performance (Training Set) - {the_fund_code}")

    # 5. 在测试集上进行评估 (查看泛化能力)
    print("\n--- Evaluating Agent on Test Data ---")
    test_history = evaluate_agent(trained_agent, test_df, CONFIG)
    plot_results(test_history, CONFIG, test_df, f"DQN Performance (Test Set) - {the_fund_code}")
