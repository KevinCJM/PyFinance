import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Dirichlet
import torch.optim as optim
import gym
from gym import spaces
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import warnings
from typing import List, Dict

# --- 全局设置 ---
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
# 设置随机种子以保证结果可复现
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# 1. 数据处理模块
# ==============================================================================
def load_and_preprocess_data(file_path: str, fund_list: List[str], window_size: int = 60) -> (
        Dict[str, pd.DataFrame], pd.DataFrame):
    """
    加载并预处理ETF日线数据。
    使用滚动窗口进行标准化，以避免前视偏差。

    :param file_path: Parquet文件路径。
    :param fund_list: 要筛选的基金代码列表。
    :param window_size: 滚动标准化的窗口大小（天）。
    :return: 一个包含各种特征数据框的字典，以及一个对数收益率数据框。
    """
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"错误：'{file_path}' 文件未找到。")
        raise

    etf_df = df[df['ts_code'].isin(fund_list)].reset_index(drop=True)
    etf_df['trade_date'] = pd.to_datetime(etf_df['trade_date'])
    etf_df['log_return'] = etf_df.groupby('ts_code')['close'].transform(lambda x: np.log(x / x.shift(1)))

    min_date = etf_df.groupby('ts_code')['trade_date'].min().reset_index()
    etf_df = etf_df[etf_df['trade_date'] >= min_date['trade_date'].max()].reset_index(drop=True)

    # 滚动标准化
    def rolling_scale(pivot_df: pd.DataFrame, window: int) -> pd.DataFrame:
        """对数据框的每一列进行滚动标准化"""
        # .shift(1) 是关键，确保我们用t-1的数据来标准化t时刻的数据
        rolling_mean = pivot_df.rolling(window=window, min_periods=1).mean().shift(1)
        rolling_std = pivot_df.rolling(window=window, min_periods=1).std().shift(1)
        # 避免除以零
        rolling_std.replace(0, 0.001, inplace=True)
        return (pivot_df - rolling_mean) / rolling_std

    # 创建一个函数来处理填充和滚动缩放
    def process_pivot_df_rolling(the_df, value_col, window, fill_val=None):
        pivot = the_df.pivot(index='trade_date', columns='ts_code', values=value_col)
        # 对价格和交易量/额进行滚动标准化
        if value_col not in ['log_return']:
            if fill_val is not None:
                pivot = pivot.fillna(fill_val)
            else:
                pivot = pivot.fillna(method='ffill').fillna(method='bfill')  # 先填充，确保滚动窗口计算有效

            pivot = rolling_scale(pivot, window)
        else:  # 对数收益率不需要标准化，但需要填充
            pivot = pivot.fillna(0)
        return pivot

    feature_cols = ['open', 'close', 'high', 'low', 'vol', 'amount']
    fill_vals = {'vol': 0, 'amount': 0}
    all_df_dict = {
        col: process_pivot_df_rolling(etf_df, col, window_size, fill_vals.get(col))
        for col in feature_cols}
    log_return_df = process_pivot_df_rolling(etf_df, 'log_return', window_size)

    # 处理滚动计算产生的NaN值: 滚动计算会在数据集的开头产生NaN值，我们需要将这些行删除; 我们以close_df为基准，找到第一个非NaN的行
    first_valid_index = all_df_dict['close'].first_valid_index()

    print(f"原始数据从 {all_df_dict['close'].index[0].date()} 开始")
    print(f"由于滚动窗口（{window_size}天），数据将从 {first_valid_index.date()} 开始")

    # 裁剪所有数据框，使其从同一起点开始
    for key in all_df_dict:
        all_df_dict[key] = all_df_dict[key].loc[first_valid_index:]
    log_return_df = log_return_df.loc[first_valid_index:]

    print("数据处理完成。")
    return all_df_dict, log_return_df


# ==============================================================================
# 2. 投资组合模型模块
# ==============================================================================
def calculate_markowitz_weights(log_returns_df: pd.DataFrame) -> np.ndarray:
    """
    计算马科维茨最优投资组合权重（最大化夏普比率）。

    :param log_returns_df: 对数收益率数据框。
    :return: 最优权重数组。
    """
    mu = log_returns_df.mean() * 252
    cov = log_returns_df.cov() * 252

    n_portfolios = 10000
    n_assets = len(mu)
    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(mu * weights)
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

        results[0, i], results[1, i], results[
            2, i] = portfolio_return, portfolio_stddev, portfolio_return / portfolio_stddev

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]

    print("马科维茨最优权重 (最大夏普比率):")
    for i, w in enumerate(optimal_weights):
        print(f"  {log_returns_df.columns[i]}: {w:.4f}")

    return optimal_weights


# ==============================================================================
# 3. 强化学习环境模块
# ==============================================================================
class AssetAllocationEnv(gym.Env):
    """用于资产配置的Gym环境"""

    def __init__(self, data_dict: Dict[str, pd.DataFrame], log_returns: pd.DataFrame, config: Dict):
        """
        资产分配环境的初始化函数

        参数:
            data_dict (Dict[str, pd.DataFrame]): 包含特征数据的字典，键为特征名，值为对应的DataFrame
            log_returns (pd.DataFrame): 对数收益率数据，行代表时间步，列代表不同资产
            config (Dict): 配置参数字典，包含环境相关参数
        """
        super(AssetAllocationEnv, self).__init__()
        # 初始化基础数据
        self.data_dict = data_dict
        self.log_returns = log_returns
        self.n_steps, self.n_assets = log_returns.shape  # 获取时间步数和资产数量
        self.n_features = len(data_dict)  # 获取特征数量

        # 从配置中获取风险惩罚系数和奖励计算窗口
        self.risk_penalty_coef = config.get('risk_penalty_coef', 0.1)
        self.reward_window = config.get('reward_window', 22)

        # 计算环境状态空间的维度: state_dim表示状态向量的总长度，包含所有资产特征和持仓信息
        state_dim = (self.n_features * self.n_assets  # 特征数量乘以资产数量得到所有资产特征的总维
                     + self.n_assets)  # 加上持仓比例维度, 即个资产数量

        # 定义环境的观测空间
        self.observation_space = spaces.Box(
            low=-np.inf,  # 观测值的最小边界
            high=np.inf,  # 观测值的最大边界
            shape=(state_dim,),  # 观测空间的维度形状, state_dim: 状态维度总数 = 特征数量×资产数 + 资产数表示持仓权重
            dtype=np.float32  # 32位浮点数类型
        )

        # 初始化动作空间为一个连续的盒状空间
        self.action_space = spaces.Box(
            low=0, high=1,  # 动作空间的下界为0，上界为1，表示可以对每个资产采取的动作强度范围
            shape=(self.n_assets,),  # 动作空间的维度与资产数量相同，每个资产对应一个动作
            dtype=np.float32  # 使用np.float32作为数据类型，以确保在计算时具有足够的精度
        )

        # 重置环境状态
        self.reset()

    def _get_observation(self):
        market_features = [self.data_dict[key].iloc[self.current_step].values for key in self.data_dict]
        obs = np.hstack((np.array(market_features).flatten(), self.weights))
        return obs

    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value_history = [self.portfolio_value]
        self.weights_history = [self.weights]

        # --- 新增：用于计算波动的收益率历史 ---
        self.portfolio_log_returns_history = []

        return self._get_observation()

    def step(self, action: np.ndarray):
        returns = self.log_returns.iloc[self.current_step].values
        portfolio_log_return = np.dot(self.weights, returns)

        # 记录当天的组合对数收益率
        self.portfolio_log_returns_history.append(portfolio_log_return)

        self.portfolio_value *= np.exp(portfolio_log_return)

        # --- 核心修改：新的奖励计算 ---
        # 基础奖励仍然是对数收益率
        base_reward = portfolio_log_return

        # 计算风险惩罚项
        if len(self.portfolio_log_returns_history) < self.reward_window:
            # 在历史数据不足时，波动率视为0，不进行惩罚
            volatility_penalty = 0
        else:
            # 计算最近 reward_window 天的收益率波动（标准差）
            rolling_std = np.std(self.portfolio_log_returns_history[-self.reward_window:])
            volatility_penalty = self.risk_penalty_coef * rolling_std

        # 最终奖励 = 基础奖励 - 风险惩罚
        reward = base_reward - volatility_penalty

        # --- 后续部分不变 ---
        self.weights = action
        self.portfolio_value_history.append(self.portfolio_value)
        self.weights_history.append(self.weights)
        self.current_step += 1
        done = self.current_step >= self.n_steps
        next_obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_obs, reward, done, {}


# ==============================================================================
# 4. PPO Agent 模块
# ==============================================================================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Softplus())

    def forward(self, state):
        return Dirichlet(self.network(state) + 1e-6)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, state):
        return self.network(state)


class PPOAgent:
    """PPO智能体，封装了训练和评估逻辑"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        self.config = config
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['critic_lr'])

        # --- 新增：学习率衰减调度器 ---
        # 效果：在 num_iterations 步内，学习率从 1.0*lr 线性衰减到 end_factor*lr
        self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.actor_optimizer,
            start_factor=1.0,
            end_factor=0.001,  # 最终学习率衰减为初始的百分之多少
            total_iters=config['num_iterations']
        )
        self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.critic_optimizer,
            start_factor=1.0,
            end_factor=0.001,
            total_iters=config['num_iterations']
        )

    # --- train 方法再次修改版 ---
    def train(self, env: gym.Env):
        history = {
            'total_rewards': [], 'actor_losses': [],
            'critic_losses': [], 'entropies': [],
            'learning_rates': []  # <--- 新增
        }
        n_steps = env.n_steps

        for iteration in range(self.config['num_iterations']):
            # --- 数据收集部分 ---
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
            state = env.reset()
            for _ in range(n_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actor(state_tensor)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = self.critic(state_tensor)

                next_state, reward, done, _ = env.step(action.squeeze().cpu().numpy())

                rewards.append(float(reward))

                states.append(state);
                actions.append(action.cpu().numpy().flatten())
                log_probs.append(log_prob.cpu().numpy());
                dones.append(done)
                values.append(value.item());
                state = next_state

            # =================== 核心修改点 ===================
            # 使用Python内置的sum()并用float()强制转换，确保得到一个纯粹的浮点数
            total_reward_this_iteration = float(sum(rewards))
            history['total_rewards'].append(total_reward_this_iteration)
            # =================================================

            if (iteration + 1) % 10 == 0:
                # 现在 history['total_rewards'][-1] 保证是float类型，可以被安全地格式化
                print(
                    f"  Iteration {iteration + 1}/{self.config['num_iterations']}, Total Reward: {history['total_rewards'][-1]:.4f}, Final Portfolio Value: {env.portfolio_value:.4f}")

            # --- GAE 和优势计算部分---
            advantages = np.zeros(n_steps, dtype=np.float32)
            last_advantage = 0
            with torch.no_grad():
                last_value = self.critic(torch.FloatTensor(next_state).unsqueeze(0)).item() if not done else 0
            for t in reversed(range(n_steps)):
                next_non_terminal = 1.0 - (dones[t + 1] if t < n_steps - 1 else done)
                next_value = values[t + 1] if t < n_steps - 1 else last_value
                delta = rewards[t] + self.config['gamma'] * next_value * next_non_terminal - values[t]
                advantages[t] = last_advantage = delta + self.config['gamma'] * self.config[
                    'gae_lambda'] * next_non_terminal * last_advantage
            returns = advantages + np.array(values)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

            states_t, actions_t, log_probs_t = torch.FloatTensor(np.array(states)), torch.FloatTensor(
                np.array(actions)), torch.FloatTensor(np.array(log_probs).flatten())
            advantages_t, returns_t = torch.FloatTensor(advantages), torch.FloatTensor(returns)

            # --- 优化网络部分---
            epoch_actor_losses, epoch_critic_losses, epoch_entropies = [], [], []
            for _ in range(self.config['n_epochs']):
                indices = np.arange(n_steps)
                np.random.shuffle(indices)
                for start in range(0, n_steps, self.config['batch_size']):
                    batch_indices = indices[start: start + self.config['batch_size']]

                    dist = self.actor(states_t[batch_indices])
                    new_log_probs = dist.log_prob(actions_t[batch_indices])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - log_probs_t[batch_indices])
                    term1 = ratio * advantages_t[batch_indices]
                    term2 = torch.clamp(ratio, 1 - self.config['epsilon'], 1 + self.config['epsilon']) * advantages_t[
                        batch_indices]
                    actor_loss = -torch.min(term1, term2).mean() - self.config['entropy_coef'] * entropy

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    epoch_actor_losses.append(actor_loss.item())

                    predicted_values = self.critic(states_t[batch_indices]).squeeze()
                    critic_loss = nn.MSELoss()(predicted_values, returns_t[batch_indices])

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    epoch_critic_losses.append(critic_loss.item())

                    epoch_entropies.append(entropy.item())

            history['actor_losses'].append(np.mean(epoch_actor_losses))
            history['critic_losses'].append(np.mean(epoch_critic_losses))
            history['entropies'].append(np.mean(epoch_entropies))

            # --- 在每次迭代的最后，更新学习率 ---
            self.actor_scheduler.step()
            self.critic_scheduler.step()

            # --- 修改打印信息，监控当前学习率 ---
            if (iteration + 1) % 10 == 0:
                current_lr = self.actor_optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                print(f"  Iteration {iteration + 1}/{self.config['num_iterations']}, "
                      f"Total Reward: {history['total_rewards'][-1]:.4f}, "
                      f"Current LR: {current_lr:.2e}")  # .2e 表示用科学计数法显示

        print("--- PPO训练完成 ---")
        return env.portfolio_value_history, history

    def evaluate(self, env: gym.Env, train_final_value: float) -> (List[float], List[np.ndarray]):
        state = env.reset()
        env.portfolio_value = train_final_value
        env.portfolio_value_history = [train_final_value]

        self.actor.eval()
        self.critic.eval()

        with torch.no_grad():
            for _ in range(env.n_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                dist = self.actor(state_tensor)
                action = dist.sample()
                state, _, done, _ = env.step(action.squeeze().cpu().numpy())
                if done: break

        print("--- PPO评估完成 ---")
        return env.portfolio_value_history, env.weights_history


# ==============================================================================
# 5. 可视化模块 (修改版)
# ==============================================================================
def plot_results(train_results, test_results, ppo_train_value):
    """
    绘制训练集和测试集的表现图表，并额外增加一张归一化的测试集表现图。
    """
    print("\n--- 6. 生成结果图表 ---")

    # --- 图一：训练集表现图 (保持不变) ---
    plt.figure(figsize=(14, 7))
    plt.plot(train_results['index'], train_results['ppo'], label='PPO Strategy (In-Sample)', linewidth=2)
    plt.plot(train_results['index'], train_results['markowitz'], label='Markowitz Static (In-Sample)', linestyle='--')
    plt.title('Performance on Training Set')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()

    # # --- 图二：测试集表现图 (连续净值，保持不变，用于展示绝对收益) ---
    # plt.figure(figsize=(14, 7))
    # plt.plot(test_results['index'], test_results['ppo'], label='PPO Strategy (Out-of-Sample)', linewidth=2)
    # plt.plot(test_results['index'], test_results['markowitz'], label='Markowitz Static (Out-of-Sample)', linestyle='--')
    # plt.title('Continuous Performance on Test Set (Out-of-Sample)')
    # plt.xlabel('Date')
    # plt.ylabel('Portfolio Value (Continuing from Train Set)')
    # plt.legend()
    # plt.show()

    # --- 新增！图三：归一化的测试集表现图 ---
    # 将测试集的净值曲线全部除以其在测试期的第一个值，使它们都从1.0开始
    ppo_test_normalized = test_results['ppo'] / test_results['ppo'][0]
    markowitz_test_normalized = test_results['markowitz'] / test_results['markowitz'][0]

    plt.figure(figsize=(14, 7))
    plt.plot(test_results['index'], ppo_test_normalized, label='PPO Strategy (Normalized)', linewidth=2)
    plt.plot(test_results['index'], markowitz_test_normalized, label='Markowitz Static (Normalized)', linestyle='--')
    plt.title('Normalized Performance on Test Set (Starting from 1.0)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    plt.show()

    # --- ces

    # # --- PPO在测试集上的动态权重分配 (保持不变) ---
    # plt.figure(figsize=(14, 7))
    # plt.stackplot(test_results['index'], np.array(test_results['weights']).T, labels=test_results['asset_labels'],
    #               alpha=0.8)
    # plt.title('PPO Dynamic Asset Allocation on Test Set')
    # plt.xlabel('Date')
    # plt.ylabel('Asset Weight')
    # plt.legend(loc='upper left', ncol=2, fontsize='small')
    # plt.margins(0, 0)
    # plt.show()


# ==============================================================================
# 新增：收敛过程可视化模块
# ==============================================================================
def plot_convergence(history: Dict):
    print("\n--- 7. 生成收敛过程图表 ---")

    # 将画布扩大，以容纳新的子图
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))  # 从2x2变为3x2
    fig.suptitle('PPO Convergence Metrics', fontsize=16)

    # 绘制前四个图 (代码不变)
    axs[0, 0].plot(history['total_rewards']);
    axs[0, 0].set_title('Total Reward per Iteration')
    axs[0, 1].plot(history['critic_losses'], color='orange');
    axs[0, 1].set_title('Critic Loss')
    axs[1, 0].plot(history['actor_losses'], color='green');
    axs[1, 0].set_title('Actor Loss')
    axs[1, 1].plot(history['entropies'], color='red');
    axs[1, 1].set_title('Policy Entropy')

    # --- 新增：绘制学习率衰减曲线 ---
    axs[2, 0].plot(history['learning_rates'], color='purple')
    axs[2, 0].set_title('Learning Rate Decay')
    axs[2, 0].set_xlabel('Iteration')
    axs[2, 0].set_ylabel('Learning Rate')

    # 隐藏最后一个空的子图
    axs[2, 1].axis('off')

    # 为所有子图添加横轴标签
    for ax in axs.flat:
        if ax.get_xlabel() == '':  # 只为没有标签的图添加
            ax.set_xlabel('Iteration')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ==============================================================================
# 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 配置参数 ---
    CONFIG = {
        'file_path': './Data/etf_daily.parquet',
        'fund_list': [
            '510050.SH', '159915.SZ', '159912.SZ', '512500.SH', '511010.SH',
            '513100.SH', '513030.SH', '513080.SH', '513520.SH', '518880.SH',
            '161226.SZ', '501018.SH', '159981.SZ', '159985.SZ', '159980.SZ',
            '511990.SH'
        ],
        'test_ratio': 0.2,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'gamma': 0.99,
        'epsilon': 0.2,
        'gae_lambda': 0.95,
        'n_epochs': 10,
        'batch_size': 64,
        'num_iterations': 2000,
        'entropy_coef': 0.01,  # <--- 新增熵系数
        'risk_penalty_coef': 0.0,  # <--- 新增：风险惩罚系数
        'reward_window': 22  # <--- 新增：计算波动的窗口
    }

    # 1. 加载和预处理数据
    print("--- 1. 开始加载和预处理数据 (使用滚动标准化) ---")
    all_data_dict, log_df = load_and_preprocess_data(CONFIG['file_path'], CONFIG['fund_list'])

    # 2. 拆分训练集和测试集
    print("\n--- 2. 拆分训练集与测试集 ---")
    n_total_steps = len(log_df)
    n_test_steps = int(n_total_steps * CONFIG['test_ratio'])
    n_train_steps = n_total_steps - n_test_steps
    print(f"总数据量: {n_total_steps} 天, 训练集: {n_train_steps} 天, 测试集: {n_test_steps} 天")

    data_dict_train = {key: df.iloc[:n_train_steps] for key, df in all_data_dict.items()}
    data_dict_test = {key: df.iloc[n_train_steps:] for key, df in all_data_dict.items()}
    log_df_train = log_df.iloc[:n_train_steps]
    log_df_test = log_df.iloc[n_train_steps:]

    # 3. 计算基准模型 (马科维茨)
    print("\n--- 3. 计算马科维茨基准模型 ---")
    mk_weights = calculate_markowitz_weights(log_df_train)
    mk_returns_train = (log_df_train * mk_weights).sum(axis=1)
    markowitz_value_train = np.exp(mk_returns_train.cumsum())
    markowitz_returns_test = (log_df_test * mk_weights).sum(axis=1)
    markowitz_value_test = np.exp(markowitz_returns_test.cumsum())
    markowitz_value_test = markowitz_value_train.iloc[-1] * markowitz_value_test

    # 4. 初始化并训练PPO Agent
    print("\n--- 4.1 构建环境并初始化 ---")
    env_train = AssetAllocationEnv(data_dict_train, log_df_train, CONFIG)
    state_dim = env_train.observation_space.shape[0]
    action_dim = env_train.action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, CONFIG)
    print("\n--- 4.2 开始PPO训练 ---")
    ppo_value_train, train_history = agent.train(env_train)

    # 5. 在测试集上评估PPO Agent
    print("\n--- 5. 在测试集上评估已训练的PPO模型 ---")
    env_test = AssetAllocationEnv(data_dict_test, log_df_test, CONFIG)
    ppo_value_test, ppo_weights_test = agent.evaluate(env_test, ppo_value_train[-1])

    # 6. 整理并可视化结果
    train_results = {
        'index': log_df_train.index.insert(0, log_df_train.index[0] - pd.Timedelta(days=1)),
        'ppo': ppo_value_train,
        'markowitz': np.insert(markowitz_value_train.values, 0, 1.0)
    }
    test_results = {
        'index': log_df_test.index.insert(0, log_df_test.index[0] - pd.Timedelta(days=1)),
        'ppo': ppo_value_test,
        'markowitz': np.insert(markowitz_value_test.values, 0, markowitz_value_train.iloc[-1]),
        'weights': ppo_weights_test,
        'asset_labels': log_df.columns
    }
    # 6. 整理并可视化最终结果
    plot_results(train_results, test_results, ppo_value_train)
    # 7. 可视化收敛过程
    plot_convergence(train_history)
