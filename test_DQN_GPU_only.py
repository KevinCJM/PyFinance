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

# --- 1. 配置中心 (CONFIG) ---
CONFIG = {
    "infra": {"seed": 42},
    "preprocessing": {"train_split_ratio": 0.8},
    "environment": {
        "initial_capital": 100000,
        "transaction_cost_pct": 0.001,
        "trade_penalty": 0.05,
        "slippage_pct": 0.0005,
        "reward_scaling": {"holding_pnl_factor": 1.0, "settlement_return_factor": 1.0}
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
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_linear_decay_steps": 250000
    },
    "training": {"num_episodes": 100}
}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_data(fund_code, basic_paths, metrics_paths):
    df_final = None
    for ftype, path in basic_paths.items():
        df = pd.read_parquet(path)
        fund = df[[fund_code]].copy()
        fund.columns = [ftype]
        df_final = fund if df_final is None else df_final.join(fund, how='outer')
    for path in metrics_paths:
        df = pd.read_parquet(path)
        df = df[df['ts_code'] == fund_code].reset_index(drop=True).drop(columns=['ts_code'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df_final = df_final.join(df, how='outer')
    return df_final


def split_data_df(df, config):
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    starts = [df[col].first_valid_index() for col in df.columns]
    start = max(dt for dt in starts if dt is not None)
    df = df.loc[start:]
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    split_i = int(len(df) * config['preprocessing']['train_split_ratio'])
    return df.iloc[:split_i].copy(), df.iloc[split_i:].copy()


def df_to_tensor(train_df, test_df, config, device):
    feat_cols = train_df.columns.drop('close')
    # to tensor on GPU
    train_X = torch.tensor(train_df[feat_cols].values, dtype=torch.float32, device=device)
    test_X = torch.tensor(test_df[feat_cols].values, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_df['close'].values, dtype=torch.float32, device=device).unsqueeze(1)
    test_y = torch.tensor(test_df['close'].values, dtype=torch.float32, device=device).unsqueeze(1)
    # normalize on GPU
    mean = train_X.mean(dim=0, keepdim=True)
    std = train_X.std(dim=0, unbiased=False, keepdim=True)
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std
    # concat features and close
    train_tensor = torch.cat([train_X, train_y], dim=1)
    test_tensor = torch.cat([test_X, test_y], dim=1)
    return train_tensor, test_tensor


class StockTradingEnv(gym.Env):
    def __init__(self, data_tensor, config, device):
        super().__init__()
        self.device = device
        self.data = data_tensor
        self.features = data_tensor[:, :-1]
        self.prices = data_tensor[:, -1]
        ec = config['environment']
        self.initial_capital = torch.tensor(ec['initial_capital'], device=device)
        self.transaction_cost_pct = ec['transaction_cost_pct']
        self.trade_penalty = torch.tensor(ec['trade_penalty'], device=device)
        self.slippage_pct = ec['slippage_pct']
        self.reward_cfg = ec['reward_scaling']
        self.action_space = spaces.Discrete(config['agent']['action_dim'])
        obs_dim = self.features.shape[1] + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.step_idx = 0
        self.cash = self.initial_capital.clone()
        self.shares = torch.tensor(0.0, device=self.device)
        self.portfolio_value = self.initial_capital.clone()
        self.position_steps = 0
        self.history = []
        return self._get_observation()

    def _get_observation(self):
        obs = self.features[self.step_idx]
        pos_flag = 1.0 if self.shares > 0 else 0.0
        pos_t = torch.tensor(pos_flag, device=self.device)
        if self.shares > 0:
            unrlzd = self.prices[self.step_idx] / self.entry_price - 1.0
        else:
            unrlzd = torch.tensor(0.0, device=self.device)
        return torch.cat([obs, pos_t.unsqueeze(0), unrlzd.unsqueeze(0)], dim=0)

    def step(self, action):
        price = self.prices[self.step_idx]
        prev_val = self.portfolio_value.clone()
        reward = torch.tensor(0.0, device=self.device)
        # holding reward
        if self.shares > 0:
            pnl = (self.cash + self.shares * price - prev_val) / prev_val
            shaped = torch.sign(pnl) * torch.sqrt(torch.abs(pnl))
            reward += shaped * self.reward_cfg['holding_pnl_factor']
        # settlement on sell
        if action == 2 and self.shares > 0:
            ret = price / self.entry_price - 1.0
            dur = torch.log(torch.tensor(self.position_steps + 1, device=self.device))
            settle = ret * dur - self.trade_penalty
            reward += settle * self.reward_cfg['settlement_return_factor']
        self._take_action(action, price)
        self.step_idx += 1
        if self.shares > 0:
            self.position_steps += 1
        done = self.step_idx >= self.features.size(0) - 1
        obs = self._get_observation() if not done else torch.zeros_like(self._get_observation())
        self.portfolio_value = self.cash + self.shares * price
        info = {
            'portfolio_value': self.portfolio_value.item(),
            'action': action,
            'price': price.item(),
            'reward': reward.item()
        }
        self.history.append(info)
        return obs, reward, done, info

    def _take_action(self, action, price):
        if action == 1 and self.shares == 0:
            buy_p = price * (1 + self.slippage_pct)
            cost = self.cash * self.transaction_cost_pct
            cap = self.cash - cost
            self.shares = cap / buy_p
            self.cash = torch.tensor(0.0, device=self.device)
            self.entry_price = buy_p
            self.position_steps = 1
        elif action == 2 and self.shares > 0:
            sell_p = price * (1 - self.slippage_pct)
            proceeds = self.shares * sell_p
            cost = proceeds * self.transaction_cost_pct
            self.cash = proceeds - cost
            self.shares = torch.tensor(0.0, device=self.device)
            self.entry_price = torch.tensor(0.0, device=self.device)
            self.position_steps = 0


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, config, device):
        acfg = config['agent']
        self.device = device
        self.state_dim = acfg['state_dim']
        self.action_dim = acfg['action_dim']
        self.gamma = acfg['gamma']
        self.batch_size = acfg['batch_size']
        self.target_update_freq = acfg['target_update_freq']
        self.epsilon = acfg['epsilon_start']
        self.eps_min = acfg['epsilon_min']
        decay_steps = acfg['epsilon_linear_decay_steps']
        self.eps_decay = (acfg['epsilon_start'] - acfg['epsilon_min']) / decay_steps
        self.memory = collections.deque(maxlen=acfg['memory_size'])
        self.policy_net = DQN(self.state_dim, self.action_dim, acfg['dqn_hidden_layers']).to(device)
        self.target_net = DQN(self.state_dim, self.action_dim, acfg['dqn_hidden_layers']).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=acfg['learning_rate'])
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0

    def act(self, state):
        pos = state[-2].item()
        valid = [0, 1] if pos == 0 else [0, 2]
        if random.random() < self.epsilon:
            return random.choice(valid)
        q = self.policy_net(state.unsqueeze(0))
        mask = torch.full((self.action_dim,), float('-inf'), device=self.device)
        for a in valid:
            mask[a] = 0.0
        q = q + mask
        return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self, step):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        states = torch.stack(s).to(self.device)
        actions = torch.tensor(a, device=self.device).unsqueeze(1)
        rewards = torch.tensor(r, device=self.device).unsqueeze(1)
        next_states = torch.stack(s2).to(self.device)
        dones = torch.tensor(done, device=self.device).unsqueeze(1)
        curr_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            nxt_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            nxt_q[dones] = 0.0
            target = rewards + self.gamma * nxt_q
        loss = self.loss_fn(curr_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_decay)
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def calculate_episode_stats(history, initial_capital):
    if not history:
        return {"final_value": initial_capital, "total_return": 0.0,
                "annual_volatility": 0.0, "total_rewards": 0.0}
    df = pd.DataFrame(history)
    final = df['portfolio_value'].iloc[-1]
    tot_r = (final / initial_capital) - 1
    dr = df['portfolio_value'].pct_change().dropna()
    vol = dr.std() * np.sqrt(252) if len(dr) > 1 and dr.std() > 0 else 0.0
    return {"final_value": final, "total_return": tot_r,
            "annual_volatility": vol, "total_rewards": df['reward'].sum()}


def train_dqn(config, data_tensor, device):
    env = StockTradingEnv(data_tensor, config, device)
    config['agent']['state_dim'] = env.observation_space.shape[0]
    agent = DQNAgent(config, device)
    global_step = 0
    for epi in range(config['training']['num_episodes']):
        state = env.reset()
        done = False
        while not done:
            global_step += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(global_step)
        if (epi + 1) % 10 == 0 or epi == config['training']['num_episodes'] - 1:
            stats = calculate_episode_stats(env.history, config['environment']['initial_capital'])
            print(f"Episode {epi + 1}/{config['training']['num_episodes']}: total_rewards={stats['total_rewards']:.4f},"
                  f" return={stats['total_return']:.2%}, vol={stats['annual_volatility']:.2%}, eps={agent.epsilon:.4f}")
    return agent


def evaluate_agent(agent, data_tensor, config, device):
    agent.epsilon = 0.0
    env = StockTradingEnv(data_tensor, config, device)
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        state, _, done, _ = env.step(action)
    return env.history


def plot_results(history, config, benchmark_df, title):
    df = pd.DataFrame(history).set_index('date') if 'date' in history[0] else pd.DataFrame(history)
    init = config['environment']['initial_capital']
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    norm_pv = df['portfolio_value'] / init
    ax1.plot(df.index, norm_pv, label='DQN Portfolio')
    bm_norm = benchmark_df['close'] / benchmark_df['close'].iloc[0]
    ax1.plot(benchmark_df.index, bm_norm, '--', label='Benchmark')
    ax1.set_ylabel('Normalized NAV')
    ax1.legend(loc='upper left')
    ax3.fill_between(df.index, 0, 1, where=(df['action'] == 1), step='post', alpha=0.3, label='Long')
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seeds(CONFIG['infra']['seed'])

    # 1. 加载并切分数据
    fund_code = '510050.SH'
    basic_paths = {
        'log': './Data/wide_log_return_df.parquet', 'high': './Data/wide_high_df.parquet',
        'low': './Data/wide_low_df.parquet', 'vol': './Data/wide_vol_df.parquet',
        'amount': './Data/wide_amount_df.parquet', 'close': './Data/wide_close_df.parquet',
        'open': './Data/wide_open_df.parquet'
    }
    metrics_paths = [
        './Data/Metrics/2d.parquet', './Data/Metrics/5d.parquet', './Data/Metrics/10d.parquet',
        './Data/Metrics/15d.parquet', './Data/Metrics/25d.parquet', './Data/Metrics/50d.parquet',
        './Data/Metrics/75d.parquet', './Data/Metrics/5m.parquet', './Data/Metrics/6m.parquet',
        './Data/Metrics/rolling_metrics.parquet'
    ]
    full_df = get_data(fund_code, basic_paths, metrics_paths)
    train_df, test_df = split_data_df(full_df, CONFIG)

    # 2. 转为 GPU tensor
    train_tensor, test_tensor = df_to_tensor(train_df, test_df, CONFIG, device)

    # 3. 训练
    agent = train_dqn(CONFIG, train_tensor, device)

    # 4. 评估 & 绘图
    train_hist = evaluate_agent(agent, train_tensor, CONFIG, device)
    plot_results(train_hist, CONFIG, train_df, f"DQN Training - {fund_code}")
    test_hist = evaluate_agent(agent, test_tensor, CONFIG, device)
    plot_results(test_hist, CONFIG, test_df, f"DQN Testing  - {fund_code}")
