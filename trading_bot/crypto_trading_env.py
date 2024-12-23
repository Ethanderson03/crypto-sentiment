import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class CryptoTradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=10000, transaction_fee=0.001, window_size=20):
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.window_size = window_size
        self.current_step = 0
        
        # Action space: [position_size]
        # position_size: -1 (full short) to 1 (full long)
        self.action_space = spaces.Box(
            low=np.array([-1], dtype=np.float32),
            high=np.array([1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Calculate normalization factors
        self.price_scaler = df['price'].std()
        self.volatility_scaler = df['volatility'].std()
        
        # Observation space (normalized)
        # [normalized_price, sentiment, momentum, rsi, normalized_volatility, position, unrealized_pnl]
        self.observation_space = spaces.Box(
            low=np.array([-10, 0, -5, 0, -5, -1, -5], dtype=np.float32),
            high=np.array([10, 100, 5, 100, 5, 1, 5], dtype=np.float32),
            dtype=np.float32
        )
        
        self.reset()

    def _normalize_observation(self, obs):
        """Normalize the observation values"""
        obs[0] = (obs[0] - self.df['price'].mean()) / self.price_scaler
        obs[4] = (obs[4] - self.df['volatility'].mean()) / self.volatility_scaler
        return obs

    def _next_observation(self):
        """Get the next observation"""
        obs = np.array([
            self.df.iloc[self.current_step]['price'],
            self.df.iloc[self.current_step]['value'],
            self.df.iloc[self.current_step]['momentum'],
            self.df.iloc[self.current_step]['rsi'],
            self.df.iloc[self.current_step]['volatility'],
            self.current_position,
            self.unrealized_pnl
        ], dtype=np.float32)
        
        return self._normalize_observation(obs)

    def _take_action(self, action):
        """Execute the action"""
        current_price = self.df.iloc[self.current_step]['price']
        position_size = action[0]  # -1 to 1
        
        # Close existing position if direction changes
        if np.sign(position_size) != np.sign(self.current_position) and self.current_position != 0:
            close_price = current_price * (1 - self.transaction_fee * np.sign(self.current_position))
            pnl = self.current_position * (close_price - self.entry_price)
            self.balance += pnl
            self.current_position = 0
            self.entry_price = 0
            self.trades.append({
                'type': 'close',
                'price': close_price,
                'pnl': pnl,
                'step': self.current_step
            })
        
        # Open new position
        if abs(position_size) > 0.1 and self.current_position == 0:
            self.entry_price = current_price * (1 + self.transaction_fee * np.sign(position_size))
            position_value = self.balance * abs(position_size)
            self.current_position = np.sign(position_size)
            self.trades.append({
                'type': 'open',
                'price': self.entry_price,
                'position': self.current_position,
                'step': self.current_step
            })
        
        # Calculate unrealized PnL
        if self.current_position != 0:
            self.unrealized_pnl = self.current_position * (current_price - self.entry_price) / self.entry_price

    def _calculate_reward(self):
        """Calculate the reward using a combination of returns and risk metrics"""
        if len(self.returns_history) < self.window_size:
            return 0
        
        # Calculate returns-based components
        returns_mean = np.mean(self.returns_history[-self.window_size:])
        returns_std = np.std(self.returns_history[-self.window_size:]) + 1e-9
        sharpe = np.sqrt(252) * returns_mean / returns_std
        
        # Calculate position-based penalty
        position_penalty = -0.001 * abs(self.current_position)  # Small penalty for holding positions
        
        # Calculate trading frequency penalty
        trade_penalty = -0.002 * (1 if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step else 0)
        
        # Combine components
        reward = sharpe + position_penalty + trade_penalty
        
        return reward

    def step(self, action):
        """Execute one time step within the environment"""
        self._take_action(action)
        
        # Calculate returns
        if self.current_position != 0:
            current_price = self.df.iloc[self.current_step]['price']
            returns = self.current_position * (current_price - self.entry_price) / self.entry_price
        else:
            returns = 0
            
        self.returns_history.append(returns)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is terminated
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # We don't truncate episodes
        
        # Get next observation
        obs = self._next_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.current_position,
            'unrealized_pnl': self.unrealized_pnl,
            'num_trades': len(self.trades)
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        """Reset the state of the environment to an initial state"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.current_step = 0
        self.current_position = 0
        self.entry_price = 0
        self.unrealized_pnl = 0
        self.returns_history = []
        self.trades = []
        
        return self._next_observation(), {}

    def render(self, mode='human'):
        """Render the environment to the screen"""
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance:.2f}')
        print(f'Position: {self.current_position}')
        print(f'Unrealized PnL: {self.unrealized_pnl:.4f}')
        print(f'Number of trades: {len(self.trades)}')
        print('----------------------------------------')