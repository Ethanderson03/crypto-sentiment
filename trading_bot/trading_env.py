import gym
from gym import spaces
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()
        
        # Data
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Calculate price normalization factors
        self.price_mean = df['price'].mean()
        self.price_std = df['price'].std()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Observation space: [normalized_price, sentiment, returns, volatility, rsi, momentum, position]
        self.observation_space = spaces.Box(
            low=np.array([-10, 0, -1, 0, 0, -1, 0], dtype=np.float32),
            high=np.array([10, 1, 1, 2, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.last_trade_price = 0
        
    def _get_observation(self):
        """Get current observation (state)"""
        current_data = self.df.iloc[self.current_step]
        
        # Normalize the features
        price = (current_data['price'] - self.price_mean) / self.price_std  # Standardize price
        sentiment = current_data['value'] / 100.0  # Normalize to [0,1]
        returns = np.clip(current_data['returns'], -1, 1)  # Clip returns to [-1,1]
        volatility = np.clip(current_data['volatility'], 0, 2)  # Clip volatility to [0,2]
        rsi = current_data['rsi'] / 100.0  # Normalize to [0,1]
        momentum = current_data['momentum']
        
        # Combine into observation
        obs = np.array([
            price,
            sentiment,
            returns,
            volatility,
            rsi,
            momentum,
            float(self.position)  # Current position
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, action):
        """Calculate reward for the current step"""
        current_price = self.df.iloc[self.current_step]['price']
        
        # Calculate portfolio value
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        
        # Calculate returns
        if self.current_step > 0:
            prev_portfolio_value = self.balance + (self.position * self.df.iloc[self.current_step-1]['price'] if self.position > 0 else 0)
            returns = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            returns = 0
        
        # Base reward is the portfolio returns
        reward = returns
        
        # Small penalty for transaction costs when trading
        if action != 0:  # If trading occurred
            reward -= self.transaction_fee
        
        return reward
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # Get current price
        current_price = self.df.iloc[self.current_step]['price']
        
        # Execute trading action
        if action == 1 and self.position <= 0:  # Buy
            # Calculate maximum shares we can buy
            max_shares = self.balance / (current_price * (1 + self.transaction_fee))
            self.position = max_shares
            self.balance = 0
            self.last_trade_price = current_price
            self.trades.append({
                'type': 'open',
                'price': current_price,
                'position': self.position
            })
            
        elif action == 2 and self.position > 0:  # Sell
            # Calculate sale proceeds
            sale_proceeds = self.position * current_price * (1 - self.transaction_fee)
            self.balance = sale_proceeds
            pnl = (current_price - self.last_trade_price) * self.position
            self.trades.append({
                'type': 'close',
                'price': current_price,
                'position': self.position,
                'pnl': pnl
            })
            self.position = 0
            self.last_trade_price = 0
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Move to next step
        self.current_step += 1
        
        # Get new observation
        obs = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        return obs, reward, done, False, self._get_info()
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.last_trade_price = 0
        
        return self._get_observation(), self._get_info()
    
    def _get_info(self):
        """Get current environment info"""
        current_price = self.df.iloc[self.current_step]['price']
        portfolio_value = self.balance + (self.position * current_price if self.position > 0 else 0)
        
        return {
            'balance': portfolio_value,
            'position': self.position,
            'step': self.current_step
        } 