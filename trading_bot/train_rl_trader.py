import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import safe_mean
from crypto_trading_env import CryptoTradingEnv
import yfinance as yf
import requests
import os
import gymnasium as gym
import time
from typing import Dict, List
import json

class ResultsWriter:
    """A simple results writer that writes training information to disk."""
    
    def __init__(self, path: str, header: Dict):
        """
        Initialize ResultsWriter
        
        :param path: Path to write results to
        :param header: Header dictionary with metadata
        """
        self.path = path
        self.header = header
        self.results = []
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        
        # Write header
        self._write_header()
    
    def _write_header(self):
        """Write header information to disk"""
        with open(self.path + '.manifest.json', 'w') as f:
            json.dump(self.header, f)
    
    def write_row(self, epinfo: Dict):
        """
        Write a training episode result to disk
        
        :param epinfo: Dictionary containing episode information
        """
        self.results.append(epinfo)
        self._write_results()
    
    def _write_results(self):
        """Write all results to disk"""
        with open(self.path + '.progress.json', 'w') as f:
            json.dump(self.results, f)

class TensorboardCallback(BaseCallback):
    """Custom callback for plotting additional values in tensorboard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_env = None
        self.last_mean_reward = -np.inf
        self.best_mean_reward = -np.inf

    def _on_training_start(self):
        """Called at the start of training"""
        self._log_freq = 1000  # log every 1000 steps
        self.training_env = self.model.get_env()
        self.results_writer = ResultsWriter(
            os.path.join(self.model.tensorboard_log, 'results'),
            header={'t_start': time.time()}
        )

    def _on_step(self) -> bool:
        """Called at each step of training"""
        if self.n_calls % self._log_freq == 0:
            # Log additional statistics
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = safe_mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
                mean_length = safe_mean([ep_info['l'] for ep_info in self.model.ep_info_buffer])
                
                self.logger.record('rollout/ep_rew_mean', mean_reward)
                self.logger.record('rollout/ep_len_mean', mean_length)
                
                # Track best mean reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
                self.logger.record('rollout/best_mean_reward', self.best_mean_reward)
                
                # Calculate improvement
                if self.last_mean_reward != -np.inf:
                    improvement = (mean_reward - self.last_mean_reward) / abs(self.last_mean_reward)
                    self.logger.record('rollout/reward_improvement', improvement)
                
                self.last_mean_reward = mean_reward
                
                # Write results
                self.results_writer.write_row({
                    'step': self.num_timesteps,
                    'reward_mean': float(mean_reward),
                    'length_mean': float(mean_length),
                    'best_reward': float(self.best_mean_reward)
                })
        
        return True

def fetch_and_prepare_data(coin, start_date, end_date):
    """Fetch and prepare data for training"""
    print(f"Fetching data for {coin} from {start_date} to {end_date}")
    
    # Convert dates to timezone-aware
    start_date = pd.Timestamp(start_date).tz_localize('UTC')
    end_date = pd.Timestamp(end_date).tz_localize('UTC')
    
    # Fetch price data
    ticker = yf.Ticker(f"{coin}-USD")
    price_data = ticker.history(start=start_date, end=end_date, interval='1d')
    
    if len(price_data) == 0:
        raise ValueError(f"No price data found for {coin}")
    
    # Fetch sentiment data
    days_needed = (end_date - start_date).days + 30
    url = f"https://api.alternative.me/fng/?limit={days_needed}&format=json"
    response = requests.get(url)
    sentiment_data = pd.DataFrame(response.json()['data'])
    
    # Convert timestamps to datetime with UTC timezone
    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'].astype(int), unit='s').dt.tz_localize('UTC')
    sentiment_data['value'] = sentiment_data['value'].astype(float)
    sentiment_data.set_index('timestamp', inplace=True)
    
    # Prepare DataFrame
    df = price_data[['Close']].copy()
    df.columns = ['price']
    
    # Ensure price data index is timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    
    # Normalize timestamps to midnight UTC
    df.index = df.index.normalize()
    sentiment_data.index = sentiment_data.index.normalize()
    
    # Sort both indices to ensure proper alignment
    df = df.sort_index()
    sentiment_data = sentiment_data.sort_index()
    
    # Merge data
    df = df.merge(sentiment_data[['value']], left_index=True, right_index=True, how='left')
    df['value'] = df['value'].ffill().bfill()
    
    # Calculate technical indicators
    df['returns'] = df['price'].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    
    # RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum
    df['sma20'] = df['price'].rolling(window=20).mean()
    df['sma50'] = df['price'].rolling(window=50).mean()
    df['momentum'] = (df['sma20'] - df['sma50']) / df['sma50']
    
    # Drop NaN values and reset index
    df = df.dropna().copy()
    
    # Ensure all columns are float32
    float_columns = ['price', 'value', 'returns', 'volatility', 'rsi', 'momentum']
    for col in float_columns:
        df[col] = df[col].astype(np.float32)
    
    # Reset index and keep only the required columns
    df = df.reset_index(drop=True)
    df = df[float_columns]
    
    print(f"Prepared {len(df)} data points for {coin}")
    return df

def split_data(df, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets"""
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]
    
    return train_data, val_data, test_data

def create_env(data, env_id):
    """Create a monitored environment"""
    try:
        os.makedirs('logs', exist_ok=True)
        
        # Debug information
        print(f"\nCreating environment {env_id}")
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head()}")
        
        # Create the base environment
        env = CryptoTradingEnv(data)
        
        # Wrap with Monitor
        env = Monitor(env, f'logs/{env_id}', allow_early_resets=True)
        
        # Test environment
        print("\nTesting environment...")
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")
        
        test_action = env.action_space.sample()
        obs, reward, done, _, info = env.step(test_action)
        print(f"Test step successful. Reward: {reward}")
        
        return env
        
    except Exception as e:
        import traceback
        print(f"Error creating environment {env_id}:")
        print(traceback.format_exc())
        raise e

def train_agent(train_env, val_env, total_timesteps=500000):
    """Train the RL agent with improved hyperparameters"""
    # Create evaluation callback
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Create PPO agent with tuned hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        ),
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        return None
    
    return model

def evaluate_agent(model, env, num_episodes=10):
    """Evaluate the trained agent"""
    episode_rewards = []
    episode_trades = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        
        episode_rewards.append(total_reward)
        episode_trades.append(info['num_trades'])
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_trades': np.mean(episode_trades),
        'std_trades': np.std(episode_trades)
    }

def main():
    # Set up parameters
    coins = ['BTC']  # Start with just BTC for testing
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    
    for coin in coins:
        print(f"\nTraining agent for {coin}...")
        
        try:
            # Prepare data
            df = fetch_and_prepare_data(coin, start_date, end_date)
            
            # Debug information
            print("\nData preparation complete:")
            print(f"Data shape: {df.shape}")
            print(f"Data columns: {df.columns.tolist()}")
            print(f"Data types:\n{df.dtypes}")
            print(f"First few rows:\n{df.head()}")
            
            train_data, val_data, test_data = split_data(df)
            
            print("\nCreating environments...")
            try:
                # Create environments with error handling
                train_env = DummyVecEnv([lambda: create_env(train_data, 'train')])
                val_env = DummyVecEnv([lambda: create_env(val_data, 'val')])
                test_env = DummyVecEnv([lambda: create_env(test_data, 'test')])
            except Exception as e:
                print(f"Failed to create environments: {str(e)}")
                continue
            
            # Train agent
            print("\nStarting training...")
            try:
                model = train_agent(train_env, val_env)
                
                if model is None:
                    print("Training failed. Skipping evaluation.")
                    continue
                
                # Evaluate on test set
                print("\nEvaluating agent...")
                eval_results = evaluate_agent(model, test_env)
                
                print("\nTest Results:")
                print(f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"Mean trades per episode: {eval_results['mean_trades']:.1f} ± {eval_results['std_trades']:.1f}")
                
                # Save model
                model_path = f"{coin}_trading_agent"
                model.save(model_path)
                print(f"Model saved as {model_path}")
                
            except Exception as e:
                print(f"Error during training/evaluation: {str(e)}")
                continue
            
        except Exception as e:
            print(f"Error processing {coin}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 