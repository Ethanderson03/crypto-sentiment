# Crypto Trading Bot

This directory contains the reinforcement learning-based cryptocurrency trading bot that uses market data and Fear & Greed index to make trading decisions.

## Components

- `trading_env.py`: The trading environment that implements the OpenAI Gym interface
- `evaluate_model.py`: Script to evaluate the trained model's performance
- `train_rl_trader.py`: Script to train the reinforcement learning model
- `crypto_trading_env.py`: Base environment class for cryptocurrency trading

## Features

- Uses PPO (Proximal Policy Optimization) algorithm for training
- Incorporates Fear & Greed index as a sentiment indicator
- Includes technical indicators (RSI, Momentum, Volatility)
- Supports position sizing and risk management
- Visualizes trading decisions and performance metrics

## Usage

1. Train the model:

```bash
python train_rl_trader.py
```

2. Evaluate the model:

```bash
python evaluate_model.py
```

The evaluation will generate an interactive HTML visualization showing:

- BTC price and trading activity
- Portfolio value over time
- Fear & Greed index values
