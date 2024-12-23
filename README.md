# Crypto Fear & Greed Trading Analysis

This project combines sentiment analysis using the Fear & Greed index with reinforcement learning to develop and test cryptocurrency trading strategies.

## Project Structure

```
.
├── fear_greed_analysis/    # Analysis of Fear & Greed index correlation with crypto prices
│   ├── crypto_sentiment_analysis.py
│   └── README.md
│
├── trading_bot/            # Reinforcement learning trading bot implementation
│   ├── trading_env.py
│   ├── evaluate_model.py
│   ├── train_rl_trader.py
│   ├── crypto_trading_env.py
│   └── README.md
│
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Features

- Comprehensive analysis of Fear & Greed index's impact on crypto prices
- Reinforcement learning-based trading bot using sentiment and technical indicators
- Interactive visualizations of analysis results and trading performance
- Support for multiple cryptocurrencies
- Backtesting framework for strategy evaluation

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run Fear & Greed Analysis:

```bash
cd fear_greed_analysis
python crypto_sentiment_analysis.py
```

2. Train the Trading Bot:

```bash
cd trading_bot
python train_rl_trader.py
```

3. Evaluate Trading Performance:

```bash
cd trading_bot
python evaluate_model.py
```

## Dependencies

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Pandas
- NumPy
- Plotly
- yfinance
- requests

See `requirements.txt` for complete list of dependencies.
