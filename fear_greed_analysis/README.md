# Crypto Fear & Greed Analysis

This directory contains tools and scripts for analyzing the relationship between cryptocurrency prices and the Fear & Greed index.

## Components

- `crypto_sentiment_analysis.py`: Script to analyze correlations between Fear & Greed index and price movements

## Features

- Fetches historical cryptocurrency price data from Yahoo Finance
- Retrieves Fear & Greed index data from alternative.me API
- Calculates rolling correlations between sentiment and price movements
- Generates interactive visualizations of the analysis
- Supports multiple cryptocurrencies (BTC, ETH, etc.)

## Analysis Insights

The analysis explores:

- How Fear & Greed index correlates with future price movements
- Optimal trading strategies based on extreme sentiment values
- Lag effects between sentiment changes and price reactions
- Comparison of sentiment impact across different cryptocurrencies

## Usage

Run the sentiment analysis:

```bash
python crypto_sentiment_analysis.py
```

The script will generate interactive visualizations showing:

- Price movements vs Fear & Greed index
- Rolling correlations
- Sentiment distribution analysis
- Performance of contrarian strategies
