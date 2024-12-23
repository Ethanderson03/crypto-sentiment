# Crypto Fear & Greed Analysis

This project analyzes the relationship between the Crypto Fear & Greed Index and cryptocurrency price movements. It creates an interactive dashboard showing how market sentiment correlates with future returns across multiple cryptocurrencies.

## Features

- Multi-cryptocurrency analysis (BTC, ETH, BNB, XRP, SOL, ADA)
- Interactive dashboard with tabbed interface
- Sentiment correlation analysis
- Return magnitude analysis by sentiment level
- Risk-adjusted returns analysis

## Installation

1. Clone the repository:

```
git clone [your-repo-url]
```

2. Install required packages:

```
pip install -r requirements.txt
```

## Usage

Run the analysis:

```
python crypto_sentiment_analysis.py
```

This will:

1. Fetch the latest Fear & Greed Index data
2. Download cryptocurrency price data
3. Generate an interactive HTML dashboard
4. Open the dashboard in your default browser

## Dashboard Components

### Predictive Analysis

- Price vs Fear & Greed Index visualization
- Rolling correlation between sentiment and future returns
- Prediction accuracy tracking over time
- Color-coded extreme fear and greed zones

### Magnitude Analysis

- Return distribution by sentiment level (box plots)
- Average returns for each sentiment category
- Risk (volatility) analysis
- Risk-adjusted return metrics

## Analysis Methodology

The analysis examines three time windows:

- 7-day forward returns
- 14-day forward returns
- 30-day forward returns

For each period, it analyzes:

- Correlation between sentiment and returns
- Return magnitude during different sentiment levels
- Risk-adjusted performance metrics
- Prediction accuracy of sentiment signals

## Sentiment Categories

The Fear & Greed Index (0-100) is divided into:

- Extreme Fear (0-25)
- Fear (26-45)
- Neutral (46-55)
- Greed (56-75)
- Extreme Greed (76-100)

## Data Sources

- Fear & Greed Index: [Alternative.me API](https://alternative.me/crypto/fear-and-greed-index/)
- Cryptocurrency prices: Yahoo Finance via `yfinance`

## Results Interpretation

The dashboard helps identify:

- Optimal trading windows based on sentiment
- Risk-adjusted return opportunities
- Market psychology patterns
- Contrarian investment opportunities

## Dependencies

- pandas: Data manipulation and analysis
- plotly: Interactive visualizations
- yfinance: Cryptocurrency price data
- requests: API calls

## Notes

- The dashboard is generated as an HTML file for interactive exploration
- Historical data is limited to 365 days
- All returns are calculated on a forward-looking basis
- Risk-adjusted returns use a Sharpe-like ratio calculation

## Contributing

Feel free to fork the repository and submit pull requests with improvements or additional features.

## License

MIT License - feel free to use this code for any purpose.
