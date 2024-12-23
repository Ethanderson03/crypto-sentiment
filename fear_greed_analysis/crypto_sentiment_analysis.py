import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import webbrowser
import os
import numpy as np
from plotly.subplots import make_subplots

# Update the crypto list
CRYPTOCURRENCIES = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'BNB-USD': 'Binance Coin',
    'XRP-USD': 'Ripple',
    'SOL-USD': 'Solana',
    'ADA-USD': 'Cardano'
}

def fetch_fear_greed_data():
    """Fetch Fear and Greed index data from alternative.me API"""
    url = "https://api.alternative.me/fng/?limit=365&format=json"
    response = requests.get(url)
    return response.json()

def fetch_crypto_prices(start_date, end_date):
    """Fetch historical prices for multiple cryptocurrencies using Yahoo Finance"""
    prices = {}
    for symbol, name in CRYPTOCURRENCIES.items():
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            coin_name = symbol.split('-')[0].lower()
            prices[coin_name] = pd.DataFrame({
                'timestamp': df.index.tz_localize(None),
                f'{coin_name}_price': df['Close'],
                f'{coin_name}_returns': df['Close'].pct_change()
            }).reset_index(drop=True)
        except Exception as e:
            print(f"Error fetching {name}: {str(e)}")
    return prices

def calculate_future_returns(df, coin, windows=[7, 14, 30]):
    """Calculate future returns for different time windows"""
    price_col = f'{coin}_price'
    for window in windows:
        df[f'future_return_{window}d'] = df[price_col].pct_change(periods=window).shift(-window)
        df[f'future_direction_{window}d'] = (df[f'future_return_{window}d'] > 0).astype(int)

def calculate_rolling_correlation(df, sentiment_window=7, windows=[7, 14, 30]):
    """Calculate rolling correlation between sentiment and future returns"""
    df['sentiment_ma'] = df['value'].rolling(window=sentiment_window).mean()
    
    correlations = {}
    for window in windows:
        correlation = df['sentiment_ma'].rolling(window=30).corr(df[f'future_return_{window}d'])
        correlations[f'correlation_{window}d'] = correlation
        
        df[f'predicted_direction_{window}d'] = (df['sentiment_ma'] > 50).astype(int)
        df[f'prediction_correct_{window}d'] = (
            df[f'predicted_direction_{window}d'] == df[f'future_direction_{window}d']
        ).astype(int)
        
    return correlations

def create_predictive_analysis(sentiment_data, crypto_prices, coin):
    """Create analysis of Fear & Greed index's predictive power"""
    # Process fear and greed data
    df_sentiment = pd.DataFrame(sentiment_data['data'])
    df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'].astype(int), unit='s')
    df_sentiment['value'] = df_sentiment['value'].astype(float)
    df_sentiment['timestamp'] = df_sentiment['timestamp'].dt.tz_localize(None)
    
    # Merge sentiment and price data
    df = df_sentiment.merge(crypto_prices[coin], on='timestamp', how='inner')
    
    # Calculate future returns and correlations
    windows = [7, 14, 30]
    calculate_future_returns(df, coin, windows)
    correlations = calculate_rolling_correlation(df, sentiment_window=7, windows=windows)
    
    price_col = f'{coin}_price'
    
    # Create visualization
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=(
                           f"{CRYPTOCURRENCIES[f'{coin.upper()}-USD']} Price vs Market Sentiment (7-day MA)",
                           "Correlation between Sentiment and Future Returns<br><sup>Higher = Sentiment predicts returns better | Lower = Sentiment predicts opposite</sup>",
                           "Prediction Accuracy Over Time<br><sup>Higher = More accurate predictions | Lower = Less accurate predictions</sup>"
                       ),
                       vertical_spacing=0.15,
                       row_heights=[0.4, 0.3, 0.3],
                       specs=[[{"secondary_y": True}],
                             [{"secondary_y": False}],
                             [{"secondary_y": False}]])
    
    # Plot 1: Price and Sentiment with color-coded background
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df[price_col],
                  name=f"{coin.upper()} Price",
                  line=dict(color='#2962FF', width=2)),
        row=1, col=1, secondary_y=False
    )
    
    # Add sentiment as filled area plot
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['sentiment_ma'],
                  name="Market Sentiment",
                  fill='tozeroy',
                  line=dict(color='#FF6D00', width=1),
                  fillcolor='rgba(255, 109, 0, 0.2)'),
        row=1, col=1, secondary_y=True
    )
    
    # Add zones for extreme fear and greed
    fig.add_hrect(y0=0, y1=25, 
                  fillcolor="green", opacity=0.1,
                  layer="below", line_width=0,
                  secondary_y=True,
                  row=1, col=1,
                  annotation_text="Extreme<br>Fear",
                  annotation=dict(font_size=10))
    
    fig.add_hrect(y0=75, y1=100,
                  fillcolor="red", opacity=0.1,
                  layer="below", line_width=0,
                  secondary_y=True,
                  row=1, col=1,
                  annotation_text="Extreme<br>Greed",
                  annotation=dict(font_size=10))
    
    # Plot 2: Rolling Correlations with zero line and colored areas
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    window_colors = {
        7: '#1E88E5',   # Blue
        14: '#43A047',  # Green
        30: '#E53935'   # Red
    }
    
    for window in windows:
        fig.add_trace(
            go.Scatter(x=df['timestamp'],
                      y=correlations[f'correlation_{window}d'],
                      name=f'{window}-day Correlation',
                      line=dict(color=window_colors[window], width=2)),
            row=2, col=1
        )
    
    # Plot 3: Prediction Accuracy with 50% baseline
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", row=3, col=1,
                  annotation_text="Random Guess (50%)")
    
    for window in windows:
        accuracy = df[f'prediction_correct_{window}d'].rolling(window=30).mean()
        fig.add_trace(
            go.Scatter(x=df['timestamp'],
                      y=accuracy,
                      name=f'{window}-day Accuracy',
                      line=dict(color=window_colors[window], width=2)),
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{CRYPTOCURRENCIES[f'{coin.upper()}-USD']} Market Sentiment Analysis<br><sup>Analyzing the predictive power of Fear & Greed Index</sup>",
            x=0.5,
            xanchor='center'
        ),
        height=1200,
        width=1400,  # Add fixed width
        showlegend=True,
        hovermode="x unified",
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    # Update axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    # Update y-axes titles and ranges
    fig.update_yaxes(title_text=f"{coin.upper()} Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Fear & Greed Index (0-100)", range=[0, 100], row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Correlation (-1 to 1)", range=[-1, 1], row=2, col=1)
    fig.update_yaxes(title_text="Prediction Accuracy (0-100%)", range=[0, 1], row=3, col=1)
    
    return fig, df

def analyze_magnitude_by_sentiment(df):
    """Analyze how large price movements are based on sentiment levels"""
    
    # Define sentiment buckets
    df['sentiment_bucket'] = pd.cut(df['value'], 
                                  bins=[0, 25, 45, 55, 75, 100],
                                  labels=['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'],
                                  ordered=True)
    
    windows = [7, 14, 30]
    analysis = {}
    
    for window in windows:
        returns = f'future_return_{window}d'
        # Simplified aggregation approach
        grouped = df.groupby('sentiment_bucket', observed=True)
        stats = pd.DataFrame({
            'mean': grouped[returns].mean() * 100,
            'std': grouped[returns].std() * 100,
            'max': grouped[returns].max() * 100,
            'min': grouped[returns].min() * 100,
            'count': grouped[returns].count()
        })
        
        # Calculate Sharpe-like ratio
        stats['risk_adjusted'] = stats['mean'] / stats['std']
        analysis[window] = stats.round(4)
    
    return analysis

def create_magnitude_analysis_fig(df, analysis, coin):
    """Create visualization for magnitude analysis"""
    
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=(
                           "Return Distribution by Sentiment Level",
                           "Average Returns by Sentiment Level",
                           "Risk (Volatility) by Sentiment Level",
                           "Risk-Adjusted Returns by Sentiment Level"
                       ),
                       vertical_spacing=0.15,
                       horizontal_spacing=0.1)
    
    sentiment_colors = {
        'Extreme Fear': '#1B5E20',   # Dark Green
        'Fear': '#4CAF50',          # Green
        'Neutral': '#9E9E9E',       # Gray
        'Greed': '#FF9800',         # Orange
        'Extreme Greed': '#B71C1C'  # Dark Red
    }
    
    window_colors = {
        7: '#1E88E5',   # Blue
        14: '#43A047',  # Green
        30: '#E53935'   # Red
    }
    
    # 1. Box Plot of Returns Distribution
    for window in [7, 14, 30]:
        returns = f'future_return_{window}d'
        fig.add_trace(
            go.Box(x=df['sentiment_bucket'],
                  y=df[returns] * 100,  # Convert to percentages
                  name=f'{window}d Returns',
                  marker_color=window_colors[window]),
            row=1, col=1
        )
    
    # 2. Average Returns by Sentiment
    for window in [7, 14, 30]:
        stats = analysis[window]
        fig.add_trace(
            go.Bar(x=stats.index,
                  y=stats['mean'],
                  name=f'{window}d Avg Return',
                  marker_color=window_colors[window],
                  showlegend=False),
            row=1, col=2
        )
    
    # 3. Risk (Volatility) by Sentiment
    for window in [7, 14, 30]:
        stats = analysis[window]
        fig.add_trace(
            go.Bar(x=stats.index,
                  y=stats['std'],
                  name=f'{window}d Volatility',
                  marker_color=window_colors[window],
                  showlegend=False),
            row=2, col=1
        )
    
    # 4. Risk-Adjusted Returns
    for window in [7, 14, 30]:
        stats = analysis[window]
        fig.add_trace(
            go.Bar(x=stats.index,
                  y=stats['risk_adjusted'],
                  name=f'{window}d Risk-Adjusted',
                  marker_color=window_colors[window],
                  showlegend=False),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{CRYPTOCURRENCIES[f'{coin.upper()}-USD']} Price Movement Magnitude Analysis<br><sup>How large are price moves following different sentiment levels?</sup>",
            x=0.5,
            xanchor='center'
        ),
        height=1000,
        width=1400,  # Add fixed width
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        boxmode='group'
    )
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', row=row, col=col)
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Future Returns (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return/Risk Ratio", row=2, col=2)
    
    return fig

def create_html_dashboard(figs, coins):
    """Create an HTML dashboard with tabs for different cryptocurrencies"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Sentiment Analysis Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .tab-container { margin-bottom: 20px; }
            .tab-button {
                padding: 10px 20px;
                border: none;
                background-color: #f0f0f0;
                cursor: pointer;
                font-size: 16px;
                margin-right: 5px;
                border-radius: 5px;
            }
            .tab-button.active {
                background-color: #2962FF;
                color: white;
            }
            .tab-content {
                display: none;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .tab-content.active { display: block; }
            .grid-container {
                display: grid;
                grid-template-columns: 1fr;
                gap: 30px;
                padding: 20px;
            }
            h1 {
                color: #2962FF;
                text-align: center;
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <h1>Cryptocurrency Sentiment Analysis Dashboard</h1>
        <div class="tab-container">
    """
    
    # Add tab buttons
    for i, coin in enumerate(coins):
        active = ' active' if i == 0 else ''
        html_content += f'<button class="tab-button{active}" onclick="openTab(event, \'{coin}\')">{CRYPTOCURRENCIES[f"{coin.upper()}-USD"]}</button>\n'
    
    # Add tab content
    for i, coin in enumerate(coins):
        active = ' active' if i == 0 else ''
        html_content += f"""
        <div id="{coin}" class="tab-content{active}">
            <div class="grid-container">
                <div id="{coin}_predictive"></div>
                <div id="{coin}_magnitude"></div>
            </div>
        </div>
        """
    
    # Add JavaScript for tab functionality
    html_content += """
        </div>
        <script>
        function openTab(evt, coinName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(coinName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        </script>
    """
    
    # Add Plotly figures
    for coin in coins:
        html_content += f"""
        <script>
            var predictive_{coin} = {figs[coin]['predictive'].to_json()};
            var magnitude_{coin} = {figs[coin]['magnitude'].to_json()};
            Plotly.newPlot('{coin}_predictive', predictive_{coin}.data, predictive_{coin}.layout);
            Plotly.newPlot('{coin}_magnitude', magnitude_{coin}.data, magnitude_{coin}.layout);
        </script>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

def main():
    try:
        # Fetch data
        sentiment_data = fetch_fear_greed_data()
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        crypto_prices = fetch_crypto_prices(start_date, end_date)
        
        # Create analyses for each cryptocurrency
        figures = {}
        analyses = {}
        
        for coin in crypto_prices.keys():
            print(f"\nAnalyzing {CRYPTOCURRENCIES[f'{coin.upper()}-USD']}...")
            
            # Create predictive analysis
            fig1, df = create_predictive_analysis(sentiment_data, crypto_prices, coin)
            
            # Create magnitude analysis
            analysis = analyze_magnitude_by_sentiment(df)
            fig2 = create_magnitude_analysis_fig(df, analysis, coin)
            
            figures[coin] = {
                'predictive': fig1,
                'magnitude': fig2
            }
            analyses[coin] = analysis
        
        # Create and save dashboard
        dashboard_html = create_html_dashboard(figures, list(crypto_prices.keys()))
        with open('crypto_dashboard.html', 'w') as f:
            f.write(dashboard_html)
        
        # Print summary statistics for each coin
        for coin in crypto_prices.keys():
            print(f"\n{CRYPTOCURRENCIES[f'{coin.upper()}-USD']} Analysis:")
            for window in [7, 14, 30]:
                print(f"\n{window}-Day Forward Returns by Sentiment Level:")
                print(analyses[coin][window].to_string())
        
        # Open the dashboard in the default browser
        webbrowser.open('file://' + os.path.realpath("crypto_dashboard.html"))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 