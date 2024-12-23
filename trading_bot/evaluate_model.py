import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from crypto_trading_env import CryptoTradingEnv
import yfinance as yf
import requests
import os

def fetch_and_prepare_data(start_date, end_date):
    """Fetch and prepare test data"""
    print(f"Fetching test data from {start_date} to {end_date}")
    
    # Convert dates to timezone-aware
    start_date = pd.Timestamp(start_date).tz_localize(None)  # Remove timezone first
    end_date = pd.Timestamp(end_date).tz_localize(None)
    
    # Fetch price data
    ticker = yf.Ticker("BTC-USD")
    price_data = ticker.history(start=start_date, end=end_date, interval='1d')
    
    # Fetch sentiment data
    days_needed = (end_date - start_date).days + 30
    url = f"https://api.alternative.me/fng/?limit={days_needed}&format=json"
    response = requests.get(url)
    sentiment_data = pd.DataFrame(response.json()['data'])
    
    # Process sentiment data
    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'].astype(int), unit='s')
    sentiment_data['value'] = sentiment_data['value'].astype(float)
    sentiment_data.set_index('timestamp', inplace=True)
    
    # Prepare DataFrame
    df = price_data[['Close']].copy()
    df.columns = ['price']
    
    # Remove timezone from price data index
    df.index = df.index.tz_localize(None)
    
    # Normalize timestamps to midnight
    df.index = df.index.normalize()
    sentiment_data.index = sentiment_data.index.normalize()
    
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
    
    # Drop NaN values
    df = df.dropna().copy()
    
    # Ensure all columns are float32
    float_columns = ['price', 'value', 'returns', 'volatility', 'rsi', 'momentum']
    for col in float_columns:
        df[col] = df[col].astype(np.float32)
    
    # Save index as timestamp column and reset index
    df['timestamp'] = df.index
    df = df.reset_index(drop=True)
    
    print("\nData Summary:")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of records: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df

def evaluate_model(model, env, df):
    """Run model evaluation and track trades"""
    obs, _ = env.reset()
    done = False
    trades = []
    portfolio_values = []
    current_position = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Track portfolio value and position
        portfolio_values.append({
            'timestamp': df.iloc[env.current_step-1]['timestamp'],
            'portfolio_value': info['balance'],
            'position': info['position']
        })
        
        # Track trades
        if len(env.trades) > len(trades):
            new_trade = env.trades[-1]
            trades.append({
                'timestamp': df.iloc[env.current_step-1]['timestamp'],
                'type': new_trade['type'],
                'price': new_trade['price'],
                'position': new_trade.get('position', 0),
                'pnl': new_trade.get('pnl', 0)
            })
        
        obs = next_obs
        done = terminated or truncated
    
    return trades, portfolio_values

def create_evaluation_plot(df, trades, portfolio_values):
    """Create visualization of model performance"""
    # Convert lists to DataFrames
    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_values)
    
    # Create figure
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=("BTC Price and Trading Activity",
                                     "Portfolio Value",
                                     "Fear & Greed Index"),
                       vertical_spacing=0.1,
                       row_heights=[0.5, 0.25, 0.25])
    
    # Plot 1: Price and trades
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['price'],
                  name="BTC Price",
                  line=dict(color='#2962FF')),
        row=1, col=1
    )
    
    # Add buy/sell markers
    if not trades_df.empty:
        # Buy points
        buy_trades = trades_df[trades_df['type'] == 'open']
        fig.add_trace(
            go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'],
                      mode='markers',
                      name="Buy",
                      marker=dict(color='green', size=10, symbol='triangle-up')),
            row=1, col=1
        )
        
        # Sell points
        sell_trades = trades_df[trades_df['type'] == 'close']
        fig.add_trace(
            go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'],
                      mode='markers',
                      name="Sell",
                      marker=dict(color='red', size=10, symbol='triangle-down')),
            row=1, col=1
        )
    
    # Plot 2: Portfolio value
    fig.add_trace(
        go.Scatter(x=portfolio_df['timestamp'], y=portfolio_df['portfolio_value'],
                  name="Portfolio Value",
                  line=dict(color='#00C853')),
        row=2, col=1
    )
    
    # Plot 3: Fear & Greed Index
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['value'],
                  name="Fear & Greed Index",
                  line=dict(color='#FF6D00')),
        row=3, col=1
    )
    
    # Add threshold lines for Fear & Greed Index
    fig.add_hline(y=25, line_dash="dash", line_color="green", row=3, col=1,
                  annotation_text="Extreme Fear")
    fig.add_hline(y=75, line_dash="dash", line_color="red", row=3, col=1,
                  annotation_text="Extreme Greed")
    
    # Update layout
    fig.update_layout(
        title="Model Evaluation Results",
        height=1200,
        showlegend=True,
        hovermode="x unified"
    )
    
    # Save plot
    fig.write_html("model_evaluation.html")
    
    return fig

def main():
    # Load the trained model
    model_path = "best_model/best_model"
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully")
        
        # Prepare test data (last 3 months)
        end_date = datetime.now()
        start_date = end_date - pd.DateOffset(months=3)
        df = fetch_and_prepare_data(start_date, end_date)
        print(f"Prepared {len(df)} data points for testing")
        
        # Create environment
        env = CryptoTradingEnv(df)
        print("Environment created")
        
        # Run evaluation
        print("Running model evaluation...")
        trades, portfolio_values = evaluate_model(model, env, df)
        
        # Calculate performance metrics
        initial_value = portfolio_values[0]['portfolio_value']
        final_value = portfolio_values[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        print("\nPerformance Summary:")
        print(f"Initial Portfolio Value: ${initial_value:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        
        # Create and save visualization
        print("\nCreating visualization...")
        fig = create_evaluation_plot(df, trades, portfolio_values)
        print("Visualization saved as 'model_evaluation.html'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 