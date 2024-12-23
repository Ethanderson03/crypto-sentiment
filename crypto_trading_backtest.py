import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
import webbrowser
import os

class CryptoSentimentTrader:
    def __init__(self, initial_capital=10000, trading_fee_pct=0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trading_fee_pct = trading_fee_pct
        self.positions = {}  # {coin: {'quantity': q, 'entry_price': p, 'entry_date': d, 'position_size': s, 'type': 'long/short'}}
        self.trades = []
        self.portfolio_values = []
        
    def calculate_position_size(self, sentiment_value, momentum_signal, available_capital, coin):
        """
        Calculate position size based on sentiment, momentum, and historical success rates
        More extreme sentiment and stronger momentum = larger position size
        """
        # Base position size on sentiment (0 to 1)
        if sentiment_value <= 35:  # Strong fear zone
            sentiment_score = (35 - sentiment_value) / 35  # Higher score for more fear
        elif sentiment_value >= 65:  # Strong greed zone
            sentiment_score = (sentiment_value - 65) / 35  # Higher score for more greed
        else:
            sentiment_score = 0.2  # Lower base score for neutral sentiment
        
        # Combine with momentum (0 to 1)
        momentum_score = (momentum_signal + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        
        # Adjust base position size by coin (ETH more aggressive)
        base_size = 0.3 if coin == 'ETH' else 0.2
        max_size = 0.9 if coin == 'ETH' else 0.7
        
        # Combined score (0 to 1)
        combined_score = (sentiment_score * 0.7 + momentum_score * 0.3)
        
        # Calculate position size (20-70% for BTC, 30-90% for ETH)
        position_size = available_capital * (base_size + (max_size - base_size) * combined_score)
        
        return position_size

    def execute_trade(self, date, coin, price, sentiment, momentum_signal, action, reason=""):
        """Execute a trade with position sizing and fee calculation"""
        if action in ['buy', 'short']:  # Opening positions
            position_size = self.calculate_position_size(sentiment, momentum_signal, self.capital, coin)
            if position_size > 0:
                fee = position_size * self.trading_fee_pct
                quantity = (position_size - fee) / price
                self.capital -= position_size
                
                position_type = 'long' if action == 'buy' else 'short'
                self.positions[coin] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'entry_date': date,
                    'position_size': position_size,
                    'type': position_type
                }
                
                self.trades.append({
                    'date': date,
                    'coin': coin,
                    'action': action,
                    'price': price,
                    'quantity': quantity,
                    'sentiment': sentiment,
                    'momentum': momentum_signal,
                    'fee': fee,
                    'position_size': position_size,
                    'reason': reason,
                    'type': position_type
                })
                
        elif action in ['sell', 'cover'] and coin in self.positions:  # Closing positions
            position = self.positions[coin]
            quantity = position['quantity']
            current_value = quantity * price
            fee = current_value * self.trading_fee_pct
            
            # Calculate profit percentage (reversed for short positions)
            if position['type'] == 'long':
                profit_pct = ((current_value - fee) - position['position_size']) / position['position_size'] * 100
            else:  # short position
                profit_pct = (position['position_size'] - (current_value + fee)) / position['position_size'] * 100
            
            # Add to capital (for shorts, we gain when price goes down)
            if position['type'] == 'long':
                self.capital += (current_value - fee)
            else:
                # For shorts, we return the borrowed coins and keep the difference
                self.capital += (position['position_size'] + (position['position_size'] * profit_pct / 100) - fee)
            
            self.trades.append({
                'date': date,
                'coin': coin,
                'action': action,
                'price': price,
                'quantity': quantity,
                'sentiment': sentiment,
                'momentum': momentum_signal,
                'fee': fee,
                'position_size': current_value,
                'profit_pct': profit_pct,
                'reason': reason,
                'type': position['type']
            })
            del self.positions[coin]

    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value including open positions"""
        portfolio_value = self.capital
        for coin, position in self.positions.items():
            if coin in current_prices:
                current_value = position['quantity'] * current_prices[coin]
                if position['type'] == 'long':
                    portfolio_value += current_value
                else:  # short position
                    # For shorts, we gain when price goes down
                    profit_pct = (position['entry_price'] - current_prices[coin]) / position['entry_price']
                    portfolio_value += position['position_size'] * (1 + profit_pct)
        return portfolio_value

def fetch_fear_greed_data():
    """Fetch Fear and Greed index data"""
    # Calculate the number of days needed to cover our test period plus some buffer
    days_needed = 365 * 3  # 3 years of data
    
    url = f"https://api.alternative.me/fng/?limit={days_needed}&format=json"
    print(f"\nFetching Fear & Greed data for the past {days_needed} days...")
    
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return None
    
    data = response.json()
    print(f"Received {len(data['data'])} days of Fear & Greed data")
    
    return data

def prepare_backtest_data(coin, start_date, end_date):
    """Prepare data for backtesting including sentiment and price data"""
    # Fetch price data
    ticker = yf.Ticker(f"{coin}-USD")
    price_data = ticker.history(start=start_date, end=end_date)
    print(f"\nPrice data range: {price_data.index[0]} to {price_data.index[-1]}")
    print(f"Number of price data points: {len(price_data)}")
    
    # Convert price data index to UTC
    price_data.index = price_data.index.tz_localize(None)
    
    # Fetch sentiment data
    sentiment_data = pd.DataFrame(fetch_fear_greed_data()['data'])
    sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'].astype(int), unit='s')
    sentiment_data['value'] = sentiment_data['value'].astype(float)
    
    # Print sentiment data info
    print(f"\nSentiment data range: {sentiment_data['timestamp'].min()} to {sentiment_data['timestamp'].max()}")
    print(f"Number of sentiment data points: {len(sentiment_data)}")
    
    # Filter sentiment data to match price data range
    sentiment_data = sentiment_data[
        (sentiment_data['timestamp'] >= start_date) & 
        (sentiment_data['timestamp'] <= end_date)
    ]
    sentiment_data.set_index('timestamp', inplace=True)
    
    print(f"\nFiltered sentiment data range: {sentiment_data.index[0]} to {sentiment_data.index[-1]}")
    print(f"Number of filtered sentiment data points: {len(sentiment_data)}")
    
    # Merge price and sentiment data
    df = price_data[['Close']].copy()
    df.columns = ['price']
    
    # Print before merge
    print(f"\nBefore merge - Price data shape: {df.shape}")
    print(f"Before merge - Sentiment data shape: {sentiment_data.shape}")
    
    # Merge with forward fill for missing sentiment values
    df = df.merge(sentiment_data, left_index=True, right_index=True, how='left')
    df['value'] = df['value'].fillna(method='ffill')  # Forward fill missing values
    
    # Print after merge
    print(f"\nAfter merge - Final data shape: {df.shape}")
    print(f"Final data range: {df.index[0]} to {df.index[-1]}")
    print(f"Number of days with missing sentiment: {df['value'].isna().sum()}")
    
    # Calculate momentum indicators
    df['sma20'] = df['price'].rolling(window=20).mean()
    df['sma50'] = df['price'].rolling(window=50).mean()
    df['momentum'] = (df['sma20'] - df['sma50']) / df['sma50']
    
    # Calculate sentiment indicators
    df['sentiment_ma5'] = df['value'].rolling(window=5).mean()
    df['sentiment_ma20'] = df['value'].rolling(window=20).mean()
    
    # Calculate sentiment momentum (rate of change)
    df['sentiment_change'] = df['sentiment_ma5'] - df['sentiment_ma20']
    
    # Calculate future returns for different periods
    df['return_1d'] = df['price'].pct_change(1).shift(-1)
    df['return_7d'] = df['price'].pct_change(7).shift(-7)
    df['return_14d'] = df['price'].pct_change(14).shift(-14)
    df['return_30d'] = df['price'].pct_change(30).shift(-30)
    
    return df

def run_backtest(coin, start_date, end_date, initial_capital=10000):
    """Run backtest for a specific coin"""
    # Prepare data
    df = prepare_backtest_data(coin, start_date, end_date)
    
    # Add volatility indicator
    df['price_change'] = df['price'].pct_change()
    df['volatility'] = df['price_change'].rolling(window=20).std() * np.sqrt(365) * 100
    
    # Add RSI
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Initialize trader
    trader = CryptoSentimentTrader(initial_capital=initial_capital)
    
    # Run simulation
    for idx, row in df.iterrows():
        current_prices = {coin: row['price']}
        
        # Check for position closure if in position
        if coin in trader.positions:
            position = trader.positions[coin]
            is_long = position['type'] == 'long'
            price_change = (row['price'] - position['entry_price']) / position['entry_price']
            profit_pct = price_change * 100 if is_long else -price_change * 100
            days_held = (idx - position['entry_date']).days
            
            # Dynamic profit targets and stop losses based on volatility and coin
            volatility_factor = min(2, max(0.5, row['volatility'] / 50))
            base_profit_target = 2.5 if coin == 'ETH' else 2.0
            base_stop_loss = -2.0 if coin == 'ETH' else -1.5
            
            should_close = False
            close_reason = ""
            
            # 1. Quick profit (prioritize this)
            if profit_pct >= base_profit_target * volatility_factor and days_held <= 2:
                should_close = True
                close_reason = "Quick Profit"
            
            # 2. Time-based scaling profit target
            elif days_held >= 1 and profit_pct >= (base_profit_target * 0.7 * (1 + days_held * 0.2)):
                should_close = True
                close_reason = "Scaled Profit"
            
            # 3. Dynamic stop loss based on volatility and time held
            elif profit_pct <= base_stop_loss * volatility_factor * (1 + days_held * 0.1):
                should_close = True
                close_reason = "Stop Loss"
            
            # 4. Take profits on strong sentiment reversal
            elif (is_long and row['sentiment_change'] >= 3 and profit_pct > 0) or \
                 (not is_long and row['sentiment_change'] <= -3 and profit_pct > 0):
                should_close = True
                close_reason = "Sentiment Reversal"
            
            # 5. Maximum hold time with any profit
            elif days_held >= 5 and profit_pct > 0:
                should_close = True
                close_reason = "Time-based Exit"
            
            if should_close:
                action = 'sell' if is_long else 'cover'
                trader.execute_trade(idx, coin, row['price'], row['value'], row['momentum'], action, close_reason)
        
        # Enter new position if conditions are met
        elif coin not in trader.positions:
            should_trade = False
            trade_type = None
            trade_reason = ""
            
            # More stringent entry conditions
            if (
                # Strong fear with positive momentum
                (row['value'] <= 35 and row['momentum'] > 0 and row['rsi'] < 40) or
                # Extreme fear regardless of momentum
                (row['value'] <= 25 and row['rsi'] < 45) or
                # Strong momentum with reasonable sentiment
                (row['momentum'] >= 0.02 and row['value'] < 60 and row['rsi'] < 60)
            ):
                should_trade = True
                trade_type = 'buy'
                trade_reason = "Long - " + (
                    "Strong Fear" if row['value'] <= 35
                    else "Extreme Fear" if row['value'] <= 25
                    else "Strong Momentum"
                )
            
            # Short only on extreme conditions
            elif (
                # Extreme greed with negative momentum
                (row['value'] >= 75 and row['momentum'] < 0 and row['rsi'] > 60) or
                # Strong negative momentum with high sentiment
                (row['momentum'] <= -0.02 and row['value'] >= 65 and row['rsi'] > 70)
            ):
                should_trade = True
                trade_type = 'short'
                trade_reason = "Short - " + (
                    "Extreme Greed" if row['value'] >= 75
                    else "Strong Downtrend"
                )
            
            if should_trade and trade_type:
                trader.execute_trade(idx, coin, row['price'], row['value'], row['momentum'], trade_type, trade_reason)
        
        # Track portfolio value
        portfolio_value = trader.get_portfolio_value(current_prices)
        trader.portfolio_values.append({
            'date': idx,
            'portfolio_value': portfolio_value
        })
    
    # Calculate buy & hold performance
    initial_price = df['price'].iloc[0]
    final_price = df['price'].iloc[-1]
    buy_hold_return = (final_price - initial_price) / initial_price * 100
    
    # Calculate strategy performance
    strategy_return = (trader.get_portfolio_value({coin: final_price}) - initial_capital) / initial_capital * 100
    
    return trader, buy_hold_return, strategy_return, df

def create_backtest_visualization(trader, df, coin, buy_hold_return, strategy_return):
    """Create visualization of backtest results"""
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=(f"{coin} Price and Trading Activity",
                                     "Sentiment and Momentum",
                                     "Portfolio Value"),
                       vertical_spacing=0.1,
                       row_heights=[0.4, 0.3, 0.3],
                       specs=[[{"secondary_y": False}],
                             [{"secondary_y": True}],
                             [{"secondary_y": False}]])

    # Price and Moving Averages
    fig.add_trace(
        go.Scatter(x=df.index, y=df['price'],
                  name=f"{coin} Price",
                  line=dict(color='#2962FF')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sma20'],
                  name="20-day MA",
                  line=dict(color='#00C853', dash='dot')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sma50'],
                  name="50-day MA",
                  line=dict(color='#FF6D00', dash='dot')),
        row=1, col=1
    )
    
    # Add trade points to price chart
    trades_df = pd.DataFrame(trader.trades)
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        fig.add_trace(
            go.Scatter(x=buy_trades['date'], y=buy_trades['price'],
                      mode='markers',
                      name="Buy Points",
                      marker=dict(color='green', size=12, symbol='triangle-up'),
                      text=[f"Buy<br>Reason: {r}<br>Sentiment: {s:.1f}<br>Size: ${ps:.0f}<br>Price: ${p:.2f}"
                           for r, s, ps, p in zip(buy_trades['reason'],
                                                buy_trades['sentiment'],
                                                buy_trades['position_size'],
                                                buy_trades['price'])],
                      hovertemplate="%{text}"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=sell_trades['date'], y=sell_trades['price'],
                      mode='markers',
                      name="Sell Points",
                      marker=dict(color='red', size=12, symbol='triangle-down'),
                      text=[f"Sell<br>Reason: {r}<br>Sentiment: {s:.1f}<br>Profit: {p:.1f}%<br>Price: ${pr:.2f}"
                           for r, s, p, pr in zip(sell_trades['reason'],
                                                sell_trades['sentiment'],
                                                sell_trades['profit_pct'],
                                                sell_trades['price'])],
                      hovertemplate="%{text}"),
            row=1, col=1
        )
    
    # Sentiment and Momentum
    fig.add_trace(
        go.Scatter(x=df.index, y=df['value'],
                  name="Fear & Greed Index",
                  line=dict(color='#2962FF')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['sentiment_change'],
                  name="Sentiment Change",
                  line=dict(color='#FF6D00')),
        row=2, col=1, secondary_y=True
    )
    
    # Add threshold lines for sentiment
    fig.add_hline(y=40, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="green", row=2, col=1)
    
    # Portfolio Value
    portfolio_df = pd.DataFrame(trader.portfolio_values)
    fig.add_trace(
        go.Scatter(x=portfolio_df['date'], y=portfolio_df['portfolio_value'],
                  name="Portfolio Value",
                  line=dict(color='#00C853')),
        row=3, col=1
    )
    
    # Add buy & hold comparison
    buy_hold_values = df['price'] * (trader.initial_capital / df['price'].iloc[0])
    fig.add_trace(
        go.Scatter(x=df.index, y=buy_hold_values,
                  name="Buy & Hold",
                  line=dict(color='#FF6D00', dash='dot')),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Backtest Results: {coin}<br>" +
                 f"Strategy Return: {strategy_return:.2f}% | Buy & Hold Return: {buy_hold_return:.2f}%<br>" +
                 f"Number of Trades: {len(trader.trades)}",
            x=0.5,
            xanchor='center'
        ),
        height=1200,
        width=1400,
        showlegend=True,
        hovermode="x unified"
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text=f"{coin} Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", row=2, col=1)
    fig.update_yaxes(title_text="Sentiment Change", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=3, col=1)
    
    return fig

def create_tabbed_visualization(results):
    """Create a tabbed visualization with results for all coins"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Trading Backtest Results</title>
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
            }
            .tab-button.active {
                background-color: #2962FF;
                color: white;
            }
            .tab-content {
                display: none;
                padding: 20px;
                border: 1px solid #ddd;
            }
            .tab-content.active { display: block; }
            .summary {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .stats-container {
                display: flex;
                justify-content: space-between;
                flex-wrap: wrap;
            }
            .stat-box {
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px;
                flex: 1;
                min-width: 200px;
            }
        </style>
    </head>
    <body>
        <div class="tab-container">
    """
    
    # Add tab buttons
    for coin in results.keys():
        html_content += f"""
            <button class="tab-button" onclick="openTab(event, '{coin}')" id="{coin}-btn">
                {coin}
            </button>
        """
    
    # Add tab content
    for coin, result in results.items():
        trader, buy_hold_return, strategy_return, df = result
        
        # Create visualization
        fig = create_backtest_visualization(trader, df, coin, buy_hold_return, strategy_return)
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
        
        trades_df = pd.DataFrame(trader.trades) if trader.trades else pd.DataFrame()
        
        html_content += f"""
        <div id="{coin}" class="tab-content">
            <div class="summary">
                <h2>{coin} Backtest Results</h2>
                <div class="stats-container">
                    <div class="stat-box">
                        <h3>Strategy Performance</h3>
                        <p>Return: {strategy_return:.2f}%</p>
                        <p>Final Value: ${trader.get_portfolio_value({coin: df['price'].iloc[-1]}):.2f}</p>
                        <p>Number of Trades: {len(trader.trades)}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Buy & Hold Performance</h3>
                        <p>Return: {buy_hold_return:.2f}%</p>
                        <p>Initial Price: ${df['price'].iloc[0]:.2f}</p>
                        <p>Final Price: ${df['price'].iloc[-1]:.2f}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Sentiment Statistics</h3>
                        <p>Average Sentiment: {df['value'].mean():.2f}</p>
                        <p>Extreme Fear Days (≤25): {len(df[df['value'] <= 25])}</p>
                        <p>Extreme Greed Days (≥75): {len(df[df['value'] >= 75])}</p>
                    </div>
                </div>
            </div>
            {plot_html}
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
            // Open first tab by default
            document.getElementById(document.getElementsByClassName("tab-button")[0].id).click();
        </script>
    </body>
    </html>
    """
    
    return html_content

def analyze_trades(trades_df):
    """Analyze trade performance by type and reason"""
    if trades_df.empty:
        return None
    
    # Convert to DataFrame if it's a list
    if isinstance(trades_df, list):
        trades_df = pd.DataFrame(trades_df)
    
    # Separate long and short trades
    long_trades = trades_df[trades_df['type'] == 'long']
    short_trades = trades_df[trades_df['type'] == 'short']
    
    # Analysis by trade type
    type_analysis = {
        'long': {
            'count': len(long_trades[long_trades['action'] == 'buy']),
            'profitable': len(long_trades[long_trades['profit_pct'] > 0]) if 'profit_pct' in long_trades.columns else 0,
            'avg_profit': long_trades['profit_pct'].mean() if 'profit_pct' in long_trades.columns else 0,
            'max_profit': long_trades['profit_pct'].max() if 'profit_pct' in long_trades.columns else 0,
            'max_loss': long_trades['profit_pct'].min() if 'profit_pct' in long_trades.columns else 0,
        },
        'short': {
            'count': len(short_trades[short_trades['action'] == 'short']),
            'profitable': len(short_trades[short_trades['profit_pct'] > 0]) if 'profit_pct' in short_trades.columns else 0,
            'avg_profit': short_trades['profit_pct'].mean() if 'profit_pct' in short_trades.columns else 0,
            'max_profit': short_trades['profit_pct'].max() if 'profit_pct' in short_trades.columns else 0,
            'max_loss': short_trades['profit_pct'].min() if 'profit_pct' in short_trades.columns else 0,
        }
    }
    
    # Analysis by reason
    reason_analysis = {}
    for reason in trades_df['reason'].unique():
        reason_trades = trades_df[trades_df['reason'] == reason]
        if 'profit_pct' in reason_trades.columns:
            reason_analysis[reason] = {
                'count': len(reason_trades),
                'profitable': len(reason_trades[reason_trades['profit_pct'] > 0]),
                'avg_profit': reason_trades['profit_pct'].mean(),
                'max_profit': reason_trades['profit_pct'].max(),
                'max_loss': reason_trades['profit_pct'].min(),
            }
    
    return {
        'by_type': type_analysis,
        'by_reason': reason_analysis
    }

def print_trade_analysis(analysis):
    """Print detailed trade analysis"""
    if not analysis:
        print("No trades to analyze")
        return
    
    print("\nTrade Analysis by Type:")
    print("-----------------------")
    for trade_type, stats in analysis['by_type'].items():
        if stats['count'] > 0:
            profit_ratio = (stats['profitable'] / stats['count']) * 100 if stats['count'] > 0 else 0
            print(f"\n{trade_type.upper()} Trades:")
            print(f"Count: {stats['count']}")
            print(f"Profitable: {stats['profitable']} ({profit_ratio:.1f}%)")
            print(f"Average Profit: {stats['avg_profit']:.2f}%")
            print(f"Max Profit: {stats['max_profit']:.2f}%")
            print(f"Max Loss: {stats['max_loss']:.2f}%")
    
    print("\nTrade Analysis by Reason:")
    print("------------------------")
    for reason, stats in analysis['by_reason'].items():
        if stats['count'] > 0:
            profit_ratio = (stats['profitable'] / stats['count']) * 100 if stats['count'] > 0 else 0
            print(f"\n{reason}:")
            print(f"Count: {stats['count']}")
            print(f"Profitable: {stats['profitable']} ({profit_ratio:.1f}%)")
            print(f"Average Profit: {stats['avg_profit']:.2f}%")
            print(f"Max Profit: {stats['max_profit']:.2f}%")
            print(f"Max Loss: {stats['max_loss']:.2f}%")

def main():
    # Set up backtest parameters
    coins = ['BTC', 'ETH']
    
    # Test period from January 2022 to December 2023 (includes crypto winter and recovery)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    initial_capital = 10000
    
    # Print test period
    print(f"\nTesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("This period includes the 2022 crypto winter and 2023 recovery\n")
    
    results = {}
    
    for coin in coins:
        print(f"\nBacktesting {coin}...")
        trader, buy_hold_return, strategy_return, df = run_backtest(
            coin, start_date, end_date, initial_capital
        )
        
        # Store results
        results[coin] = (trader, buy_hold_return, strategy_return, df)
        
        # Analyze trades
        trades_df = pd.DataFrame(trader.trades)
        trade_analysis = analyze_trades(trades_df)
        
        # Print basic results
        print(f"\n{coin} Backtest Results:")
        print(f"Strategy Return: {strategy_return:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"Number of Trades: {len(trader.trades)}")
        print(f"Final Portfolio Value: ${trader.get_portfolio_value({coin: df['price'].iloc[-1]}):.2f}")
        
        # Print detailed trade analysis
        print_trade_analysis(trade_analysis)
    
    # Create and save tabbed visualization
    html_content = create_tabbed_visualization(results)
    with open('backtest_results.html', 'w') as f:
        f.write(html_content)
    
    # Open visualization
    webbrowser.open(f"file://{os.path.realpath('backtest_results.html')}")

if __name__ == "__main__":
    main() 