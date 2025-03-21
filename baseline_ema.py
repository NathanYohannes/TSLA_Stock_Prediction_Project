import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Constants
OUTPUT_PLOT_FILE = 'ema_baseline_results.png'

def load_data(file_path):
    """Load and preprocess the TSLA stock data."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def calculate_ema(data, span):
    """Calculate Exponential Moving Average."""
    return data.ewm(span=span, adjust=False).mean()

def create_predictions(df, spans):
    """Create predictions using multiple EMAs."""
    for span in spans:
        # Calculate EMA for the closing price
        df[f'EMA_{span}'] = calculate_ema(df['Close'], span)
        
        # Use EMA as prediction for next day
        df[f'Prediction_{span}'] = df[f'EMA_{span}'].shift(1)
        
        # Calculate predicted price change
        df[f'Predicted_Change_{span}'] = df[f'Prediction_{span}'] - df['Close']
        
        # Calculate actual price change
        df['Actual_Change'] = df['Close'].shift(-1) - df['Close']
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df

def calculate_trading_metrics(df, spans, initial_investment):
    """Calculate trading metrics for each EMA period."""
    for span in spans:
        # Initialize portfolio value
        portfolio_value = initial_investment
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        
        for i in range(len(df)-1):
            current_price = df['Close'].iloc[i]
            predicted_change = df[f'Predicted_Change_{span}'].iloc[i]
            actual_change = df['Actual_Change'].iloc[i]
            
            # Trading logic based on prediction
            if predicted_change > 0 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    profit = (df['Close'].iloc[i-1] - current_price) * abs(position)
                    portfolio_value += profit
                position = 1
                trades.append(('BUY', current_price, df.index[i]))
            elif predicted_change < 0 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    profit = (current_price - df['Close'].iloc[i-1]) * position
                    portfolio_value += profit
                position = -1
                trades.append(('SELL', current_price, df.index[i]))
        
        # Calculate final portfolio value
        final_value = portfolio_value
        total_return = ((final_value - initial_investment) / initial_investment) * 100
        num_trades = len(trades)
        
        print(f"\nTrading Metrics for {span}-day EMA:")
        print(f"Initial Investment: ${initial_investment:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {num_trades}")
        
        # Calculate win rate
        correct_predictions = sum(1 for i in range(len(df)-1) 
                                if (df[f'Predicted_Change_{span}'].iloc[i] > 0 and df['Actual_Change'].iloc[i] > 0) or
                                   (df[f'Predicted_Change_{span}'].iloc[i] < 0 and df['Actual_Change'].iloc[i] < 0))
        win_rate = (correct_predictions / (len(df)-1)) * 100
        print(f"Prediction Win Rate: {win_rate:.2f}%")

def evaluate_model(df, spans):
    """Evaluate the model's performance for each EMA period."""
    for span in spans:
        # Calculate metrics using numpy
        actual = df['Close'].values
        predicted = df[f'Prediction_{span}'].values
        
        # Mean Squared Error
        mse = np.mean((actual - predicted) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        print(f"\nModel Performance Metrics for {span}-day EMA:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")

def plot_results(df, spans):
    """Plot actual vs predicted prices for both EMAs."""
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Close'], label='Actual Price', alpha=0.7, color='black')
    
    colors = ['#FF6B6B', '#4ECDC4']  # Different colors for different EMAs
    for span, color in zip(spans, colors):
        plt.plot(df.index, df[f'Prediction_{span}'], 
                label=f'{span}-day EMA Prediction', 
                alpha=0.7, 
                color=color)
    
    plt.title('TSLA Stock Price: Actual vs Predicted (8-day and 20-day EMA)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILE)
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='TSLA Stock Price Prediction using EMA')
    parser.add_argument('--input', default='Datasets/raw/TSLA.csv',
                      help='Path to input CSV file')
    parser.add_argument('--periods', nargs='+', type=int, default=[8, 20],
                      help='EMA periods to use (space-separated integers)')
    parser.add_argument('--investment', type=float, default=10000,
                      help='Initial investment amount for trading simulation')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    
    # Create predictions
    print(f"Creating predictions using {args.periods}-day EMAs...")
    df = create_predictions(df, args.periods)
    
    # Evaluate model
    print("\nEvaluating models...")
    evaluate_model(df, args.periods)
    
    # Calculate trading metrics
    print("\nCalculating trading metrics...")
    calculate_trading_metrics(df, args.periods, args.investment)
    
    # Plot results
    print(f"\nGenerating visualization...")
    plot_results(df, args.periods)
    print(f"\nResults have been saved to '{OUTPUT_PLOT_FILE}'")

if __name__ == "__main__":
    main() 