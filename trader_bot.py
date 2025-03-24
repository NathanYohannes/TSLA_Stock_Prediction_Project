import argparse
import subprocess
import json
from datetime import datetime
import os
from Scripts import sentiment_analysis

class TraderBot:
    def __init__(self, capital, shares_owned, transaction_fee_percent=1.0):
        self.capital = capital
        self.shares_owned = shares_owned
        self.transaction_fee_percent = transaction_fee_percent
        self.transaction_fee_multiplier = 1 + (transaction_fee_percent / 100)
    
    def calculate_max_shares_to_buy(self, current_price):
        """Calculate maximum shares that can be bought with available capital"""
        return int(self.capital / (current_price * self.transaction_fee_multiplier))
    
    def execute_trade(self, current_price, predicted_price, sentiment_price):
        """Execute trade based on predictions and current portfolio"""
        # Simple trading logic: buy if predicted price is higher, sell if lower
        price_diff = sentiment_price - current_price
        
        if price_diff > 0:  # Buy signal
            # Use 50% of available capital for each trade
            shares_to_buy = int(self.calculate_max_shares_to_buy(current_price) * 0.5)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * self.transaction_fee_multiplier
                self.capital -= cost
                self.shares_owned += shares_to_buy
                return f"BUY {shares_to_buy} shares at ${current_price:.2f} (with fees: ${cost:.2f})"
        else:  # Sell signal
            # Sell 50% of owned shares
            shares_to_sell = int(self.shares_owned * 0.5)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price / self.transaction_fee_multiplier
                self.capital += revenue
                self.shares_owned -= shares_to_sell
                return f"SELL {shares_to_sell} shares at ${current_price:.2f} (after fees: ${revenue:.2f})"
        
        return "HOLD - No trade executed"
    
    def get_portfolio_value(self, current_price):
        """Calculate total portfolio value"""
        stock_value = self.shares_owned * current_price
        return self.capital + stock_value
    
    def save_state(self, filename="trader_state.json"):
        """Save current state to file"""
        state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "capital": self.capital,
            "shares_owned": self.shares_owned
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=4)
    
    @classmethod
    def load_state(cls, filename="trader_state.json", initial_capital=10000, transaction_fee_percent=1.0):
        """Load state from file or create new instance with default values"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                state = json.load(f)
                return cls(state["capital"], state["shares_owned"], transaction_fee_percent)
        return cls(initial_capital, 0, transaction_fee_percent)

def run_prediction(model_path, input_file):
    """Run predict_future.py and parse its output"""
    cmd = f"python predict_future.py --model_path {model_path} --input_file {input_file} --days_ahead 1"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse the output to get prices
    lines = result.stdout.split('\n')
    current_price = None
    predicted_price = None
    sentiment_price = None
    
    for line in lines:
        if "Current Price: $" in line:
            current_price = float(line.split("$")[1])
        elif "Predicted Price: $" in line:
            predicted_price = float(line.split("$")[1])
        elif "Final Predicted Price: $" in line:
            sentiment_price = float(line.split("$")[1])
    
    # If no sentiment-adjusted price available, use the base prediction
    if predicted_price and not sentiment_price:
        sentiment_price = predicted_price
    
    return current_price, predicted_price, sentiment_price

def main():
    parser = argparse.ArgumentParser(description='TSLA Trading Bot')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved LSTM model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--initial_capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--transaction_fee', type=float, default=1.0, help='Transaction fee percentage')
    
    args = parser.parse_args()
    
    # Initialize or load trader bot
    trader = TraderBot.load_state(initial_capital=args.initial_capital, 
                                transaction_fee_percent=args.transaction_fee)
    
    # Get predictions
    current_price, predicted_price, sentiment_price = run_prediction(args.model_path, args.input_file)
    
    if all(price is not None for price in [current_price, predicted_price, sentiment_price]):
        # Print current portfolio status
        print("\n=== Current Portfolio Status ===")
        print(f"Available Capital: ${trader.capital:.2f}")
        print(f"Shares Owned: {trader.shares_owned}")
        print(f"Portfolio Value: ${trader.get_portfolio_value(current_price):.2f}")
        
        # Execute trade
        print("\n=== Trade Decision ===")
        print(f"Current Price: ${current_price:.2f}")
        if sentiment_price != predicted_price:
            print(f"Base Predicted Price: ${predicted_price:.2f}")
            print(f"Final Predicted Price: ${sentiment_price:.2f}")
        else:
            print(f"Predicted Price: ${predicted_price:.2f}")
        trade_result = trader.execute_trade(current_price, predicted_price, sentiment_price)
        print(trade_result)
        
        # Print updated portfolio status
        print("\n=== Updated Portfolio Status ===")
        print(f"Available Capital: ${trader.capital:.2f}")
        print(f"Shares Owned: {trader.shares_owned}")
        print(f"Portfolio Value: ${trader.get_portfolio_value(current_price):.2f}")
        
        # Save state
        trader.save_state()
    else:
        print("Error: Could not get valid predictions from predict_future.py")

if __name__ == "__main__":
    main() 