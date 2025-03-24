import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import argparse
import os
from Scripts import sentiment_analysis
from lstm_model import LSTMModel, StockDataset, normalize_data, denormalize_data

def load_and_preprocess_data(file_path):
    """Load and preprocess the stock data"""
    # Load the data
    df = pd.read_csv(file_path)
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Select features for prediction
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    # Normalize the data
    normalized_data, min_vals, max_vals = normalize_data(data)
    
    return normalized_data, min_vals, max_vals, df

def make_future_prediction(model, last_sequence, device):
    """Make a prediction for the next day"""
    model.eval()
    with torch.no_grad():
        # Add batch dimension
        last_sequence = last_sequence.unsqueeze(0).to(device)
        prediction = model(last_sequence)
        return prediction.cpu().numpy()[0][0]

def get_valued_sentiment():
    """Get sentiment value, return 0 if NewsAPI key is not available"""
    try:
        articles = sentiment_analysis.get_tesla_articles(days_back=7)
        if not articles:  # If no articles found (likely due to missing API key)
            print("NewsAPI key not found or invalid. Running without sentiment analysis.")
            return 0
        df = sentiment_analysis.process_articles(articles)
        metrics = sentiment_analysis.calculate_metrics(df)
        print(metrics)
        sentiment_score = metrics['avg_weighted_sentiment']
        valued_sentiment = sentiment_score * 100
        return valued_sentiment
    except Exception as e:
        print("Error in sentiment analysis. Running without sentiment adjustment.")
        return 0

def buy_sell_decision(predicted_price, current_price, threshold = 5):
    if predicted_price > current_price + threshold :
        return 'buy'
    elif predicted_price < current_price - threshold:
        return 'sell'
    else:
        return 'hold'
    
def get_current_price(df):
    return df.iloc[-1, 1]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Make future predictions using a saved LSTM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--sequence_length', type=int, default=60, help='Number of time steps to look back')
    parser.add_argument('--days_ahead', type=int, default=1, help='Number of days to predict ahead')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    data, min_vals, max_vals, df = load_and_preprocess_data(args.input_file)
    
    # Initialize model
    model = LSTMModel(input_size=data.shape[1]).to(device)
    
    # Load the saved model
    model.load_state_dict(torch.load(args.model_path))
    
    # Get the last sequence of data
    last_sequence = torch.FloatTensor(data[-args.sequence_length:])
    
    # Make predictions for future days
    predictions = []
    dates = []
    last_date = df['Date'].iloc[-1]
    
    print("\nFuture Predictions:")
    for i in range(args.days_ahead):
        # Make prediction
        pred = make_future_prediction(model, last_sequence, device)
        
        # Create full array for denormalization
        pred_full = np.zeros((1, data.shape[1]))
        pred_full[0, 3] = pred  # 3 is the index for Close price
        
        # Denormalize prediction
        denormalized_pred = denormalize_data(pred_full, min_vals, max_vals)[0, 3]
        valued_sentiment = get_valued_sentiment()
        sentiment_weighted_price = denormalized_pred + valued_sentiment

        # Calculate next date
        next_date = last_date + timedelta(days=i+1)
        
        predictions.append(denormalized_pred)
        dates.append(next_date)
        
        print(f"\nDate: {next_date.strftime('%Y-%m-%d')}")
        print(f"Current Price: ${get_current_price(df):.2f}")
        print(f"Predicted Price: ${denormalized_pred:.2f}")
        if valued_sentiment != 0:
            print(f"Sentiment Adjustment: ${valued_sentiment:.2f}")
            print(f"Final Predicted Price: ${sentiment_weighted_price:.2f}")
        
        if valued_sentiment != 0:
            print(f"Buy/Sell Decision: {buy_sell_decision(sentiment_weighted_price, get_current_price(df))}")
        else:
            print(f"Buy/Sell Decision: {buy_sell_decision(denormalized_pred, get_current_price(df))}")

        # Update last_sequence for next prediction
        # We'll use the predicted value as the next day's close price
        new_row = np.zeros((1, data.shape[1]))
        new_row[0, 3] = pred  # Use predicted close price
        # Use the previous day's values for other features
        new_row[0, 0] = data[-1, 0]  # Open
        new_row[0, 1] = data[-1, 1]  # High
        new_row[0, 2] = data[-1, 2]  # Low
        new_row[0, 4] = data[-1, 4]  # Volume
        
        # Update data and last_sequence
        data = np.vstack([data, new_row])
        last_sequence = torch.FloatTensor(data[-args.sequence_length:])
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Close'], label='Historical')
    plt.plot(dates, predictions, 'r--', label='Predictions')
    plt.title('TSLA Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create LSTM directory if it doesn't exist
    os.makedirs('graphs/LSTM', exist_ok=True)
    plt.savefig('graphs/LSTM/future_predictions.png')
    plt.close()

if __name__ == "__main__":
    main() 