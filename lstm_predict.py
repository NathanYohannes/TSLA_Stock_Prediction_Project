import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os
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

def plot_predictions(y_true, y_pred, dates, save_path):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual')
    plt.plot(dates, y_pred, label='Predicted')
    plt.title('TSLA Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Make predictions using a saved LSTM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--sequence_length', type=int, default=60, help='Number of time steps to look back')
    parser.add_argument('--output_plot', type=str, default='prediction_results.png', help='Path to save the prediction plot')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    data, min_vals, max_vals, df = load_and_preprocess_data(args.input_file)
    
    # Create dataset and dataloader
    dataset = StockDataset(data, args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LSTMModel(input_size=data.shape[1]).to(device)
    
    # Load the saved model
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    # Make predictions
    predictions = []
    actual_values = []
    dates = []
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(batch_y.numpy())
    
    # Denormalize predictions and actual values
    predictions = np.array(predictions).reshape(-1, 1)
    actual_values = np.array(actual_values).reshape(-1, 1)
    
    # Create full arrays for denormalization
    pred_full = np.zeros((len(predictions), data.shape[1]))
    actual_full = np.zeros((len(actual_values), data.shape[1]))
    pred_full[:, 3] = predictions.flatten()
    actual_full[:, 3] = actual_values.flatten()
    
    # Denormalize
    predictions = denormalize_data(pred_full, min_vals, max_vals)[:, 3]
    actual_values = denormalize_data(actual_full, min_vals, max_vals)[:, 3]
    
    # Get dates for plotting
    test_dates = df['Date'].iloc[-len(actual_values):]
    
    # Plot results
    plot_predictions(actual_values, predictions, test_dates, args.output_plot)
    
    # Calculate and print metrics
    mse = np.mean((actual_values - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - predictions))
    
    print(f"\nModel Performance Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    # Print the last prediction
    print(f"\nLast Prediction:")
    print(f"Date: {test_dates.iloc[-1]}")
    print(f"Actual Price: ${actual_values[-1]:.2f}")
    print(f"Predicted Price: ${predictions[-1]:.2f}")

if __name__ == "__main__":
    main() 