import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import os
import glob

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        return (self.data[idx:idx + self.seq_length], 
                self.data[idx + self.seq_length, 3])  # 3 is the index for Close price

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        
        # Fully connected layers
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def normalize_data(data):
    """Normalize data using min-max scaling"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data, min_vals, max_vals

def denormalize_data(normalized_data, min_vals, max_vals):
    """Denormalize data back to original scale"""
    return normalized_data * (max_vals - min_vals) + min_vals

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
    plt.plot(dates, y_pred, 'r--', label='Predicted')
    plt.title('TSLA Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create LSTM directory if it doesn't exist
    os.makedirs('graphs/LSTM', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def get_next_model_number(model_name):
    """Get the next available model number for the given model name"""
    # Look for existing models with the same name pattern
    existing_models = glob.glob(os.path.join('models', f'{model_name}_*.pth'))
    
    if not existing_models:
        return 1
    
    # Extract numbers from existing filenames
    numbers = []
    for model_path in existing_models:
        try:
            # Get the filename without extension and split by underscore
            base_name = os.path.splitext(os.path.basename(model_path))[0]
            parts = base_name.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        except:
            continue
    
    if not numbers:
        return 1
    
    return max(numbers) + 1

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_name):
    """Train the model"""
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Get the next available model number
    model_number = get_next_model_number(model_name)
    model_path = os.path.join('models', f'{model_name}_{model_number}.pth')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs.squeeze(), batch_y).item()
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return model_path

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train LSTM model for stock price prediction')
    parser.add_argument('--input_file', type=str, help='Path to the input CSV file')
    parser.add_argument('--sequence_length', type=int, default=60, help='Number of time steps to look back')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_name', type=str, default='lstm_model', help='Name for the saved model')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    data, min_vals, max_vals, df = load_and_preprocess_data(args.input_file)
    
    # Create dataset and dataloaders
    dataset = StockDataset(data, args.sequence_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = LSTMModel(input_size=data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    model_path = train_model(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.model_name)
    print(f"\nBest model saved to: {model_path}")
    
    # Load best model for predictions
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Make predictions
    predictions = []
    actual_values = []
    dates = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
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
    plot_predictions(actual_values, predictions, test_dates, 'graphs/LSTM/lstm_predictions_pytorch.png')
    
    # Calculate and print metrics
    mse = np.mean((actual_values - predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - predictions))
    
    print(f"\nModel Performance Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

if __name__ == "__main__":
    main() 