# TSLA Stock Price Prediction and Trading System

This project implements a machine learning-based trading system for Tesla (TSLA) stock using LSTM neural networks and sentiment analysis. The system consists of two main components:
1. A price prediction model using LSTM
2. An automated trading bot that makes decisions based on predictions

## Setup

1. Install required dependencies:
```bash
pip install torch pandas numpy matplotlib newsapi-python
```

2. Set up your NewsAPI key for sentiment analysis:
   - Get an API key from [NewsAPI](https://newsapi.org/)
   - Store it securely (the sentiment analysis module will guide you)

## Using the Trading Bot

The trading bot (`trader_bot.py`) uses predictions from the LSTM model to make trading decisions. It maintains a state of your portfolio and considers transaction fees.

### Basic Usage

```bash
python trader_bot.py --model_path models/tsla_lstm_1.pth --input_file Project_Datasets/Cleaned_and_Formatted/March20_1year.csv
```

### Parameters

- `--model_path`: Path to your trained LSTM model
- `--input_file`: Path to your input data CSV file
- `--initial_capital`: Starting capital (default: $10,000)
- `--transaction_fee`: Transaction fee percentage (default: 1.0%)

### Output Example

```
=== Current Portfolio Status ===
Available Capital: $10000.00
Shares Owned: 0
Portfolio Value: $10000.00

=== Trade Decision ===
Current Price: $248.71
Predicted Price (with sentiment): $239.58
SELL - No shares owned

=== Updated Portfolio Status ===
Available Capital: $10000.00
Shares Owned: 0
Portfolio Value: $10000.00
```

### Trading Logic

- Buys when predicted price > current price (uses 50% of available capital)
- Sells when predicted price < current price (sells 50% of owned shares)
- Considers transaction fees in calculations
- Maintains portfolio state between runs

## Making Future Predictions (Optional)

You can also use the prediction system directly without trading:

```bash
python predict_future.py --model_path models/tsla_lstm_1.pth --input_file Project_Datasets/Cleaned_and_Formatted/March20_1year.csv --days_ahead 1
```

### Parameters

- `--model_path`: Path to your trained LSTM model
- `--input_file`: Path to your input data CSV file
- `--days_ahead`: Number of days to predict (default: 1)
- `--sequence_length`: Number of past days to consider (default: 60)

### Output Example

```
Date: 2024-03-21
Current Price: $248.71
Predicted Price Before Sentiment: $244.25
Predicted Price After Sentiment: $239.58
```

## Project Structure

```
├── models/                  # Saved LSTM models
├── graphs/                  # Generated visualizations
│   ├── EMA/                # EMA analysis graphs
│   └── LSTM/               # LSTM prediction graphs
├── Scripts/                # Core modules
│   ├── lstm_model.py       # LSTM model implementation
│   ├── predict_future.py   # Future price prediction
│   └── sentiment_analysis.py # News sentiment analysis
└── trader_bot.py           # Automated trading system
```

## Important Notes

1. The system uses both price predictions and sentiment analysis from news
2. Trading decisions are simplified to reduce risk:
   - Only uses 50% of available capital for buying
   - Only sells 50% of owned shares when selling
3. Transaction fees are considered in all calculations
4. Portfolio state is saved between sessions in `trader_state.json`

## Disclaimer

This is an experimental trading system. Always perform your own due diligence before making real trading decisions. The system's predictions should not be considered as financial advice. 