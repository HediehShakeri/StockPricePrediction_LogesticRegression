## Stock Price Direction Prediction Model
# Overview
This project implements a robust machine learning pipeline for predicting stock price direction using logistic regression. The model leverages historical stock data from Yahoo Finance, incorporates technical indicators, and provides performance evaluation and visualization capabilities. The codebase is designed to be modular, extensible, and suitable for both educational and practical applications in financial data analysis.
Features

### Data Acquisition: Fetches historical stock data using the yfinance library for a user-specified stock symbol and time period.
### Feature Engineering: Computes technical indicators such as Simple Moving Averages (SMA), Relative Strength Index (RSI), and binary overbought/oversold signals.
### Model Training: Employs logistic regression with feature scaling to predict whether the stock price will increase or decrease on the next trading day.
### Evaluation: Assesses model performance using metrics like accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
### Visualization: Generates a plot showing actual stock prices alongside predicted probabilities of price increases, saved as a high-resolution image.
### Persistence: Saves the trained model and scaler for future use using joblib.

# Requirements
To run this project, ensure the following Python packages are installed:

yfinance: For fetching stock market data.  
pandas: For data manipulation and storage.  
numpy: For numerical computations.  
scikit-learn: For machine learning model training and evaluation.  
matplotlib: For plotting results.  
joblib: For model serialization.  

# Install dependencies using:
pip install yfinance pandas numpy scikit-learn matplotlib joblib

# Usage

Run the Script:Execute the main script (stock_prediction.py) in a Python environment:python stock_prediction.py


# Input Parameters:
Stock Symbol: Enter a valid stock ticker (e.g., AAPL for Apple Inc.).  
Time Period: Specify the number of months of historical data to fetch (e.g., 12 for one year).  


# Output:
A CSV file (<symbol>_stock_data.csv) containing the downloaded stock data.  
A trained model and scaler saved as logistic_model.pkl and scaler.pkl.  
A performance report printed to the console, including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.  
A visualization saved as logistic_price_prediction.png, showing stock prices and predicted probabilities.  



# Code Structure
The codebase is organized into modular functions for clarity and reusability:

download_stock_data(symbol, period): Retrieves and saves historical stock data.  
calculate_rsi(prices, periods): Computes the Relative Strength Index (RSI) for a price series.  
prepare_features(df): Generates technical indicators and prepares features and target variables.  
train_model(X, y): Trains a logistic regression model with scaled features.  
evaluate_model(model, X_test_scaled, y_test): Evaluates model performance using multiple metrics.  
plot_results(df, model, X_test_scaled, y_test): Visualizes actual prices and predicted probabilities.  
main(): Orchestrates the pipeline, handling user input and function execution.  

# Technical Details

## Data Preprocessing:
Missing values in technical indicators are handled using a hybrid fill strategy: SMA values are filled with the closing price, and RSI defaults to a neutral value (50).  
Features include closing price, volume, 5-day SMA, 20-day SMA, RSI, and binary overbought/oversold indicators.  
The target variable is a binary indicator of whether the next day's closing price is higher than the current day's.  


## Model Training:
Data is split into training (80%) and testing (20%) sets without shuffling to preserve temporal order.  
Features are standardized using StandardScaler to ensure consistent model performance.  
Logistic regression is used with a maximum of 500 iterations for convergence.  


## Evaluation:
Metrics focus on classification performance, with ROC-AUC providing insight into probabilistic predictions.  
The confusion matrix details true positives, false positives, true negatives, and false negatives.  


## Visualization:
The plot uses a color gradient to represent the probability of price increases, overlaid on the actual closing prices.  
The output is saved as a high-resolution PNG file for clarity.  



## Limitations

The model predicts price direction (up/down) rather than magnitude, limiting its use for precise price forecasting.  
Logistic regression assumes linear separability, which may not capture complex market dynamics.  
Historical data from Yahoo Finance may occasionally contain gaps or inconsistencies.  
The model does not account for external factors like news or macroeconomic events.  

## Future Improvements

Incorporate additional technical indicators (e.g., MACD, Bollinger Bands) or fundamental data.  
Experiment with advanced models like Random Forest, Gradient Boosting, or neural networks.  
Add cross-validation to improve model robustness.  
Implement real-time data fetching for live predictions.  
Enhance visualization with interactive plots using libraries like plotly.  

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as needed.
