import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

# Load CPI data
cpi_data = pd.read_excel("CPI.xlsx")
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
cpi_data.set_index('Date', inplace=True)

# Load stock data
stock_folder = "stock_folder"
stock_files = [f for f in os.listdir(stock_folder) if f.endswith(".xlsx")]

# Function to calculate correlation and build models
def analyze_stock(stock_data, cpi_data, expected_inflation, min_max_scaler):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

    # Merge stock and CPI data on Date
    merged_data = pd.merge(stock_data, cpi_data, left_index=True, right_index=True, how='inner')

    # Handle NaN values in CPI column
    if merged_data['CPI'].isnull().any():
        st.write(f"Warning: NaN values found in 'CPI' column for {stock_data.name}. Dropping NaN values.")
        merged_data = merged_data.dropna(subset=['CPI'])

    # Calculate CPI change
    merged_data['CPI Change'] = merged_data['CPI'].pct_change()

    # Drop NaN values after calculating percentage change
    merged_data = merged_data.dropna()

    # Show correlation between 'Close' column and 'CPI Change'
    correlation_close_cpi = merged_data['Close'].corr(merged_data['CPI Change'])
    correlation_actual = merged_data['Close'].corr(merged_data['CPI'])

    stock_name = getattr(stock_data, 'name', None)
    if stock_name is None:
        # Use file name as a fallback if 'name' attribute is not available
        stock_name = os.path.basename(stock_file)

    st.write(f"Correlation between 'Close' and 'CPI Change' for {stock_name}: {correlation_close_cpi}")
    st.write(f"Actual Correlation between 'Close' and 'CPI' for {stock_name}: {correlation_actual}")

    # Train Linear Regression model
    model_lr = LinearRegression()
    X_lr = merged_data[['CPI']]
    y_lr = merged_data['Close']
    model_lr.fit(X_lr, y_lr)

    # Train ARIMA model using auto_arima
    model_arima = auto_arima(y_lr, seasonal=False, suppress_warnings=True)

    # Train LSTM model
    scaled_data = min_max_scaler.fit_transform(y_lr.values.reshape(-1, 1))

    x_train, y_train = prepare_data_for_lstm(scaled_data)
    model_lstm = build_lstm_model(x_train.shape[1])
    model_lstm.fit(x_train, y_train, epochs=50, batch_size=32)

    # Predict future prices based on Linear Regression
    future_prices_lr = model_lr.predict([[expected_inflation]])
    st.write(f"Predicted Price Change for Future Inflation (Linear Regression): {future_prices_lr[0]}")

    # Predict future prices based on ARIMA
    future_prices_arima = model_arima.predict(1).values[0]
  # 1 is the number of steps to forecast
    st.write(f"Predicted Price Change for Future Inflation (ARIMA): {future_prices_arima}")

    # Predict future prices using LSTM
    last_observed_price = scaled_data[-1]
    future_price_lstm = predict_future_lstm(last_observed_price, model_lstm, min_max_scaler)
    st.write(f"Predicted Stock Price for Future Inflation (LSTM): {future_price_lstm}")

    # Display the latest actual price
    latest_actual_price = merged_data['Close'].iloc[-1]
    st.write(f"Latest Actual Price for {stock_name}: {latest_actual_price}")

    return correlation_close_cpi, future_prices_lr[0], future_prices_arima, latest_actual_price, future_price_lstm, stock_name

def prepare_data_for_lstm(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_lstm(last_observed_price, model, min_max_scaler, num_steps=1):
    predicted_prices = []
    input_data = last_observed_price.reshape(1, -1, 1)

    for _ in range(num_steps):
        predicted_price = model.predict(input_data)
        predicted_prices.append(predicted_price[0, 0])
        input_data = np.append(input_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

    return min_max_scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))[-1, 0]

# Function to perform sentiment analysis using VADER
def perform_sentiment_analysis(stock_names):
    st.write("\nPerforming Sentiment Analysis...")
    analyzer = SentimentIntensityAnalyzer()

    # Fetch news articles for each stock using News API
    newsapi = NewsApiClient(api_key='YOUR_NEWS_API_KEY')  # Replace 'YOUR_NEWS_API_KEY' with your actual API key

    sentiment_scores = []
    for stock_name in stock_names:
        st.write(f"\nFetching news for {stock_name}...")
        try:
            articles = newsapi.get_everything(q=stock_name, language='en', sort_by='relevancy')
            headlines = [article['title'] for article in articles['articles']]
            sentiment_scores.append(analyze_sentiment(analyzer, headlines))
        except Exception as e:
            st.write(f"Error fetching news for {stock_name}: {e}")
            sentiment_scores.append(None)

    return sentiment_scores

def analyze_sentiment(analyzer, headlines):
    compound_scores = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    average_score = np.mean(compound_scores)
    return average_score

# Streamlit UI
st.title("Stock-CPI Correlation Analysis with Expected Inflation and Price Prediction")
expected_inflation = st.number_input("Enter Expected Upcoming Inflation:", min_value=0.0, step=0.01)

# Select tenure for training the model
tenure_options = ['1 year', '3 years', '5 years', '10 years']
selected_tenure = st.selectbox("Select Tenure for Training Model:", tenure_options)

# Convert tenure to timedelta for filtering data
tenure_mapping = {'1 year': pd.DateOffset(years=1),
                  '3 years': pd.DateOffset(years=3),
                  '5 years': pd.DateOffset(years=5),
                  '10 years': pd.DateOffset(years=10)}

selected_tenure_offset = tenure_mapping[selected_tenure]
end_date = pd.to_datetime("2023-11-01")  # Last date available in the data
start_date = end_date - selected_tenure_offset

# Train Model Button
train_model_button = st.button("Train Model")

if train_model_button:
    st.write(f"Training model with Expected Inflation: {expected_inflation} and Tenure: {selected_tenure}...")

    correlations = []
    future_prices_lr_list = []
    future_prices_arima_list = []
    latest_actual_prices = []
    future_price_lstm_list = []
    stock_names = []

    for stock_file in stock_files:
        st.write(f"\nTraining for {stock_file}...")
        selected_stock_data = pd.read_excel(os.path.join(stock_folder, stock_file))
        selected_stock_data.name = stock_file  # Assign a name to the stock_data for reference

        # Filter stock data based on selected tenure
        selected_stock_data = selected_stock_data[(selected_stock_data['Date'] >= start_date) & (selected_stock_data['Date'] <= end_date)]

        min_max_scaler = MinMaxScaler()  # Move the MinMaxScaler initialization inside the loop
        correlation_close_cpi, future_price_lr, future_price_arima, latest_actual_price, future_price_lstm, stock_name = analyze_stock(selected_stock_data, cpi_data, expected_inflation, min_max_scaler)

        correlations.append(correlation_close_cpi)
        future_prices_lr_list.append(future_price_lr)
        future_prices_arima_list.append(future_price_arima)
        latest_actual_prices.append(latest_actual_price)
        future_price_lstm_list.append(future_price_lstm)
        stock_names.append(stock_name)

    # Display overall summary in a table
    summary_data = {
        'Stock': stock_names,
        'Correlation with CPI Change': correlations,
        'Predicted Price Change (Linear Regression)': future_prices_lr_list,
        'Predicted Price Change (ARIMA)': future_prices_arima_list,
        'Latest Actual Price': latest_actual_prices,
        'Predicted Stock Price (LSTM)': future_price_lstm_list
    }
    summary_df = pd.DataFrame(summary_data)
    st.write("\nCorrelation and Price Prediction Summary:")
    st.table(summary_df)

    # Perform sentiment analysis
    sentiment_scores = perform_sentiment_analysis(stock_names)

    # Display sentiment scores in a table
    sentiment_data = {
        'Stock': stock_names,
        'Sentiment Score': sentiment_scores
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    st.write("\nSentiment Analysis Summary:")
    st.table(sentiment_df)
