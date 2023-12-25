import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load CPI data
cpi_data = pd.read_excel("CPI.xlsx")
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
cpi_data.set_index('Date', inplace=True)

# Load stock data
stock_folder = "stock_folder"
stock_files = [f for f in os.listdir(stock_folder) if f.endswith(".xlsx")]

# Load stock news data
stock_news_data = pd.read_excel("stock_news.xlsx")

# Function to calculate correlation and build models
def analyze_stock(stock_data, cpi_data, expected_inflation):
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
    arima_predictions = model_arima.predict(1)
    if isinstance(arima_predictions, pd.Series):
        future_prices_arima = arima_predictions.iloc[0]
    else:
        future_prices_arima = arima_predictions[0]
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

# Function to perform sentiment analysis using VADER on stock_news_data
def perform_sentiment_analysis(stock_name):
    st.write(f"\nPerforming Sentiment Analysis for {stock_name}...")

    # Check if the 'Stocks' column is present in stock_news_data
    if 'Stocks' not in stock_news_data.columns:
        st.write("Error: 'Stocks' column not found in stock_news_data. Please check your data.")
        return []

    # Extract news articles for the given stock_name
    stock_news = stock_news_data[stock_news_data['Stocks'] == stock_name]

    # Check if the 'News' column is present in stock_news_data
    if 'News' not in stock_news.columns:
        st.write("Error: 'News' column not found in stock_news_data. Please check your data.")
        return []

    news_titles = stock_news['News'].tolist()

    # Analyze sentiment using VADER
    sentiment_scores = []
    analyzer = SentimentIntensityAnalyzer()

    for title in news_titles:
        st.write(f"\nAnalyzing sentiment for a news article...")
        try:
            sentiment_scores.append(analyze_sentiment(analyzer, title))
        except Exception as e:
            st.write(f"Error analyzing sentiment for the article: {e}")
            sentiment_scores.append(None)

    return sentiment_scores

def analyze_sentiment(analyzer, text):
    compound_score = analyzer.polarity_scores(text)['compound']
    return compound_score

# Streamlit UI
st.title("Stock-CPI Correlation Analysis with Expected Inflation, Price Prediction, and Sentiment Analysis")
expected_inflation = st.number_input("Enter Expected Upcoming Inflation:", min_value=0.0, step=0.01)

# Train Model Button
train_model_button = st.button("Train Model")

if train_model_button:
    st.write(f"Training model with Expected Inflation: {expected_inflation}...")

    correlations = []
    future_prices_lr_list = []
    future_prices_arima_list = []
    latest_actual_prices = []
    future_price_lstm_list = []
    sentiment_scores_list = []
    stock_names = []

    for stock_file in stock_files:
        st.write(f"\nTraining for {stock_file}...")
        selected_stock_data = pd.read_excel(os.path.join(stock_folder, stock_file))
        selected_stock_data.name = stock_file  # Assign a name to the stock_data for reference

        min_max_scaler = MinMaxScaler()  # Move the MinMaxScaler initialization inside the loop
        correlation_close_cpi, future_price_lr, future_price_arima, latest_actual_price, future_price_lstm, stock_name = analyze_stock(selected_stock_data, cpi_data, expected_inflation)

        # Perform sentiment analysis
        sentiment_scores = perform_sentiment_analysis(stock_name)
        sentiment_scores_list.append(sentiment_scores)

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

    # Display sentiment scores in a table
    sentiment_data = {
        'Stock': stock_names,
        'Sentiment Scores': sentiment_scores_list
    }
    sentiment_df = pd.DataFrame(sentiment_data)
    st.write("\nSentiment Analysis Summary:")
    st.table(sentiment_df)
