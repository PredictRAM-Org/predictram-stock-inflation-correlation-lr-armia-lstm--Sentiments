import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

# Load stock news data
stock_news_data = pd.read_excel("stock_news.xlsx")

# Initialize NewsAPI client (replace 'YOUR_NEWSAPI_KEY' with your actual NewsAPI key)
newsapi = NewsApiClient(api_key='5843e8b1715a4c1fb6628befb47ca1e8')

# Function to calculate correlation and build models
def analyze_stock(stock_data, cpi_data, expected_inflation, selected_tenure_offset):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data.set_index('Date', inplace=True)

    # Filter stock data based on selected tenure
    end_date = stock_data.index.max()
    start_date = end_date - selected_tenure_offset
    stock_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]

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
    scaled_data = y_lr.values.reshape(-1, 1)

    if scaled_data.shape[1] != 1:
        st.write(f"Error: Expected 1 column in scaled_data, but got {scaled_data.shape[1]} columns.")
        return None

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
    future_price_lstm = predict_future_lstm(last_observed_price, model_lstm)
    st.write(f"Predicted Stock Price for Future Inflation (LSTM): {future_price_lstm}")

    # Display the latest actual price
    latest_actual_price = merged_data['Close'].iloc[-1]
    st.write(f"Latest Actual Price for {stock_name}: {latest_actual_price}")

    return correlation_close_cpi, future_prices_lr[0], future_prices_arima, latest_actual_price, future_price_lstm, stock_name

def prepare_data_for_lstm(data, time_steps=1):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(input_shape, 1)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_future_lstm(last_observed_price, model):
    num_steps = 1  # number of steps to forecast
    predicted_prices = []

    for _ in range(num_steps):
        input_data = last_observed_price.reshape((1, len(last_observed_price), 1))
        predicted_price = model.predict(input_data, verbose=0)
        predicted_prices.append(predicted_price[0, 0])
        last_observed_price = np.append(last_observed_price[1:], predicted_price[0, 0])

    return predicted_prices[-1]

def perform_sentiment_analysis(stock_name):
    stock_news = stock_news_data[stock_news_data['Stock'] == stock_name]
    sentiment_scores = []

    for index, row in stock_news.iterrows():
        # Fetch news articles for the stock from NewsAPI
        news_articles = newsapi.get_everything(q=row['Stock'], from_param=row['Date'], to=row['Date'])

        if news_articles['totalResults'] > 0:
            # Extract text from news articles
            articles_text = ' '.join([article['description'] for article in news_articles['articles']])

            # Perform sentiment analysis using VADER
            analyzer = SentimentIntensityAnalyzer()
            sentiment_score = analyzer.polarity_scores(articles_text)
            sentiment_scores.append(sentiment_score['compound'])
        else:
            sentiment_scores.append(0.0)

    return sentiment_scores

# Streamlit UI
st.title("Stock-CPI Correlation Analysis with Expected Inflation, Price Prediction, and Sentiment Analysis")
expected_inflation = st.number_input("Enter Expected Upcoming Inflation:", min_value=0.0, step=0.01)

# Select tenure for training the model
tenure_options = ['6 months', '1 year', '3 years', '5 years']
selected_tenure = st.selectbox("Select Tenure for Training Model:", tenure_options)

# Convert tenure to timedelta for filtering data
tenure_mapping = {'6 months': pd.DateOffset(months=6),
                  '1 year': pd.DateOffset(years=1),
                  '3 years': pd.DateOffset(years=3),
                  '5 years': pd.DateOffset(years=5)}

selected_tenure_offset = tenure_mapping[selected_tenure]

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
    sentiment_scores_list = []

    for stock_file in stock_files:
        st.write(f"\nTraining for {stock_file}...")
        selected_stock_data = pd.read_excel(os.path.join(stock_folder, stock_file))
        selected_stock_data.name = stock_file  # Assign a name to the stock_data for reference

        # Perform analysis based on user-selected tenure
        analysis_result = analyze_stock(selected_stock_data, cpi_data, expected_inflation, selected_tenure_offset)
        if analysis_result is not None:
            correlation_close_cpi, future_price_lr, future_price_arima, latest_actual_price, future_price_lstm, stock_name = analysis_result

            # Perform sentiment analysis using VADER
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

    # Display sentiment analysis results
    st.write("\nSentiment Analysis Results:")
    sentiment_df = pd.DataFrame(sentiment_scores_list, columns=[f"Day {i + 1}" for i in range(len(sentiment_scores_list[0]))])
    sentiment_df.insert(0, 'Stock', stock_names)
    st.table(sentiment_df)
