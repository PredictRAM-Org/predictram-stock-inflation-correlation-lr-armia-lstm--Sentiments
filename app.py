import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from pmdarima import auto_arima
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

# Download the VADER lexicon
import nltk
nltk.download('vader_lexicon')

# Load CPI data
cpi_data = pd.read_excel("CPI.xlsx")
cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
cpi_data.set_index('Date', inplace=True)

# Load stock data
stock_folder = "stock_folder"
stock_files = [f for f in os.listdir(stock_folder) if f.endswith(".xlsx")]

# Function to perform sentiment analysis using VADER
def perform_sentiment_analysis(reports):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for report in reports:
        score = sid.polarity_scores(report)['compound']
        sentiment_scores.append(score)

    # Categorize sentiment scores as positive, neutral, or negative
    positive_score = sum(1 for score in sentiment_scores if score > 0)
    neutral_score = sum(1 for score in sentiment_scores if score == 0)
    negative_score = sum(1 for score in sentiment_scores if score < 0)

    return positive_score, neutral_score, negative_score

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

    # Calculate the adjusted correlation based on expected inflation impact
    adjusted_correlation = correlation_close_cpi + 0.1 * expected_inflation  # You can adjust the multiplier as needed

    # Train Linear Regression model
    model_lr = LinearRegression()
    X_lr = merged_data[['CPI']]
    y_lr = merged_data['Close']
    model_lr.fit(X_lr, y_lr)

    # Train ARIMA model using auto_arima
    model_arima = auto_arima(y_lr, seasonal=False, suppress_warnings=True)
    
    # Predict future prices based on Linear Regression
    future_prices_lr = model_lr.predict([[expected_inflation]])

    # Predict future prices based on ARIMA
    future_prices_arima = model_arima.predict(1).iloc[0]  # 1 is the number of steps to forecast

    # Display the latest actual price
    latest_actual_price = merged_data['Close'].iloc[-1]

    return correlation_close_cpi, adjusted_correlation, future_prices_lr[0], future_prices_arima, latest_actual_price

# Function to get sentiment scores for a given stock using News API
def get_sentiment_scores(stock_name, api_key, num_reports=20):
    st.write(f"\nPerforming Sentiment Analysis for {stock_name}...")

    # Replace this with actual logic to fetch news reports for the stock using News API
    api_url = "https://newsapi.org/v2/everything"
    params = {
        'apiKey': api_key,
        'q': stock_name,
        'pageSize': num_reports,
        'sortBy': 'publishedAt',
        'language': 'en'
    }

    response = requests.get(api_url, params=params)
    news_data = response.json()

    # Extract headlines from the news data
    headlines = [article['title'] for article in news_data['articles']]

    # Perform sentiment analysis using VADER
    positive_score, neutral_score, negative_score = perform_sentiment_analysis(headlines)

    st.write(f"Positive Score: {positive_score}")
    st.write(f"Neutral Score: {neutral_score}")
    st.write(f"Negative Score: {negative_score}")

    return positive_score, neutral_score, negative_score

# Streamlit UI
st.title("Stock-CPI Correlation Analysis with Expected Inflation and Price Prediction")

# Date range selector for CPI data
cpi_end_date = pd.to_datetime("Nov 2023")  # Set the end date as the latest date in your data

# Options to select a fixed date range
date_range_options = {
    "Last One Month": cpi_end_date - pd.DateOffset(months=1),
    "Last 3 Months": cpi_end_date - pd.DateOffset(months=3),
    "Last 6 Months": cpi_end_date - pd.DateOffset(months=6),
    "Last 1 Year": cpi_end_date - pd.DateOffset(years=1),
    "Last 2 Years": cpi_end_date - pd.DateOffset(years=2),
    "Last 5 Years": cpi_end_date - pd.DateOffset(years=5),
}

selected_date_range = st.selectbox("Select Fixed Date Range for Training Data:", list(date_range_options.keys()))

# Calculate start date based on the selected date range
cpi_start_date = date_range_options[selected_date_range]

# Date range selector for stock data
stock_start_date = st.date_input("Select Start Date for Stock Data:", min_value=cpi_data.index.min(), max_value=cpi_end_date)
stock_end_date = st.date_input("Select End Date for Stock Data:", min_value=cpi_data.index.min(), max_value=cpi_end_date)

expected_inflation = st.number_input("Enter Expected Upcoming Inflation:", min_value=0.0, step=0.01)
news_api_key = st.text_input("Enter your News API Key:", type="password")
train_model_button = st.button("Train Model")

if train_model_button:
    st.write(f"Training model with Expected Inflation: {expected_inflation} and selected date ranges...")

    actual_correlations = []
    adjusted_correlations = []
    future_prices_lr_list = []
    future_prices_arima_list = []
    latest_actual_prices = []
    stock_names = []

    for stock_file in stock_files:
        st.write(f"\nTraining for {stock_file}...")
        selected_stock_data = pd.read_excel(os.path.join(stock_folder, stock_file))
        selected_stock_data.name = stock_file  # Assign a name to the stock_data for reference

        # Filter data based on selected date ranges
        selected_stock_data = selected_stock_data[(selected_stock_data['Date'] >= stock_start_date) & (selected_stock_data['Date'] <= stock_end_date)]
        selected_cpi_data = cpi_data[(cpi_data.index >= cpi_start_date) & (cpi_data.index <= cpi_end_date)]

        actual_corr, adjusted_corr, future_price_lr, future_price_arima, latest_actual_price = analyze_stock(selected_stock_data, selected_cpi_data, expected_inflation)

        actual_correlations.append(actual_corr)
        adjusted_correlations.append(adjusted_corr)
        future_prices_lr_list.append(future_price_lr)
        future_prices_arima_list.append(future_price_arima)
        latest_actual_prices.append(latest_actual_price)
        stock_names.append(stock_file)

    # Create a DataFrame with the results
    result_data = {
        'Stock': stock_names,
        'Actual Correlation': actual_correlations,
        'Adjusted Correlation': adjusted_correlations,
        'Predicted Price Change (Linear Regression)': future_prices_lr_list,
        'Predicted Price Change (ARIMA)': future_prices_arima_list,
        'Latest Actual Price': latest_actual_prices
    }
    result_df = pd.DataFrame(result_data)

    # Sort stocks by correlation with CPI Change
    result_df = result_df.sort_values(by='Actual Correlation', ascending=False)

    # Display the sorted results
    st.write("\nStocks Sorted by Correlation with CPI Change:")
    st.table(result_df)

    # Sentiment Analysis
    st.title("Sentiment Analysis")
    st.write("Performing sentiment analysis for selected stocks...")

    # Allow user to input stocks separated by comma
    selected_stocks_input = st.text_input("Enter stocks (separated by comma):")
    selected_stocks = [stock.strip() for stock in selected_stocks_input.split(',')]

    sentiment_results = []

    for stock in selected_stocks:
        if stock in result_df['Stock'].values:
            positive_score, neutral_score, negative_score = get_sentiment_scores(stock, news_api_key, num_reports=20)

            # Calculate change in correlation with CPI Change
            original_corr = result_df[result_df['Stock'] == stock]['Actual Correlation'].values[0]
            new_corr = original_corr + 0.1 * (positive_score - negative_score)
            change_in_corr = new_corr - original_corr

            sentiment_results.append({
                'Stock': stock,
                'Actual Correlation': original_corr,
                'Change in Correlation with CPI Change': change_in_corr,
                'Positive Score': positive_score,
                'Neutral Score': neutral_score,
                'Negative Score': negative_score
            })

    # Create DataFrame for sentiment analysis results
    sentiment_df = pd.DataFrame(sentiment_results)
    sentiment_df = sentiment_df.sort_values(by='Change in Correlation with CPI Change', ascending=False)

    # Display sentiment analysis results
    st.write("\nSentiment Analysis Results:")
    st.table(sentiment_df)

    # Calculate Total Score
    total_score = (sentiment_df['Positive Score'] - sentiment_df['Negative Score']).sum() + 0.1

    st.write("\nTotal Score:")
    st.write(total_score)
