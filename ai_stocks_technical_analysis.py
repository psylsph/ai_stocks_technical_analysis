## Source: Based on work by @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

from pickle import TRUE
import yfinance
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from google import genai
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.googlesearch import GoogleSearch

import streamlit as st
import os

def get_stock_data(ticker, start_date, end_date, indicators, model):
    ticker = ticker.strip()

    start = start_date-pd.DateOffset(months=1)

    # Fetch stock data
    stock_data = yfinance.download(ticker, start=start, end=end_date, ignore_tz=True,auto_adjust=True,
                                   back_adjust=True, threads=True, rounding=True, multi_level_index=False)

    # Check if data is available
    if not stock_data.empty:

        # Plot candlestick chart (only show data from start_date onwards)
        display_data = stock_data.loc[start_date:]
        fig = go.Figure(data=[
            go.Candlestick(
                x=display_data.index,
                open=display_data['Open'],
                high=display_data['High'],
                low=display_data['Low'],
                close=display_data['Close'],
                name="Candlestick"  # Replace "trace 0" with "Candlestick",
                
            )
        ])

        # Helper function to add indicators to the chart
        def add_indicator(indicator):
            sma = stock_data['Close'].rolling(window=20).mean()
            ema = stock_data['Close'].ewm(span=20).mean()
            std = stock_data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std
            bb_lower = sma - 2 * std

            if indicator == "20-Day SMA":
                fig.add_trace(go.Scatter(x=display_data.index, y=sma.loc[start_date:], mode='lines', name='SMA (20)'))
            elif indicator == "20-Day EMA":
                fig.add_trace(go.Scatter(x=display_data.index, y=ema.loc[start_date:], mode='lines', name='EMA (20)'))
            elif indicator == "20-Day Bollinger Bands":
                fig.add_trace(go.Scatter(x=display_data.index, y=bb_upper.loc[start_date:], mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=display_data.index, y=bb_lower.loc[start_date:], mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                stock_data['VWAP'] = (stock_data['Close'] * stock_data['Volume']).cumsum() / stock_data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=display_data.index, y=stock_data['VWAP'].loc[start_date:], mode='lines', name='VWAP'))

        # Add selected indicators to the chart
        for indicator in indicators:
            add_indicator(indicator)


        delta = stock_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_text = "current 14 day Relative Strength Index is " + str(rsi[:-1].iloc[-1].round(2))
        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(title=ticker + " Candlestick Chart with Technical Indicators (" + rsi_text + ")")
        fig.update_layout(height=int(600))

        st.plotly_chart(fig)

        # Save the chart as an image
        image_data= fig.to_image(format="png")

        if model is None:
            st.stop()

        with st.spinner("Searching and interpreting news articles using " + model  + ", please wait..."):

            # Sentiment Agent
            sentiment_agent = Agent(
                name="Sentiment Agent",
                role="Search and interpret news articles.",
                model=Gemini(id=model),
                tools=[GoogleSearch(timeout=60)],
                instructions=[
                    "Find relevant news articles for " + ticker + " and analyze the sentiment.",
                    "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources."
                    "Cite your sources. Be specific and provide links."
                ],
                show_tool_calls=False,
                markdown=True,
            )

            sentiment_agent_response = ""
            for delta in sentiment_agent.run(
                "Analyze the sentiment for the following " + ticker + " during the previous 6 months.\n\n" +
                "1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each stock. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n" +
                "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates, weighting the response to the last months information.", stream=True):
                sentiment_agent_response += delta.get_content_as_string()
            st.markdown("## Sentiment Analysis Results")
            st.markdown( sentiment_agent_response, unsafe_allow_html=True)


        with st.spinner("Analyzing the sentiment and stock chart's technical indicators, please wait..."):



            content = "You are a Stock Trader specializing in Technical Analysis at a top financial institution. \n\n" + \
                            "Analyze the stock chart's technical indicators and the Sentiment Analysis then provide a buy/hold/sell recommendation. \n\n" + \
                            "Base your recommendation only on the candlestick chart, the displayed technical indicators and the Sentiment Analysis \n\n" + \
                            "First, provide the recommendation, then, provide your detailed reasoning." +  "\n\n"
            full_analysis_response = client.models.generate_content(
                model=model,
                contents=[
                    {
                        "parts": [
                            {"inline_data": {"mime_type": "image/png", "data": image_data}},
                            {"text": "Also consider the " + rsi_text},
                            {"text": sentiment_agent_response},
                            {"text": content}
                        ]
                    }
                ]
            )
            st.markdown("## Sentiment and Stock Chart Technical Indicators Analysis Results")
            st.markdown(full_analysis_response.text)

    else:
        st.markdown(ticker + " does not have pricing information available, for crypto try adding ***-USD***")

# Set up Streamlit app
st.set_page_config(layout="wide")
st.markdown("### AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock / Crypto Ticker (e.g., DJT, BA.L, BTC-USD):", value='BA.L')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(datetime.now().date()-pd.DateOffset(months=3)))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now().date()-pd.DateOffset(days=1)))

# Sidebar: Select technical indicators
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"]
)

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        st.sidebar.subheader("Analysis AI Agent")
        model = st.sidebar.selectbox("Select AI Agent:", options=["gemini-2.5-pro-exp-03-25", "gemini-2.5-flash-preview-04-17",
                                                                  "gemini-2.0-flash"], index=1)
    else:
        st.sidebar.warning("Google API key not found - sentiment analysis disabled")
        model = None
except Exception as e:
    st.sidebar.warning(f"Google API initialization failed: {str(e)}")
    model = None

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    get_stock_data(ticker.upper().strip(), start_date, end_date, indicators, model)
            