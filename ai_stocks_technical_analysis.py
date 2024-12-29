## Source: Baed on work by @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import google.generativeai as genai
import base64
from datetime import datetime
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.googlesearch import GoogleSearch
import streamlit as st
import os

def get_stock_data(ticker, start_date, end_date, indicators, agent):
    ticker = ticker.strip()

    # Fetch stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data is available
    if  not stock_data.empty:

        data = stock_data

        # Plot candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"  # Replace "trace 0" with "Candlestick"
            )
        ])

        # Helper function to add indicators to the chart
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
            elif indicator == "20-Day EMA":
                ema = data['Close'].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

        # Add selected indicators to the chart
        for indicator in indicators:
            add_indicator(indicator)

        fig.update_layout(xaxis_rangeslider_visible=False)
        fig.update_layout(title=ticker + " Candlestick Chart with Technical Indicators")

        st.plotly_chart(fig)

        # Save the chart as an image
        image_data= fig.to_image(format="png")

        with st.spinner("Searching and interpreting news articles using " + agent  + ", please wait..."):

            # Sentiment Agent
            sentiment_agent = Agent(
                name="Sentiment Agent",
                role="Search and interpret news articles.",
                model=Gemini(id=agent),
                tools=[GoogleSearch()],
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
                "Analyze the sentiment for the following " + ticker + " during the previous 12 months.\n\n" +
                "1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each stock. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n" +
                "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates, weighting the response to the last 6 months information.", stream=True):
                sentiment_agent_response += delta.get_content_as_string()

        with st.spinner("Analyzing the sentiment and stock chart's technical indicators, please wait..."):

            GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBud2LfN_RDdqpWGrlfwnR7Ya86Jo32Iag")
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(agent)
            content = "You are a Stock Trader specializing in Technical Analysis at a top financial institution. \n\n" + \
                            "Analyze the stock chart's technical indicators and the Sentiment Analysis then provide a buy/hold/sell recommendation. \n\n" + \
                            "Base your recommendation only on the candlestick chart, the displayed technical indicators and the Sentiment Analysis \n\n" + \
                            "First, provide the recommendation, then, provide your detailed reasoning." +  "\n\n"
            full_analysis_response = model.generate_content([{'mime_type':'image/png', 'data': image_data}, sentiment_agent_response, content])
            st.markdown("#### Sentiment and Stock Chart Technical Indicators Analysis Results")
            st.markdown(full_analysis_response.text)
            st.markdown("\n---\n")
            st.markdown("#### Details of the Sentiment Analysis")
            st.markdown( sentiment_agent_response)


# Set up Streamlit app
st.set_page_config(layout="wide")
st.markdown("### AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# Input for stock ticker and date range
ticker = st.sidebar.text_input("Enter Stock / Crypto Ticker (e.g., DJT, BA.L, BTC-USD):", value='BA.L')
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(datetime.now().date()-pd.DateOffset(months=12)))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(datetime.now().date()-pd.DateOffset(days=1)))

# Sidebar: Select technical indicators
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands"]
)

st.sidebar.subheader("Analysis AI Agent")
#agent = st.sidebar.text_input("Enter AI Agent (e.g., gemini-1.5-flash, gemini-2.0-flash-exp):", value='gemini-2.0-flash-exp')
agent = st.sidebar.selectbox("Select AI Agent:", options=["gemini-1.5-flash", "gemini-2.0-flash-exp"], index=1)

# Fetch stock data
# Fetch stock data
if st.sidebar.button("Fetch Data"):
    get_stock_data(ticker, start_date, end_date, indicators, agent)
            