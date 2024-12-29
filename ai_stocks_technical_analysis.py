## Source: @DeepCharts Youtube Channel (https://www.youtube.com/@DeepCharts)

#### NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ollama
import google.generativeai as genai
import base64
from datetime import datetime
from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.googlesearch import GoogleSearch

# Input for stock ticker and date range
tickers = input("Enter Stock Ticker(s) (e.g., BTC-USD, ETH-USD):")
if tickers == "":
    #tickers = "BTC-USD, ETH-USD, BA.L"
    tickers = "ETH-USD"
print("Stock Tickers: " + tickers)
start_date = pd.to_datetime(datetime.now().date()-pd.DateOffset(months=12))
end_date = pd.to_datetime(datetime.now().date()-pd.DateOffset(days=1))

for ticker in tickers.split(","):

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

        indicators = ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"]
        #indicators = ["20-Day SMA"]

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

        png_renderer = pio.renderers["png"]
        png_renderer.width = 1024
        png_renderer.height = 768
        pio.renderers.default = "png"

        graph_file = "analysis_results_" + ticker + ".png"

        # Save chart as a temporary image
        fig.write_image(graph_file, "png")

        # Read image and encode to Base64
        with open(graph_file, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')

        print("AI Analysis Request for " + ticker)

        md_file = "analysis_results_" + ticker + ".md"

        # Sentiment Agent
        sentiment_agent = Agent(
            name="Sentiment Agent",
            role="Search and interpret news articles.",
            model=Ollama(id="llama3.1"),
            tools=[GoogleSearch()],
            instructions=[
                "Find relevant news articles for " + ticker + " and analyze the sentiment.",
                "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources."
                "Cite your sources. Be specific and provide links."
            ],
            show_tool_calls=False,
            markdown=True,
        )

        response = ""
        for delta in sentiment_agent.run(
            "Analyze the sentiment for the following " + ticker + " during the previous 6 months.\n\n" +
            "1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each stock. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n" +
            "Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.", stream=True):
            response += delta.get_content_as_string()

        content = "You are a Stock Trader specializing in Technical Analysis at a top financial institution. \n\n" + \
                        "Analyze the stock chart's technical indicators and the Sentiment Analysis then provide a buy/hold/sell recommendation. \n\n" + \
                        "Base your recommendation only on the candlestick chart, the displayed technical indicators and the Sentiment Analysis \n\n" + \
                        "First, provide the recommendation, then, provide your detailed reasoning." +  "\n\n"
        
        # Prepare AI analysis request
        messages = [{
            'role': 'user',
            'content': content + response,
            'images': [image_data]
        }]

        genai.configure(api_key="AIzaSyBud2LfN_RDdqpWGrlfwnR7Ya86Jo32Iag")
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Display AI analysis result
        with open(md_file, "w") as analysis_results:
            analysis_results.write("**AI Analysis Results for " + ticker + "**\n\n")
            analysis_results.write("![" + graph_file + "](" + graph_file + ")\n\n")
            analysis_results.flush()
            response = ollama.chat(model='llama3.2-vision', messages=messages)
            analysis_results.write("## llama3.2-vision\n")
            analysis_results.write(response["message"]["content"])
            analysis_results.flush()
            response = ollama.chat(model='vanilj/Phi-4', messages=messages)
            analysis_results.write("## vanilj/Phi-4\n")
            analysis_results.write(response["message"]["content"])
            analysis_results.flush()
            response = model.generate_content([{'mime_type':'image/png', 'data': image_data}, response, messages[0]['content']])
            analysis_results.write("## gemini-1.5-flash\n")
            analysis_results.write(response.text)
            analysis_results.flush()


            