# AI Stocks Technical Analysis

A Python application that generates AI-powered technical analysis for stocks and cryptocurrencies using candlestick charts, technical indicators, sentiment analysis, and Google's Gemini AI models.

## Features

- Interactive candlestick charts with multiple technical indicators:
  - 20-Day Simple Moving Average (SMA)
  - 20-Day Exponential Moving Average (EMA)
  - 20-Day Bollinger Bands
  - Volume Weighted Average Price (VWAP)
- AI-powered analysis using Google's Gemini models (gemini-1.5-pro or gemini-2.0-flash-exp)
- Sentiment analysis from news sources with scoring
- Support for multiple assets:
  - US Stocks (e.g., DJT)
  - UK Stocks (e.g., BA.L)
  - Cryptocurrencies (e.g., BTC-USD)
- Interactive Streamlit web interface
- Detailed sentiment analysis with source citations
- Combined technical and sentiment analysis recommendations

## Prerequisites

- Python 3.x
- Google API Key for Gemini AI models

## Installation

1. Clone repository:
```sh
git clone https://github.com/psylsph/ai_stocks_technical_analysis
```
2. Install packages:
```sh
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir kaleido==0.1.0.post1 # force install this version on Windows OS
```

## Usage

1. Set your Google API key:

```sh
export GOOGLE_API_KEY=<Your Google API Key>
```

2. Run the Streamlit application:

```sh
streamlit run ai_stocks_technical_analysis.py
```
3. In the web interface:
* Enter a stock/crypto ticker
* Select date range
* Choose technical indicators
* Select Gemini AI model version
* Click "Fetch Data" to generate analysis

## Credits
Based on [@DeepCharts](https://github.com/deepcharts)

## License
[MIT License](./LICENSE.md)