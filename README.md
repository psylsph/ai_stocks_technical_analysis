# AI Cryptocurrency Technical Analysis

Python application that generates AI-powered technical analysis for cryptocurrency pairs using candlestick charts and machine learning.

## Features

- Candlestick charts with technical indicators (SMA-20)
- AI analysis using LLaMA 3.2 Vision model
- Multi-cryptocurrency support (BTC-USD, ETH-USD, etc.)
- PNG chart and Markdown report exports

## Prerequisites

- Python 3.x
- [Ollama](https://ollama.com/) with [LLaMA 3.2 Vision](https://ollama.com/library/llama3.2-vision)
- [Dependencies(requirements.txt)](./requirements.txt)

## Installation

1. Clone repository
```sh
git clone https://github.com/psylsph/ai_stocks_technical_analysis
```
2. Install packages:
```sh
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
setx GOOGLE_API_KEY <Your Google API Key>
```
## Usage

```sh
streamlit run .\ai_stocks_technical_analysis.py
```

## Credits
Based on [@DeepCharts](https://github.com/deepcharts)

## License
[MIT License](./LICENSE.md)