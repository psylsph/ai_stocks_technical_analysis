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
git clone https://github.com/psylsph/ai-crypto-apps
```
2. Install packages:
```sh
cd ai_technical_analysis
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
3. Install Ollama with LLaMA 3.2 Vision

## Install Ollama

1. Download and install via https://ollama.com/download
2. `ollama pull llama3.2-vision`

## Usage

### Run analysis:

Enter tickers when prompted (e.g., "BTC-USD, ETH-USD")

## Output
### Generated files:

* analysis_results_<TICKER>.png - Chart
* analysis_results_<TICKER>.md - Analysis

## Credits
Based on [@DeepCharts](https://github.com/deepcharts)

## License
[MIT License](./LICENSE.md)