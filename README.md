# Monte-Carlo-Based-Stock-Price-Prediction-using-Geometric-Brownian-Motion
This project predicts future stock prices using Geometric Brownian Motion (GBM) and Monte Carlo simulations.
A detailed 3 month forecast of stock behaviour is obtained through this project.
### This prediction tool is for listings on the Indian Stock Market. 


### Concept

For a stock entered by the user, the drift and volatility are calculated from that stock's historical data (1 year). These two parameters are used in the formula of Geometric Brownian Motion along with the present day price of the stock which is taken as the initial price. Monte Carlo simulations are implemented 100,000 times with the use of Geometric Brownian Motion to predict the price of the stock in each run with a different shock, thereby providing a comprehensive probability distribution of potential future outcomes.

The mathematical foundation relies on the stochastic differential equation: S(t) = S(t-1) × exp((μ - 0.5σ²)Δt + σ√Δt×Z), where S(t) is the stock price at time t, μ is the annualized drift representing the expected return, σ is the annualized volatility measuring price fluctuations, Δt is the time increment (1/252 for daily trading), and Z is a random variable drawn from a standard normal distribution. The drift (μ) is calculated as the mean of daily log returns multiplied by 252 trading days, while volatility (σ) is the standard deviation of these returns scaled by √252. Each of the 100,000 simulations generates a unique price path by sampling different random shocks (Z values) at each time step, creating a distribution that captures the full spectrum of possible future prices. The average of these 100,000 paths provides the most accurate 3 month forecast.

The model accounts for both the deterministic trend component (drift) and the random fluctuation component (volatility), producing statistical measures like mean and median predicted prices, confidence intervals (5th, 25th, 75th, 95th percentiles), standard deviation, expected returns, probability distributions, and risk-reward ratios. Additionally, the model generates four comprehensive visualizations: Monte Carlo simulation paths showing sample trajectories, confidence interval bands illustrating prediction uncertainty over time, price distribution histograms revealing the most likely outcomes, and historical price charts providing context for the forecasts. Statistics of the historical data (1 year) which is the basis of each stock prediction, is also displayed. 

The final 3-month forecast outputs include the predicted mean price, median price, price range (minimum to maximum), standard deviation indicating volatility, confidence intervals showing the range within which prices are likely to fall with 50% and 90% probability, and the expected percentage return from the current price, providing investors with a complete statistical profile of potential future stock performance.


## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Key
Get a free Gemini API key from Google AI Studio and add it to `config.py`:
```python
GEMINI_API_KEY = 'your_api_key_here'
```

### 3. Run the Application
```bash
python app.py
```

Open your browser to the address displayed 

## Optional Configuration

Edit `config.py` to customize:
- `DEFAULT_NUM_SIMULATIONS = 100000` - Number of Monte Carlo paths
- `DEFAULT_FORECAST_DAYS = 63` - Prediction period (~3 months)
- `DEFAULT_HISTORICAL_DAYS = 252` - Historical data to analyze

## 📊 Features

### Prediction Results (3 month forecast)
- Current price
- Mean & median predictions
- Price range and percentiles
- Expected return percentage

### Historical Data Summary
- Price statistics (mean, min, max, std dev)
- Period performance
- Calculated parameters (drift μ, volatility σ)

### 4 Visualization Plots
1. **Monte Carlo Simulation Paths** - Sample trajectories showing price possibilities
2. **Confidence Intervals** - 50% and 90% confidence bands over time
3. **Price Distribution** - Histogram of predicted final prices
4. **Historical Prices** - Past year's actual stock performance


## 🛠️ Tech Stack

### Backend:

Python
Flask (Web Framework)
Gunicorn (WSGI Server)

### AI & Data:

Google Gemini API (AI-powered ticker lookup)
Yahoo Finance API (Historical stock data)

### Data Processing & Analysis:

NumPy (Numerical computations)
Pandas (Data manipulation)
Matplotlib (Visualization)

### Frontend:

HTML5
CSS3
JavaScript (Vanilla)

### Session Management:

Flask Sessions
Browser SessionStorage

### Libraries:

yfinance (Yahoo Finance data fetching)
google-generativeai (Gemini API integration)
werkzeug (Flask utilities)


