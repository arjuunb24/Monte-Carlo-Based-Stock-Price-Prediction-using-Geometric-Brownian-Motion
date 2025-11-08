import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("⚠️ GEMINI_API_KEY not found. Please set it as an environment variable.")


# Simulation Parameters
DEFAULT_HISTORICAL_DAYS = 252
DEFAULT_NUM_SIMULATIONS = 100000
DEFAULT_FORECAST_DAYS = 63

# Stock Exchanges
EXCHANGES = {
    'NSE': '.NS',
    'BSE': '.BO'
}