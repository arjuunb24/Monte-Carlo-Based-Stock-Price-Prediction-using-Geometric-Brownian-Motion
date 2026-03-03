import yfinance as yf
import requests
import numpy as np
from datetime import datetime, timedelta
from config import DEFAULT_HISTORICAL_DAYS

def _get_yf_session():
    """
    Return a requests Session with browser-like headers to bypass
    Yahoo Finance's bot-blocking on cloud provider IPs (Render, AWS, etc.).
    """
    session = requests.Session()
    session.headers.update({
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/122.0.0.0 Safari/537.36'
        ),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    })
    return session

class DataFetcher:
    """
    Fetches historical stock data and calculates statistical parameters
    for Monte Carlo simulation (drift and volatility).
    """
    
    def __init__(self, ticker, historical_days=DEFAULT_HISTORICAL_DAYS):
        """
        Initialize the data fetcher
        
        Parameters:
        ticker (str): Stock ticker symbol (e.g., 'RELIANCE.NS' for NSE stocks)
        historical_days (int): Number of historical days to fetch (default: 252 trading days)
        """
        self.ticker = ticker
        self.historical_days = historical_days
        self.data = None
        self.returns = None
        self.mu = None  # drift
        self.sigma = None  # volatility
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.historical_days + 120)  # Extra buffer

        print(f"\n📊 Fetching data for {self.ticker}...")
        session = _get_yf_session()
        stock = yf.Ticker(self.ticker, session=session)
        self.data = stock.history(start=start_date, end=end_date)
        
        if self.data.empty:
            raise ValueError(f"No data found for {self.ticker}. Check ticker symbol.")
        
        # Keep only the last 'historical_days' rows
        self.data = self.data.tail(self.historical_days)
        print(f"✅ Fetched {len(self.data)} days of historical data")
        
        return self.data
        
    def calculate_parameters(self):
        """Calculate drift (mu) and volatility (sigma) from historical data"""
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        # Calculate daily returns
        self.returns = np.log(self.data['Close'] / self.data['Close'].shift(1)).dropna()
        
        # Calculate annualized drift and volatility
        self.mu = self.returns.mean() * 252  # Annualized drift
        self.sigma = self.returns.std() * np.sqrt(252)  # Annualized volatility
        
        print(f"\n📈 Calculated Parameters:")
        print(f"   Annualized Drift (μ): {self.mu:.4f} ({self.mu*100:.2f}% per year)")
        print(f"   Annualized Volatility (σ): {self.sigma:.4f} ({self.sigma*100:.2f}% per year)")
        
        return self.mu, self.sigma
        
    def get_current_price(self):
        """
        Get the most recent closing price
        
        Returns:
        float: Current stock price
        """
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        return self.data['Close'].iloc[-1]
    
    def get_price_history(self):
        """
        Get the complete price history
        
        Returns:
        pandas.Series: Historical closing prices
        """
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        return self.data['Close']
    
    def get_summary_statistics(self):
        """
        Get summary statistics of historical data
        
        Returns:
        dict: Summary statistics
        """
        if self.data is None:
            raise ValueError("No data available. Run fetch_data() first.")
        
        prices = self.data['Close']
        
        summary = {
            'current_price': prices.iloc[-1],
            'historical_mean': prices.mean(),
            'historical_std': prices.std(),
            'historical_min': prices.min(),
            'historical_max': prices.max(),
            'start_date': self.data.index[0].strftime('%Y-%m-%d'),
            'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
            'total_days': len(self.data),
            'price_change_pct': ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100
        }
        
        return summary
    
    def print_summary(self):
        """Print a summary of the fetched data"""
        if self.data is None:
            print("❌ No data available. Run fetch_data() first.")
            return
        
        summary = self.get_summary_statistics()
        
        print("\n")
        print(f"HISTORICAL DATA SUMMARY FOR {self.ticker}")
        print("\n")
        print(f"Period: {summary['start_date']} to {summary['end_date']}")
        print(f"Trading Days: {summary['total_days']}")
        print(f"\nPrice Statistics:")
        print(f"  Current Price: ₹{summary['current_price']:.2f}")
        print(f"  Mean Price: ₹{summary['historical_mean']:.2f}")
        print(f"  Std Deviation: ₹{summary['historical_std']:.2f}")
        print(f"  Min Price: ₹{summary['historical_min']:.2f}")
        print(f"  Max Price: ₹{summary['historical_max']:.2f}")
        print(f"\nPeriod Performance: {summary['price_change_pct']:+.2f}%")