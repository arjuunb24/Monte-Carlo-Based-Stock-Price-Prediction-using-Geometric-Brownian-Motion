import numpy as np
from config import DEFAULT_NUM_SIMULATIONS, DEFAULT_FORECAST_DAYS

class MonteCarloSimulator:
    """
    Performs Monte Carlo simulation using Geometric Brownian Motion (GBM)
    to forecast future stock prices.
    """
    
    def __init__(self, current_price, mu, sigma):
        """
        Initialize the Monte Carlo simulator
        
        Parameters:
        current_price (float): Current stock price
        mu (float): Annualized drift (expected return)
        sigma (float): Annualized volatility (standard deviation)
        """
        self.current_price = current_price
        self.mu = mu
        self.sigma = sigma
        
    def run_simulation(self, num_simulations=DEFAULT_NUM_SIMULATIONS, 
                       forecast_days=DEFAULT_FORECAST_DAYS):
        """
        Run Monte Carlo simulation using Geometric Brownian Motion
        
        Parameters:
        num_simulations (int): Number of simulation paths (default: 100000)
        forecast_days (int): Number of days to forecast (default: 63 for ~3 months)
        
        Returns:
        np.array: Simulated price paths (shape: forecast_days x num_simulations)
        """
        print(f"\n🎲 Running Monte Carlo Simulation...")
        print(f"   Simulations: {num_simulations:,}")
        print(f"   Forecast Period: {forecast_days} days (~{forecast_days/21:.1f} months)")
        print(f"   Starting Price: ₹{self.current_price:.2f}")
        
        # Time increment (daily)
        dt = 1/252  # One trading day
        
        # Initialize array for simulations
        simulations = np.zeros((forecast_days, num_simulations))
        simulations[0] = self.current_price
        
        # Generate random walks using vectorized operations for efficiency
        print(f"   Generating {num_simulations:,} price paths...", end=" ")
        
        for t in range(1, forecast_days):
            # Generate random shocks (standard normal distribution)
            Z = np.random.standard_normal(num_simulations)
            
            # Geometric Brownian Motion formula:
            # S(t) = S(t-1) * exp((μ - 0.5*σ²)Δt + σ*√Δt*Z)
            simulations[t] = simulations[t-1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + 
                self.sigma * np.sqrt(dt) * Z
            )
        print("✅ Done!")
        
        # Calculate some quick statistics
        final_mean = simulations[-1].mean()
        final_std = simulations[-1].std()
        
        print(f"\n📊 Simulation Results Preview:")
        print(f"   Final Mean Price: ₹{final_mean:.2f}")
        print(f"   Final Std Dev: ₹{final_std:.2f}")
        print(f"   Expected Return: {((final_mean/self.current_price) - 1)*100:+.2f}%")
        
        return simulations
    
    def get_simulation_summary(self, simulations):
        """
        Get a quick summary of simulation results
        
        Parameters:
        simulations (np.array): Simulation results
        
        Returns:
        dict: Summary statistics
        """
        final_prices = simulations[-1]
        
        summary = {
            'mean': final_prices.mean(),
            'median': np.median(final_prices),
            'std': final_prices.std(),
            'min': final_prices.min(),
            'max': final_prices.max(),
            'num_simulations': simulations.shape[1],
            'forecast_days': simulations.shape[0]
        }
        
        return summary