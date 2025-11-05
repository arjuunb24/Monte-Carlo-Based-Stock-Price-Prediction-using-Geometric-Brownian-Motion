import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    """
    Generates individual visualization plots for Monte Carlo simulation results.
    Each plot is displayed separately for better focus and analysis.
    """
    
    def __init__(self, simulations, historical_data, ticker):
        """
        Initialize the visualizer
        
        Parameters:
        simulations (np.array): Monte Carlo simulation results
        historical_data (pandas.DataFrame): Historical stock data
        ticker (str): Stock ticker symbol
        """
        self.simulations = simulations
        self.historical_data = historical_data
        self.ticker = ticker
    
    def plot_simulation_paths(self, num_paths=100):
        """Generate Plot 1: Sample simulation paths"""
        print("\n📈 Generating Plot 1: Monte Carlo Simulation Paths...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot sample paths
        sample_indices = np.random.choice(self.simulations.shape[1], num_paths, replace=False)
        for idx in sample_indices:
            ax.plot(self.simulations[:, idx], alpha=0.1, color='blue', linewidth=0.5)
        
        # Plot mean path
        ax.plot(self.simulations.mean(axis=1), color='red', linewidth=2.5, label='Mean Path', zorder=5)
        
        ax.set_title(f'Monte Carlo Simulation Paths for {self.ticker}\n(Sample of {num_paths} paths)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('plot1_simulation_paths.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Plot 1 complete!")
    
    def plot_confidence_intervals(self):
        """Generate Plot 2: Confidence intervals"""
        print("\n📊 Generating Plot 2: Price Forecast with Confidence Intervals...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        days = np.arange(self.simulations.shape[0])
        mean_path = self.simulations.mean(axis=1)
        p5 = np.percentile(self.simulations, 5, axis=1)
        p25 = np.percentile(self.simulations, 25, axis=1)
        p75 = np.percentile(self.simulations, 75, axis=1)
        p95 = np.percentile(self.simulations, 95, axis=1)
        
        # Plot confidence intervals
        ax.fill_between(days, p5, p95, alpha=0.2, color='blue', label='90% Confidence Interval')
        ax.fill_between(days, p25, p75, alpha=0.35, color='blue', label='50% Confidence Interval')
        ax.plot(mean_path, color='red', linewidth=2.5, label='Mean Forecast', zorder=5)
        
        ax.set_title(f'Price Forecast with Confidence Intervals for {self.ticker}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('plot2_confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Plot 2 complete!")
    
    def plot_price_distribution(self):
        """Generate Plot 3: Final price distribution"""
        print("\n📉 Generating Plot 3: Distribution of Predicted Prices...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        final_prices = self.simulations[-1]
        
        # Create histogram
        n, bins, patches = ax.hist(final_prices, bins=100, color='skyblue', 
                                    edgecolor='black', alpha=0.7, density=False)
        
        # Add vertical lines for key statistics
        mean_price = final_prices.mean()
        p5 = np.percentile(final_prices, 5)
        p95 = np.percentile(final_prices, 95)
        
        ax.axvline(mean_price, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: ₹{mean_price:.2f}')
        ax.axvline(p5, color='orange', linestyle='--', linewidth=2, 
                   label=f'5th Percentile: ₹{p5:.2f}')
        ax.axvline(p95, color='orange', linestyle='--', linewidth=2, 
                   label=f'95th Percentile: ₹{p95:.2f}')
        
        ax.set_title(f'Distribution of Predicted Prices (3 months) for {self.ticker}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Price (₹)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        plt.savefig('plot3_price_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Plot 3 complete!")
    
    def plot_historical_prices(self):
        """Generate Plot 4: Historical prices"""
        print("\n📅 Generating Plot 4: Historical Stock Prices...")
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(self.historical_data.index, self.historical_data['Close'], 
                color='green', linewidth=2)
        
        # Add a horizontal line for current price
        current_price = self.historical_data['Close'].iloc[-1]
        ax.axhline(current_price, color='red', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'Current Price: ₹{current_price:.2f}')
        
        ax.set_title(f'Historical Stock Prices for {self.ticker}\n(Past {len(self.historical_data)} Trading Days)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (₹)', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('plot4_historical_prices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Plot 4 complete!")
    
    def generate_all_plots(self, num_paths=200):
        """Generate all 4 plots individually"""
        print("GENERATING VISUALIZATIONS")
        
        self.plot_simulation_paths(num_paths)
        self.plot_confidence_intervals()
        self.plot_price_distribution()
        self.plot_historical_prices()
        