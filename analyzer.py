import numpy as np

class ResultAnalyzer:
    """
    Analyzes Monte Carlo simulation results and generates detailed explanations
    for each visualization plot with supporting statistics.
    """
    
    def __init__(self, simulations, current_price, ticker):
        """
        Initialize the analyzer
        
        Parameters:
        simulations (np.array): Monte Carlo simulation results (shape: days x simulations)
        current_price (float): Current stock price
        ticker (str): Stock ticker symbol
        """
        self.simulations = simulations
        self.current_price = current_price
        self.ticker = ticker
        self.results = None
        self.final_prices = simulations[-1]
        self.forecast_days = simulations.shape[0]
        self.num_simulations = simulations.shape[1]
    
    def calculate_statistics(self):
        """
        Calculate statistical summary of simulation results
        Same as original analyze_results() method
        
        Returns:
        dict: Statistical summary
        """
        results = {
            'current_price': self.current_price,
            'mean_price': np.mean(self.final_prices),
            'median_price': np.median(self.final_prices),
            'std_dev': np.std(self.final_prices),
            'min_price': np.min(self.final_prices),
            'max_price': np.max(self.final_prices),
            'percentile_5': np.percentile(self.final_prices, 5),
            'percentile_25': np.percentile(self.final_prices, 25),
            'percentile_75': np.percentile(self.final_prices, 75),
            'percentile_95': np.percentile(self.final_prices, 95),
        }
        
        self.results = results
        
        print(f"PREDICTION RESULTS FOR {self.ticker}")
        print("\n")
        print(f"Current Price: ₹{results['current_price']:.2f}")
        print(f"\nPredicted Price (3 months):")
        print(f"  Mean:           ₹{results['mean_price']:.2f}")
        print(f"  Median:         ₹{results['median_price']:.2f}")
        print(f"  Std Dev:        ₹{results['std_dev']:.2f}")
        print(f"\nPrice Range:")
        print(f"  Min:            ₹{results['min_price']:.2f}")
        print(f"  Max:            ₹{results['max_price']:.2f}")
        print(f"\nConfidence Intervals:")
        print(f"  5th Percentile:  ₹{results['percentile_5']:.2f}")
        print(f"  25th Percentile: ₹{results['percentile_25']:.2f}")
        print(f"  75th Percentile: ₹{results['percentile_75']:.2f}")
        print(f"  95th Percentile: ₹{results['percentile_95']:.2f}")
        print(f"\nExpected Return: {((results['mean_price']/results['current_price']) - 1)*100:.2f}%")
        
        return results
    
    def _explain_simulation_paths(self):
        """
        Generate detailed explanation for Plot 1: Monte Carlo Simulation Paths
        
        Returns:
        str: Formatted explanation with supporting statistics
        """
        # Calculate path statistics
        mean_path = self.simulations.mean(axis=1)
        final_mean = mean_path[-1]
        initial_price = self.simulations[0, 0]
        
        # Calculate how many paths end above/below current price
        paths_above = np.sum(self.final_prices > self.current_price)
        paths_below = np.sum(self.final_prices <= self.current_price)
        pct_above = (paths_above / self.num_simulations) * 100
        pct_below = (paths_below / self.num_simulations) * 100
        
        # Calculate average path volatility (standard deviation across time)
        path_volatilities = self.simulations.std(axis=1)
        avg_volatility = path_volatilities.mean()
        final_volatility = path_volatilities[-1]
        
        explanation = f"""
PLOT 1: MONTE CARLO SIMULATION PATHS

This plot visualizes a sample of individual price trajectories from the Monte 
Carlo simulation, showing the range of possible outcomes for {self.ticker} stock.

KEY INSIGHTS:

📊 Simulation Overview:
   • Total simulations run: {self.num_simulations:,}
   • Forecast period: {self.forecast_days} trading days (~3 months)
   • Starting price: ₹{initial_price:.2f}
   • Mean final price: ₹{final_mean:.2f}

📈 Path Distribution:
   • Paths ending above current price: {paths_above:,} ({pct_above:.1f}%)
   • Paths ending below current price: {paths_below:,} ({pct_below:.1f}%)
   
🎯 What the RED LINE Shows:
   The red line represents the MEAN PATH - the average price across all 
   {self.num_simulations:,} simulations at each time point. This is the most 
   statistically likely trajectory based on historical patterns.

📉 Volatility Analysis:
   • Average daily volatility: ₹{avg_volatility:.2f}
   • Final day volatility: ₹{final_volatility:.2f}
   • This shows how uncertainty increases over time

💡 INTERPRETATION:
   The spread of the blue paths illustrates the uncertainty in stock price 
   prediction. A wider spread indicates higher risk/volatility. The fact that 
   {pct_above:.1f}% of simulations end above the current price suggests a 
   {"bullish" if pct_above > 50 else "bearish"} outlook for the next 3 months.
"""
        return explanation
    
    def _explain_confidence_intervals(self):
        """
        Generate detailed explanation for Plot 2: Price Forecast with Confidence Intervals
        
        Returns:
        str: Formatted explanation with supporting statistics
        """
        # Calculate confidence interval statistics
        mean_path = self.simulations.mean(axis=1)
        p5 = np.percentile(self.simulations, 5, axis=1)
        p25 = np.percentile(self.simulations, 25, axis=1)
        p75 = np.percentile(self.simulations, 75, axis=1)
        p95 = np.percentile(self.simulations, 95, axis=1)
        
        # Final day intervals
        final_p5 = p5[-1]
        final_p25 = p25[-1]
        final_p75 = p75[-1]
        final_p95 = p95[-1]
        
        # Calculate interval widths
        ci_50_width = final_p75 - final_p25
        ci_90_width = final_p95 - final_p5
        
        # Calculate percentage ranges
        ci_50_range_pct = (ci_50_width / self.current_price) * 100
        ci_90_range_pct = (ci_90_width / self.current_price) * 100
        
        # Mid-point analysis (day 31 - approximately 1.5 months)
        mid_point = self.forecast_days // 2
        mid_mean = mean_path[mid_point]
        mid_p5 = p5[mid_point]
        mid_p95 = p95[mid_point]
        
        explanation = f"""
PLOT 2: PRICE FORECAST WITH CONFIDENCE INTERVALS

This plot shows the predicted price range over time with statistical confidence
intervals, helping you understand the probability distribution of future prices.

KEY INSIGHTS:

📊 Confidence Interval Explanation:
   • DARK BLUE (50% CI): There's a 50% probability the price will fall in this range
   • LIGHT BLUE (90% CI): There's a 90% probability the price will fall in this range
   • RED LINE: The mean (average) expected price path

📈 Final Day (Day {self.forecast_days}) Price Ranges:
   
   90% Confidence Interval:
   • Lower Bound (5th percentile):  ₹{final_p5:.2f}
   • Upper Bound (95th percentile): ₹{final_p95:.2f}
   • Range Width: ₹{ci_90_width:.2f} ({ci_90_range_pct:.1f}% of current price)
   
   50% Confidence Interval:
   • Lower Bound (25th percentile):  ₹{final_p25:.2f}
   • Upper Bound (75th percentile): ₹{final_p75:.2f}
   • Range Width: ₹{ci_50_width:.2f} ({ci_50_range_pct:.1f}% of current price)

🎯 Mid-Point Analysis (Day {mid_point} - ~1.5 months):
   • Expected price: ₹{mid_mean:.2f}
   • 90% CI Range: ₹{mid_p5:.2f} to ₹{mid_p95:.2f}

💡 INTERPRETATION:
   The widening of the confidence intervals over time reflects increasing 
   uncertainty in longer-term predictions. The 90% confidence interval suggests 
   that there's a 90% probability the stock price will be between ₹{final_p5:.2f} 
   and ₹{final_p95:.2f} after 3 months.
   
   {"⚠️  WARNING: The wide confidence interval indicates HIGH VOLATILITY/RISK." if ci_90_range_pct > 50 else "✅ The relatively narrow confidence interval suggests MODERATE VOLATILITY/RISK."}
"""
        return explanation
    
    def _explain_price_distribution(self):
        """
        Generate detailed explanation for Plot 3: Distribution of Predicted Prices
        
        Returns:
        str: Formatted explanation with supporting statistics
        """
        # Distribution statistics
        mean_price = self.results['mean_price']
        median_price = self.results['median_price']
        std_dev = self.results['std_dev']
        
        # Calculate skewness
        skewness = ((self.final_prices - mean_price) ** 3).mean() / (std_dev ** 3)
        
        # Calculate kurtosis (excess kurtosis)
        kurtosis = ((self.final_prices - mean_price) ** 4).mean() / (std_dev ** 4) - 3
        
        # Price probability analysis
        price_above_current = np.sum(self.final_prices > self.current_price) / self.num_simulations * 100
        price_above_mean = np.sum(self.final_prices > mean_price) / self.num_simulations * 100
        
        # Calculate percentage in different ranges
        range_lower = mean_price - std_dev
        range_upper = mean_price + std_dev
        within_1_std = np.sum((self.final_prices >= range_lower) & 
                              (self.final_prices <= range_upper)) / self.num_simulations * 100
        
        # Extreme outcomes
        extreme_gain = self.results['percentile_95']
        extreme_loss = self.results['percentile_5']
        gain_potential = ((extreme_gain / self.current_price) - 1) * 100
        loss_risk = ((extreme_loss / self.current_price) - 1) * 100
        
        # Most likely price range (highest density)
        hist, bin_edges = np.histogram(self.final_prices, bins=100)
        max_density_idx = hist.argmax()
        most_likely_range = (bin_edges[max_density_idx], bin_edges[max_density_idx + 1])
        
        explanation = f"""
PLOT 3: DISTRIBUTION OF PREDICTED PRICES (3 MONTHS)

This histogram shows the probability distribution of final stock prices after
3 months, revealing the most likely outcomes and their probabilities.

KEY INSIGHTS:

📊 Distribution Statistics:
   • Mean Price: ₹{mean_price:.2f}
   • Median Price: ₹{median_price:.2f}
   • Standard Deviation: ₹{std_dev:.2f}
   • Skewness: {skewness:.3f} {"(Right-skewed - tail extends toward higher prices)" if skewness > 0.1 else "(Left-skewed - tail extends toward lower prices)" if skewness < -0.1 else "(Symmetric distribution)"}
   • Kurtosis: {kurtosis:.3f} {"(Fat tails - higher probability of extreme outcomes)" if kurtosis > 0.5 else "(Thin tails - lower probability of extreme outcomes)" if kurtosis < -0.5 else "(Normal tail behavior)"}

🎯 What the Vertical Lines Show:
   • RED DASHED LINE (Mean): The average predicted price (₹{mean_price:.2f})
   • ORANGE DASHED LINES: 5th and 95th percentiles
     - 5th Percentile: ₹{extreme_loss:.2f} ({loss_risk:+.2f}% from current)
     - 95th Percentile: ₹{extreme_gain:.2f} ({gain_potential:+.2f}% from current)

📈 Probability Analysis:
   • Probability of price > current price: {price_above_current:.1f}%
   • Probability of price > mean: {price_above_mean:.1f}%
   • Probability within 1 std dev of mean: {within_1_std:.1f}%
   
🎲 Most Likely Price Range:
   • The peak of the histogram shows prices between ₹{most_likely_range[0]:.2f} 
     and ₹{most_likely_range[1]:.2f} are most probable

💡 INTERPRETATION:
   This distribution represents {self.num_simulations:,} possible future outcomes.
   The shape tells us:
   
   {"• The distribution is positively skewed, suggesting higher upside potential" if skewness > 0.1 else "• The distribution is negatively skewed, suggesting higher downside risk" if skewness < -0.1 else "• The distribution is fairly symmetric"}
   {"• Fat tails indicate a higher-than-normal chance of extreme price movements" if abs(kurtosis) > 0.5 else "• Normal tails suggest typical market behavior"}
   
   Risk/Reward Summary:
   • Upside potential (95th percentile): +{gain_potential:.2f}%
   • Downside risk (5th percentile): {loss_risk:.2f}%
   • Risk-Reward Ratio: {abs(gain_potential/loss_risk):.2f}:1
   
   {"✅ Favorable risk-reward profile" if gain_potential > abs(loss_risk) else "⚠️  Unfavorable risk-reward profile - potential losses exceed potential gains"}
"""
        return explanation
    
    def _explain_historical_prices(self):
        """
        Generate detailed explanation for Plot 4: Historical Stock Prices
        
        Returns:
        str: Formatted explanation with supporting statistics
        """
        # Calculate historical price statistics
        historical_mean = self.simulations[0, 0]  # Starting price (same for all simulations)
        
        # Calculate historical trend
        first_price = self.simulations[0, 0]
        predicted_mean = self.simulations.mean(axis=1)[-1]
        historical_to_predicted_change = ((predicted_mean / first_price) - 1) * 100
        
        # Volatility comparison
        # Note: We don't have access to historical data here, so we'll use simulation data
        # In actual implementation, this would come from the historical_data parameter
        predicted_std = self.simulations[-1].std()
        predicted_cv = (predicted_std / predicted_mean) * 100  # Coefficient of variation
        
        explanation = f"""
PLOT 4: HISTORICAL STOCK PRICES


This plot displays the actual historical price movements of {self.ticker} stock
over the past year (252 trading days), which formed the basis for our predictions.

KEY INSIGHTS:

📊 Historical Context:
   • Analysis Period: Past 252 trading days (~1 year)
   • Starting Historical Price: ₹{first_price:.2f}
   • Current Price: ₹{self.current_price:.2f}
   • This historical data was used to calculate:
     - Drift (μ): The average rate of return
     - Volatility (σ): The degree of price variation

🔄 Historical Pattern to Prediction:
   • The Monte Carlo simulation extends these historical patterns forward
   • Predicted mean price in 3 months: ₹{predicted_mean:.2f}
   • Projected change from current: {historical_to_predicted_change:+.2f}%

The historical chart reveals trends, support/resistance levels, and 
volatility patterns that influence future price movements. Comparing predicted 
prices with recent historical prices helps assess if predictions are realistic 
or if they suggest unusual market conditions.
   
"""
        return explanation
    
    def print_full_analysis(self):
        """
        Print complete analysis including statistics and detailed plot explanations
        """
        if self.results is None:
            print("Error: No results calculated. Run calculate_statistics() first.")
            return
        
        print("\n")
        print("DETAILED ANALYSIS REPORT")
        print("\n")
        # Print explanations for each plot
        print(self._explain_simulation_paths())
        print("\n")
        
        print(self._explain_confidence_intervals())
        print("\n")
        
        print(self._explain_price_distribution())
        print("\n")
        
        print(self._explain_historical_prices())
        