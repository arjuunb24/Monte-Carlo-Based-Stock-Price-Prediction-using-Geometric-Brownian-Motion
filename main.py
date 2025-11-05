from ticker_finder import TickerFinder
from data_fetcher import DataFetcher
from simulator import MonteCarloSimulator
from analyzer import ResultAnalyzer
from visualizer import Visualizer
from config import DEFAULT_NUM_SIMULATIONS, DEFAULT_FORECAST_DAYS
import matplotlib.pyplot as plt
import os

def main():
    """Main orchestration function for stock price prediction"""
    
    print("\n")
    print("STOCK PRICE PREDICTION SYSTEM")
    print("Monte Carlo Simulation with Geometric Brownian Motion")
    


    # Delete old plots if they exist
    for i in range(1, 5):
        plot_file = f'plot{i}_*.png'
        import glob
        for file in glob.glob(plot_file):
            try:
                os.remove(file)
            except:
                pass
    
    try:
        # Step 1: Get company name from user
        print("\n")
        company_name = input("Enter company name (e.g., Reliance, TCS, Infosys): ").strip()
        
        if not company_name:
            print("❌ Error: Company name cannot be empty")
            return
        
        # Step 2: Find ticker using Gemini
        print("\n")
        
        
        finder = TickerFinder()
        ticker = finder.find_ticker(company_name)
        
        if not ticker:
            print("\n❌ Unable to proceed without a valid ticker symbol.")
            print("   Please try again with a different company name.")
            return
        
        # Step 3: Fetch data and calculate parameters
        print("\n")
        
        fetcher = DataFetcher(ticker)
        fetcher.fetch_data()
        fetcher.print_summary()
        fetcher.calculate_parameters()
        
        # Step 4: Run Monte Carlo simulation
        
        
        
        simulator = MonteCarloSimulator(
            current_price=fetcher.get_current_price(),
            mu=fetcher.mu,
            sigma=fetcher.sigma
        )
        
        simulations = simulator.run_simulation(
            num_simulations=DEFAULT_NUM_SIMULATIONS,
            forecast_days=DEFAULT_FORECAST_DAYS
        )
        
        # Step 5: Analyze results
        print("\n")
        
        analyzer = ResultAnalyzer(
            simulations=simulations,
            current_price=fetcher.get_current_price(),
            ticker=ticker
        )
        
        analyzer.calculate_statistics()
        analyzer.print_full_analysis()
        
        # Step 6: Generate visualizations
        print("\n")
        
        visualizer = Visualizer(
            simulations=simulations,
            historical_data=fetcher.data,
            ticker=ticker
        )
        
        visualizer.generate_all_plots(num_paths=200)
        
        # Final message
        print("\n")
        print("✅ ANALYSIS COMPLETE!")
        print("\n")
        plt.show(block=False)

    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        print("Exiting...")
    except Exception as e:
        print(f"\n\n❌ An error occurred: {str(e)}")
        print("Please check your inputs and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()