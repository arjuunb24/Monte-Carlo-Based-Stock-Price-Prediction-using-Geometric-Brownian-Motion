from ticker_finder import TickerFinder
from data_fetcher import DataFetcher
from simulator import MonteCarloSimulator
from analyzer import ResultAnalyzer
from visualizer import Visualizer
from config import DEFAULT_NUM_SIMULATIONS, DEFAULT_FORECAST_DAYS

def main():
    """Main orchestration function for stock price prediction"""
    
    print("\n" + "="*70)
    print(" " * 15 + "STOCK PRICE PREDICTION SYSTEM")
    print(" " * 10 + "Monte Carlo Simulation with Gemini AI Integration")
    print("="*70)
    
    try:
        # Step 1: Get company name from user
        print("\n📝 Step 1: Enter Company Information")
        print("-" * 70)
        company_name = input("Enter company name (e.g., Reliance, TCS, Infosys): ").strip()
        
        if not company_name:
            print("❌ Error: Company name cannot be empty")
            return
        
        # Step 2: Find ticker using Gemini
        print("\n" + "="*70)
        print("🔍 Step 2: Finding Ticker Symbol")
        print("="*70)
        
        finder = TickerFinder()
        ticker = finder.find_ticker(company_name)
        
        if not ticker:
            print("\n❌ Unable to proceed without a valid ticker symbol.")
            print("   Please try again with a different company name.")
            return
        
        # Step 3: Fetch data and calculate parameters
        print("\n" + "="*70)
        print("📊 Step 3: Fetching Historical Data")
        print("="*70)
        
        fetcher = DataFetcher(ticker)
        fetcher.fetch_data()
        fetcher.print_summary()
        fetcher.calculate_parameters()
        
        # Step 4: Run Monte Carlo simulation
        print("\n" + "="*70)
        print("🎲 Step 4: Running Monte Carlo Simulation")
        print("="*70)
        
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
        print("\n" + "="*70)
        print("📈 Step 5: Analyzing Results")
        print("="*70)
        
        analyzer = ResultAnalyzer(
            simulations=simulations,
            current_price=fetcher.get_current_price(),
            ticker=ticker
        )
        
        analyzer.calculate_statistics()
        analyzer.print_full_analysis()
        
        # Step 6: Generate visualizations
        print("\n" + "="*70)
        print("📊 Step 6: Generating Visualizations")
        print("="*70)
        print("\n⚠️  Note: Plots will open one at a time. Close each plot to see the next one.")
        
        input("\nPress Enter to start generating plots...")
        
        visualizer = Visualizer(
            simulations=simulations,
            historical_data=fetcher.data,
            ticker=ticker
        )
        
        visualizer.generate_all_plots(num_paths=200)
        
        # Final message
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("\n💡 Remember: This is a statistical prediction based on historical")
        print("   patterns. Always consult with financial advisors before investing.")
        print("\n" + "="*70)
        
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