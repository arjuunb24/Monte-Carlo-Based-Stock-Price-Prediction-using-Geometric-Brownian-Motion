"""
Flask Web Application for Stock Price Predictor
Connects the backend prediction system with the web frontend
"""

from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import yfinance as yf

from ticker_finder import TickerFinder
from data_fetcher import DataFetcher
from simulator import MonteCarloSimulator
from analyzer import ResultAnalyzer
from visualizer import Visualizer
from config import DEFAULT_NUM_SIMULATIONS, DEFAULT_FORECAST_DAYS

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this!

# Configure upload folder for plots
UPLOAD_FOLDER = 'static/plots'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def run_prediction(company_name):
    """
    Run the complete prediction pipeline
    
    Returns:
    dict: Contains all results, statistics, and plot paths
    """
    try:
        # Step 1: Find ticker
        finder = TickerFinder()
        ticker = finder.find_ticker(company_name)
        
        if not ticker:
            return {
                'success': False,
                'error': f"Could not find a valid ticker for '{company_name}'. Please ensure the company is listed on NSE or BSE."
            }
        
        # Step 2: Fetch data
        fetcher = DataFetcher(ticker)
        fetcher.fetch_data()
        fetcher.calculate_parameters()
        
        # Step 3: Run simulation
        simulator = MonteCarloSimulator(
            current_price=fetcher.get_current_price(),
            mu=fetcher.mu,
            sigma=fetcher.sigma
        )
        
        simulations = simulator.run_simulation(
            num_simulations=DEFAULT_NUM_SIMULATIONS,
            forecast_days=DEFAULT_FORECAST_DAYS
        )
        
        # Step 4: Analyze results
        analyzer = ResultAnalyzer(
            simulations=simulations,
            current_price=fetcher.get_current_price(),
            ticker=ticker
        )
        
        results = analyzer.calculate_statistics()
        
        # Step 5: Generate plots and save them
        visualizer = Visualizer(
            simulations=simulations,
            historical_data=fetcher.data,
            ticker=ticker
        )
        
        # Generate plots with custom save paths
        plot_paths = generate_plots_for_web(visualizer, ticker)
        
        # Step 6: Get plot explanations
        explanations = {
            'plot1': analyzer._explain_simulation_paths(),
            'plot2': analyzer._explain_confidence_intervals(),
            'plot3': analyzer._explain_price_distribution(),
            'plot4': analyzer._explain_historical_prices()
        }
        
        # Get summary statistics
        summary = fetcher.get_summary_statistics()
        
        company_info = yf.Ticker(ticker).info
        company_full_name = company_info.get('longName', company_info.get('shortName', ticker))

        return {
            'success': True,
            'ticker': ticker,
            'company_full_name': company_full_name,
            'results': results,
            'summary': summary,
            'plots': plot_paths,
            'explanations': explanations,
            'mu': fetcher.mu,
            'sigma': fetcher.sigma
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }


def generate_plots_for_web(visualizer, ticker):
    """
    Generate all plots and return their file paths
    
    Returns:
    dict: Paths to all generated plots
    """
    import numpy as np
    
    plot_paths = {}
    timestamp = str(int(os.time.time())) if hasattr(os, 'time') else '1'
    
    # Plot 1: Simulation Paths
    fig, ax = plt.subplots(figsize=(12, 7))
    sample_indices = np.random.choice(visualizer.simulations.shape[1], 200, replace=False)
    for idx in sample_indices:
        ax.plot(visualizer.simulations[:, idx], alpha=0.1, color='blue', linewidth=0.5)
    ax.plot(visualizer.simulations.mean(axis=1), color='red', linewidth=2.5, label='Mean Path', zorder=5)
    ax.set_title(f'Monte Carlo Simulation Paths for {ticker}\n(Sample of 200 paths)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot1_path = f'plots/plot1_{timestamp}.png'
    plt.savefig(os.path.join('static', plot1_path), dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['plot1'] = plot1_path
    
    # Plot 2: Confidence Intervals
    fig, ax = plt.subplots(figsize=(12, 7))
    days = np.arange(visualizer.simulations.shape[0])
    mean_path = visualizer.simulations.mean(axis=1)
    p5 = np.percentile(visualizer.simulations, 5, axis=1)
    p25 = np.percentile(visualizer.simulations, 25, axis=1)
    p75 = np.percentile(visualizer.simulations, 75, axis=1)
    p95 = np.percentile(visualizer.simulations, 95, axis=1)
    ax.fill_between(days, p5, p95, alpha=0.2, color='blue', label='90% Confidence Interval')
    ax.fill_between(days, p25, p75, alpha=0.35, color='blue', label='50% Confidence Interval')
    ax.plot(mean_path, color='red', linewidth=2.5, label='Mean Forecast', zorder=5)
    ax.set_title(f'Price Forecast with Confidence Intervals for {ticker}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plot2_path = f'plots/plot2_{timestamp}.png'
    plt.savefig(os.path.join('static', plot2_path), dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['plot2'] = plot2_path
    
    # Plot 3: Price Distribution
    fig, ax = plt.subplots(figsize=(12, 7))
    final_prices = visualizer.simulations[-1]
    n, bins, patches = ax.hist(final_prices, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    mean_price = final_prices.mean()
    p5 = np.percentile(final_prices, 5)
    p95 = np.percentile(final_prices, 95)
    ax.axvline(mean_price, color='red', linestyle='--', linewidth=2.5, label=f'Mean: ₹{mean_price:.2f}')
    ax.axvline(p5, color='orange', linestyle='--', linewidth=2, label=f'5th Percentile: ₹{p5:.2f}')
    ax.axvline(p95, color='orange', linestyle='--', linewidth=2, label=f'95th Percentile: ₹{p95:.2f}')
    ax.set_title(f'Distribution of Predicted Prices (3 months) for {ticker}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Price (₹)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plot3_path = f'plots/plot3_{timestamp}.png'
    plt.savefig(os.path.join('static', plot3_path), dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['plot3'] = plot3_path
    
    # Plot 4: Historical Prices
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(visualizer.historical_data.index, visualizer.historical_data['Close'], color='green', linewidth=2)
    current_price = visualizer.historical_data['Close'].iloc[-1]
    ax.axhline(current_price, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Current Price: ₹{current_price:.2f}')
    ax.set_title(f'Historical Stock Prices for {ticker}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot4_path = f'plots/plot4_{timestamp}.png'
    plt.savefig(os.path.join('static', plot4_path), dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['plot4'] = plot4_path
    
    return plot_paths


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.get_json()
        company_name = data.get('company_name', '').strip()
        
        if not company_name:
            return jsonify({
                'success': False,
                'error': 'Please enter a company name'
            })
        
        # Run prediction
        result = run_prediction(company_name)
        
        # Store result in session for plot detail pages
        if result['success']:
            session['last_result'] = result
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })


@app.route('/plot/<plot_id>')
def plot_detail(plot_id):
    """Render individual plot detail page"""
    result = session.get('last_result')
    
    if not result or not result.get('success'):
        return render_template('error.html', 
                             error='No prediction data available. Please run a prediction first.')
    
    plot_titles = {
        'plot1': 'Monte Carlo Simulation Paths',
        'plot2': 'Price Forecast with Confidence Intervals',
        'plot3': 'Distribution of Predicted Prices (3 Months)',
        'plot4': 'Historical Stock Prices'
    }
    
    if plot_id not in plot_titles:
        return render_template('error.html', error='Invalid plot ID')
    
    return render_template('plot_detail.html',
                         plot_id=plot_id,
                         plot_title=plot_titles[plot_id],
                         plot_path=result['plots'][plot_id],
                         explanation=result['explanations'][plot_id],
                         ticker=result['ticker'])


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
