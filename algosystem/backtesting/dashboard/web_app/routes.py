from flask import render_template, request, jsonify, redirect, url_for, flash, send_from_directory
import os
import json
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from algosystem.backtesting.dashboard.web_app.app import app, DEFAULT_CONFIG_PATH, USER_CONFIG_PATH, uploaded_data, engine, dashboard_path
from algosystem.backtesting.engine import Engine


AVAILABLE_METRICS = [
    # Basic performance metrics
    {"id": "total_return", "type": "Percentage", "title": "Total Return", "value_key": "total_return", 
     "description": "Total return over the full period", "category": "performance"},
    {"id": "annual_return", "type": "Percentage", "title": "Annualized Return", "value_key": "annualized_return", 
     "description": "Annualized return of the strategy", "category": "performance"},
    {"id": "volatility", "type": "Percentage", "title": "Volatility", "value_key": "annualized_volatility", 
     "description": "Annualized volatility of the strategy", "category": "performance"},
     
    # Risk metrics
    {"id": "max_drawdown", "type": "Percentage", "title": "Max Drawdown", "value_key": "max_drawdown", 
     "description": "Maximum drawdown of the strategy", "category": "risk"},
    {"id": "var_95", "type": "Percentage", "title": "Value at Risk (95%)", "value_key": "var_95", 
     "description": "95% Value at Risk", "category": "risk"},
    {"id": "cvar_95", "type": "Percentage", "title": "Conditional VaR (95%)", "value_key": "cvar_95", 
     "description": "95% Conditional Value at Risk (Expected Shortfall)", "category": "risk"},
    {"id": "skewness", "type": "Value", "title": "Skewness", "value_key": "skewness", 
     "description": "Skewness of returns distribution", "category": "risk"},
     
    # Ratio metrics
    {"id": "sharpe_ratio", "type": "Value", "title": "Sharpe Ratio", "value_key": "sharpe_ratio", 
     "description": "Sharpe ratio of the strategy", "category": "ratio"},
    {"id": "sortino_ratio", "type": "Value", "title": "Sortino Ratio", "value_key": "sortino_ratio", 
     "description": "Sortino ratio of the strategy", "category": "ratio"},
    {"id": "calmar_ratio", "type": "Value", "title": "Calmar Ratio", "value_key": "calmar_ratio", 
     "description": "Calmar ratio of the strategy", "category": "ratio"},
     
    # Trade statistics
    {"id": "positive_days", "type": "Value", "title": "Positive Days", "value_key": "positive_days", 
     "description": "Number of days with positive returns", "category": "trade"},
    {"id": "negative_days", "type": "Value", "title": "Negative Days", "value_key": "negative_days", 
     "description": "Number of days with negative returns", "category": "trade"},
    {"id": "win_rate", "type": "Percentage", "title": "Win Rate", "value_key": "pct_positive_days", 
     "description": "Percentage of days with positive returns", "category": "trade"},
     
    # Monthly statistics
    {"id": "best_month", "type": "Percentage", "title": "Best Month", "value_key": "best_month", 
     "description": "Best monthly return", "category": "trade"},
    {"id": "worst_month", "type": "Percentage", "title": "Worst Month", "value_key": "worst_month", 
     "description": "Worst monthly return", "category": "trade"},
    {"id": "avg_monthly_return", "type": "Percentage", "title": "Avg Monthly Return", "value_key": "avg_monthly_return", 
     "description": "Average monthly return", "category": "trade"},
    {"id": "monthly_volatility", "type": "Percentage", "title": "Monthly Volatility", "value_key": "monthly_volatility", 
     "description": "Standard deviation of monthly returns", "category": "trade"},
    {"id": "monthly_win_rate", "type": "Percentage", "title": "Monthly Win Rate", "value_key": "pct_positive_months", 
     "description": "Percentage of months with positive returns", "category": "trade"},
     
    # Benchmark-relative metrics (conditionally available)
    {"id": "alpha", "type": "Percentage", "title": "Alpha", "value_key": "alpha", 
     "description": "Alpha relative to benchmark", "category": "benchmark"},
    {"id": "beta", "type": "Value", "title": "Beta", "value_key": "beta", 
     "description": "Beta relative to benchmark", "category": "benchmark"},
    {"id": "correlation", "type": "Value", "title": "Correlation", "value_key": "correlation", 
     "description": "Correlation with benchmark", "category": "benchmark"},
    {"id": "tracking_error", "type": "Percentage", "title": "Tracking Error", "value_key": "tracking_error", 
     "description": "Tracking error relative to benchmark", "category": "benchmark"},
    {"id": "information_ratio", "type": "Value", "title": "Information Ratio", "value_key": "information_ratio", 
     "description": "Information ratio relative to benchmark", "category": "benchmark"},
    {"id": "capture_ratio_up", "type": "Percentage", "title": "Upside Capture", "value_key": "capture_ratio_up", 
     "description": "Upside capture ratio", "category": "benchmark"},
    {"id": "capture_ratio_down", "type": "Percentage", "title": "Downside Capture", "value_key": "capture_ratio_down", 
     "description": "Downside capture ratio", "category": "benchmark"}
]

AVAILABLE_CHARTS = [
    # Basic performance charts
    {"id": "equity_curve", "type": "LineChart", "title": "Equity Curve", "data_key": "equity_curve", 
     "description": "Shows the growth of portfolio value over time", "category": "performance"},
    {"id": "drawdown", "type": "LineChart", "title": "Drawdown Chart", "data_key": "drawdown_series", 
     "description": "Shows the drawdown periods for the strategy", "category": "performance"},
    {"id": "daily_returns", "type": "LineChart", "title": "Daily Returns", "data_key": "daily_returns", 
     "description": "Shows daily returns of the strategy", "category": "performance"},
    {"id": "monthly_returns", "type": "HeatmapTable", "title": "Monthly Returns Heatmap", "data_key": "monthly_returns", 
     "description": "Displays monthly returns as a heatmap", "category": "returns"},
    {"id": "yearly_returns", "type": "BarChart", "title": "Yearly Returns", "data_key": "yearly_returns", 
     "description": "Shows yearly returns as a bar chart", "category": "returns"},
     
    # Rolling metrics charts
    {"id": "rolling_sharpe", "type": "LineChart", "title": "Rolling Sharpe Ratio", "data_key": "rolling_sharpe", 
     "description": "Shows the rolling Sharpe ratio over time", "category": "rolling"},
    {"id": "rolling_sortino", "type": "LineChart", "title": "Rolling Sortino Ratio", "data_key": "rolling_sortino", 
     "description": "Shows the rolling Sortino ratio over time", "category": "rolling"},
    {"id": "rolling_volatility", "type": "LineChart", "title": "Rolling Volatility", "data_key": "rolling_volatility", 
     "description": "Shows the rolling volatility over time", "category": "rolling"},
    {"id": "rolling_skew", "type": "LineChart", "title": "Rolling Skewness", "data_key": "rolling_skew", 
     "description": "Shows the rolling skewness of returns", "category": "rolling"},
    {"id": "rolling_var", "type": "LineChart", "title": "Rolling VaR (5%)", "data_key": "rolling_var", 
     "description": "Shows the rolling 5% Value at Risk", "category": "rolling"},
    {"id": "rolling_drawdown_duration", "type": "LineChart", "title": "Rolling Max Drawdown Duration", "data_key": "rolling_drawdown_duration", 
     "description": "Shows the rolling maximum drawdown duration in days", "category": "rolling"},
     
    # Rolling returns charts
    {"id": "rolling_3m_returns", "type": "LineChart", "title": "Rolling 3-Month Returns", "data_key": "rolling_3m_returns", 
     "description": "Shows rolling 3-month returns", "category": "returns"},
    {"id": "rolling_6m_returns", "type": "LineChart", "title": "Rolling 6-Month Returns", "data_key": "rolling_6m_returns", 
     "description": "Shows rolling 6-month returns", "category": "returns"},
    {"id": "rolling_1y_returns", "type": "LineChart", "title": "Rolling 1-Year Returns", "data_key": "rolling_1y_returns", 
     "description": "Shows rolling 1-year returns", "category": "returns"},
     
    # Benchmark comparison charts (conditionally available)
    {"id": "benchmark_comparison", "type": "LineChart", "title": "Strategy vs Benchmark", 
     "data_key": "benchmark_comparison", 
     "description": "Compares strategy and benchmark performance", "category": "benchmark"},
    {"id": "relative_performance", "type": "LineChart", "title": "Relative Performance", 
     "data_key": "relative_performance", 
     "description": "Shows performance relative to benchmark", "category": "benchmark"},
    {"id": "benchmark_drawdown", "type": "LineChart", "title": "Benchmark Drawdown", 
     "data_key": "benchmark_drawdown_series", 
     "description": "Shows benchmark drawdown periods", "category": "benchmark"},
    {"id": "benchmark_volatility", "type": "LineChart", "title": "Benchmark Rolling Volatility", 
     "data_key": "benchmark_rolling_volatility", 
     "description": "Shows benchmark rolling volatility", "category": "benchmark"}
]

def load_config(config_path=None):
    """Load configuration from file or use default if file doesn't exist."""
    if config_path is None:
        config_path = USER_CONFIG_PATH
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Use default config
        with open(DEFAULT_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Save to user config
        with open(USER_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config

def save_config(config, config_path=None):
    """Save configuration to file."""
    if config_path is None:
        config_path = USER_CONFIG_PATH
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

@app.route('/')
def index():
    """Render the dashboard editor."""
    config = load_config()
    return render_template('index.html', 
                          config=config, 
                          available_charts=AVAILABLE_CHARTS,
                          available_metrics=AVAILABLE_METRICS,
                          data_loaded=(uploaded_data is not None))

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get the current configuration."""
    config = load_config()
    return jsonify(config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update the configuration."""
    config = request.json
    save_config(config)
    return jsonify({"status": "success"})

@app.route('/api/reset-config', methods=['POST'])
def reset_config():
    """Reset to the default configuration."""
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    save_config(config)
    return jsonify({"status": "success"})

@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    """Upload a CSV file and process it."""
    global uploaded_data, engine, dashboard_path
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"})
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load the CSV data
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # If it's a multi-column DataFrame, use the first column as price data
            if isinstance(data, pd.DataFrame) and data.shape[1] > 1:
                price_data = data.iloc[:, 0]
            else:
                price_data = data
            
            # Create a backtest engine
            engine = Engine(data=price_data, benchmark=None)
            results = engine.run()
            
            # Generate dashboard
            dashboard_path = engine.generate_dashboard(
                output_dir=os.path.join(app.config['UPLOAD_FOLDER'], 'dashboard'),
                open_browser=False,
                config_path=USER_CONFIG_PATH
            )
            
            # Store the uploaded data for later use
            uploaded_data = data
            
            return jsonify({
                "status": "success", 
                "message": f"Successfully processed {filename}",
                "dashboard_url": f"/dashboard"
            })
        
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"})
    
    return jsonify({"status": "error", "message": "Invalid file format"})

@app.route('/dashboard')
def view_dashboard():
    """View the generated dashboard."""
    if dashboard_path:
        dashboard_dir = os.path.dirname(dashboard_path)
        return send_from_directory(dashboard_dir, 'dashboard.html')
    else:
        return redirect(url_for('index'))

@app.route('/dashboard/<path:filename>')
def dashboard_files(filename):
    """Serve dashboard files."""
    if dashboard_path:
        dashboard_dir = os.path.dirname(dashboard_path)
        return send_from_directory(dashboard_dir, filename)
    else:
        return redirect(url_for('index'))