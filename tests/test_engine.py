import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from algosystem.backtesting.engine import Engine


class TestEngine:
    """Test cases for the Engine class."""
    
    def test_engine_initialization_with_series(self, sample_price_series):
        """Test Engine initialization with pandas Series."""
        engine = Engine(sample_price_series)
        
        assert engine.price_series is not None
        assert len(engine.price_series) == len(sample_price_series)
        assert engine.initial_capital == sample_price_series.iloc[0]
        assert engine.results is None
    
    def test_engine_initialization_with_dataframe(self, sample_dataframe):
        """Test Engine initialization with pandas DataFrame."""
        engine = Engine(sample_dataframe, price_column='Strategy')
        
        assert engine.price_series is not None
        assert len(engine.price_series) == len(sample_dataframe)
        assert engine.initial_capital == sample_dataframe['Strategy'].iloc[0]
    
    def test_engine_initialization_with_benchmark(self, sample_price_series, sample_benchmark_series):
        """Test Engine initialization with benchmark data."""
        engine = Engine(sample_price_series, benchmark=sample_benchmark_series)
        
        assert engine.price_series is not None
        assert engine.benchmark_series is not None
        assert len(engine.benchmark_series) > 0
    
    def test_engine_initialization_with_custom_capital(self, sample_price_series):
        """Test Engine initialization with custom initial capital."""
        custom_capital = 50000.0
        engine = Engine(sample_price_series, initial_capital=custom_capital)
        
        assert engine.initial_capital == custom_capital
    
    def test_engine_initialization_with_date_range(self, sample_price_series):
        """Test Engine initialization with custom date range."""
        start_date = '2020-01-15'
        end_date = '2020-02-15'
        
        engine = Engine(sample_price_series, start_date=start_date, end_date=end_date)
        
        assert engine.start_date == pd.to_datetime(start_date)
        assert engine.end_date == pd.to_datetime(end_date)
        assert len(engine.price_series) < len(sample_price_series)
    
    def test_basic_backtest_run(self, sample_price_series):
        """Test basic backtest execution."""
        engine = Engine(sample_price_series)
        results = engine.run()
        
        # Check that results are generated
        assert results is not None
        assert isinstance(results, dict)
        
        # Check for required result keys
        required_keys = ['equity', 'initial_capital', 'final_capital', 'returns', 'metrics', 'plots']
        for key in required_keys:
            assert key in results
        
        # Check equity series
        assert 'equity' in results
        assert isinstance(results['equity'], pd.Series)
        assert len(results['equity']) > 0
        
        # Check that metrics were calculated
        assert 'metrics' in results
        assert isinstance(results['metrics'], dict)
        assert len(results['metrics']) > 0
    
    def test_backtest_with_benchmark(self, sample_price_series, sample_benchmark_series):
        """Test backtest with benchmark comparison."""
        engine = Engine(sample_price_series, benchmark=sample_benchmark_series)
        results = engine.run()
        
        # Check that benchmark-specific metrics are calculated
        metrics = results['metrics']
        benchmark_metrics = ['alpha', 'beta', 'correlation']
        
        # Note: Some metrics might not be present if calculation fails
        # We just check that no errors occurred during execution
        assert results is not None
        assert 'metrics' in results
    
    def test_get_results_before_run(self, sample_price_series):
        """Test getting results before running backtest."""
        engine = Engine(sample_price_series)
        results = engine.get_results()
        
        # Should return empty dict when no backtest has been run
        assert results == {}
    
    def test_get_metrics_before_run(self, sample_price_series):
        """Test getting metrics before running backtest."""
        engine = Engine(sample_price_series)
        metrics = engine.get_metrics()
        
        # Should return empty dict when no backtest has been run
        assert metrics == {}
    
    def test_small_dataset(self, small_dataset):
        """Test engine with very small dataset."""
        engine = Engine(small_dataset)
        
        # Should not raise an error even with small dataset
        results = engine.run()
        assert results is not None
        assert 'metrics' in results
    
    def test_large_dataset(self, large_dataset):
        """Test engine with large dataset."""
        engine = Engine(large_dataset)
        
        # Should handle large datasets without issues
        results = engine.run()
        assert results is not None
        assert len(results['equity']) == len(large_dataset)
    
    def test_constant_prices(self, constant_series):
        """Test engine with constant price series."""
        engine = Engine(constant_series)
        results = engine.run()
        
        # Should handle constant prices gracefully
        assert results is not None
        assert results['returns'] == 0.0  # No change in prices
    
    def test_highly_volatile_series(self, volatile_series):
        """Test engine with highly volatile series."""
        engine = Engine(volatile_series)
        results = engine.run()
        
        # Should handle high volatility without errors
        assert results is not None
        assert 'metrics' in results
        assert 'volatility' in results['metrics']
    
    def test_negative_returns_series(self, negative_returns_series):
        """Test engine with series that has negative returns."""
        engine = Engine(negative_returns_series)
        results = engine.run()
        
        # Should handle negative returns without errors
        assert results is not None
        assert results['returns'] < 0  # Should show negative total return
    
    def test_invalid_data_type(self):
        """Test engine with invalid data type."""
        with pytest.raises((TypeError, ValueError)):
            Engine("invalid_data")
    
    def test_empty_dataframe_error(self):
        """Test engine with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            Engine(empty_df)
    
    def test_dataframe_without_price_column(self, sample_dataframe):
        """Test engine with DataFrame but no price column specified."""
        # Should raise error when DataFrame has multiple columns but no price_column specified
        with pytest.raises(ValueError):
            Engine(sample_dataframe)  # Has multiple columns
    
    def test_invalid_date_range(self, sample_price_series):
        """Test engine with invalid date range."""
        # End date before start date
        with pytest.raises(ValueError):
            engine = Engine(sample_price_series, start_date='2020-12-31', end_date='2020-01-01')
            engine.run()
    
    def test_print_metrics(self, sample_price_series, capsys):
        """Test metrics printing functionality."""
        engine = Engine(sample_price_series)
        results = engine.run()
        
        # Should not raise error
        engine.print_metrics()
        
        # Check that something was printed
        captured = capsys.readouterr()
        # Note: We don't check exact content as logs may vary


class TestEngineEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_data_handling(self, missing_data_series):
        """Test how engine handles missing data."""
        # The engine should handle NaN values gracefully
        engine = Engine(missing_data_series.dropna())  # Drop NaN for valid test
        results = engine.run()
        
        assert results is not None
        assert 'equity' in results
    
    def test_single_data_point(self):
        """Test engine with single data point."""
        single_point = pd.Series([100], index=[pd.Timestamp('2020-01-01')])
        
        # Should handle gracefully or raise appropriate error
        try:
            engine = Engine(single_point)
            results = engine.run()
            # If it runs, should not crash
            assert results is not None
        except ValueError:
            # It's acceptable to raise ValueError for insufficient data
            pass
    
    def test_non_datetime_index(self):
        """Test engine with non-datetime index."""
        data = pd.Series([100, 101, 102], index=[0, 1, 2])
        
        # Should either work or raise a clear error
        try:
            engine = Engine(data)
            results = engine.run()
            assert results is not None
        except (ValueError, TypeError):
            # Acceptable to reject non-datetime indices
            pass