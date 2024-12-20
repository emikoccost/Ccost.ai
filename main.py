"""
Demand Forecasting Production Pipeline
A comprehensive solution for demand forecasting using SARIMA models
"""

import logging
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('demand_forecast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    A class for demand forecasting using SARIMA models
    Includes data preprocessing, model training, evaluation, and visualization
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the forecaster with configuration
        Args:config_path (str): Path to the configuration file
        """
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = None
        self.metrics_history = []

    @staticmethod
    def load_config(config_path: str) -> dict:
        """
        Load configuration from YAML file
        Args:config_path (str): Path to configuration file
        Returns: dict: Configuration parameters
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    @staticmethod
    def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
        """
        Load and preprocess the input data
        Args: file_path (str): Path to the input CSV file
        Returns: pd.DataFrame: Processed dataframe with additional features:
        - Time-based features (Year, Month, Day, DayOfWeek, Quarter)
        - Lag features (1-day and 7-day lags)
        """
        try:
            logger.info("Loading data...")
            df = pd.read_csv(file_path)

            # Basic preprocessing
            df['Date'] = pd.to_datetime(df['Date'])
            daily_sales = df.groupby('Date')['Total'].sum().reset_index()
            daily_sales = daily_sales.set_index('Date')

            # Time-based features
            daily_sales['Year'] = daily_sales.index.year
            daily_sales['Month'] = daily_sales.index.month
            daily_sales['Day'] = daily_sales.index.day
            daily_sales['DayOfWeek'] = daily_sales.index.dayofweek
            daily_sales['Quarter'] = daily_sales.index.quarter

            # Lag features
            daily_sales['Sales_Lag1'] = daily_sales['Total'].shift(1)
            daily_sales['Sales_Lag7'] = daily_sales['Total'].shift(7)
            daily_sales = daily_sales.dropna()

            logger.info(f"Data loaded and preprocessed. Shape: {daily_sales.shape}")
            return daily_sales

        except Exception as e:
            logger.error(f"Error in data loading and preprocessing: {e}")
            raise

    @staticmethod
    def evaluate_metrics(y_true: np.array, y_pred: np.array,
                         weights: Optional[np.array] = None) -> Dict:
        """
        Calculate forecast quality metrics
        Args:
            y_true (np.array): Actual values
            y_pred (np.array): Predicted values
            weights (np.array, optional): Weights for weighted metrics
        Returns:
            Dict: Dictionary containing various metrics:
                - MAE: Mean Absolute Error
                - RMSE: Root Mean Square Error
                - Bias: Systematic bias in predictions
                - Weighted versions if weights provided
                - Combined Score: MAE + |Bias|
        """
        try:
            metrics = {'MAE': mean_absolute_error(y_true, y_pred),
                       'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                       'Bias': np.mean(y_pred - y_true)}
            # Weighted metrics
            if weights is not None:
                metrics['Weighted_MAE'] = np.average(np.abs(y_true - y_pred), weights=weights)
                metrics['Weighted_Bias'] = np.average(y_pred - y_true, weights=weights)

            # Combined score
            metrics['Combined_Score'] = metrics['MAE'] + abs(metrics['Bias'])

            return metrics

        except Exception as e:
            logger.error(f"Error in metrics calculation: {e}")
            raise

    def train_model(self, data: pd.DataFrame, train_size: float = 0.8) -> Tuple:
        """
        Train the SARIMA model and evaluate its performance
        Args:
            data (pd.DataFrame): Preprocessed data
            train_size (float): Proportion of data to use for training
        Returns:
            Tuple: (forecast, metrics, train_data, test_data)
                - forecast: Model predictions
                - metrics: Evaluation metrics
                - train_data: Training dataset
                - test_data: Test dataset
        """
        try:
            logger.info("Starting model training...")

            # Train/test split
            train_size = int(len(data) * train_size)
            train = data[:train_size]
            test = data[train_size:]

            # Model training
            model = SARIMAX(train['Total'],
                            order=self.config.get('sarima_order', (1, 1, 1)),
                            seasonal_order=self.config.get('seasonal_order', (1, 1, 1, 12)))

            self.model = model.fit()

            # Make predictions
            forecast = self.model.forecast(steps=len(test))

            # Evaluate performance
            metrics = self.evaluate_metrics(test['Total'], forecast)
            self.metrics_history.append(metrics)

            logger.info(f"Model trained. Metrics: {metrics}")
            return forecast, metrics, train, test

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def create_future_forecast(self, steps: int, save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create and save forecast for future periods
        Args:
            steps (int): Number of periods to forecast
            save_path (str, optional): Path to save the forecast
        Returns:
            pd.DataFrame: Forecast with dates and confidence intervals
        """
        try:
            if self.model is None:
                raise ValueError("Model is not trained")

            last_date = self.model.data.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                         periods=steps, freq='D')

            forecast = self.model.get_forecast(steps=steps)
            forecast_mean = forecast.predicted_mean
            confidence_int = forecast.conf_int()

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast_mean,
                'Lower_CI': confidence_int.iloc[:, 0],
                'Upper_CI': confidence_int.iloc[:, 1]
            })

            if save_path:
                forecast_df.to_csv(save_path, index=False)
                logger.info(f"Future forecast saved to {save_path}")

            return forecast_df

        except Exception as e:
            logger.error(f"Error in creating future forecast: {e}")
            raise

    def plot_results(self, train: pd.DataFrame, test: pd.DataFrame,
                     forecast: np.array, save_path: Optional[str] = None):
        """
        Visualize forecast results
        Args:
            train (pd.DataFrame): Training data
            test (pd.DataFrame): Test data
            forecast (np.array): Model predictions
            save_path (str, optional): Path to save the plots
        """
        try:
            plt.figure(figsize=(15, 7))
            plt.plot(train.index, train['Total'], label='Training Data')
            plt.plot(test.index, test['Total'], label='Actual Data')
            plt.plot(test.index, forecast, label='Forecast', color='red')
            plt.title('Demand Forecast Results')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.legend()
            plt.grid(True)

            if save_path:
                plt.savefig(save_path)
            plt.close()

            # Residuals plot
            residuals = test['Total'] - forecast
            self.plot_residuals(residuals, save_path)

        except Exception as e:
            logger.error(f"Error in plotting results: {e}")
            raise

    @staticmethod
    def plot_residuals(residuals: np.array, save_path: Optional[str] = None):
        """
        Plot forecast residuals
        Args:
            residuals (np.array): Forecast residuals
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(15, 7))
        plt.plot(residuals)
        plt.title('Forecast Residuals')
        plt.xlabel('Date')
        plt.ylabel('Residual Value')
        plt.grid(True)

        if save_path:
            path = Path(save_path)
            plt.savefig(str(path.parent / f"{path.stem}_residuals{path.suffix}"))
        plt.close()

    def save_model(self, path: str):
        """
        Save trained model to disk
        Args:
            path (str): Path to save the model
        """
        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str):
        """
        Load trained model from disk
        Args:
            path (str): Path to the saved model
        """
        try:
            self.model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise


def main():
    """
    Main execution pipeline:
    1. Initialize forecaster
    2. Load and preprocess data
    3. Train model
    4. Evaluate and visualize results
    5. Create future forecast
    6. Save results
    """
    try:
        # Initialize
        forecaster = DemandForecaster()

        # Load and prepare data
        data = forecaster.load_and_preprocess_data('supermarket_sales.csv')

        # Train model
        forecast, metrics, train, test = forecaster.train_model(data)

        # Visualize results
        forecaster.plot_results(train, test, forecast, 'forecast_results.png')

        forecaster.create_future_forecast(steps=30,
                                          save_path='future_forecast.csv'
                                          )

        forecaster.save_model('demand_forecast_model.joblib')

        logger.info("Pipeline completed successfully")
        logger.info(f"Final metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise
