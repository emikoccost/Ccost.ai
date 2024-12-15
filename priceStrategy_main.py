"""
Price Strategy Production Pipeline
A comprehensive solution for Price Strategy module to interface with AI Engine
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


warnings.filterwarnings('ignore')

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('price_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PriceStrategy:
    """
    A class for price strategy API
    Interface with AI Intelligence Engine
    """
    # Define default configuration values
    DEFAULT_CONFIG = {
        'minimum_margin': 3.0,         # Default minimum margin percentage
        'desired_margin': 10.0,        # Default desired margin percentage
        'check_period': 7,            # Default check period in days
        'margin_default_step': 1.0,    # Default step for margin adjustments
        'turnover_high': 0.8,         # High turnover threshold
        'turnover_low': 0.2,          # Low turnover threshold
        'marketing_campaign': 1.0,    #Marketing Campaign margin
        'min_price': 0.01,            # Minimum allowed price
        'max_price': 1000000.0,       # Maximum allowed price
        'data_path': 'data/',         # Default data directory
        'static_database': 'data/products.csv'  # Default product database
    }

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the Price Strategy module with configuration
        Args:config_path (str): Path to the configuration file
        """
        loaded_config = self.load_config(config_path)
        self.config = {**self.DEFAULT_CONFIG, **loaded_config}
        self.model = None
        self.scaler = None
        self.metrics_history = []

 # Store configuration values as instance attributes for easy access
        self.data_path = self.config['data_path']
        self.static_database = self.config['static_database']
        self.minimum_margin = self.config['minimum_margin']
        self.desired_margin = self.config['desired_margin']
        self.check_period = self.config['check_period']
        self.margin_default_step = self.config['margin_default_step']
        self.turnover_high = self.config['turnover_high']
        self.turnover_low = self.config['turnover_low']
        self.min_price = self.config['min_price']
        self.max_price = self.config['max_price']

    # Initialize turnover ratio (will be updated per product)
        self.turnover_ratio = 0.0

     # Validate paths on initialization
        if not self.validate_paths():
            logger.warning("Some paths are invalid. Please check configuration.")

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
        
    def validate_paths(self):
        """
        Validate that all configured paths exist
        Returns: bool: True if all paths are valid, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            # Path(self.data_path).mkdir(parents=True, exist_ok=True)
  
            # Check if data files exist
            if not Path(self.data_path).exists():
                logger.warning(f"Static Database  path does not exist: {self.static_database}")
                # Create parent directories if they don't exist
                Path(self.static_database).parent.mkdir(parents=True, exist_ok=True)
                # Create empty CSV with required columns
                df = pd.DataFrame(columns=[
                    'Universal_Product_Code',
                    'Current_Unit_Price',
                    'Unit_Cost',
                    'current_Margin',
                    'Marketing_Campaign',
                    'Turnover_Ratio'
                ])
                df.to_csv(self.static_database, index=False)
                logger.info(f"Created empty database at {self.static_database}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating paths: {e}")
            return False
            

    def apply_price_strategy(self, current_margin: float) -> bool:
        """
        Example method showing how to use the configured values
        """
        if current_margin < self.minimum_margin:
            logger.warning(f"Current margin {current_margin}% is below minimum {self.minimum_margin}%")
            return False
            
        if current_margin < self.desired_margin:
            logger.info(f"Current margin {current_margin}% is below desired {self.desired_margin}%")
            # Implement price adjustment logic here
            
        return True

    def search_by_upc(self, upc_number):
        """
        Search for a product in CSV file using UPC number and return specific financial metrics
    
        Parameters:
        file_path = : Path to the CSV file
        upc_number (str): SKU number to search for
    
        Returns:
        dict: Dictionary containing the requested product information
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.static_database)
            # DEBUG - printing
            # print(df.columns)
            # Convert UPC in dataframe to integer
            df['Universal_Product_Code'] = df['Universal_Product_Code'].astype(int)

            # Convert input UPC to integer
            upc_number = int(upc_number)

        
            # Search for the UPC number
            product = df[df['Universal_Product_Code'] == upc_number]
        

            if len(product) == 0:
                return f"No product found with UPC: {upc_number}"
        
            # Get the specific columns you requested
            needed_columns = [
                'Universal_Product_Code',
                'Current_Unit_Price',
                'Unit_Cost',
                'current_Margin',
                'Marketing_Campaign',
                'Turnover_Ratio'
            ]
        
            product_info = product[needed_columns].iloc[0]
        
            # Update instance turnover ratio for the current product
            self.turnover_ratio = float(product_info['Turnover_Ratio'])
            
            # Convert to dictionary for easy access
            result = product_info.to_dict()
        
            # Format the values for better readability
            if 'Current_Unit_Price' in result:
                result['Current_Unit_Price'] = f"${result['Current_Unit_Price']:.2f}"

            # if 'Current_Unit_Price' in result:
                # result['Current_Unit_Price'] = f"${result[Current_Unit_Price]}"
            if 'current_Margin' in result:
                result['current_Margin'] = f"{result['current_Margin']:.2f}%"
                # result['current_Margin'] = f"{result['current_Margin']}%"
            # if 'Unit_Cost' in result:
                # result['Unit_Cost'] = f"${result['Unit Cost']:.2f}"
            if 'Turnover_Ratio' in result:
                result['Turnover_Ratio'] = f"{result['Turnover_Ratio']:.2f}"
            if 'Product_Category' in result:
                result['Product_Category'] = f"{result['Product_Category']}"
        
            return result
    
        except FileNotFoundError:
            return "Error: CSV file not found"
        except KeyError as e:
            return f"Error: Column not found in CSV file: {str(e)}"
        except Exception as e:
            return f"Error occurred: {str(e)}"

    def calculate_margin(self, cost: float, price: float) -> float:
        """
        Calculate margin percentage
        """
        actual_margins = [
            0,
            0,
            0
        ]
        if cost == 0:
            return actual_margins
        
        actual_margins[0] = ((price[0] - cost) / price[0]) * 100

        if price[1] > 0:
             actual_margins[1] = ((price[1] - cost) / price[1]) * 100
        if price[2] > 0:
             actual_margins[2] = ((price[2] - cost) / price[2]) * 100
        
        return actual_margins
    
    def suggest_price(self, cost: float, current_margin: float, Marketing_Campaign: str, marketing_margin: float) -> float:
        """
        Suggest a price based on cost and category

        Parameters:
        cost (float): Unit cost of the product
        current_margin (float): Current margin percentage
        
        Returns:
        float: Suggested price rounded to 2 decimal places
        """
        print(f"Current margin: {current_margin}%")


        suggested_prices = [
            0,
            0,
            0
        ]

        target_margin = current_margin

        if self.turnover_ratio >= self.turnover_high:
            target_margin = current_margin + self.margin_default_step
        if self.turnover_ratio <= self.turnover_low:
            target_margin = current_margin - self.margin_default_step
            if target_margin <= self.minimum_margin:
                target_margin = current_margin
        print(f"Target margin: {target_margin}%")
        suggested_prices[0] = cost / (1 - target_margin/100)

        """ 
        Check Minimum margin and desired margin 
        """
        # Ensure price is within bounds
        
        suggested_prices[0] = round(suggested_prices[0], 2)

        if target_margin >= self.desired_margin:
            target_margin = self.desired_margin
        suggested_prices[1] = cost / (1 - target_margin/100)
        suggested_prices[1] = round(suggested_prices[1], 2)

        if target_margin <= self.minimum_margin:
            target_margin = self.minimum_margin
        suggested_prices[1] = cost / (1 - target_margin/100)
        suggested_prices[1] = round(suggested_prices[1], 2)

        """ 
        Check Marketing Campaign 
        """
        if Marketing_Campaign == True & marketing_margin > 0:
            target_margin = marketing_margin
        suggested_prices[2] = cost / (1 - target_margin/100) 
        suggested_prices[2] = round(suggested_prices[2], 2) 

        return suggested_prices  
       

    

def run_price_strategy():
    """
    Example usage of the PriceStrategy class
    """
    
    # Create instance
    strategy = PriceStrategy('config.yaml')
    
    # Print configuration
    print("\nConfiguration Values:")
    print(f"Minimum margin: {strategy.minimum_margin}%")
    print(f"Desired margin: {strategy.desired_margin}%")
    print(f"Check period: {strategy.check_period} days")
    print(f"Turnover thresholds: Low={strategy.turnover_low}, High={strategy.turnover_high}")
     
    # Search for the product
    upc_to_find = input("Enter UPC string: ")
    result = strategy.search_by_upc(upc_to_find)
    
    # Print results
    if isinstance(result, dict):
        print("\nProduct Information:")
        for key, value in result.items():
            print(f"{key}: {value}")

                   
        # Extract values for price calculation
        if isinstance(result['Unit_Cost'], str):
            unit_cost = float(result['Unit_Cost'].replace('$', ''))
        else:
            unit_cost = float(result['Unit_Cost'])
            # unit_cost = float(result['Unit_Cost'].replace('$', ''))
        
        if isinstance(result['current_Margin'], str):
            current_margin = float(result['current_Margin'].replace('%', ''))
        else:
            current_margin = float(result['current_Margin'])

            # current_margin = float(result['current_Margin'].replace('%', ''))

        # Calculate suggested price
        suggested_prices = strategy.suggest_price(unit_cost, current_margin, result['Marketing_Campaign'], result['Marketing_Campaign'])
        actual_margins = strategy.calculate_margin(unit_cost, suggested_prices)
        print(f"Current price: {result['Current_Unit_Price']}")
        print(f"Current margin: {result['current_Margin']}")
        print(f"\nPrice Analysis")
        print(f"Suggested price #1: ${suggested_prices[0]:.2f}")
        print(f"Resulting margin: #1 {actual_margins[0]:.1f}%")
        if suggested_prices[1] > 0:
            print(f"Suggested price #2: ${suggested_prices[1]:.2f}")
            print(f"Resulting margin: #2 {actual_margins[1]:.1f}%")
        if suggested_prices[2] > 0:
            print(f"Suggested price #3: ${suggested_prices[2]:.2f}")
            print(f"Resulting margin: #3 {actual_margins[2]:.1f}%")

      

    else:
        print(result)

if __name__ == "__main__":
    # Example UPC
    run_price_strategy()
    
  
