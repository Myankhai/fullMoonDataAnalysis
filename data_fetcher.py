import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from urllib.parse import quote
import scipy.stats as stats

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.chicago_api_url = "https://data.cityofchicago.org/resource/9hwr-2zxp.json"  # 2022 crimes API
        self.nyc_api_url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"  # Historical complaints API

    def fetch_chicago_homicides(self, start_date, end_date):
        """Fetch homicide data from Chicago Data Portal"""
        query = f"?$where=date between '{start_date}T00:00:00' and '{end_date}T23:59:59' AND primary_type='HOMICIDE'"
        query += "&$order=date DESC"
        
        response = requests.get(self.chicago_api_url + query)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if len(df) == 0:
                print("Warning: No Chicago homicide data found for the specified date range")
                return pd.DataFrame(columns=['date'])  # Return empty DataFrame with required column
            if 'date' not in df.columns:
                print(f"Available columns: {df.columns.tolist()}")
                raise Exception("'date' column not found in response")
            df['date'] = pd.to_datetime(df['date'])
            return df
        else:
            print(f"Chicago API Response: {response.text}")
            raise Exception(f"Failed to fetch Chicago data: {response.status_code}")

    def fetch_nyc_homicides(self, start_date, end_date):
        """Fetch homicide data from NYC OpenData"""
        # Build the query with proper URL encoding
        conditions = [
            f"cmplnt_fr_dt between '{start_date}T00:00:00' and '{end_date}T23:59:59'",
            "law_cat_cd='FELONY'",
            "ofns_desc like '%MURDER%'"  # More flexible search for murder-related crimes
        ]
        where_clause = ' AND '.join(conditions)
        
        query = f"?$where={quote(where_clause)}"
        query += "&$order=cmplnt_fr_dt DESC"
        
        url = self.nyc_api_url + query
        print(f"Querying NYC API with URL: {url}")  # Debug output
        
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            if len(df) == 0:
                print("Warning: No NYC homicide data found for the specified date range")
                return pd.DataFrame(columns=['date'])
            
            # Print available columns for debugging
            print("Available columns in NYC data:", df.columns.tolist())
            
            # Convert date column - handle both possible column names
            date_col = 'cmplnt_fr_dt'
            if date_col in df.columns:
                df['date'] = pd.to_datetime(df[date_col]).dt.date.apply(lambda x: datetime.combine(x, datetime.min.time()))
            else:
                print(f"Warning: No date column found in NYC data. Available columns: {df.columns.tolist()}")
                return pd.DataFrame(columns=['date'])
                
            return df
        else:
            print(f"NYC API Response: {response.text}")
            raise Exception(f"Failed to fetch NYC data: {response.status_code}")

    def get_city_coordinates(self, city):
        """Return coordinates for supported cities"""
        coordinates = {
            'CHICAGO': {'lat': 41.8781, 'lon': -87.6298},
            'NYC': {'lat': 40.7128, 'lon': -74.0060}
        }
        return coordinates.get(city.upper())

    def analyze_satellite_proximity(self, satellite_data, crime_data):
        """Analyze relationship between satellite proximity and crime frequency"""
        if not satellite_data or len(crime_data) == 0:
            print("No data available for satellite proximity analysis")
            return {
                'correlation': 0,
                'p_value': 1.0,
                'merged_data': pd.DataFrame(columns=['date', 'distance'])
            }

        # Convert satellite data to DataFrame if it's not empty
        if isinstance(satellite_data, dict):
            if not satellite_data:  # Empty dict
                return {
                    'correlation': 0,
                    'p_value': 1.0,
                    'merged_data': pd.DataFrame(columns=['date', 'distance'])
                }
            sat_df = pd.DataFrame([satellite_data])
        else:
            sat_df = pd.DataFrame(satellite_data)

        # Ensure we have the required columns
        if 'distance' not in sat_df.columns or 'date' not in sat_df.columns:
            print("Missing required columns in satellite data")
            return {
                'correlation': 0,
                'p_value': 1.0,
                'merged_data': pd.DataFrame(columns=['date', 'distance'])
            }

        # Group crimes by date for correlation analysis
        daily_crimes = crime_data.groupby('date').size().reset_index(name='crime_count')
        
        # Merge with satellite data
        merged_data = pd.merge(
            daily_crimes,
            sat_df[['date', 'distance']],
            on='date',
            how='left'
        )
        
        # Fill missing distances with mean or 0
        if len(merged_data) > 0:
            merged_data['distance'].fillna(merged_data['distance'].mean() if len(merged_data['distance'].dropna()) > 0 else 0, inplace=True)
        
        # Calculate correlation if we have enough data points
        if len(merged_data) >= 2:
            correlation = stats.pearsonr(
                merged_data['distance'],
                merged_data['crime_count']
            )
        else:
            print("Insufficient data points for satellite correlation analysis")
            correlation = (0, 1.0)
        
        return {
            'correlation': correlation[0],
            'p_value': correlation[1],
            'merged_data': merged_data
        } 