from datetime import datetime, timedelta
from data_fetcher import DataFetcher
from moon_calculator import MoonCalculator
from analyzer import CrimeAnalyzer
import os
from dotenv import load_dotenv
import json
import pandas as pd

def convert_timestamps(obj):
    """Convert timestamps to string format for JSON serialization"""
    if isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, dict):
        return {k: convert_timestamps(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps(i) for i in obj]
    return obj

def main():
    # Load environment variables
    load_dotenv()
    # NASA API key no longer needed
    # if not os.getenv('NASA_API_KEY'):
    #     raise ValueError("NASA API key not found. Please set NASA_API_KEY in .env file")

    # Initialize components
    data_fetcher = DataFetcher()
    moon_calculator = MoonCalculator()
    analyzer = CrimeAnalyzer(moon_calculator)

    # Create output directory for JSON files if it doesn't exist
    os.makedirs('analysis_data', exist_ok=True)

    # Set date range for analysis (use 2022 for all cities)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    
    print(f"Analyzing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Store moon analysis results for all cities
    city_data = {}
    all_analysis_results = {}
    
    # Analysis for each city
    cities = ['CHICAGO', 'NYC', 'LA']
    for city in cities:
        print(f"\nAnalyzing data for {city}...")
        
        try:
            # Fetch crime data
            if city == 'CHICAGO':
                crime_data = data_fetcher.fetch_chicago_homicides(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            elif city == 'NYC':
                crime_data = data_fetcher.fetch_nyc_homicides(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
            else:  # LA
                crime_data = data_fetcher.fetch_la_homicides(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

            if len(crime_data) == 0:
                print(f"No crime data found for {city} in the specified date range.")
                continue

            # Analyze moon phase correlation
            try:
                moon_analysis = analyzer.analyze_crime_moon_correlation(
                    crime_data,
                    start_date,
                    end_date
                )
                # Store the analysis results
                city_data[city] = moon_analysis['daily_data']
                all_analysis_results[city] = convert_timestamps(moon_analysis)
                
                # Export individual city analysis to JSON
                output_file = os.path.join('analysis_data', f'{city.lower()}_analysis.json')
                with open(output_file, 'w') as f:
                    json.dump(convert_timestamps(moon_analysis), f, indent=2)
                print(f"Exported analysis results to: {output_file}")
                
            except Exception as e:
                print(f"Error in moon phase analysis for {city}: {str(e)}")
                continue

            # Create and save individual city visualization
            try:
                fig = analyzer.create_moon_visualization(
                    crime_data,
                    moon_analysis['daily_data'],
                    city
                )
                fig.write_html(f"{city.lower()}_moon_analysis.html")
                print(f"Created visualization file: {city.lower()}_moon_analysis.html")
            except Exception as e:
                print(f"Error creating visualization for {city}: {str(e)}")

            # Print results
            print(f"\nResults for {city}:")
            print(f"Total crimes in period: {len(crime_data)}")
            
            # Print moon phase analysis
            print("\nMoon Phase Analysis:")
            print(f"Overall correlation: {moon_analysis['correlation']:.3f}")
            print(f"Overall p-value: {moon_analysis['p_value']:.3f}")
            
            if moon_analysis['full_moon_stats']:
                stats = moon_analysis['full_moon_stats']
                print("\nFull Moon vs Non-Full Moon Comparison:")
                print(f"Full moon days: {stats['full_moon_days_count']} ({stats['full_moon_proportion']:.1%} of total)")
                print(f"Non-full moon days: {stats['non_full_moon_days_count']} ({stats['non_full_moon_proportion']:.1%} of total)")
                
                print(f"\nRaw Statistics:")
                print(f"Total crimes on full moon days: {stats['full_moon_total_crimes']}")
                print(f"Total crimes on non-full moon days: {stats['non_full_moon_total_crimes']}")
                print(f"Average crimes on full moon days: {stats['full_moon_avg_crimes']:.2f}")
                print(f"Average crimes on non-full moon days: {stats['non_full_moon_avg_crimes']:.2f}")
                
                print(f"\nNormalized Statistics (accounting for day proportion):")
                print(f"Crime rate on full moon days: {stats['full_moon_rate']:.2f} crimes/day")
                print(f"Crime rate on non-full moon days: {stats['non_full_moon_rate']:.2f} crimes/day")
                print(f"Normalized full moon crime rate: {stats['normalized_full_moon_rate']:.2f}")
                print(f"Normalized non-full moon crime rate: {stats['normalized_non_full_moon_rate']:.2f}")
                print(f"Normalized percent difference: {stats['percent_difference']:.1f}%")
                print(f"Statistical significance (t-test p-value): {stats['p_value']:.3f}")

        except Exception as e:
            print(f"Error analyzing {city}: {str(e)}")

    # Create and save cities comparison visualization
    if len(city_data) >= 2:  # If we have data for at least two cities
        try:
            comparison_fig = analyzer.create_cities_comparison(
                city_data.get('CHICAGO', pd.DataFrame()),
                city_data.get('NYC', pd.DataFrame()),
                city_data.get('LA', pd.DataFrame())
            )
            comparison_fig.write_html("cities_comparison.html")
            print("\nCreated cities comparison visualization: cities_comparison.html")
            
            # Export combined analysis results
            combined_output = {
                'cities': all_analysis_results,
                'analysis_period': {
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d')
                }
            }
            combined_output_file = os.path.join('analysis_data', 'combined_analysis.json')
            with open(combined_output_file, 'w') as f:
                json.dump(combined_output, f, indent=2)
            print(f"Exported combined analysis results to: {combined_output_file}")
            
        except Exception as e:
            print(f"\nError creating cities comparison: {str(e)}")

if __name__ == "__main__":
    main() 