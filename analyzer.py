import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

class CrimeAnalyzer:
    def __init__(self, moon_calculator):
        self.moon_calculator = moon_calculator
        # Set fixed random seed for reproducibility
        np.random.seed(42)
        
    def analyze_crime_moon_correlation(self, crime_data, start_date, end_date):
        """Analyze correlation between crime frequency and moon phases, with specific focus on full moon days"""
        if len(crime_data) == 0:
            print("No crime data available for moon phase analysis")
            return {
                'correlation': 0,
                'p_value': 1.0,
                'daily_data': pd.DataFrame(columns=['date', 'count', 'moon_phase', 'is_full_moon']),
                'full_moon_stats': None
            }

        # Group crimes by date and count
        daily_crimes = crime_data.groupby('date').size().reset_index(name='count')
        
        # Add moon phase information
        daily_crimes['moon_phase'] = daily_crimes['date'].apply(
            self.moon_calculator.get_moon_phase
        )
        
        # Convert numeric columns to native Python types
        daily_crimes['count'] = daily_crimes['count'].astype(float)
        daily_crimes['moon_phase'] = daily_crimes['moon_phase'].astype(float)
        
        # Mark full moon days (phase > 0.95)
        daily_crimes['is_full_moon'] = daily_crimes['moon_phase'] > 0.95
        
        # Calculate statistics for full moon vs non-full moon days
        full_moon_days = daily_crimes[daily_crimes['is_full_moon']]
        non_full_moon_days = daily_crimes[~daily_crimes['is_full_moon']]
        
        if len(full_moon_days) > 0 and len(non_full_moon_days) > 0:
            from scipy import stats
            
            # Calculate total crimes and days for each group
            full_moon_total_crimes = float(full_moon_days['count'].sum())
            non_full_moon_total_crimes = float(non_full_moon_days['count'].sum())
            full_moon_days_count = int(len(full_moon_days))
            non_full_moon_days_count = int(len(non_full_moon_days))
            
            # Calculate average crimes per day for each group
            full_moon_avg = float(full_moon_days['count'].mean())
            non_full_moon_avg = float(non_full_moon_days['count'].mean())
            
            # Calculate normalized rates (crimes per day)
            full_moon_rate = full_moon_total_crimes / full_moon_days_count if full_moon_days_count > 0 else 0
            non_full_moon_rate = non_full_moon_total_crimes / non_full_moon_days_count if non_full_moon_days_count > 0 else 0
            
            # Calculate proportion of full moon days to total days
            total_days = full_moon_days_count + non_full_moon_days_count
            full_moon_proportion = full_moon_days_count / total_days if total_days > 0 else 0
            non_full_moon_proportion = non_full_moon_days_count / total_days if total_days > 0 else 0
            
            # Calculate normalized crime rates accounting for day proportion
            normalized_full_moon_rate = full_moon_rate / full_moon_proportion if full_moon_proportion > 0 else 0
            normalized_non_full_moon_rate = non_full_moon_rate / non_full_moon_proportion if non_full_moon_proportion > 0 else 0
            
            # Calculate percent difference safely using normalized rates
            try:
                if normalized_non_full_moon_rate != 0:
                    percent_diff = ((normalized_full_moon_rate - normalized_non_full_moon_rate) / normalized_non_full_moon_rate) * 100
                else:
                    percent_diff = 0 if normalized_full_moon_rate == 0 else float('inf')
            except:
                percent_diff = 0
            
            # Perform t-test to check if difference is statistically significant
            try:
                # Normalize the crime counts by the proportion of days before t-test
                normalized_full_moon_counts = full_moon_days['count'] / full_moon_proportion
                normalized_non_full_moon_counts = non_full_moon_days['count'] / non_full_moon_proportion
                
                t_stat, p_value = stats.ttest_ind(
                    normalized_full_moon_counts,
                    normalized_non_full_moon_counts
                )
                t_stat = float(t_stat)
                p_value = float(p_value)
            except:
                t_stat, p_value = 0.0, 1.0
            
            full_moon_stats = {
                'full_moon_days_count': full_moon_days_count,
                'non_full_moon_days_count': non_full_moon_days_count,
                'full_moon_total_crimes': full_moon_total_crimes,
                'non_full_moon_total_crimes': non_full_moon_total_crimes,
                'full_moon_avg_crimes': full_moon_avg,
                'non_full_moon_avg_crimes': non_full_moon_avg,
                'full_moon_rate': float(full_moon_rate),
                'non_full_moon_rate': float(non_full_moon_rate),
                'full_moon_proportion': float(full_moon_proportion),
                'non_full_moon_proportion': float(non_full_moon_proportion),
                'normalized_full_moon_rate': float(normalized_full_moon_rate),
                'normalized_non_full_moon_rate': float(normalized_non_full_moon_rate),
                'percent_difference': float(percent_diff),
                't_statistic': t_stat,
                'p_value': p_value
            }
        else:
            full_moon_stats = None
            print("Insufficient data to compare full moon vs non-full moon days")

        # Calculate overall correlation if enough data points
        if len(daily_crimes) >= 2:
            try:
                correlation = stats.pearsonr(
                    daily_crimes['moon_phase'],
                    daily_crimes['count']
                )
                correlation = (float(correlation[0]), float(correlation[1]))
            except:
                print("Error calculating moon phase correlation")
                correlation = (0.0, 1.0)
        else:
            print("Insufficient data points for correlation analysis")
            correlation = (0.0, 1.0)
        
        # Convert boolean column to native Python bool
        daily_crimes['is_full_moon'] = daily_crimes['is_full_moon'].astype(bool)
        
        return {
            'correlation': correlation[0],
            'p_value': correlation[1],
            'daily_data': daily_crimes,
            'full_moon_stats': full_moon_stats
        }
        
    def create_moon_visualization(self, crime_data, moon_data, city):
        """Create comprehensive visualization of moon phase analysis with normalized data"""
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{city} Crime Frequency and Moon Phases (2022)',
                'Normalized Crime Rates: Full Moon vs Non-Full Moon',
                'Crime Rate Distribution (Normalized)'
            ),
            specs=[[{"secondary_y": True}], [{}], [{}]],
            vertical_spacing=0.15,
            row_heights=[0.4, 0.3, 0.3]
        )

        # 1. Crime frequency over time with moon phases
        if isinstance(moon_data, pd.DataFrame) and len(moon_data) > 0:
            # Calculate normalized daily crime rates
            total_days = len(moon_data)
            moon_data['normalized_rate'] = moon_data['count'] / (moon_data['is_full_moon'].mean() if moon_data['is_full_moon'].any() else 1)
            moon_data['rolling_avg_norm'] = moon_data['normalized_rate'].rolling(window=7, center=True).mean()
            
            # Main normalized crime rate line
            fig.add_trace(
                go.Scatter(
                    x=moon_data['date'],
                    y=moon_data['normalized_rate'],
                    name='Daily Normalized Crime Rate',
                    line=dict(color='blue', width=1),
                    opacity=0.6
                ),
                row=1, col=1,
                secondary_y=False
            )
            
            # Rolling average line
            fig.add_trace(
                go.Scatter(
                    x=moon_data['date'],
                    y=moon_data['rolling_avg_norm'],
                    name='7-day Moving Average (Normalized)',
                    line=dict(color='darkblue', width=2)
                ),
                row=1, col=1,
                secondary_y=False
            )
            
            # Moon phases
            fig.add_trace(
                go.Scatter(
                    x=moon_data['date'],
                    y=moon_data['moon_phase'],
                    name='Moon Phase',
                    line=dict(color='orange', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(255, 165, 0, 0.1)'
                ),
                row=1, col=1,
                secondary_y=True
            )
            
            # Highlight full moon days with vertical lines
            full_moon_days = moon_data[moon_data['is_full_moon']]
            for date in full_moon_days['date']:
                fig.add_vline(
                    x=date,
                    line_width=1,
                    line_dash="dash",
                    line_color="rgba(255, 165, 0, 0.5)",
                    row=1
                )

        # 2. Bar chart comparing normalized rates
        if isinstance(moon_data, pd.DataFrame) and 'is_full_moon' in moon_data.columns:
            full_moon_stats = {
                'full_moon_days': len(moon_data[moon_data['is_full_moon']]),
                'non_full_moon_days': len(moon_data[~moon_data['is_full_moon']]),
                'full_moon_total': moon_data[moon_data['is_full_moon']]['count'].sum(),
                'non_full_moon_total': moon_data[~moon_data['is_full_moon']]['count'].sum()
            }
            
            # Calculate normalized rates
            full_moon_prop = full_moon_stats['full_moon_days'] / len(moon_data)
            non_full_moon_prop = full_moon_stats['non_full_moon_days'] / len(moon_data)
            
            normalized_full_moon = (full_moon_stats['full_moon_total'] / full_moon_stats['full_moon_days']) / full_moon_prop if full_moon_prop > 0 else 0
            normalized_non_full_moon = (full_moon_stats['non_full_moon_total'] / full_moon_stats['non_full_moon_days']) / non_full_moon_prop if non_full_moon_prop > 0 else 0
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=['Full Moon Days', 'Non-Full Moon Days'],
                    y=[normalized_full_moon, normalized_non_full_moon],
                    name='Normalized Crime Rate',
                    marker_color=['orange', 'blue'],
                    text=[f'{normalized_full_moon:.2f}', f'{normalized_non_full_moon:.2f}'],
                    textposition='auto',
                ),
                row=2, col=1
            )

        # 3. Violin plot of crime rate distribution
        if isinstance(moon_data, pd.DataFrame) and 'is_full_moon' in moon_data.columns:
            # Calculate normalized rates for each day
            moon_data['norm_rate'] = moon_data['count'] / moon_data['is_full_moon'].map({True: full_moon_prop, False: non_full_moon_prop})
            
            fig.add_trace(
                go.Violin(
                    x=['Full Moon']*len(moon_data[moon_data['is_full_moon']]),
                    y=moon_data[moon_data['is_full_moon']]['norm_rate'],
                    name='Full Moon',
                    box_visible=True,
                    meanline_visible=True,
                    line_color='orange',
                    fillcolor='rgba(255, 165, 0, 0.3)'
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Violin(
                    x=['Non-Full Moon']*len(moon_data[~moon_data['is_full_moon']]),
                    y=moon_data[~moon_data['is_full_moon']]['norm_rate'],
                    name='Non-Full Moon',
                    box_visible=True,
                    meanline_visible=True,
                    line_color='blue',
                    fillcolor='rgba(0, 0, 255, 0.3)'
                ),
                row=3, col=1
            )

        # Update layout and styling
        fig.update_layout(
            height=1200,
            title=dict(
                text=f"{city} Moon Phase Analysis Dashboard (2022)",
                x=0.5,
                font=dict(size=24)
            ),
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )

        # Update axes styling
        for i in range(1, 4):
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i,
                col=1
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                row=i,
                col=1
            )

        # Update specific axes titles
        fig.update_yaxes(title_text="Normalized Crime Rate", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Moon Phase", secondary_y=True, row=1, col=1)
        fig.update_yaxes(title_text="Normalized Crime Rate", row=2, col=1)
        fig.update_yaxes(title_text="Normalized Crime Rate", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Period", row=2, col=1)
        fig.update_xaxes(title_text="Period", row=3, col=1)

        return fig 

    def create_cities_comparison(self, chicago_data, nyc_data, la_data=None):
        """Create a comparative visualization between Chicago, NYC, and LA moon phase analysis"""
        # Determine number of cities with data
        cities_data = []
        if not chicago_data.empty:
            cities_data.append(('Chicago', chicago_data))
        if not nyc_data.empty:
            cities_data.append(('NYC', nyc_data))
        if la_data is not None and not la_data.empty:
            cities_data.append(('LA', la_data))
        
        num_cities = len(cities_data)
        if num_cities < 2:
            raise ValueError("Need at least two cities with data for comparison")

        # Create subplots layout
        fig = make_subplots(
            rows=3, cols=num_cities,
            subplot_titles=(
                [f'{city} Crime Rate vs Moon Phase' for city, _ in cities_data] +
                ['Normalized Crime Rates Comparison'] * num_cities +
                [f'Monthly Pattern Comparison - {city}' for city, _ in cities_data]
            ),
            specs=[
                [{"secondary_y": True} for _ in range(num_cities)],
                [{"colspan": num_cities}] + [None] * (num_cities - 1),
                [{"type": "polar"} for _ in range(num_cities)]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Helper function to process city data
        def process_city_data(moon_data):
            if not isinstance(moon_data, pd.DataFrame) or len(moon_data) == 0:
                return None
            
            # Convert date strings back to datetime if needed
            if isinstance(moon_data['date'].iloc[0], str):
                moon_data['date'] = pd.to_datetime(moon_data['date'])
            
            # Calculate normalized rates
            moon_data['normalized_rate'] = moon_data['count'] / (moon_data['is_full_moon'].mean() if moon_data['is_full_moon'].any() else 1)
            moon_data['rolling_avg_norm'] = moon_data['normalized_rate'].rolling(window=7, center=True).mean()
            
            # Calculate monthly averages
            moon_data['month'] = moon_data['date'].dt.month
            monthly_stats = moon_data.groupby('month').agg({
                'normalized_rate': 'mean',
                'moon_phase': 'mean'
            }).reset_index()
            
            # Calculate stats for full moon vs non-full moon
            full_moon_stats = {
                'full_moon_days': len(moon_data[moon_data['is_full_moon']]),
                'non_full_moon_days': len(moon_data[~moon_data['is_full_moon']]),
                'full_moon_total': moon_data[moon_data['is_full_moon']]['count'].sum(),
                'non_full_moon_total': moon_data[~moon_data['is_full_moon']]['count'].sum()
            }
            
            full_moon_prop = full_moon_stats['full_moon_days'] / len(moon_data)
            non_full_moon_prop = full_moon_stats['non_full_moon_days'] / len(moon_data)
            
            normalized_full_moon = (full_moon_stats['full_moon_total'] / full_moon_stats['full_moon_days']) / full_moon_prop if full_moon_prop > 0 else 0
            normalized_non_full_moon = (full_moon_stats['non_full_moon_total'] / full_moon_stats['non_full_moon_days']) / non_full_moon_prop if non_full_moon_prop > 0 else 0
            
            return {
                'data': moon_data,
                'monthly_stats': monthly_stats,
                'normalized_rates': {
                    'full_moon': normalized_full_moon,
                    'non_full_moon': normalized_non_full_moon
                }
            }

        # Process data for all cities
        processed_data = {}
        for city_name, city_data in cities_data:
            processed = process_city_data(city_data)
            if processed:
                processed_data[city_name] = processed

        if processed_data:
            # 1. Time series plots for each city
            for idx, (city_name, city_data) in enumerate(processed_data.items(), 1):
                # Main normalized crime rate line
                fig.add_trace(
                    go.Scatter(
                        x=city_data['data']['date'],
                        y=city_data['data']['normalized_rate'],
                        name=f'{city_name} Daily Rate',
                        line=dict(color='blue', width=1),
                        opacity=0.6
                    ),
                    row=1, col=idx,
                    secondary_y=False
                )
                
                # Rolling average line
                fig.add_trace(
                    go.Scatter(
                        x=city_data['data']['date'],
                        y=city_data['data']['rolling_avg_norm'],
                        name=f'{city_name} 7-day Avg',
                        line=dict(color='darkblue', width=2)
                    ),
                    row=1, col=idx,
                    secondary_y=False
                )
                
                # Moon phases
                fig.add_trace(
                    go.Scatter(
                        x=city_data['data']['date'],
                        y=city_data['data']['moon_phase'],
                        name='Moon Phase',
                        line=dict(color='orange', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255, 165, 0, 0.1)',
                        showlegend=idx==1  # Only show in legend once
                    ),
                    row=1, col=idx,
                    secondary_y=True
                )

            # 2. Bar chart comparing normalized rates between cities
            cities_comparison = []
            for city_name, city_data in processed_data.items():
                cities_comparison.extend([
                    {
                        'City': city_name,
                        'Period': 'Full Moon',
                        'Rate': city_data['normalized_rates']['full_moon']
                    },
                    {
                        'City': city_name,
                        'Period': 'Non-Full Moon',
                        'Rate': city_data['normalized_rates']['non_full_moon']
                    }
                ])

            cities_comparison = pd.DataFrame(cities_comparison)

            # Create grouped bar chart
            for period, color in [('Full Moon', 'orange'), ('Non-Full Moon', 'blue')]:
                fig.add_trace(
                    go.Bar(
                        x=cities_comparison[cities_comparison['Period'] == period]['City'],
                        y=cities_comparison[cities_comparison['Period'] == period]['Rate'],
                        name=period,
                        marker_color=color,
                        text=cities_comparison[cities_comparison['Period'] == period]['Rate'].round(2),
                        textposition='auto',
                        offsetgroup=period
                    ),
                    row=2, col=1
                )

            # 3. Monthly pattern comparison (polar plots)
            for idx, (city_name, city_data) in enumerate(processed_data.items(), 1):
                monthly_stats = city_data['monthly_stats']
                
                # Convert month numbers to angles (in radians)
                angles = (monthly_stats['month'] - 1) * (2 * np.pi / 12)
                
                # Close the circle by appending the first value to the end
                angles = np.append(angles, angles[0])
                values = np.append(monthly_stats['normalized_rate'], monthly_stats['normalized_rate'].iloc[0])
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=angles * 180 / np.pi,  # Convert to degrees
                        name=f'{city_name} Monthly Pattern',
                        line_color='blue'
                    ),
                    row=3, col=idx
                )

            # Update layout
            fig.update_layout(
                height=1200,
                title=dict(
                    text="Multi-City Moon Phase Analysis Comparison (2022)",
                    x=0.5,
                    font=dict(size=24)
                ),
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1
            )

            # Update polar plot settings for each city
            for i in range(1, num_cities + 1):
                polar_key = f'polar{i if i > 1 else ""}'
                fig.update_layout(**{
                    polar_key: dict(
                        radialaxis=dict(
                            tickfont_size=10,
                            title='Normalized Rate'
                        ),
                        angularaxis=dict(
                            tickfont_size=10,
                            direction='clockwise',
                            period=12,
                            tickmode='array',
                            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                            tickvals=list(range(0, 360, 30))
                        )
                    )
                })

            # Update axes
            for i in range(1, num_cities + 1):
                fig.update_yaxes(title_text="Normalized Crime Rate", secondary_y=False, row=1, col=i)
                fig.update_yaxes(title_text="Moon Phase", secondary_y=True, row=1, col=i)
                fig.update_xaxes(title_text="Date", row=1, col=i)

            fig.update_yaxes(title_text="Normalized Crime Rate", row=2, col=1)
            fig.update_xaxes(title_text="City", row=2, col=1)

        return fig

    def export_analysis_to_json(self, analysis_results, output_file):
        """
        Export the analysis results to a JSON file.
        
        Args:
            analysis_results (dict): The results from analyze_crime_moon_correlation
            output_file (str): Path to the output JSON file
        """
        # Convert DataFrame to dict for JSON serialization
        if 'daily_data' in analysis_results and isinstance(analysis_results['daily_data'], pd.DataFrame):
            analysis_results['daily_data'] = analysis_results['daily_data'].apply(
                lambda x: x.strftime('%Y-%m-%d') if isinstance(x, datetime) else x
            )
            analysis_results['daily_data'] = analysis_results['daily_data'].to_dict(orient='records')
        
        # Convert numpy types to Python native types
        def convert_to_native_types(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                              np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native_types(item) for item in obj]
            return obj
        
        # Convert all numpy types to native Python types
        serializable_results = convert_to_native_types(analysis_results)
        
        # Export to JSON
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2) 