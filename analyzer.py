import pandas as pd
import numpy as np
from datetime import datetime
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        
        # Mark full moon days (phase > 0.95)
        daily_crimes['is_full_moon'] = daily_crimes['moon_phase'] > 0.95
        
        # Calculate statistics for full moon vs non-full moon days
        full_moon_days = daily_crimes[daily_crimes['is_full_moon']]
        non_full_moon_days = daily_crimes[~daily_crimes['is_full_moon']]
        
        if len(full_moon_days) > 0 and len(non_full_moon_days) > 0:
            from scipy import stats
            
            # Calculate total crimes and days for each group
            full_moon_total_crimes = full_moon_days['count'].sum()
            non_full_moon_total_crimes = non_full_moon_days['count'].sum()
            full_moon_days_count = len(full_moon_days)
            non_full_moon_days_count = len(non_full_moon_days)
            
            # Calculate average crimes per day for each group
            full_moon_avg = full_moon_days['count'].mean()
            non_full_moon_avg = non_full_moon_days['count'].mean()
            
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
            except:
                t_stat, p_value = 0, 1.0
            
            full_moon_stats = {
                'full_moon_days_count': full_moon_days_count,
                'non_full_moon_days_count': non_full_moon_days_count,
                'full_moon_total_crimes': full_moon_total_crimes,
                'non_full_moon_total_crimes': non_full_moon_total_crimes,
                'full_moon_avg_crimes': full_moon_avg,
                'non_full_moon_avg_crimes': non_full_moon_avg,
                'full_moon_rate': full_moon_rate,
                'non_full_moon_rate': non_full_moon_rate,
                'full_moon_proportion': full_moon_proportion,
                'non_full_moon_proportion': non_full_moon_proportion,
                'normalized_full_moon_rate': normalized_full_moon_rate,
                'normalized_non_full_moon_rate': normalized_non_full_moon_rate,
                'percent_difference': percent_diff,
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
            except:
                print("Error calculating moon phase correlation")
                correlation = (0, 1.0)
        else:
            print("Insufficient data points for correlation analysis")
            correlation = (0, 1.0)
        
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

    def create_cities_comparison(self, chicago_data, nyc_data):
        """Create a comparative visualization between Chicago and NYC moon phase analysis"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Chicago Crime Rate vs Moon Phase',
                'NYC Crime Rate vs Moon Phase',
                'Normalized Crime Rates Comparison',
                'Crime Rate Distribution Comparison',
                'Monthly Pattern Comparison - Chicago',
                'Monthly Pattern Comparison - NYC'
            ),
            specs=[
                [{"secondary_y": True}, {"secondary_y": True}],
                [{"colspan": 2}, None],
                [{"type": "polar"}, {"type": "polar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Helper function to process city data
        def process_city_data(moon_data):
            if not isinstance(moon_data, pd.DataFrame) or len(moon_data) == 0:
                return None
            
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

        # Process data for both cities
        chicago_processed = process_city_data(chicago_data)
        nyc_processed = process_city_data(nyc_data)

        if chicago_processed and nyc_processed:
            # 1. Time series plots for both cities
            cities_data = [
                (chicago_processed['data'], 1, 'Chicago'),
                (nyc_processed['data'], 2, 'NYC')
            ]

            for city_data, col, city_name in cities_data:
                # Main normalized crime rate line
                fig.add_trace(
                    go.Scatter(
                        x=city_data['date'],
                        y=city_data['normalized_rate'],
                        name=f'{city_name} Daily Rate',
                        line=dict(color='blue', width=1),
                        opacity=0.6
                    ),
                    row=1, col=col,
                    secondary_y=False
                )
                
                # Rolling average line
                fig.add_trace(
                    go.Scatter(
                        x=city_data['date'],
                        y=city_data['rolling_avg_norm'],
                        name=f'{city_name} 7-day Avg',
                        line=dict(color='darkblue', width=2)
                    ),
                    row=1, col=col,
                    secondary_y=False
                )
                
                # Moon phases
                fig.add_trace(
                    go.Scatter(
                        x=city_data['date'],
                        y=city_data['moon_phase'],
                        name='Moon Phase',
                        line=dict(color='orange', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(255, 165, 0, 0.1)',
                        showlegend=col==1  # Only show in legend once
                    ),
                    row=1, col=col,
                    secondary_y=True
                )

            # 2. Bar chart comparing normalized rates between cities
            cities_comparison = pd.DataFrame([
                {
                    'City': 'Chicago',
                    'Period': 'Full Moon',
                    'Rate': chicago_processed['normalized_rates']['full_moon']
                },
                {
                    'City': 'Chicago',
                    'Period': 'Non-Full Moon',
                    'Rate': chicago_processed['normalized_rates']['non_full_moon']
                },
                {
                    'City': 'NYC',
                    'Period': 'Full Moon',
                    'Rate': nyc_processed['normalized_rates']['full_moon']
                },
                {
                    'City': 'NYC',
                    'Period': 'Non-Full Moon',
                    'Rate': nyc_processed['normalized_rates']['non_full_moon']
                }
            ])

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
            for city_data, col, city_name in [(chicago_processed, 1, 'Chicago'), (nyc_processed, 2, 'NYC')]:
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
                    row=3, col=col
                )

            # Update layout
            fig.update_layout(
                height=1200,
                title=dict(
                    text="Chicago vs NYC Moon Phase Analysis Comparison (2022)",
                    x=0.5,
                    font=dict(size=24)
                ),
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                # Update polar plot settings
                polar=dict(
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
                ),
                polar2=dict(
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
            )

            # Update axes
            for col in [1, 2]:
                fig.update_yaxes(title_text="Normalized Crime Rate", secondary_y=False, row=1, col=col)
                fig.update_yaxes(title_text="Moon Phase", secondary_y=True, row=1, col=col)
                fig.update_xaxes(title_text="Date", row=1, col=col)

            fig.update_yaxes(title_text="Normalized Crime Rate", row=2, col=1)
            fig.update_xaxes(title_text="City", row=2, col=1)

        return fig 