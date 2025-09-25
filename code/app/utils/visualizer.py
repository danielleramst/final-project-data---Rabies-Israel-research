"""
Visualization utilities for the Rabies prediction dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Any


class Visualizer:
    """Handles data visualization for the dashboard"""
    
    def __init__(self):
        """Initialize visualizer with default styling"""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_temporal_analysis(self, df: pd.DataFrame):
        """Create temporal analysis visualizations"""
        st.subheader("Temporal Analysis")
        
        # Prepare data
        if 'Date' not in df.columns:
            st.warning("Date column not found for temporal analysis")
            return
        
        if len(df) == 0:
            st.warning("No data available for temporal analysis")
            return
        
        df_temp = df.copy()
        df_temp['Date'] = pd.to_datetime(df_temp['Date'])
        df_temp['Year'] = df_temp['Date'].dt.year
        df_temp['Month'] = df_temp['Date'].dt.month
        df_temp['MonthName'] = df_temp['Date'].dt.month_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly trend
            monthly_counts = df_temp.groupby('Month').size().reset_index(name='Count')
            
            fig = px.bar(
                monthly_counts, 
                x='Month', 
                y='Count',
                title='Rabies Cases by Month',
                labels={'Month': 'Month', 'Count': 'Number of Cases'}
            )
            fig.update_layout(xaxis=dict(tickmode='linear'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Yearly trend
            yearly_counts = df_temp.groupby('Year').size().reset_index(name='Count')
            
            fig = px.line(
                yearly_counts, 
                x='Year', 
                y='Count',
                title='Rabies Cases by Year',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap of month vs year
        st.subheader("Monthly Pattern Across Years")
        
        if len(df_temp) > 0:
            pivot_data = df_temp.groupby(['Year', 'Month']).size().unstack(fill_value=0)
            
            fig = px.imshow(
                pivot_data.T,
                labels=dict(x="Year", y="Month", color="Cases"),
                title="Rabies Cases Heatmap (Month vs Year)",
                aspect="auto"
            )
            fig.update_layout(
                yaxis=dict(tickmode='linear', tick0=1, dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_geographical_analysis(self, df: pd.DataFrame):
        """Create geographical analysis visualizations"""
        st.subheader("Geographical Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Region distribution
            if 'Region' in df.columns:
                region_counts = df['Region'].value_counts()
                
                fig = px.pie(
                    values=region_counts.values,
                    names=region_counts.index,
                    title='Distribution of Cases by Region'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Settlement distribution (top 10)
            if 'Settlement' in df.columns:
                settlement_counts = df['Settlement'].value_counts().head(10)
                
                fig = px.bar(
                    x=settlement_counts.values,
                    y=settlement_counts.index,
                    orientation='h',
                    title='Top 10 Settlements by Cases',
                    labels={'x': 'Number of Cases', 'y': 'Settlement'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot of coordinates
        if 'x' in df.columns and 'y' in df.columns:
            st.subheader("Geographical Distribution of Cases")
            
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='Region' if 'Region' in df.columns else None,
                title='Geographical Distribution of Rabies Cases',
                labels={'x': 'X Coordinate', 'y': 'Y Coordinate'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_weather_analysis(self, df: pd.DataFrame):
        """Create weather-related analysis visualizations"""
        st.subheader("Weather Impact Analysis")
        
        weather_cols = ['Avg Temperature', 'Monthly Precipitation (mm)', 'Rainy Days']
        available_cols = [col for col in weather_cols if col in df.columns]
        
        if not available_cols:
            st.warning("No weather data columns found")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature distribution
            if 'Avg Temperature' in df.columns:
                fig = px.histogram(
                    df,
                    x='Avg Temperature',
                    title='Distribution of Average Temperature',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Precipitation vs Cases
            if 'Monthly Precipitation (mm)' in df.columns:
                # Group by precipitation ranges
                df_temp = df.copy()
                df_temp['Precip_Range'] = pd.cut(
                    df_temp['Monthly Precipitation (mm)'], 
                    bins=5, 
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                )
                precip_counts = df_temp['Precip_Range'].value_counts()
                
                fig = px.bar(
                    x=precip_counts.index,
                    y=precip_counts.values,
                    title='Cases by Precipitation Level',
                    labels={'x': 'Precipitation Level', 'y': 'Number of Cases'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Weather correlation matrix
        if len(available_cols) > 1:
            st.subheader("Weather Variables Correlation")
            
            corr_matrix = df[available_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title='Weather Variables Correlation Matrix',
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )
            fig.update_layout(
                xaxis_title="Variables",
                yaxis_title="Variables"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_species_analysis(self, df: pd.DataFrame):
        """Create species-related analysis visualizations"""
        st.subheader("Species Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Animal species distribution
            if 'Animal Species' in df.columns:
                species_counts = df['Animal Species'].value_counts()
                
                fig = px.pie(
                    values=species_counts.values,
                    names=species_counts.index,
                    title='Distribution by Animal Species'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rabies species distribution
            if 'Rabies Species' in df.columns:
                rabies_species_counts = df['Rabies Species'].value_counts()
                
                fig = px.bar(
                    x=rabies_species_counts.index,
                    y=rabies_species_counts.values,
                    title='Distribution by Rabies Species',
                    labels={'x': 'Rabies Species', 'y': 'Number of Cases'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    def plot_war_impact(self, df: pd.DataFrame):
        """Create war impact analysis visualizations"""
        st.subheader("War Impact Analysis")
        
        if 'War in Israel' not in df.columns:
            st.warning("War data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cases during war vs peace
            war_counts = df['War in Israel'].value_counts()
            war_labels = ['No War', 'War'] if 0 in war_counts.index else war_counts.index
            
            fig = px.pie(
                values=war_counts.values,
                names=war_labels,
                title='Cases During War vs Peace Time'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Events per year distribution
            if 'Event Per Year' in df.columns:
                fig = px.histogram(
                    df,
                    x='Event Per Year',
                    title='Distribution of Events Per Year',
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def plot_feature_importance(self, importance_dict: Dict[str, float], title: str = "Feature Importance"):
        """Plot feature importance"""
        st.subheader(title)
        
        # Convert to DataFrame and sort
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        # Take top 10 features
        importance_df = importance_df.tail(10)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=title,
            labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_prediction_comparison(self, region_importance: Dict[str, float], 
                                 month_importance: Dict[str, float]):
        """Compare feature importance between region and month predictions"""
        st.subheader("Feature Importance Comparison: Region vs Month")
        
        # Get common features
        all_features = set(region_importance.keys()) | set(month_importance.keys())
        
        comparison_data = []
        for feature in all_features:
            comparison_data.append({
                'Feature': feature,
                'Region_Importance': region_importance.get(feature, 0),
                'Month_Importance': month_importance.get(feature, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Region_Importance', ascending=True).tail(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=comparison_df['Feature'],
            x=comparison_df['Region_Importance'],
            name='Region',
            orientation='h',
            marker_color='orange'
        ))
        
        fig.add_trace(go.Bar(
            y=comparison_df['Feature'],
            x=comparison_df['Month_Importance'],
            name='Month',
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig.update_layout(
            title='Feature Importance Comparison: Region vs Month',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_summary_statistics(self, df: pd.DataFrame):
        """Create summary statistics table"""
        st.subheader("Dataset Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Overview:**")
            st.metric("Total Records", len(df))
            
            if 'Date' in df.columns:
                df_temp = df.copy()
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                date_range = f"{df_temp['Date'].min().strftime('%Y-%m-%d')} to {df_temp['Date'].max().strftime('%Y-%m-%d')}"
                st.markdown(f"**Date Range:** {date_range}")
            
            if 'Region' in df.columns:
                st.metric("Unique Regions", df['Region'].nunique())
            
            if 'Settlement' in df.columns:
                st.metric("Unique Settlements", df['Settlement'].nunique())
        
        with col2:
            # Numeric columns summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.markdown("**Numeric Variables Summary:**")
                st.dataframe(df[numeric_cols].describe().round(2))
    
    def plot_comprehensive_dashboard(self, df: pd.DataFrame):
        """Create a comprehensive visualization dashboard"""
        
        # Summary statistics
        self.create_summary_statistics(df)
        
        # Temporal analysis
        self.plot_temporal_analysis(df)
        
        # Geographical analysis
        self.plot_geographical_analysis(df)
        
        # Weather analysis
        self.plot_weather_analysis(df)
        
        # Species analysis
        self.plot_species_analysis(df)
        
        # War impact analysis
        self.plot_war_impact(df)
