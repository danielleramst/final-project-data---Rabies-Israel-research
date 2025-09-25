"""
Visualization page for the Rabies Dashboard
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from utils.model_handler import ModelHandler
from utils.visualizer import Visualizer


def show_visualization_page():
    """Display the visualization page"""
    
    st.header("📊 Data Visualization & Analysis")
    st.markdown("Explore patterns and insights in the rabies outbreak data.")
    
    # Initialize components
    try:
        data_loader = DataLoader()
        model_handler = ModelHandler()
        visualizer = Visualizer()
        
    except Exception as e:
        st.error(f"Error initializing visualization components: {e}")
        return
    
    # Data source selection
    data_source = st.radio(
        "Choose data source for visualization:",
        ["Existing Dataset", "Upload Custom File"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if data_source == "Existing Dataset":
        show_existing_data_visualization(data_loader, model_handler, visualizer)
    
    elif data_source == "Upload Custom File":
        show_custom_data_visualization(data_loader, model_handler, visualizer)


def show_existing_data_visualization(data_loader: DataLoader, model_handler: ModelHandler, visualizer: Visualizer):
    """Show visualization using existing dataset"""
    
    try:
        # Load existing data
        df = data_loader.load_excel_file("data/Rabies__Weather__War_Combined_1.4.25.xlsx")
        
        if df.empty:
            st.error("Could not load existing dataset")
            return
        
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Create visualization tabs
        viz_tabs = st.tabs([
            "📈 Overview", 
            "⏰ Temporal Analysis", 
            "🌍 Geographic Analysis", 
            "🌡️ Weather Analysis",
            "🦎 Species Analysis",
            "⚔️ War Impact",
            "🤖 Model Insights"
        ])
        
        with viz_tabs[0]:  # Overview
            try:
                st.subheader("Dataset Overview")
                visualizer.create_summary_statistics(df)
                
                # Basic data exploration
                st.subheader("Data Sample")
                st.dataframe(df.head(10))
                
                # Column information
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info)
            except Exception as e:
                st.error(f"Error in overview: {e}")
        
        with viz_tabs[1]:  # Temporal Analysis
            try:
                visualizer.plot_temporal_analysis(df)
            except Exception as e:
                st.error(f"Error in temporal analysis: {e}")
        
        with viz_tabs[2]:  # Geographic Analysis
            try:
                visualizer.plot_geographical_analysis(df)
            except Exception as e:
                st.error(f"Error in geographic analysis: {e}")
        
        with viz_tabs[3]:  # Weather Analysis
            try:
                visualizer.plot_weather_analysis(df)
            except Exception as e:
                st.error(f"Error in weather analysis: {e}")
        
        with viz_tabs[4]:  # Species Analysis
            try:
                visualizer.plot_species_analysis(df)
            except Exception as e:
                st.error(f"Error in species analysis: {e}")
        
        with viz_tabs[5]:  # War Impact
            try:
                visualizer.plot_war_impact(df)
            except Exception as e:
                st.error(f"Error in war impact analysis: {e}")
        
        with viz_tabs[6]:  # Model Insights
            try:
                show_model_insights(df, data_loader, model_handler, visualizer)
            except Exception as e:
                st.error(f"Error in model insights: {e}")
            
    except Exception as e:
        st.error(f"Error in visualization: {e}")


def show_custom_data_visualization(data_loader: DataLoader, model_handler: ModelHandler, visualizer: Visualizer):
    """Show visualization for uploaded custom file"""
    
    st.subheader("Upload Custom Data for Visualization")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'csv'],
        help="Upload an Excel (.xlsx) or CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Load file
            df = data_loader.load_uploaded_file(uploaded_file)
            
            if df.empty:
                return
            
            st.success(f"File loaded successfully! Shape: {df.shape}")
            
            # Show available visualizations based on columns
            available_cols = df.columns.tolist()
            
            st.subheader("Available Visualizations")
            st.markdown(f"**Available columns:** {', '.join(available_cols)}")
            
            # Basic visualizations that work with any data
            viz_options = st.multiselect(
                "Select visualizations to display:",
                [
                    "Data Summary",
                    "Column Distributions", 
                    "Correlation Matrix",
                    "Missing Data Analysis"
                ],
                default=["Data Summary"]
            )
            
            if "Data Summary" in viz_options:
                st.subheader("Data Summary")
                visualizer.create_summary_statistics(df)
            
            if "Column Distributions" in viz_options:
                st.subheader("Column Distributions")
                show_column_distributions(df)
            
            if "Correlation Matrix" in viz_options:
                st.subheader("Correlation Matrix")
                show_correlation_matrix(df)
            
            if "Missing Data Analysis" in viz_options:
                st.subheader("Missing Data Analysis")
                show_missing_data_analysis(df)
            
            # Try to show domain-specific visualizations if columns match
            if all(col in available_cols for col in ['Date', 'Region']):
                st.markdown("---")
                st.subheader("Domain-Specific Visualizations")
                st.info("Your data appears to contain rabies-related columns. Showing specialized visualizations.")
                
                # Use the comprehensive dashboard
                visualizer.plot_comprehensive_dashboard(df)
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


def show_model_insights(df: pd.DataFrame, data_loader: DataLoader, model_handler: ModelHandler, visualizer: Visualizer):
    """Show model-related insights and feature importance"""
    
    st.subheader("🤖 Model Insights")
    
    if model_handler.models is None:
        st.error("Models not loaded")
        return
    
    # Model information
    model_info = model_handler.get_model_info()
    
    if "error" not in model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Region Prediction Model")
            st.markdown(f"**Model Type:** {model_info['region_model']['type']}")
            st.markdown(f"**Number of Classes:** {model_info['region_model']['n_classes']}")
            st.markdown(f"**Features Used:** {model_info['region_model']['n_features']}")
            
            # Show classes
            st.markdown("**Predicted Classes:**")
            for i, cls in enumerate(model_info['region_model']['classes']):
                st.markdown(f"- {cls}")
        
        with col2:
            st.markdown("### Month Prediction Model")
            st.markdown(f"**Model Type:** {model_info['month_model']['type']}")
            st.markdown(f"**Number of Classes:** {model_info['month_model']['n_classes']}")
            st.markdown(f"**Features Used:** {model_info['month_model']['n_features']}")
            
            # Show classes
            st.markdown("**Predicted Classes:**")
            for i, cls in enumerate(model_info['month_model']['classes']):
                st.markdown(f"- {cls}")
    
    # Feature importance analysis
    st.markdown("---")
    st.subheader("Feature Importance Analysis")
    
    try:
        # Get a sample for feature importance calculation
        sample_df = df.sample(n=min(10, len(df)), random_state=42)
        processed_sample = data_loader.preprocess_data(sample_df)
        
        if not processed_sample.empty:
            # Get feature importance for both models
            region_result = model_handler.predict_region(processed_sample.iloc[[0]])
            month_result = model_handler.predict_month(processed_sample.iloc[[0]])
            
            if "error" not in region_result and "error" not in month_result:
                # Show individual feature importance
                col1, col2 = st.columns(2)
                
                with col1:
                    visualizer.plot_feature_importance(
                        region_result["feature_importance"], 
                        "Region Prediction - Feature Importance"
                    )
                
                with col2:
                    visualizer.plot_feature_importance(
                        month_result["feature_importance"], 
                        "Month Prediction - Feature Importance"
                    )
                
                # Show comparison
                visualizer.plot_prediction_comparison(
                    region_result["feature_importance"],
                    month_result["feature_importance"]
                )
            
    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
    
    # Prediction distribution analysis
    st.markdown("---")
    st.subheader("Prediction Distribution Analysis")
    
    if st.button("Analyze Prediction Patterns", type="secondary"):
        with st.spinner("Analyzing prediction patterns..."):
            try:
                # Sample more data for analysis
                sample_size = min(50, len(df))
                sample_df = df.sample(n=sample_size, random_state=42)
                processed_sample = data_loader.preprocess_data(sample_df)
                
                if not processed_sample.empty:
                    # Get batch predictions
                    results_df = model_handler.batch_predict(processed_sample, "both")
                    
                    if not results_df.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Region Prediction Distribution")
                            if 'predicted_region' in results_df.columns:
                                region_dist = results_df['predicted_region'].value_counts()
                                st.bar_chart(region_dist)
                        
                        with col2:
                            st.markdown("#### Month Prediction Distribution")
                            if 'predicted_month' in results_df.columns:
                                month_dist = results_df['predicted_month'].value_counts()
                                st.bar_chart(month_dist)
                        
                        # Confidence analysis
                        st.markdown("#### Prediction Confidence Analysis")
                        if 'region_confidence' in results_df.columns:
                            st.markdown("**Region Prediction Confidence:**")
                            st.write(f"Mean: {results_df['region_confidence'].mean():.3f}")
                            st.write(f"Std: {results_df['region_confidence'].std():.3f}")
                            
                        if 'month_confidence' in results_df.columns:
                            st.markdown("**Month Prediction Confidence:**")
                            st.write(f"Mean: {results_df['month_confidence'].mean():.3f}")
                            st.write(f"Std: {results_df['month_confidence'].std():.3f}")
                
            except Exception as e:
                st.error(f"Error in prediction analysis: {e}")


def show_column_distributions(df: pd.DataFrame):
    """Show distribution plots for numeric columns"""
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for distribution analysis")
        return
    
    selected_cols = st.multiselect(
        "Select columns to visualize:",
        numeric_cols.tolist(),
        default=numeric_cols.tolist()[:3] if len(numeric_cols) >= 3 else numeric_cols.tolist()
    )
    
    for col in selected_cols:
        st.markdown(f"#### {col}")
        st.histogram_chart(df[col].dropna())


def show_correlation_matrix(df: pd.DataFrame):
    """Show correlation matrix for numeric columns"""
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis")
        return
    
    import plotly.express as px
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Correlation Matrix',
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_missing_data_analysis(df: pd.DataFrame):
    """Show missing data analysis"""
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    }).sort_values('Missing Count', ascending=False)
    
    # Filter only columns with missing data
    missing_df = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_df) == 0:
        st.success("✅ No missing data found in the dataset!")
    else:
        st.dataframe(missing_df, hide_index=True)
        
        # Visualization
        import plotly.express as px
        
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Percentage',
            title='Missing Data by Column'
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
