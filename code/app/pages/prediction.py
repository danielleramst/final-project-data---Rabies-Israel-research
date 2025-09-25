"""
Prediction page for the Rabies Dashboard
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from utils.model_handler import ModelHandler


def show_prediction_page():
    """Display the prediction page"""
    
    st.header("🔮 Rabies Outbreak Prediction")
    st.markdown("Make predictions for rabies outbreaks using our trained machine learning models.")
    
    # Initialize components
    try:
        data_loader = DataLoader()
        model_handler = ModelHandler()
        
        if data_loader.preprocessor is None or model_handler.models is None:
            st.error("System not properly initialized. Please check the models and preprocessor.")
            return
            
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return
    
    # Prediction mode selection
    prediction_mode = st.radio(
        "Choose prediction mode:",
        ["Single Observation", "Upload File", "Use Existing Data"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if prediction_mode == "Single Observation":
        show_single_prediction(data_loader, model_handler)
    
    elif prediction_mode == "Upload File":
        show_file_prediction(data_loader, model_handler)
    
    elif prediction_mode == "Use Existing Data":
        show_existing_data_prediction(data_loader, model_handler)


def show_single_prediction(data_loader: DataLoader, model_handler: ModelHandler):
    """Show single observation prediction interface"""
    
    st.subheader("Single Observation Prediction")
    st.markdown("Enter the details of a new observation to get predictions.")
    
    # Feature selection option
    st.markdown("### Feature Selection (Optional)")
    
    available_features = model_handler.get_available_features()
    if available_features:
        use_feature_selection = st.checkbox("🎯 Use Feature Selection", help="Select specific features to use for prediction")
        
        selected_features = None
        if use_feature_selection:
            selected_features = st.multiselect(
                "Select features to use:",
                available_features,
                default=available_features[:5],  # Default to first 5 features
                help="Choose which features to include in the prediction. Fewer features may be faster but less accurate."
            )
            
            if selected_features:
                st.success(f"Using {len(selected_features)} selected features: {', '.join(selected_features)}")
            else:
                st.warning("Please select at least one feature.")
                return
    else:
        selected_features = None
        st.info("Feature selection not available - using all features.")
    
    st.markdown("---")
    
    # Get input from form
    input_data = data_loader.create_input_form()
    
    if input_data is not None:
        st.markdown("---")
        
        # Process the input
        try:
            processed_data = data_loader.process_single_input(input_data)
            
            if processed_data.empty:
                st.error("Error processing input data")
                return
            
            # Make predictions with selected features
            with st.spinner("Making predictions..."):
                results = model_handler.predict_both(processed_data, selected_features)
            
            # Display results
            model_handler.display_prediction_results(results)
            
            # Show which features were used
            if selected_features:
                st.markdown("### Features Used in Prediction")
                st.write(f"**Selected Features ({len(selected_features)}):** {', '.join(selected_features)}")
            
            # Show processed input data
            with st.expander("View Processed Input Data"):
                st.dataframe(processed_data)
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def show_file_prediction(data_loader: DataLoader, model_handler: ModelHandler):
    """Show file upload prediction interface"""
    
    st.subheader("File Upload Prediction")
    st.markdown("Upload an Excel or CSV file to make batch predictions.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['xlsx', 'csv'],
        help="Upload an Excel (.xlsx) or CSV file with the required columns"
    )
    
    if uploaded_file is not None:
        try:
            # Load file
            df = data_loader.load_uploaded_file(uploaded_file)
            
            if df.empty:
                return
            
            # Show file info
            st.success(f"File loaded successfully! Shape: {df.shape}")
            
            # Validate data
            if not data_loader.validate_input_data(df):
                return
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Prediction target selection
            target = st.selectbox(
                "Select prediction target:",
                ["both", "region", "month"],
                help="Choose what to predict: both region and month, only region, or only month"
            )
            
            # Make predictions button
            if st.button("Make Batch Predictions", type="primary"):
                with st.spinner("Processing data and making predictions..."):
                    # Preprocess data
                    processed_df = data_loader.preprocess_data(df)
                    
                    if processed_df.empty:
                        st.error("Error preprocessing data")
                        return
                    
                    # Make predictions
                    results_df = model_handler.batch_predict(processed_df, target)
                    
                    if results_df.empty:
                        st.error("Error making predictions")
                        return
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv,
                        file_name="rabies_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")


def show_existing_data_prediction(data_loader: DataLoader, model_handler: ModelHandler):
    """Show existing data prediction interface"""
    
    st.subheader("Use Existing Dataset")
    st.markdown("Make predictions using the existing rabies dataset.")
    
    try:
        # Load existing data
        df = data_loader.load_excel_file("data/Rabies__Weather__War_Combined_1.4.25.xlsx")
        
        if df.empty:
            st.error("Could not load existing dataset")
            return
        
        st.success(f"Existing dataset loaded! Shape: {df.shape}")
        
        # Show data info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            if 'Region' in df.columns:
                st.metric("Unique Regions", df['Region'].nunique())
        
        with col3:
            if 'Settlement' in df.columns:
                st.metric("Unique Settlements", df['Settlement'].nunique())
        
        # Sample size selection
        sample_size = st.slider(
            "Select sample size for prediction (for demo purposes):",
            min_value=1,
            max_value=min(100, len(df)),
            value=min(10, len(df)),
            help="Select how many records to use for prediction demonstration"
        )
        
        # Sample the data
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # Show sample
        st.subheader("Sample Data")
        st.dataframe(sample_df)
        
        # Prediction target selection
        target = st.selectbox(
            "Select prediction target:",
            ["both", "region", "month"],
            help="Choose what to predict"
        )
        
        # Make predictions
        if st.button("Make Predictions on Sample", type="primary"):
            with st.spinner("Making predictions..."):
                try:
                    # Preprocess sample
                    processed_sample = data_loader.preprocess_data(sample_df)
                    
                    if processed_sample.empty:
                        st.error("Error preprocessing sample data")
                        return
                    
                    # Make predictions
                    results_df = model_handler.batch_predict(processed_sample, target)
                    
                    if results_df.empty:
                        st.error("Error making predictions")
                        return
                    
                    # Combine original and predictions
                    combined_results = sample_df.reset_index(drop=True)
                    for col in results_df.columns:
                        if col != 'index':
                            combined_results[col] = results_df[col]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(combined_results)
                    
                    # Download option
                    csv = combined_results.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="sample_predictions.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error in prediction: {e}")
                    
    except Exception as e:
        st.error(f"Error loading existing data: {e}")
