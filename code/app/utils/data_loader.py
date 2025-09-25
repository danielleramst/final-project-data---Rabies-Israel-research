"""
Data loading and preprocessing utilities for the Rabies prediction dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Union
import streamlit as st
import joblib


class DataLoader:
    """Handles data loading, validation, and preprocessing"""
    
    def __init__(self, preprocessor_path: str = "models/preprocessor.pkl"):
        """Initialize with preprocessor"""
        try:
            self.preprocessor = joblib.load(preprocessor_path)
            self.label_encoders = self.preprocessor['label_encoders']
            self.scaler = self.preprocessor['scaler']
            self.numeric_columns = self.preprocessor['numeric_columns']
            self.categorical_columns = self.preprocessor['categorical_columns']
        except Exception as e:
            st.error(f"Error loading preprocessor: {e}")
            self.preprocessor = None
    
    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load Excel file and return DataFrame"""
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return pd.DataFrame()
    
    def load_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Load uploaded file from Streamlit file uploader"""
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload .xlsx or .csv files.")
                return pd.DataFrame()
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return pd.DataFrame()
    
    def validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate that input data has required columns"""
        required_columns = [
            'Date', 'Animal Species', 'Rabies Species', 'Region', 
            'Settlement', 'Region_Weather', 'x', 'y', 'Avg Temperature',
            'Monthly Precipitation (mm)', 'Rainy Days', 'War in Israel', 'Event Per Year'
        ]
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def process_single_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert single input form data to processed DataFrame"""
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Process the data
        processed_df = self.preprocess_data(df)
        
        return processed_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to input data"""
        if self.preprocessor is None:
            st.error("Preprocessor not loaded")
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying original
            df_processed = df.copy()
            
            # Handle Date column
            if 'Date' in df_processed.columns:
                df_processed['Date'] = pd.to_datetime(df_processed['Date'])
                df_processed['Year'] = df_processed['Date'].dt.year
                df_processed['Month'] = df_processed['Date'].dt.month
            
            # Count rabies cases per year-month (for batch data)
            if len(df_processed) > 1:
                rabies_counts = df_processed.groupby(['Year', 'Month']).size().reset_index(name='Rabies_Cases_Per_Month')
                df_processed = df_processed.merge(rabies_counts, on=['Year', 'Month'], how='left')
            else:
                # For single predictions, set a default value
                df_processed['Rabies_Cases_Per_Month'] = 1
            
            # Drop unnecessary columns if they exist
            columns_to_drop = ['Date', 'War Name', 'Index Event ID']
            for col in columns_to_drop:
                if col in df_processed.columns:
                    df_processed = df_processed.drop(columns=[col])
            
            # Handle War in Israel column
            if 'War in Israel' in df_processed.columns:
                df_processed['War in Israel'] = df_processed['War in Israel'].map({
                    'Yes': 1, 'No': 0, True: 1, False: 0, 1: 1, 0: 0
                })
            
            # Apply label encoding to categorical columns
            for col in self.categorical_columns:
                if col in df_processed.columns:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    df_processed[col] = df_processed[col].astype(str)
                    
                    # For unseen categories, assign most frequent category
                    unseen_mask = ~df_processed[col].isin(le.classes_)
                    if unseen_mask.any():
                        most_frequent = le.classes_[0]  # Use first class as default
                        df_processed.loc[unseen_mask, col] = most_frequent
                        st.warning(f"Unknown values in '{col}' column replaced with '{most_frequent}'")
                    
                    df_processed[col] = le.transform(df_processed[col])
            
            # Apply scaling to numeric columns
            numeric_cols_present = [col for col in self.numeric_columns if col in df_processed.columns]
            if numeric_cols_present:
                df_processed[numeric_cols_present] = self.scaler.transform(df_processed[numeric_cols_present])
            
            return df_processed
            
        except Exception as e:
            st.error(f"Error preprocessing data: {e}")
            return pd.DataFrame()
    
    def get_feature_options(self) -> Dict[str, list]:
        """Get available options for categorical features"""
        if self.preprocessor is None:
            return {}
        
        options = {}
        for col, encoder in self.label_encoders.items():
            options[col] = list(encoder.classes_)
        
        return options
    
    def create_input_form(self) -> Dict[str, Any]:
        """Create Streamlit input form and return the collected data"""
        st.subheader("Input New Observation")
        
        # Get feature options
        feature_options = self.get_feature_options()
        
        # Create form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                date = st.date_input("Date", value=datetime.now())
                animal_species = st.selectbox("Animal Species", 
                                            feature_options.get('Animal Species', []))
                rabies_species = st.selectbox("Rabies Species", 
                                            feature_options.get('Rabies Species', []))
                region = st.selectbox("Region", 
                                    feature_options.get('Region', []))
                settlement = st.selectbox("Settlement", 
                                        feature_options.get('Settlement', []))
            
            with col2:
                region_weather = st.selectbox("Region Weather", 
                                            feature_options.get('Region_Weather', []))
                x_coord = st.number_input("X Coordinate", value=0.0)
                y_coord = st.number_input("Y Coordinate", value=0.0)
                avg_temp = st.number_input("Average Temperature (°C)", value=20.0)
                precipitation = st.number_input("Monthly Precipitation (mm)", value=50.0)
            
            with col3:
                rainy_days = st.number_input("Rainy Days", value=10, min_value=0, max_value=31)
                war_in_israel = st.checkbox("War in Israel")
                event_per_year = st.number_input("Events Per Year", value=10, min_value=0)
            
            submitted = st.form_submit_button("Make Prediction")
            
            if submitted:
                input_data = {
                    'Date': date,
                    'Animal Species': animal_species,
                    'Rabies Species': rabies_species,
                    'Region': region,
                    'Settlement': settlement,
                    'Region_Weather': region_weather,
                    'x': x_coord,
                    'y': y_coord,
                    'Avg Temperature': avg_temp,
                    'Monthly Precipitation (mm)': precipitation,
                    'Rainy Days': rainy_days,
                    'War in Israel': war_in_israel,
                    'Event Per Year': event_per_year
                }
                return input_data
        
        return None
