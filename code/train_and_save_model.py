"""
Script to train and save the rabies prediction model
Extracted from the evaluation code to create reusable model artifacts
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def load_and_prepare_data(path):
    """Load and preprocess the data"""
    df = pd.read_excel(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Count rabies cases per year-month
    rabies_counts = df.groupby(['Year', 'Month']).size().reset_index(name='Rabies_Cases_Per_Month')
    df = df.merge(rabies_counts, on=['Year', 'Month'], how='left')

    # Drop unnecessary columns
    df = df.drop(columns=['Date', 'War Name', 'Index Event ID'])
    df['War in Israel'] = df['War in Israel'].map({'Yes': 1, 'No': 0})

    return df

def create_preprocessors(df):
    """Create and fit preprocessors for categorical and numeric data"""
    
    # Categorical columns
    label_cols = ['Animal Species', 'Rabies Species', 'Region', 'Settlement', 'Region_Weather', 'Month']
    label_encoders = {}
    
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Scale numeric columns
    num_cols = ['x', 'y', 'Avg Temperature', 'Monthly Precipitation (mm)',
                'Rainy Days', 'Rabies_Cases_Per_Month', 'Event Per Year']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    preprocessors = {
        'label_encoders': label_encoders,
        'scaler': scaler,
        'numeric_columns': num_cols,
        'categorical_columns': label_cols
    }
    
    return df, preprocessors

def train_models(df):
    """Train models for both Region and Month prediction"""
    models = {}
    
    # Train Region prediction model
    X_region = df.drop(columns=['Region'])
    y_region = df['Region']
    
    region_model = GradientBoostingClassifier(random_state=42)
    region_model.fit(X_region, y_region)
    models['region'] = region_model
    models['region_features'] = list(X_region.columns)  # Save feature names and order
    
    # Train Month prediction model
    X_month = df.drop(columns=['Month'])
    y_month = df['Month']
    
    month_model = GradientBoostingClassifier(random_state=42)
    month_model.fit(X_month, y_month)
    models['month'] = month_model
    models['month_features'] = list(X_month.columns)  # Save feature names and order
    
    return models

def main():
    """Main function to train and save models"""
    print("Loading and preprocessing data...")
    
    # Load data
    file_path = "data/Rabies__Weather__War_Combined_1.4.25.xlsx"
    df = load_and_prepare_data(file_path)
    
    # Create preprocessors
    df_processed, preprocessors = create_preprocessors(df)
    
    print("Training models...")
    
    # Train models
    models = train_models(df_processed)
    
    # Save preprocessors
    joblib.dump(preprocessors, 'models/preprocessor.pkl')
    print("✓ Preprocessors saved to models/preprocessor.pkl")
    
    # Save models
    joblib.dump(models['region'], 'models/region_model.pkl')
    print("✓ Region model saved to models/region_model.pkl")
    
    joblib.dump(models['month'], 'models/month_model.pkl')
    print("✓ Month model saved to models/month_model.pkl")
    
    # Save both models in one file for convenience - WITH FEATURE NAMES
    all_models = {
        'region_model': models['region'],
        'month_model': models['month'],
        'preprocessor': preprocessors,
        'region_features': models['region_features'],  # Include feature names
        'month_features': models['month_features']      # Include feature names
    }
    joblib.dump(all_models, 'models/trained_model.pkl')
    print("✓ All models saved to models/trained_model.pkl")
    
    print("\nModel training and saving completed successfully!")
    print(f"Data shape: {df.shape}")
    print(f"Region model classes: {len(models['region'].classes_)}")
    print(f"Month model classes: {len(models['month'].classes_)}")
    print(f"Region features ({len(models['region_features'])}): {models['region_features']}")
    print(f"Month features ({len(models['month_features'])}): {models['month_features']}")

if __name__ == "__main__":
    main()
