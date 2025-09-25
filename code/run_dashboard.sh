#!/bin/bash

# Rabies Dashboard Launcher Script
echo "🦠 Starting Rabies Prediction Dashboard..."

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
echo "Activating virtual environment..."
source rabies_dashboard_venv/bin/activate

# Check if models exist
if [ ! -f "models/trained_model.pkl" ]; then
    echo "⚠️  Models not found. Training models first..."
    python train_and_save_model.py
fi

# Launch Streamlit app
echo "🚀 Launching dashboard..."
streamlit run app/main.py

echo "Dashboard stopped."
