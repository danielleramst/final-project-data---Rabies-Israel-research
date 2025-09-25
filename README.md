# Rabies Prediction Dashboard

A Streamlit-based dashboard for predicting and analyzing rabies outbreaks in Israel. The system combines historical rabies cases with weather patterns and war events data to predict outbreak regions and timing.

## Project Structure

```
rabies-prediction-dashboard/
├── code/                         # Streamlit Dashboard Implementation
│   ├── app/                     # Main Application Code
│   │   ├── main.py             # Dashboard Entry Point
│   │   ├── config/
│   │   │   └── config.yaml     # App Configuration
│   │   ├── pages/              # Dashboard Pages
│   │   │   ├── prediction.py   # Prediction Interface
│   │   │   └── visualization.py # Data Analysis Views
│   │   └── utils/              # Utility Modules
│   │       ├── data_loader.py  # Data Loading & Processing
│   │       ├── model_handler.py # ML Model Management
│   │       └── visualizer.py   # Visualization Functions
│   ├── data/                    # Dataset Directory
│   │   └── Rabies__Weather__War_Combined_1.4.25.xlsx
│   ├── models/                  # Trained ML Models
│   │   ├── month_model.pkl     # Month Prediction Model
│   │   ├── preprocessor.pkl    # Data Preprocessor
│   │   ├── region_model.pkl    # Region Prediction Model
│   │   └── trained_model.pkl   # Combined Models
│   ├── requirements.txt        # Python Dependencies
│   ├── run_dashboard.sh        # Launch Script
│   ├── test_fixes.py          # Testing Script
│   └── train_and_save_model.py # Model Training Script
├── reports/                     # Research Documentation
│   ├── Business Understanding.pdf
│   ├── Data Understanding.pdf
│   ├── Data Preparation.pdf
│   ├── Modeling.pdf
│   └── Evaluation.pdf
└── venv/                       # Python Virtual Environment

```

## Component Description

### Dashboard Implementation (`code/`)
- **Main Application (`app/`)**: Core dashboard implementation
  - `main.py`: Application entry point and main UI
  - `config/`: Configuration files
  - `pages/`: Separate dashboard views
  - `utils/`: Helper modules and functions

- **Data (`data/`)**: Combined dataset including:
  - Rabies outbreak records
  - Weather data (temperature, precipitation)
  - War events information
  - Geographic coordinates

- **Models (`models/`)**: Pre-trained ML models
  - Region prediction model
  - Month prediction model
  - Data preprocessor
  - Combined model package

### Documentation (`reports/`)
Comprehensive research documentation covering:
- Business Understanding
- Data Understanding
- Data Preparation
- Modeling Process
- Model Evaluation

## Setup and Installation

1. **Virtual Environment**
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/MacOS
```

2. **Install Dependencies**
```bash
cd code
pip install -r requirements.txt
```

3. **Run Dashboard**
```bash
streamlit run app/main.py
```

## Features

### Prediction Capabilities
- Region prediction
- Month prediction
- Feature selection
- Batch processing

### Data Analysis
- Temporal analysis
- Geographic distribution
- Weather impact analysis
- Species analysis
- War impact analysis

### Interactive Dashboard
- Modern Streamlit UI
- Interactive visualizations
- Feature importance insights
- Data upload support

## Technology Stack

- **Python 3.8+**
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, folium
- **Frontend**: streamlit
- **Data Format**: Excel (.xlsx)

## Dependencies

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.21.6
scikit-learn==1.3.0
plotly==5.15.0
folium==0.14.0
streamlit-folium==0.13.0
pyyaml==6.0
joblib==1.2.0
openpyxl==3.0.10
matplotlib==3.5.3
seaborn==0.11.2
```