# 🦠 Rabies Prediction System

A machine learning-based dashboard for predicting and analyzing rabies outbreaks in Israel. The system uses historical data combining rabies cases, weather patterns, and war events to predict outbreak regions and timing.

## 🎯 Features

### Prediction Capabilities
- **Region Prediction**: Predict geographical areas at risk for rabies outbreaks
- **Month Prediction**: Forecast temporal patterns of outbreaks
- **Feature Selection**: Choose specific features for prediction
- **Batch Processing**: Support for both single and multiple predictions

### Data Analysis
- **Temporal Analysis**: Monthly and yearly trends
- **Geographic Distribution**: Regional outbreak patterns
- **Weather Impact**: Temperature and precipitation effects
- **Species Analysis**: Animal and rabies species distribution
- **War Impact**: Analysis of conflict effects on outbreaks

### Interactive Dashboard
- **Modern UI**: Built with Streamlit
- **Interactive Visualizations**: Using Plotly and Folium
- **Feature Importance**: Understanding model decisions
- **Data Upload**: Support for new data input

## 🛠️ Technology Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn (GradientBoostingClassifier)
- **Data Processing**: pandas, numpy
- **Visualization**: plotly, folium, matplotlib, seaborn
- **Frontend**: streamlit
- **Data Format**: Excel (.xlsx)

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rabies-prediction.git
cd rabies-prediction
```

2. **Create virtual environment**
```bash
python -m venv rabies_dashboard_venv
source rabies_dashboard_venv/bin/activate  # On Windows: rabies_dashboard_venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Running the Dashboard

1. **Quick Start**
```bash
./run_dashboard.sh
```

2. **Manual Start**
```bash
source rabies_dashboard_venv/bin/activate
streamlit run app/main.py
```

### Making Predictions

1. **Single Prediction**
   - Navigate to "🔮 Prediction" tab
   - Choose "Single Observation"
   - Optionally select specific features
   - Fill in the observation details
   - Click "Make Prediction"

2. **Batch Prediction**
   - Choose "Upload File"
   - Upload an Excel file with required columns
   - Select prediction target (Region/Month/Both)
   - View and download results

3. **Feature Selection**
   - Enable "🎯 Use Feature Selection"
   - Choose desired features from the list
   - Default: top 5 most important features

### Analyzing Data

Navigate through visualization tabs:
- 📈 Overview
- ⏰ Temporal Analysis
- 🌍 Geographic Analysis
- 🌡️ Weather Analysis
- 🦎 Species Analysis
- ⚔️ War Impact
- 🤖 Model Insights

## 📁 Project Structure

```
rabies_dashboard/
├── app/
│   ├── main.py               # Main Streamlit application
│   ├── pages/
│   │   ├── prediction.py     # Prediction interface
│   │   └── visualization.py  # Data visualization
│   ├── utils/
│   │   ├── data_loader.py   # Data loading and preprocessing
│   │   ├── model_handler.py # Model loading and prediction
│   │   └── visualizer.py    # Visualization functions
│   └── config/
│       └── config.yaml      # Configuration settings
├── data/
│   └── Rabies__Weather__War_Combined_1.4.25.xlsx
├── models/                  # Trained ML models
├── requirements.txt
└── README.md
```

## 📊 Data Description

The system uses a combined dataset including:
- Rabies outbreak records
- Weather data (temperature, precipitation)
- War events in Israel
- Geographic coordinates
- Animal and rabies species information

### Features
- Animal Species
- Rabies Species
- Region
- Settlement
- Weather conditions
- Geographic coordinates (x, y)
- Temperature and precipitation
- War events
- Temporal features

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sources: Veterinary Services, Israel Meteorological Service
- Original research team: Daniel and Iman
- Project supervision: [Supervisor Name]