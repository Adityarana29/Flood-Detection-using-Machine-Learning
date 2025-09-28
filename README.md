🌊 Flood Detection & Weather Risk Dashboard with ML Forecasting

🚨 AI + Earth Observation for Real-Time Flood Monitoring in India

This project is an interactive dashboard that integrates:

🌐 Google Earth Engine (GEE) for Sentinel-1 SAR flood detection

💧 JRC Global Surface Water dataset for permanent water removal

🌦 Open-Meteo API for real-time weather risks

🤖 Machine Learning (2014–2024) for 2025 flood forecasting

📊 Streamlit dashboard with live maps, district-level risk, and charts

✨ Features

✅ Flood Detection

Uses Sentinel-1 SAR (VV backscatter) before/after flood events

Masks permanent water bodies with JRC GSW dataset

Generates flood extent maps

✅ Weather Risk Mapping

Fetches real-time weather for selected districts via Open-Meteo API

Computes 3-day rainfall accumulation

Categorizes risk as High / Medium / Low

Displays district risk in maps and bar charts

✅ Machine Learning Prediction (2014–2024 → 2025)

Trains Random Forest classifier on historical features:

SAR backscatter (VV/VH)

Rainfall (CHIRPS/ERA5)

Temperature (ERA5/Open-Meteo)

Produces:

📈 Performance metrics (classification report, confusion matrix)

🔎 Feature importance plots

🔮 Forecast for 2025 flood-prone districts

✅ Interactive Streamlit Dashboard

Sidebar controls for dates, thresholds, states

Tabs for Flood Map | Weather Risk Map | ML Prediction

Dynamic plots with Plotly, Pydeck, and Geemap

🗂️ Data Sources

🛰️ Sentinel-1 SAR (VV/VH) – COPERNICUS/S1_GRD (Google Earth Engine)

🌊 JRC Global Surface Water – JRC/GSW1_4/GlobalSurfaceWater (GEE)

🌧 CHIRPS Rainfall – UCSB-CHG/CHIRPS/DAILY (1981–present, GEE)

🌍 ERA5 Climate Reanalysis – ECMWF/ERA5/DAILY (GEE)


Usage

Select States/UTs from sidebar (J&K, Himachal, Uttarakhand, NE states, etc.)

Set Pre/Post Flood Dates to compare SAR backscatter

Adjust thresholds (VV dB, JRC permanent water %)

Run Analysis →

Flood Map Tab → visualizes flood extent

Weather Risk Tab → shows district-level risk + bar chart

ML Prediction Tab → trains model (2014–2024) & forecasts 2025 risks

📊 Example Outputs

🗺 Flood Map → Sentinel-1 SAR derived flood extent

🌦 Weather Risk Map → District risk (High/Medium/Low)

📉 Risk Bar Chart → Number of districts at risk by state

🔮 ML Forecast → Predicted flood-prone districts for 2025

🧠 Machine Learning Workflow

Extract features (2014–2024):

SAR backscatter (VV/VH change)

Rainfall accumulation

Temperature anomalies

Train Random Forest classifier

Validate with classification report + confusion matrix

Forecast 2025 flood-prone districts

🔮 Future Work

Replace simulated ML training with real Sentinel-1 + CHIRPS/ERA5 features

Extend to pixel-level flood risk prediction with ConvLSTM/UNet

Add real-time flood alert API integration

Deploy on cloud platforms (AWS/GCP) for scalability

🏗️ Tech Stack

Languages: Python

Libraries: Streamlit, Earth Engine API, Geemap, Pydeck, Plotly, Sklearn, Seaborn, Pandas, Matplotlib

Data: Sentinel-1 SAR, CHIRPS Rainfall, ERA5, JRC GSW, Open-Meteo

👨‍💻 Author

Developed by Aditya Rana
M.Tech Thesis Project – Terrain-Based Route Planning & Flood Risk AI

⚡ Bridging satellite data, weather intelligence, and AI for disaster resilience.
⚡ Open-Meteo API – Real-time temperature, wind, and rainfall forecasts

🗺 FAO GAUL Boundaries – District/state boundaries for India
