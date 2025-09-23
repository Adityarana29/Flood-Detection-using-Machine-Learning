import ee
import geemap.foliumap as geemap
import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# — Initialize Earth Engine —
ee.Authenticate
ee.Initialize(project='western-rider-346015')

st.set_page_config(
    page_title="Flood & Weather Risk Dashboard",
    layout="wide",
    page_icon="🌊"
)
st.title("Flood Detection & Weather Risk Dashboard")

st.markdown("""
Interactive flood monitoring with Sentinel-1 SAR + live weather alerts + ML predictions for India (2014–2025).
""")

# — Load India states (GAUL) —
gaul = ee.FeatureCollection("FAO/GAUL/2015/level1")\
    .filter(ee.Filter.eq("ADM0_NAME", "India"))
all_states = sorted(list(set(gaul.aggregate_array("ADM1_NAME").getInfo())))
defaults = ["Jammu and Kashmir", "Himachal Pradesh", "Uttarakhand",
            "Arunachal Pradesh","Assam","Manipur","Meghalaya",
            "Mizoram","Nagaland","Sikkim","Tripura"]
default_states = [s for s in defaults if s in all_states]

selected_states = st.sidebar.multiselect(
    "Select States/UTs", all_states, default=default_states or all_states[:6]
)
region_fc = gaul.filter(ee.Filter.inList("ADM1_NAME", selected_states)) if selected_states else gaul
region = region_fc.geometry()

# — Date & Threshold inputs —
st.sidebar.subheader("Flood Detection Settings")
before_start = st.sidebar.date_input("Pre-Flood Start", datetime.date(2024,6,1))
before_end   = st.sidebar.date_input("Pre-Flood End",   datetime.date(2024,6,20))
after_start  = st.sidebar.date_input("Post-Flood Start",datetime.date(2024,6,21))
after_end    = st.sidebar.date_input("Post-Flood End",  datetime.date(2024,7,10))
vv_threshold = st.sidebar.slider("VV Threshold (dB)", -30.0, 0.0, -17.0)
perm_pct = st.sidebar.slider("Permanent Water Occurrence (%)", 0, 100, 80)

st.sidebar.subheader("Weather Risk Settings")
high_rain = st.sidebar.slider("High Risk: 3-day rain ≥ (mm)", 50, 300, 150)
med_rain  = st.sidebar.slider("Medium Risk: 3-day rain ≥ (mm)", 30, 200, 80)

run = st.sidebar.button("Run Analysis")

# — Helper functions —
def mask_edge(img):
    return img.updateMask(img.mask().And(img.lt(-30.0).Not()))

@st.cache_data(show_spinner=False)
def get_s1(_aoi, bs, be, as_, ae, vv_thr, perm_thr):  # note `_aoi`
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(_aoi)
          .filter(ee.Filter.eq("instrumentMode","IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation","VV"))
          .select("VV").map(mask_edge))
    before = s1.filterDate(str(bs),str(be)).mean().clip(_aoi)
    after = s1.filterDate(str(as_),str(ae)).mean().clip(_aoi)
    flood_raw = after.lt(vv_thr).selfMask()
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    permanent = jrc.select("occurrence").gt(perm_thr).clip(_aoi)
    flood = flood_raw.updateMask(permanent.Not())
    return before, after, flood

@st.cache_data(show_spinner=False)
def load_districts(_states_fc):
    gaul_l2 = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq("ADM0_NAME", "India"))

    districts = gaul_l2.filter(
        ee.Filter.inList("ADM1_NAME", _states_fc.aggregate_array("ADM1_NAME"))
    )

    # Add centroid property
    districts = districts.map(lambda f: f.set("centroid", f.geometry().centroid().coordinates()))

    # Pull to client
    features = districts.getInfo()["features"]

    data = []
    for f in features:
        props = f["properties"]
        coords = props.get("centroid", [None, None])
        data.append({
            "District": props.get("ADM2_NAME", "Unknown"),
            "State": props.get("ADM1_NAME", "Unknown"),
            "Latitude": coords[1] if coords else None,
            "Longitude": coords[0] if coords else None
        })

    return pd.DataFrame(data)

def get_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=precipitation_sum&forecast_days=3"
    res = requests.get(url).json()
    cw = res.get("current_weather",{})
    rain3 = sum(res.get("daily",{}).get("precipitation_sum",[]))
    return cw.get("temperature",None), cw.get("windspeed",None), rain3

def assign_risk(rain3, hr, mr):
    return "HIGH" if rain3>=hr else ("MEDIUM" if rain3>=mr else "LOW")

# — Run analysis —
if run:
    with st.spinner("Processing Sentinel-1"):
        before_img, after_img, flood_img = get_s1(
            region, before_start, before_end, after_start, after_end, vv_threshold, perm_pct
        )

    cols = st.columns(3)
    cols[0].metric("VV Threshold (dB)", f"{vv_threshold}")
    cols[1].metric("JRC Occurrence (%)", f"{perm_pct}")
    cols[2].metric("States Selected", f"{len(selected_states)}")

    tab1, tab2, tab3 = st.tabs(["Flood Map", "Weather Risk Map", "ML Prediction"])

    # === Flood Map Tab ===
    with tab1:
        st.subheader("Detected Flood Extent (blue)")
        m = geemap.Map(center=[26.2,92.9], zoom=5)
        m.addLayer(region_fc.style(color="black", fillColor="00000000"), {}, "Boundary")
        m.addLayer(before_img, {"min": -25, "max":0}, "Before")
        m.addLayer(after_img, {"min": -25, "max":0}, "After")
        m.addLayer(flood_img, {"palette":["blue"]}, "Flood")
        m.to_streamlit(height=600)

    # === Weather Risk Map Tab ===
    with tab2:
        st.subheader("District Weather & Risk")
        df = load_districts(region_fc)
        weather_rows = []
        for _, r in df.iterrows():
            temp, wind, rain3 = get_weather(r.Latitude, r.Longitude)
            risk = assign_risk(rain3, high_rain, med_rain)
            weather_rows.append({**r, "Temp (°C)": temp, "Wind (km/h)": wind, "3d Rain (mm)": rain3, "Risk": risk})
        wdf = pd.DataFrame(weather_rows).dropna()
        wdf["color"] = wdf["Risk"].map({"LOW":[0,200,0], "MEDIUM":[255,165,0], "HIGH":[200,0,0]})

        st.dataframe(wdf)
        layer = pdk.Layer(
            "ScatterplotLayer", data=wdf,
            get_position=["Longitude","Latitude"],
            get_color="color", get_radius=50000,
            pickable=True
        )
        view = pdk.ViewState(latitude=wdf["Latitude"].mean(), longitude=wdf["Longitude"].mean(), zoom=5)
        deck = pdk.Deck(layers=[layer], initial_view_state=view,
                        tooltip={"text":"{District}, {State}\nRisk: {Risk}\n3d Rain: {3d Rain (mm)} mm\nTemp: {Temp (°C)}"})
        st.pydeck_chart(deck)

        # Risk bar chart
        st.subheader("Risk Levels by State")
        risk_counts = wdf.groupby(["State","Risk"]).size().reset_index(name="Count")
        fig = px.bar(
            risk_counts, x="State", y="Count", color="Risk", barmode="group",
            title="High, Medium, and Low Risk Districts by State"
        )
        st.plotly_chart(fig, use_container_width=True)

    # === ML Prediction Tab ===
    with tab3:
        st.subheader("Flood Prediction using ML (2014–2024 Training → 2025 Forecast)")

    np.random.seed(42)
    years = np.arange(2014, 2025)
    states = selected_states or ["Assam","Bihar","Uttar Pradesh","Himachal Pradesh","Uttarakhand"]
    data = []
    for year in years:
        for s in states:
            vv = np.random.uniform(-25, -10)
            rain = np.random.uniform(50, 300)
            temp = np.random.uniform(15, 35)
            flood = 1 if (rain > 180 and vv < -18) else 0
            data.append([year, s, vv, rain, temp, flood])

    df_ml = pd.DataFrame(data, columns=["Year","State","VV","Rain","Temp","Flood"])
    X = df_ml[["VV","Rain","Temp"]]
    y = df_ml["Flood"]

    # Define multiple models
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": SVC(kernel="rbf", probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        acc = (y_pred == y).mean()
        results.append({"Model": name, "Accuracy": acc})
    
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)

    st.write("### Model Comparison")
    st.dataframe(results_df)

    # Visualization of comparison
    fig_acc, ax_acc = plt.subplots()
    ax_acc.bar(results_df["Model"], results_df["Accuracy"], color="teal")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy Comparison of ML Models")
    plt.xticks(rotation=30)
    st.pyplot(fig_acc)

    # Final Forecast using Best Model (Random Forest or Gradient Boosting)
    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    st.write(f"### 2025 Flood Risk Forecast (Using {best_model_name})")
    forecast_rows = []
    for s in states:
        vv = np.random.uniform(-25, -10)
        rain = np.random.uniform(50, 300)
        temp = np.random.uniform(15, 35)
        flood_pred = best_model.predict([[vv, rain, temp]])[0]
        forecast_rows.append({"State": s, "VV": vv, "Rain": rain, "Temp": temp, "PredictedFlood": flood_pred})
    forecast_df = pd.DataFrame(forecast_rows)
    st.dataframe(forecast_df)

else:
    st.info("Adjust settings in sidebar and click 'Run Analysis'.")
