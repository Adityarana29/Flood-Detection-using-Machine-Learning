import ee
import json
import streamlit as st

@st.cache_resource
def init_ee():
    service_account_info = dict(st.secrets["gee"])

    credentials = ee.ServiceAccountCredentials(
        service_account_info["client_email"],
        key_data=json.dumps(service_account_info)
    )
    ee.Initialize(credentials)

init_ee()

def get_region(selected_states):
    gaul = (
        ee.FeatureCollection("FAO/GAUL/2015/level1")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
    )

    fc = gaul.filter(ee.Filter.inList("ADM1_NAME", selected_states))
    return fc, fc.geometry()

def sentinel1_flood(aoi, start, end, vv_thr):
    s1 = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .select("VV")
    )

    mean_img = s1.mean().clip(aoi)
    flood = mean_img.lt(vv_thr).selfMask()

    return mean_img, flood

def jrc_mask(aoi, occ=80):
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    return jrc.select("occurrence").gt(occ).clip(aoi)

import geemap
import pandas as pd

@st.cache_data(ttl=3600)
def extract_ml_dataset(aoi, flood_img):
    """
    Extract flood + auxiliary features directly from GEE
    """

    # District boundaries
    districts = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM0_NAME", "India"))
        .filterBounds(aoi)
    )

    # Weather proxy (ERA5 rainfall)
    rain = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .select("total_precipitation_sum")
        .filterDate("2024-06-01", "2024-06-10")
        .sum()
        .clip(aoi)
    )

    # JRC Water
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
        .select("occurrence").clip(aoi)

    # Stack features
    stack = flood_img.rename("flood") \
        .addBands(rain.rename("rain_3d")) \
        .addBands(water.rename("water_occurrence"))

    # Reduce to districts
    stats = stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=500
    )

    # Convert to pandas (ONE SERVER CALL)
    df = geemap.ee_to_df(stats)

    # Cleanup
    df = df.dropna()
    df["flood_label"] = (df["flood"] > 0).astype(int)

    return df

import geemap.foliumap as geemap

with st.tab("🌍 Flood Map (GEE)"):

    states = st.multiselect(
        "Select States",
        [
            "Jammu and Kashmir","Himachal Pradesh","Uttarakhand",
            "Assam","Bihar","Uttar Pradesh",
            "Arunachal Pradesh","Meghalaya","Manipur",
            "Mizoram","Nagaland","Tripura","Sikkim"
        ],
        default=["Assam","Uttarakhand"]
    )

    if states and st.button("Run Flood Detection"):
        fc, region = get_region(states)

        vv_thr = st.slider("VV Threshold (dB)", -30.0, 0.0, -17.0)

        backscatter, flood = sentinel1_flood(
            region, "2024-06-01", "2024-07-10", vv_thr
        )

        perm = jrc_mask(region)

        m = geemap.Map(center=[26, 92], zoom=5)
        m.addLayer(fc.style(color="black", fillColor="00000000"), {}, "States")
        m.addLayer(backscatter, {"min": -25, "max": 0}, "Sentinel-1 VV")
        m.addLayer(flood.updateMask(perm.Not()), {"palette": ["blue"]}, "Flood")
        m.addLayer(perm, {"palette": ["cyan"]}, "Permanent Water")

        m.to_streamlit(height=600)

with st.tab("🤖 ML Prediction"):

    if st.button("Extract Data & Run ML"):
        with st.spinner("Extracting features from GEE..."):
            df = extract_ml_dataset(region, flood)

        from ML_integration import run_ml_pipeline

        df_pred, model, metrics = run_ml_pipeline(df)

        st.metric("Accuracy", round(metrics["accuracy"], 3))
        st.metric("Recall", round(metrics["recall"], 3))

        st.dataframe(df_pred.head(50))

        st.download_button(
            "Download Dataset",
            df_pred.to_csv(index=False),
            "flood_predictions.csv",
            "text/csv"
        )
