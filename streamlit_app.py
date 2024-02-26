import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# Sidebar content
markdown = "GitHub Repository: <https://github.com/codingpanda19>"
st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://i.imgur.com/UbOXYAU.png"
st.sidebar.image(logo)

# Navigation
pages = {
    "Home": "Home.py",
    "HGM": "1_📉_HGM.py",
    "Rainfall Runoff Module": "2_💦_RainfallRunoffModule.py",
    "Surface Water Module": "3_🛰️_SurfaceWaterModule.py",
    "Baseflow Module": "4_🎛️_BaseflowModule.py",
    "Flood Forecasting Module": "5_🌋_FloodForecastingModule.py",
    "Interactive Map": "6_🌍_Interactive_Map.py",
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Page title
st.title(f"{selection}")

# Dynamic page loading based on selection
page = pages[selection]

with open(page, "r", encoding="utf-8") as file:
    page_code = file.read()

exec(page_code, globals())
