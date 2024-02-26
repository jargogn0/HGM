import streamlit as st
import leafmap.foliumap as leafmap
import matplotlib.dates as mdates
import geopandas as gpd
import folium
from datetime import timedelta

st.set_page_config(layout="wide")

# Customize the sidebar
markdown = """
Forest Monitor App: <https://say2byjargon-20ee8e68f2dd.herokuapp.com/>
GitHub Repository: <https://github.com/giswqs/streamlit-multipage-template>
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://i.imgur.com/UbOXYAU.png"
st.sidebar.image(logo)

# Customize page title
st.title("HGM: Towards Fevelopment of a Hydrological General Model") 
shapefile_path = 'geo/E04_Vodomerne_stanice.shp'
gdf = gpd.read_file(shapefile_path)

# Ensure the GeoDataFrame is in WGS84 (latitude and longitude)
if gdf.crs != 'epsg:4326':
    gdf = gdf.to_crs(epsg=4326)

# Calculate longitude and latitude from geometry
gdf['lon'] = gdf.geometry.x
gdf['lat'] = gdf.geometry.y

# Initialize a Folium map centered on the data
# Use the full width of the Streamlit page by not specifying width in folium.Map()
m = folium.Map(location=[gdf['lat'].mean(), gdf['lon'].mean()], zoom_start=7, control_scale=True)

# Add points to the map with tooltip for labels
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=5,  # Adjust the size as necessary
        color='blue',  # Dot color
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        tooltip=f"Catchment: {row['DBC']}"  # Tooltip content
    ).add_to(m)

# Render the Folium map in Streamlit using full page width
st.components.v1.html(m._repr_html_(), height=700, scrolling=False)
 

st.markdown(
    """
    This multipage app template demonstrates various interactive web apps created using ). It is an open-source project and you are very welcome to contribute to the [GitHub repository](https://github.com/giswqs/streamlit-multipage-template).
    """
)

st.header("Instructions")

markdown = """
1. For the [GitHub repository](https://github.com/codingpanda19) 

"""

st.markdown(markdown)

m = leafmap.Map(minimap_control=True)
m.add_basemap("OpenTopoMap")
m.to_streamlit(height=500)

