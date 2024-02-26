import streamlit as st
import leafmap.foliumap as leafmap
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# Customize the sidebar
markdown = """
GitHub Repository: <https://github.com/codingpanda19>
"""

st.sidebar.title("About")
st.sidebar.info(markdown)
logo = "https://i.imgur.com/UbOXYAU.png"
st.sidebar.image(logo)

# Customize page title
st.title("HGM: Towards Fevelopment of a Hydrological General Model")


st.title("Forest Monitor App")

# Specify the URL of the website you want to embed
website_url = "https://say2byjargon-20ee8e68f2dd.herokuapp.com/"

# Embed the website within an iframe in your Streamlit app
components.iframe(website_url, height=1000)
