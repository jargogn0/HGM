import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from catboost import CatBoostRegressor
import base64
import calendar
import warnings
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
import time  # Import time module
from tqdm import tqdm  # Import tqdm for progress bar
import matplotlib.animation as animation
from IPython.display import HTML
import matplotlib.dates as mdates
import geopandas as gpd
import folium
from datetime import timedelta

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@st.cache_data
def plot_data_for_basin_and_year(df, selected_basin, selected_year):
    # Filter the DataFrame for the selected basin and year
    filtered_df = df[(df['Basin'] == selected_basin) & (df['Date'].dt.year == selected_year)]

    # Check if the filtered dataframe is empty
    if filtered_df.empty:
        st.write("No data available for the selected basin and year.")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Assuming 'Q' is a flow rate column you want to plot
    if 'Q' in filtered_df.columns:
        ax.plot(filtered_df['Date'], filtered_df['Q'], label='Flow Rate (Q)')
    
    # Additional parameters can be added here based on availability
    # Example for 'P' (precipitation)
    if 'P' in filtered_df.columns:
        ax.plot(filtered_df['Date'], filtered_df['P'], label='Precipitation (P)')

    # Formatting the plot
    ax.set_title(f"Data for {selected_basin} Basin in {selected_year}")
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Display the plot
    st.pyplot(fig)


@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.interpolate(method='linear', inplace=True)
    return df

@st.cache_data
def load_and_preprocess_data():
    # Specifying date format directly for consistency
    df = pd.read_csv("merged_data.csv", parse_dates=['Date'], dayfirst=True)
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Q'].replace(-9999, np.nan, inplace=True)
    df['Q'].interpolate(method='linear', inplace=True)
    return df

def download_link(object_to_download, download_filename, download_link_text):
    csv = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

class ExtendedEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(),
            'Gradient Boosting': GradientBoostingRegressor(),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror'),
            'CatBoost': CatBoostRegressor(silent=True),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor()
        }

    def fit(self, X, y):
        for _, model in self.models.items():
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for _, model in self.models.items()])
        return np.mean(predictions, axis=1)

def train_model(X, y):
    ensemble = ExtendedEnsembleRegressor()  # Initialize ensemble object
    st.write("Training...")
    total_models = len(ensemble.models)
    mse_results = {}  # Store MSE results for each model
    for i, (name, model) in enumerate(ensemble.models.items(), start=1):
        st.write(f"Training {name}...")
        
        # Start a progress bar for each model training
        with st.empty():
            progress_bar = st.progress(0)
            for percent_complete in tqdm(range(0, 101, 5), leave=False):
                time.sleep(0.1)  # Simulate training time
                progress_bar.progress(percent_complete)
        
        model.fit(X, y)
        st.write(f"Model {name} trained.")
        
        # Display results for each model
        st.write(f"Results for {name}:")
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        mse_results[name] = mse  # Store MSE for this model
        st.write(f"Mean Squared Error: {mse}")

    st.success("All models trained successfully.")
    
    # Determine best model based on MSE
    best_model_name = min(mse_results, key=mse_results.get)
    st.write(f"Best model based on Mean Squared Error: {best_model_name}")
    
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(mse_results.keys(), mse_results.values())
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Mean Squared Error for Each Model')
    st.pyplot(fig)

def parse_test_data(test_data, columns):
    missing_columns = set(columns) - set(test_data.columns)
    if missing_columns:
        st.warning(f"Test data is missing columns: {missing_columns}. Adding missing columns with default values.")
        for col in missing_columns:
            test_data[col] = 0  # Add missing columns with default value (you can customize this as needed)
    return test_data[columns]

def main():
    df = load_and_preprocess_data()

    st.title("HGM: Rainfall Runoff Module 💦")

    # Load the shapefile
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
    st.components.v1.html(m._repr_html_(),  height=400, scrolling=False)

    st.sidebar.header('Filter Options')
    selected_basins = st.sidebar.multiselect(
        "Select Basin(s)",
        options=['All'] + sorted(df['Basin'].unique().tolist()),
        default=['All']
    )
    
    selected_years = st.sidebar.multiselect(
        'Select Year(s)',
        options=['All'] + sorted(df['Year'].unique().tolist()),
        default=['All']
    )
    
    month_options = ['All'] + [calendar.month_name[m] for m in range(1, 13)]
    selected_month = st.sidebar.selectbox(
        'Select Month',
        options=month_options,
        index=0,
        format_func=lambda x: 'All' if x == 'All' else x
    )

    # Apply filters
    if 'All' not in selected_basins:
        df = df[df['Basin'].isin(selected_basins)]
    if 'All' not in selected_years:
        df = df[df['Year'].isin([int(year) for year in selected_years if year != 'All'])]
    if selected_month != 'All':
        df = df[df['Month'] == month_options.index(selected_month)]

    parameters = st.sidebar.multiselect(
        'Select Parameters',
        ['P', 'T', 'Q'],
        default=['P', 'T', 'Q']
    )
    df_filtered = df[['Date', 'Basin'] + parameters].dropna()

    if st.checkbox("Show Data"):
        st.dataframe(df_filtered)

    if st.checkbox("Download Data"):
        st.markdown(download_link(df_filtered, "filtered_data.csv", "Click here to download your data!"), unsafe_allow_html=True)

    if st.checkbox("Show Analysis"):
        for param in parameters:
            with st.expander(f"{param} Distribution Analysis"):
                fig, ax = plt.subplots()
                sns.histplot(df_filtered[param].dropna(), kde=True, bins=30, ax=ax)
                st.pyplot(fig)

    if st.checkbox("Show Basin Summary"):
        basin_summary = df_filtered.groupby('Basin').describe()
        st.dataframe(basin_summary)

    if st.checkbox("Normality Test for 'P' values in a specific Basin"):
        basin_choice = st.selectbox(
            'Choose a Basin for Normality Test on P values:',
            options=df_filtered['Basin'].unique()
        )
        data = df_filtered[df_filtered['Basin'] == basin_choice]['P'].dropna().sample(min(1000, len(df_filtered)))
        stat, p = shapiro(data)
        if p > 0.05:
            st.write(f"P values for {basin_choice} Basin are likely from a normal distribution.")
        else:
            st.write(f"P values for {basin_choice} Basin are likely not from a normal distribution.")



    if st.sidebar.checkbox('Select Year and Basin for Detailed Analysis'):
        selected_basin = st.sidebar.selectbox('Select Basin for Detailed Analysis:', sorted(df['Basin'].unique()))
        available_years = sorted(df[df['Basin'] == selected_basin]['Date'].dt.year.unique())

        # This button will now only prepare for animation and not show a static plot
        animate = st.button("Play Animation")

        # Placeholder for the plot
        plot_placeholder = st.empty()

        if animate:
            with plot_placeholder.container():
                fig, ax = plt.subplots(figsize=(10, 6))
                for year in available_years:
                    filtered_df = df[(df['Basin'] == selected_basin) & (df['Date'].dt.year == year)]
                    if not filtered_df.empty:
                        ax.clear()
                        if 'Q' in filtered_df.columns:
                            ax.plot(filtered_df['Date'], filtered_df['Q'], label='Flow Rate (Q)')
                        if 'P' in filtered_df.columns:
                            ax.plot(filtered_df['Date'], filtered_df['P'], label='Precipitation (P)')
                        ax.set_title(f"Data for {selected_basin} Basin in {year}")
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Value')
                        ax.legend()
                        ax.xaxis.set_major_locator(mdates.MonthLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
                        
                        # Update the plot in the placeholder
                        plot_placeholder.pyplot(fig)
                        time.sleep(0.5)  # Adjust as needed for animation speed

    # Machine learning model training
    if st.sidebar.checkbox("Train Machine Learning Model"):
        st.sidebar.subheader("Model Training")
        custom_data_option = st.sidebar.checkbox("Load Custom Training Data")
        if custom_data_option:
            uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                custom_df = pd.read_csv(uploaded_file)
                X_custom = custom_df.drop(columns=['Q', 'Date', 'Basin'])
                y_custom = custom_df['Q']
                if st.sidebar.button("Train Model"):
                    train_model(X_custom, y_custom)
        else:
            if st.sidebar.button("Train Model"):
                train_model(df_filtered.drop(columns=['Q', 'Date', 'Basin']), df_filtered['Q'])

   # Test with selected basins or custom data
    test_option = st.sidebar.checkbox("Test with Selected Basins or Custom Data")
    if test_option:
        st.sidebar.subheader("Test Options")
        test_basins = st.sidebar.multiselect(
            "Select Test Basin(s)",
            options=sorted(df['Basin'].unique()),
            default=[]
        )
        upload_custom_data = st.sidebar.checkbox("Load Custom Test Data")
        if upload_custom_data:
            uploaded_test_file = st.sidebar.file_uploader("Choose a CSV file for testing", type="csv")
            if uploaded_test_file is not None:
                custom_test_df = pd.read_csv(uploaded_test_file)
                st.write(custom_test_df)
        if st.sidebar.button("Start Testing"):
            if test_basins:
                best_model_name = "Random Forest"
                ensemble = ExtendedEnsembleRegressor()
                best_model = ensemble.models[best_model_name]

                X_train = df_filtered.drop(columns=['Q', 'Date', 'Basin'])
                y_train = df_filtered['Q']
                best_model.fit(X_train, y_train)

                plot_placeholder = st.empty()

                with plot_placeholder.container():
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for basin in test_basins:
                        mask = df['Basin'] == basin
                        X_test_basin = df[mask].drop(['Date', 'Basin', 'Q', 'Qsim'] if 'Qsim' in df.columns else ['Date', 'Basin', 'Q'], axis=1)
                        X_test_basin_processed = parse_test_data(X_test_basin, X_train.columns)

                        with st.spinner(f"Testing {basin} Basin..."):
                            df.loc[mask, 'Qsim'] = best_model.predict(X_test_basin_processed)

                        ax.clear()
                        ax.plot(df[mask]['Date'], df[mask]['Q'], label=f'Qobs - {basin}')
                        ax.plot(df[mask]['Date'], df[mask]['Qsim'], label=f'Qsim ({best_model_name}) - {basin}', linestyle='--')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Flow Rate')
                        ax.set_title(f'Observed vs. Predicted Flow Rate for {basin}')
                        ax.legend()
                        plot_placeholder.pyplot(fig)
                        time.sleep(0.5)  # Adjust as needed for visualization

                st.success("Testing for all selected basins completed.")


if __name__ == "__main__":
    main()
