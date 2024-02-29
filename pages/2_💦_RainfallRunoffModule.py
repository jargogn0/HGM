import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
import time
import matplotlib.dates as mdates
import geopandas as gpd
import folium
from datetime import timedelta
import base64
import calendar
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
import matplotlib.animation as animation
from IPython.display import HTML

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

# Define the function to calculate Nash-Sutcliffe Efficiency (NSE)
def calculate_nse(observed, simulated):
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

# Define the function to calculate Kling-Gupta Efficiency (KGE)
def calculate_kge(observed, simulated):
    mean_observed = np.mean(observed)
    mean_simulated = np.mean(simulated)
    std_observed = np.std(observed)
    std_simulated = np.std(simulated)
    correlation = np.corrcoef(observed, simulated)[0, 1]
    kge = 1 - np.sqrt((correlation - 1) ** 2 + (std_simulated / std_observed - 1) ** 2 + (mean_simulated / mean_observed - 1) ** 2)
    return kge

# Define the function to calculate Mean Squared Error (MSE)
def calculate_mse(observed, simulated):
    return mean_squared_error(observed, simulated)

# Define the function to calculate Root Mean Squared Error (RMSE)
def calculate_rmse(observed, simulated):
    return np.sqrt(mean_squared_error(observed, simulated))

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
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    ensemble = ExtendedEnsembleRegressor()
    st.write("Training...")
    train_evaluation_metrics = {}  # Store evaluation metrics for the training set
    test_evaluation_metrics = {}   # Store evaluation metrics for the test set
    train_nse_values = {}  # Store NSE values for training set
    test_nse_values = {}   # Store NSE values for test set
    
    # Plotting Qobs vs Qsim for each model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, model in ensemble.models.items():
        st.write(f"Training {name}...")
        progress_bar = st.progress(0)
        for percent_complete in range(0, 101, 5):
            time.sleep(0.1)
            progress_bar.progress(percent_complete)
        model.fit(X_train, y_train)
        progress_bar.empty()
        st.write(f"Model {name} trained.")
        
        # Evaluate on the training set
        st.write(f"Evaluating {name} on the training set...")
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_nse = calculate_nse(y_train, y_train_pred)
        train_kge = calculate_kge(y_train, y_train_pred)
        train_evaluation_metrics[name] = {'MSE': train_mse, 'RMSE': train_rmse, 'NSE': train_nse, 'KGE': train_kge}
        train_nse_values[name] = train_nse  # Store NSE value for training set
        st.write(f"Training Evaluation Metrics for {name}:")
        st.write(train_evaluation_metrics[name])
        
        # Evaluate on the test set
        st.write(f"Evaluating {name} on the test set...")
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_nse = calculate_nse(y_test, y_test_pred)
        test_kge = calculate_kge(y_test, y_test_pred)
        test_evaluation_metrics[name] = {'MSE': test_mse, 'RMSE': test_rmse, 'NSE': test_nse, 'KGE': test_kge}
        test_nse_values[name] = test_nse  # Store NSE value for test set
        st.write(f"Test Evaluation Metrics for {name}:")
        st.write(test_evaluation_metrics[name])
        
        # Plot Qobs vs Qsim
        ax.scatter(y_test, y_test_pred, label=name)
    
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray', label='1:1 line')
    ax.set_xlabel('Observed Flow Rate (Qobs)')
    ax.set_ylabel('Simulated Flow Rate (Qsim)')
    ax.set_title('Qobs vs Qsim for Different Models')
    ax.legend()
    st.pyplot(fig)
    
    # Plot NSE values for training and test sets
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(train_nse_values.keys(), train_nse_values.values(), label='Training NSE')
    ax.bar(test_nse_values.keys(), test_nse_values.values(), label='Test NSE')
    ax.set_xlabel('Model')
    ax.set_ylabel('NSE')
    ax.set_title('NSE Comparison for Training and Test Sets')
    ax.legend()
    st.pyplot(fig)
    
    st.success("All models trained and evaluated successfully.")
    
    # Calculate the difference in NSE between the train and test sets for each model
    for name in train_evaluation_metrics.keys():
        train_nse = train_evaluation_metrics[name]['NSE']
        test_nse = test_evaluation_metrics[name]['NSE']
        difference = test_nse - train_nse
        test_evaluation_metrics[name]['Difference'] = difference
    
    # Find the best model based on NSE and the least overfitting
    best_model_name = max(test_evaluation_metrics, key=lambda k: test_evaluation_metrics[k]['NSE'] - abs(test_evaluation_metrics[k]['Difference']))
    st.write(f"Best model based on Nash-Sutcliffe Efficiency (NSE) and minimal overfitting/underfitting: {best_model_name}")
    
    # List models showing signs of overfitting or underfitting
    overfitting_models = [name for name, metrics in test_evaluation_metrics.items() if metrics['Difference'] < 0]
    underfitting_models = [name for name, metrics in test_evaluation_metrics.items() if metrics['Difference'] > 0]
    st.write("Models showing signs of overfitting:")
    st.write(overfitting_models)
    st.write("Models showing signs of underfitting:")
    st.write(underfitting_models)




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

                st.success("Animation completed.")

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
                X_train, X_test, y_train, y_test = train_test_split(X_custom, y_custom, test_size=0.2, random_state=42)
                if st.sidebar.button("Train Model"):
                    train_and_evaluate_model(X_train, y_train, X_test, y_test)
        else:
            # Split your data into training and testing sets
            X = df_filtered.drop(columns=['Q', 'Date', 'Basin'])
            y = df_filtered['Q']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if st.sidebar.button("Train Model"):
                train_and_evaluate_model(X_train, y_train, X_test, y_test)
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
