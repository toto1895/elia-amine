import streamlit as st
import pandas as pd
import requests
from predico import PredicoAPI
import downloader
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_files_connection import FilesConnection
from google.cloud import run_v2
from google.oauth2 import service_account
import json
import time
import tempfile
# Clear cache on startup
st.cache_data.clear()

# Page configuration
st.set_page_config(
    page_title="Predico monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark theme for the app
st.markdown(
    """
    <style>
    body {
        color: #fff;
        background-color: #1e1e1e;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Environment variables
user_env = os.getenv("USER") 
pwd_env = os.getenv("PWD")
pwd_view = os.getenv("PWD_VIEW")
GCLOUD = os.getenv("service_account_json")

# API and data functions
def get_api():
    """Create and authenticate API instance."""
    try:
        api = PredicoAPI(user_env, pwd_env)
        api.authenticate()
        return api
    except Exception as e:
        st.error(f"Error authenticating API: {e}")
        return None

def list_submissions(_api):
    """Return submission info sorted by most recent submission time."""
    try:
        url = "https://predico-elia.inesctec.pt/api/v1/market/challenge/submission"
        resp = requests.get(url, headers=_api._headers())
        data = resp.json()["data"]

        df_subs = pd.DataFrame(data)
        df_subs["registered_at"] = pd.to_datetime(df_subs["registered_at"])
        # Sort descending by submission time
        df_subs = df_subs.sort_values("registered_at", ascending=False)
        df_subs["registered_at"] = df_subs["registered_at"].dt.round("T")
        df_subs = df_subs.drop_duplicates(subset=["registered_at"], keep="first")
        return df_subs
    except Exception as e:
        st.error(f"Error listing submissions: {e}")
        return pd.DataFrame()

def fetch_submission_data(_api, _challenge_id, _submission_time):
    try:
        # --- Scores ---
        url_sc = "https://predico-elia.inesctec.pt/api/v1/market/challenge/submission-scores"
        sc_resp = requests.get(url_sc, params={"challenge": _challenge_id}, headers=_api._headers())
        sc_data = sc_resp.json()["data"]["personal_metrics"]
        df_scores = pd.DataFrame(sc_data)
        # Add 'market_date' as day after submission_time
        df_scores["market_date"] = (_submission_time + pd.Timedelta(days=1)).date()

        # --- Forecasts ---
        url_fc = "https://predico-elia.inesctec.pt/api/v1/market/challenge/submission/forecasts"
        fc_resp = requests.get(url_fc, params={"challenge": _challenge_id}, headers=_api._headers())
        fc = pd.DataFrame(fc_resp.json()["data"])
        fc = fc.sort_values("registered_at").drop_duplicates(subset=["datetime","variable"], keep="last")
        pivoted = fc.pivot(index="datetime", columns="variable", values="value")
        pivoted.index = pd.to_datetime(pivoted.index)

        # Merge with actual
        act = downloader.get_actual_wind_offshore_from_date(pivoted.index[0])
        act.index = act.index
        pivoted = pd.concat([act[["actual elia","DA elia (11AM)","latest elia"]], pivoted], axis=1)

        return df_scores, pivoted
    except Exception as e:
        st.error(f"Error fetching submission data: {e}")
        return pd.DataFrame(), pd.DataFrame()

def add_daily_payout(df, daily_pool=225.8065):
    try:
        # Mapping of rank to fraction
        distribution = {1: 0.40, 2: 0.27, 3: 0.18, 4: 0.10, 5: 0.05}
        
        def calc_payout(r_rmse, r_winkler):
            frac_rmse = distribution.get(r_rmse, 0)
            frac_winkler = distribution.get(r_winkler, 0)
            return round((0.5 * frac_rmse + 0.5 * frac_winkler) * daily_pool, 2)

        payouts = {}
        for date, group in df.groupby("market_date"):
            rank_rmse = group.loc[group["metric"] == "rmse", "rank"]
            rank_winkler = group.loc[group["metric"] == "winkler", "rank"]
            r_rmse = rank_rmse.iloc[0] if not rank_rmse.empty else 99
            r_winkler = rank_winkler.iloc[0] if not rank_winkler.empty else 99
            payouts[date] = calc_payout(r_rmse, r_winkler)

        df["daily_payout"] = df["market_date"].map(payouts)
        return df
    except Exception as e:
        st.error(f"Error calculating daily payout: {e}")
        return df

def fetch_last_50_scores(_api):
    """
    Fetch the last 50 submissions and retrieve their scores.
    Returns a combined DataFrame of all personal_metrics found.
    """
    try:
        # Get list of submissions
        df_subs = list_submissions(_api)
        
        # Take the latest 50
        latest_50 = df_subs.head(60)  # top 50 rows if sorted by descending time
        
        all_scores = []
        for _, row in latest_50.iterrows():
            try:
                challenge_id = row["market_session_challenge"]
                sub_time = row["registered_at"].tz_convert('CET')
                sub_id = row["id"]

                # Request scores for this specific challenge
                url_sc = "https://predico-elia.inesctec.pt/api/v1/market/challenge/submission-scores"
                resp = requests.get(url_sc, params={"challenge": challenge_id}, headers=_api._headers())
                if resp.status_code != 200:
                    continue  # skip if no data

                sc_data = resp.json()["data"]["personal_metrics"]
                df_scores = pd.DataFrame(sc_data)

                # Add helpful columns
                df_scores["submission_id"] = sub_id
                df_scores["submission_time"] = sub_time
                df_scores["market_date"] = (sub_time + pd.Timedelta(days=1)).date()

                all_scores.append(df_scores)
            except Exception as e:
                st.warning(f"Error processing score for submission {row.get('id', 'unknown')}: {e}")
                continue

        # Combine into one DataFrame
        if all_scores:
            final_scores = pd.concat(all_scores, ignore_index=True)
        else:
            final_scores = pd.DataFrame()
            
        filter_score = final_scores.loc[final_scores.metric.isin(['winkler','rmse']),
                                        ['variable','metric','value','rank','market_date']]
        df_lowest = (filter_score.loc[filter_score.groupby(["market_date", "metric"])["rank"].idxmin()]
                    .drop_duplicates().reset_index(drop=True)
                    .sort_values('market_date')
                   )

        award = add_daily_payout(df_lowest)
        return award
    except Exception as e:
        st.error(f"Error fetching last 50 scores: {e}")
        return pd.DataFrame()

def plot_rank_and_payout_separate(df):
    try:
        # Prepare data
        df_rmse = df[df["metric"] == "rmse"].copy()
        df_winkler = df[df["metric"] == "winkler"].copy()
        df_payout = df.groupby("market_date", as_index=False)["daily_payout"].first()

        # Compute monthly payout
        df_payout["market_date"] = pd.to_datetime(df_payout["market_date"])
        df_payout["month"] = df_payout["market_date"].dt.to_period("M")
        df_monthly = df_payout.groupby("month", as_index=False)["daily_payout"].sum()
        df_monthly["month_dt"] = df_monthly["month"].dt.to_timestamp()

        # Create 3 subplots with spacing
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=["", "", ""],
            vertical_spacing=0.1
        )

        # 1) Rank subplot
        fig.add_trace(go.Bar(x=df_rmse["market_date"], y=df_rmse["rank"], name="RMSE Rank"), row=1, col=1)
        fig.add_trace(go.Bar(x=df_winkler["market_date"], y=df_winkler["rank"], name="Winkler Rank"), row=1, col=1)

        # 2) Daily payout subplot
        fig.add_trace(go.Bar(x=df_payout["market_date"], y=df_payout["daily_payout"], name="Daily Payout"), row=2, col=1)

        # 3) Monthly payout subplot
        fig.add_trace(go.Bar(x=df_monthly["month_dt"], y=df_monthly["daily_payout"], name="Monthly Payout"), row=3, col=1)

        # Update layout and axes
        fig.update_layout(barmode="group")
        fig.update_yaxes(title_text="Rank", row=1, col=1)
        fig.update_yaxes(title_text="PnL (€)", row=2, col=1)
        fig.update_yaxes(title_text="PnL (€)", row=3, col=1)
        fig.update_layout(
            barmode="group",
            width=800,
            height=900,
            showlegend=False
        )

        return fig
    except Exception as e:
        st.error(f"Error plotting rank and payout: {e}")
        # Return an empty figure
        return go.Figure()

# Error metrics functions
def calculate_rmse(actual, predicted):
    """Compute Root Mean Square Error (RMSE)."""
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        return np.sqrt(np.mean((predicted - actual) ** 2))
    except Exception as e:
        st.warning(f"Error calculating RMSE: {e}")
        return np.nan

def calculate_mase(actual, predicted, training_actual=None):
    """Compute Mean Absolute Scaled Error (MASE)."""
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Use training data for scaling if provided; otherwise, use the actual series
        train = np.array(training_actual) if training_actual is not None else actual
        
        # Compute naive forecast errors: absolute differences of successive observations
        naive_errors = np.abs(np.diff(train))
        
        scale = np.mean(naive_errors)
        if scale == 0:
            return np.nan

        # Compute MASE
        mase = np.mean(np.abs(actual - predicted)) / scale
        return mase
    except Exception as e:
        st.warning(f"Error calculating MASE: {e}")
        return np.nan

def mean_pinball_loss(actual, forecast, alpha=0.5):
    """Compute pinball loss."""
    try:
        return np.mean(np.maximum(alpha*(actual - forecast), (alpha-1)*(actual - forecast)))
    except Exception as e:
        st.warning(f"Error calculating pinball loss: {e}")
        return np.nan

def get_latest_da_fcst_file(selected_date, files):
    """
    Get the latest day-ahead forecast file for the selected date.
    If no file is found for the selected date, return the most recent file available.
    Uses a simpler approach for more reliable date parsing.
    """
    try:
        # Filter for parquet files only and remove directory entries
        valid_files = [f for f in files if f.endswith(".parquet") and not f.endswith('/')]
        
        # Parse date from each filename and create a list of (date, file) tuples
        date_file_pairs = []
        
        for file in valid_files:
            try:
                # Extract just the filename without path
                filename = file.split('/')[-1]
                
                # Split by underscore to get parts
                parts = filename.split('_')
                
                # Basic validation - we need at least year, month, day
                if len(parts) < 3:
                    continue
                    
                # Extract year, month, day
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                
                # Extract hour, minute if available
                hour = int(parts[3]) if len(parts) > 3 else 0
                minute = int(parts[4]) if len(parts) > 4 else 0
                
                # Create a datetime object
                file_date = pd.Timestamp(year=year, month=month, day=day, 
                                        hour=hour, minute=minute)
                
                date_file_pairs.append((file_date, file))
            except (ValueError, IndexError):
                # Skip files that don't match the expected format
                continue
        
        # If no valid files found
        if not date_file_pairs:
            return None
            
        # Sort all files by date (newest first)
        date_file_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Convert selected_date to a Timestamp for comparison
        selected_ts = pd.Timestamp(selected_date)
        selected_date_start = pd.Timestamp(year=selected_ts.year, month=selected_ts.month, day=selected_ts.day)
        selected_date_end = selected_date_start + pd.Timedelta(days=1)
        
        # First look for files from the selected date
        for date, file in date_file_pairs:
            if selected_date_start <= date < selected_date_end:
                return file
                
        # If no file for selected date, return the most recent file
        return date_file_pairs[0][1] if date_file_pairs else None
    except Exception as e:
        st.error(f"Error getting latest forecast file: {e}")
        return None

def get_latest_wind_offshore(start) -> pd.DataFrame:
    """Get the latest wind offshore data."""
    try:
        start = start + pd.Timedelta(days=1)
        end = start
        start = start.strftime('%Y-%m-%d')
        end = end.strftime('%Y-%m-%d')

        url = (f'https://griddata.elia.be/eliabecontrols.prod/interface/windforecasting/'
        f'forecastdata?beginDate={start}&endDate={end}&region=1&'
        f'isEliaConnected=&isOffshore=True')

        d = pd.read_json(url).rename(
            columns={
                'dayAheadConfidence10':'DA elia (11AM) P10',
                'dayAheadConfidence90':'DA elia (11AM) P90',
                'dayAheadForecast':'DA elia (11AM)',
                'monitoredCapacity':'capa',
                'mostRecentForecast':'latest elia forecast',
                'realtime':'actual elia',
                'startsOn':'Datetime',
                })[['DA elia (11AM)','actual elia','Datetime','latest elia forecast']]
        d['Datetime'] = pd.to_datetime(d['Datetime'])
        d.index = d['Datetime']
        return d.rename(columns={'actual elia':'actual'})
    except Exception as e:
        st.error(f"Error getting latest wind offshore data: {e}")
        return pd.DataFrame()

def list_gcs_files(connection, prefix):
    """
    List GCS files using native method with debugging information.
    Returns a list of file paths matching the given prefix.
    """
    try:
        # Get the GCS bucket name from your connection
        bucket_name = connection._path.split('/')[0] if '/' in connection._path else connection._path
        
        # Debug information
        st.write(f"Debug - Bucket: {bucket_name}, Prefix: {prefix}")
        
        # Use Google Cloud Storage client directly
        from google.cloud import storage
        
        # Use credentials from the connection if available
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List all blobs with the given prefix
        blobs = list(client.list_blobs(bucket, prefix=prefix))
        
        # Get the names of all blobs
        file_paths = [blob.name for blob in blobs]
        
        # Debug information
        st.write(f"Found {len(file_paths)} files")
        if len(file_paths) > 0:
            st.write(f"Sample file: {file_paths[0]}")
        
        return file_paths
    
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

# Application pages
def submission_viewer():
    """Display submission details and visualizations."""
    st.subheader("Submission Viewer")

    try:
        # Authenticate & get submissions
        api = get_api()
        if api is None:
            st.error("Failed to authenticate API. Please try again.")
            return
            
        df_subs = list_submissions(api)
        if df_subs.empty:
            st.error("No submissions found. Please check your credentials and connection.")
            return

        df_subs["registered_at"] = df_subs["registered_at"].dt.tz_convert('CET')
        
        # Create the label column
        df_subs["label"] = (
            "Market date " + ((df_subs["registered_at"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
            + " | ID: " + df_subs["id"].astype(str)
            + " | Time: " + df_subs["registered_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
        )
        df_subs["dt"] = ((df_subs["registered_at"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"))
        df_subs = df_subs.drop_duplicates(subset=["dt"], keep="last")

        selected_label = st.selectbox("Select submission", df_subs["label"])

        # Extract the row matching the chosen label
        chosen_row = df_subs.loc[df_subs["label"] == selected_label].iloc[0]
        submission_time = chosen_row["registered_at"]
        challenge_id = chosen_row["market_session_challenge"]

        # Fetch data for the chosen submission (scores + forecast)
        scores, forecasts = fetch_submission_data(api, challenge_id, submission_time)

        # Force the forecast index to be timezone-naive for consistent comparisons
        if not forecasts.empty:
            forecasts.index = forecasts.index.tz_localize(None)

        # Display data
        st.subheader("Scores for This Submission")
        if scores.empty:
            st.warning("No scoring data for this submission. Showing forecasts only.")
        else:
            st.dataframe(scores)

        st.subheader("Forecast vs Actual")

        if scores.empty and not forecasts.empty:
            data_slice = forecasts
            market_date = forecasts["q10"].dropna().index[0] if "q10" in forecasts.columns else None
        elif not scores.empty:
            market_date = scores["market_date"].iloc[0]
            data_slice = forecasts
        else:
            st.error("No forecast or score data available")
            return
            
        if forecasts.empty:
            st.error("No forecast data available")
            return
            
        data_slice = forecasts.dropna(subset='q10') if 'q10' in forecasts.columns else forecasts
        
        if 'q50' in data_slice.columns and 'DA elia (11AM)' in data_slice.columns and 'actual elia' in data_slice.columns:
            df_sc = data_slice[['q50','DA elia (11AM)','actual elia']].copy().dropna()
            
            # Calculate metrics
            myrmse = round(calculate_rmse(df_sc['q50'], df_sc['actual elia']),1)
            eliarmse = round(calculate_rmse(df_sc['DA elia (11AM)'], df_sc['actual elia']),1)
            mymase = round(calculate_mase(df_sc['q50'], df_sc['actual elia']),1)
            eliamase = round(calculate_mase(df_sc['DA elia (11AM)'], df_sc['actual elia']),1)

            # Display metrics
            st.markdown(f"**RMSE (q50):** {myrmse} , MASE : {mymase}")
            st.markdown(f"**RMSE (DA elia):** {eliarmse} , MASE : {eliamase}")
        else:
            st.warning("Missing columns needed for metrics calculation")

        # Create visualization
        fig = go.Figure()

        # Add the uncertainty band (q10 - q90)
        if "q90" in data_slice.columns and "q10" in data_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice["q90"],
                    name="q90",
                    mode="lines",
                    line_color="rgba(0,0,0,0)",
                    showlegend=True
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice["q10"],
                    name="Uncertainty [q10–q90]",
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(0, 100, 80, 0.4)",
                    line_color="rgba(0,0,0,0)",
                    showlegend=True
                )
            )

        # Add forecasts
        if "q50" in data_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice["q50"],
                    name="q50",
                    mode="lines",
                    line_color="rgb(5, 222, 255)"
                )
            )
        if "DA elia (11AM)" in data_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice["DA elia (11AM)"],
                    name="DA elia (11AM)",
                    mode="lines",
                    line_color="orange"
                )
            )
        if "latest elia" in data_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice["latest elia"],
                    name="latest elia",
                    mode="lines",
                    line_color="red",
                    visible='legendonly'
                )
            )

        # Add actuals
        if "actual elia" in data_slice.columns:
            fig.add_trace(
                go.Scatter(
                    x=data_slice.index,
                    y=data_slice["actual elia"],
                    name="actual elia",
                    mode="lines",
                    line_color="white"
                )
            )

        # Final styling
        fig.update_layout(
            xaxis_title="Datetime",
            yaxis_title="MW",
            yaxis=dict(range=[0, 2300]),
            template="plotly_dark",
            showlegend=False
        )

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error in submission viewer: {e}")
        import traceback
        st.error(traceback.format_exc())


from google.cloud import storage
def list_blobs_in_bucket(bucket_name, prefix=None):
    """Lists all the blobs in the bucket with the given prefix."""
    # Instantiate a client
    service_account_info = json.loads(GCLOUD)
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    storage_client = storage.Client(credentials=credentials)
    
    # Get the bucket
    bucket = storage_client.bucket(bucket_name)
    
    # List blobs in the bucket with the specified prefix
    blobs = bucket.list_blobs(prefix=prefix)
    
    print(f"Files in bucket {bucket_name} with prefix {prefix}:")
    return [blob.name for blob in blobs]  # Return the list of blob names

def benchmark():
    """Benchmark different forecasting models."""
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.write("Cache cleared!")
        time.sleep(1)  # Reduced sleep time

    st.title("Benchmark Models")
    
    try:
        selected_date = st.date_input("Submission date", pd.to_datetime("today"))
        
        # Use caching for expensive operations
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def get_cached_actual(date):
            try:
                return get_latest_wind_offshore(date)
            except Exception as e:
                st.error(f"Error getting latest offshore data: {e}")
                return pd.DataFrame()
        
        latest_actual = get_cached_actual(selected_date)

        # Fetch model forecasts
        models = ['avg', 'oracle', 'metno', 'dmi_seamless', 'meteofrance', 'icon', 'knmi','tech']
        forecasts = {}  # Use dict for faster lookups
        
        # Create a client once outside the loop
        service_account_info = json.loads(GCLOUD)
        credentials = service_account.Credentials.from_service_account_info(service_account_info)
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket('oracle_predictions')
        
        # Process models in parallel
        @st.cache_data(ttl=60, show_spinner=False)  # Cache for 1 hour
        def process_model(model, selected_date):
            try:
                # List files for this model
                prefix = f'predico-elia/forecasts/{model}'
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                # Filter for parquet files
                valid_files = [blob.name for blob in blobs if blob.name.endswith(".parquet")]
                
                if not valid_files:
                    return None
                
                # Extract date info and find relevant file
                date_file_pairs = []
                for file in valid_files:
                    try:
                        filename = file.split('/')[-1]
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            year, month, day, hour = map(int, parts[:4])
                            file_date = pd.Timestamp(year=year, month=month, day=day, hour=hour)
                            date_file_pairs.append((file_date, file))
                    except Exception:
                        continue
                
                if not date_file_pairs:
                    return None
                
                # Sort and select best match
                date_file_pairs.sort(key=lambda x: x[0], reverse=False)
                selected_ts = pd.Timestamp(selected_date)
                selected_date_start = pd.Timestamp(year=selected_ts.year, month=selected_ts.month, day=selected_ts.day)
                selected_date_end = selected_date_start + pd.Timedelta(days=1)
                
                best_match = None
                for date, file in date_file_pairs:
                    if selected_date_start <= date < selected_date_end:
                        best_match = file
                        break
                
                # Use most recent if no match
                if best_match is None and date_file_pairs:
                    best_match = date_file_pairs[0][1]
                
                if not best_match:
                    return None
                
                # Download and load the file
                blob = bucket.blob(best_match)
                
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_parquet(temp_file.name)
                
                # Process columns
                if all(col in df.columns for col in [0.1, 0.5, 0.9]):
                    df = df[[0.1, 0.5, 0.9]]
                elif all(col in df.columns for col in ['0.1', '0.5', '0.9']):
                    df = df[['0.1', '0.5', '0.9']]
                    df.columns = [0.1, 0.5, 0.9]
                else:
                    return None
                
                if df.empty:
                    return None
                
                return df.add_prefix(f'{model}_')
                
            except Exception:
                return None
                
        # Function to load UK data from neso_data
        
        def get_uk_data(selected_date):
            try:
                # List files from the neso_data/dayahead directory
                prefix = 'neso_data/dayahead'
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                # Filter for parquet files
                valid_files = [blob.name for blob in blobs if blob.name.endswith(".parquet")]
                
                if not valid_files:
                    return None
                
                # Extract date info and find relevant file
                date_file_pairs = []
                for file in valid_files:
                    try:
                        # Try to extract date from filename pattern
                        filename = file.split('/')[-1]
                        parts = filename.split('_')
                        # Different file naming pattern might be present
                        # Try to find date components in the filename
                        if len(parts) >= 2:
                            # This is a simplified approach - adapt based on actual file naming
                            date_parts = [part for part in parts if len(part) >= 8 and part.isdigit()]
                            if date_parts:
                                date_str = date_parts[0]
                                year = int(date_str[:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                                file_date = pd.Timestamp(year=year, month=month, day=day)
                                date_file_pairs.append((file_date, file))
                    except Exception as e:
                        continue
                
                if not date_file_pairs:
                    # If no date parsing worked, just use the latest file
                    latest_file = sorted(valid_files)[-1]
                    return process_uk_file(latest_file)
                
                # Sort by date and find closest match
                date_file_pairs.sort(key=lambda x: abs((x[0] - pd.Timestamp(selected_date)).total_seconds()))
                best_match = date_file_pairs[0][1] if date_file_pairs else None
                
                if not best_match:
                    return None
                    
                return process_uk_file(best_match)
                
            except Exception as e:
                st.error(f"Error retrieving UK data: {str(e)}")
                return None
                
        def process_uk_file(file_path):
            try:
                # Download and load the file
                blob = bucket.blob(file_path)
                
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_parquet(temp_file.name)
                
                # Check if the required column exists (try both possible naming conventions)
                column_name = None
                if 'ratio_GAOFO-1' in df.columns:
                    column_name = 'ratio_GAOFO-1'
                elif 'ratio-GAOFO-1' in df.columns:
                    column_name = 'ratio-GAOFO-1'
                
                #column_name = 'ratio_EAAO-1'
                    
                if column_name:
                    # Make sure we have a datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        # Check if there's a datetime column we can use
                        datetime_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
                        if datetime_cols:
                            # Use the first column that looks like a datetime column
                            df.index = pd.to_datetime(df[datetime_cols[0]])
                        else:
                            # Create a synthetic datetime index based on selected_date
                            start_time = pd.Timestamp(selected_date) + pd.Timedelta(days=1)
                            # Assume hourly data if we can't determine the frequency
                            df.index = pd.date_range(start=start_time, periods=len(df), freq='30min')
                            st.warning("UK data has no datetime index. Created synthetic hourly timestamps.")
                    
                    # Extract just the needed column and multiply by 2263
                    uk_data = df[[column_name]].copy()
                    uk_data['uk-test'] = uk_data[column_name] * 2263 * 0.5
                    
                    # Drop the original column, keeping only the calculated one
                    uk_data = uk_data[['uk-test']]
                    
                    # Ensure the index is timezone-aware to match other data
                    if uk_data.index.tz is None:
                        uk_data.index = uk_data.index.tz_localize('UTC')
                    
                    # Resample to 15-minute intervals
                    # First, ensure the index is sorted
                    uk_data = uk_data.sort_index()
                    
                    # Resample to 15-minute intervals
                    uk_resampled = uk_data.resample('15min').interpolate(method='linear')
                    st.info(f"UK data resampled from {len(uk_data)} to {len(uk_resampled)} rows (15-minute intervals)")
                    
                    return uk_resampled
                else:
                    # If column doesn't exist, log error and return None
                    column_list = ", ".join(df.columns.tolist())
                    st.warning(f"Columns 'ratio_GAOFO-1' or 'ratio-GAOFO-1' not found in UK data file. Available columns: {column_list}")
                    return None
            except Exception as e:
                st.error(f"Error processing UK data file: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return None
        
        # Use a progress bar and process models
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i, model in enumerate(models):
            if (model =='oracle') and (pd.to_datetime(selected_date)<= pd.to_datetime('2025-03-04')):
                pass
            else:
                progress_text.text(f"Processing model: {model}")
                result = process_model(model, selected_date)
                if result is not None:
                    forecasts[model] = result
                progress_bar.progress((i + 1) / len(models))
        
        # Load UK data after processing models
        progress_text.text("Processing UK data...")
        uk_data = get_uk_data(selected_date) 
        print(uk_data)
        
        forecasts['uk'] = uk_data[result.index[0]:result.index[-1]]
            
        progress_bar.empty()
        progress_text.empty()
        
        # Combine and visualize forecasts
        if forecasts:
            try:
                df = pd.concat(list(forecasts.values()), axis=1)
                df.index = pd.to_datetime(df.index)
                
                # Store the meteofrance DataFrame for download
                meteofrance_df = None
                if 'meteofrance' in forecasts:
                    meteofrance_df = forecasts['meteofrance'].copy().round(1)
                    meteofrance_df.index = pd.to_datetime(meteofrance_df.index)
                    meteofrance_df.columns = ['q10','q50','q90']
                
                try:
                    if not latest_actual.empty:
                        df = pd.concat([latest_actual.drop(columns='Datetime'), df], axis=1)
                        default_cols = ['actual', 'DA elia (11AM)', 'oracle_0.5','avg_0.5', 'icon_0.5', 'metno_0.5', 
                                      'dmi_seamless_0.5', 'meteofrance_0.5', 'knmi_0.5','tech_0.5']
                        
                        # Add UK data to default columns if available
                        if 'uk-test' in df.columns:
                            default_cols.append('uk-test')
                    else:
                        default_cols = ['DA elia (11AM)', 'oracle_0.5', 'avg_0.5', 'icon_0.5', 'metno_0.5', 
                                       'dmi_seamless_0.5', 'meteofrance_0.5', 'knmi_0.5','tech_0.5']
                        
                        # Add UK data to default columns if available
                        if 'uk-test' in df.columns:
                            default_cols.append('uk-test')
                except Exception as e:
                    st.error(f"Error merging latest actual data: {e}")
                    default_cols = ['avg_0.5', 'icon_0.5', 'metno_0.5', 
                                   'dmi_seamless_0.5', 'meteofrance_0.5', 'knmi_0.5','tech_0.5']
                    
                    # Add UK data to default columns if available
                    if 'uk-test' in df.columns:
                        default_cols.append('uk-test')

                df = df.iloc[-96-8-3:-8].copy()
                
                # Pre-compute color mapping for faster plotting
                color_map = {
                    'actual': 'white',
                    'DA elia (11AM)': 'orange',
                    'avg_0.5': "rgb(5, 222, 255)",
                    'metno_0.5': 'red',
                    'dmi_seamless_0.5': 'green',
                    'meteofrance_0.5': 'purple',
                    'knmi_0.5': 'grey',
                    'icon_0.5': 'yellow',
                    'oracle_0.5': 'blue',
                    'uk-test': 'pink',  # Add color for UK data
                    'tech_0.5': 'cyan'  # Add color for UK data
                }

                # Create plot more efficiently
                fig = go.Figure()
                traces = []
                
                for col in df.columns:
                    color = color_map.get(col, 'blue')  # Default color
                    
                    traces.append(go.Scatter(
                        x=df.index,
                        y=df[col],
                        mode='lines',
                        name=col,
                        visible=(col in default_cols),
                        line_color=color,
                        showlegend=True
                    ))
                
                fig.add_traces(traces)
                
                fig.update_layout(
                    xaxis_title="Datetime",
                    yaxis_title="MW",
                    yaxis=dict(range=[0, 2300]),
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig)

                # Add download button for meteofrance data
                if meteofrance_df is not None:
                    # Function to convert dataframe to CSV for download
                    def convert_df_to_csv(df):
                        return df.to_csv().encode('utf-8')
                    
                    # Create a download button
                    meteofrance_csv = convert_df_to_csv(meteofrance_df)
                    st.download_button(
                        label="Download Data as CSV",
                        data=meteofrance_csv,
                        file_name=f"{selected_date.strftime('%Y-%m-%d')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Meteofrance data is not available for download.")


                # Compute scores only if actual data exists
                if 'actual' in df.columns:
                    # Optimize score computation
                    @st.cache_data(ttl=3600)
                    def compute_all_scores(df_json, cols):
                        df = pd.read_json(df_json)
                        results = {}
                        
                        for col in cols:
                            if col not in df.columns:
                                continue
                                
                            # Vectorized operations instead of apply
                            valid_mask = ~np.isnan(df['actual']) & ~np.isnan(df[col])
                            if valid_mask.sum() == 0:
                                results[f'{col}_RMSE'] = np.nan
                                results[f'{col}_MAE'] = np.nan
                                continue
                                
                            actual = df.loc[valid_mask, 'actual']
                            pred = df.loc[valid_mask, col]
                            
                            error = actual - pred
                            rmse = np.sqrt(np.mean(error**2))
                            mae = np.mean(np.abs(error))
                            
                            results[f'{col}_RMSE'] = rmse
                            results[f'{col}_MAE'] = mae
                            
                        return pd.DataFrame([results])
                    
                    # Convert columns to evaluate
                    all_cols = df.columns.tolist()
                    cols = [
                        'DA elia (11AM)', 'metno_0.5', 'meteofrance_0.5', 'avg_0.5',
                        'icon_0.5', 'knmi_0.5', 'dmi_seamless_0.5', 'oracle_0.5','tech_0.5'
                    ]
                    
                    # Add UK test to columns for evaluation
                    if 'uk-test' in all_cols:
                        cols.append('uk-test')
                        
                    cols = [col for col in cols if col in all_cols]
                    
                    # Get scores using cached function
                    scores_df = compute_all_scores(df.reset_index().to_json(), cols)
                    
                    # Split and display results
                    rmse_cols = [col for col in scores_df.columns if 'RMSE' in col]
                    mae_cols = [col for col in scores_df.columns if 'MAE' in col]
                    
                    st.subheader("RMSE Scores")
                    st.dataframe(scores_df[rmse_cols])
                    
                    st.subheader("MAE Scores")
                    st.dataframe(scores_df[mae_cols])
                else:
                    st.warning("Cannot compute scores without 'actual' column")
                    
            except Exception as e:
                st.error(f"Error processing forecasts: {str(e)}")
        else:
            st.warning("No forecast data available for the selected date.")
    except Exception as e:
        st.error(f"Error in benchmark function: {e}")
        import traceback
        st.error(traceback.format_exc())

def overview():
    """Show rankings and profit/loss overview."""
    # Add password protection
    PASSWORD = pwd_view
    if "authenticated_overview" not in st.session_state:
        st.session_state.authenticated_overview = False

    if not st.session_state.authenticated_overview:
        password = st.text_input("Enter Password to Access Overview:", type="password")
        if st.button("Login"):
            if password == PASSWORD:
                st.session_state.authenticated_overview = True
                st.success("Access granted!")
            else:
                st.error("Invalid password!")
        return

    # If authenticated, proceed with the overview logic
    st.title("Ranking & PnL")
    
    try:
        api = get_api()
        if api is None:
            st.error("Failed to authenticate API")
            return
            
        data = fetch_last_50_scores(api)
        if data.empty:
            st.error("No score data available")
            return
            
        fig = plot_rank_and_payout_separate(data)
        st.plotly_chart(fig)
        st.dataframe(data.sort_values(by='market_date', ascending=False).drop(columns='variable'))
    except Exception as e:
        st.error(f"Error in overview: {e}")
        import traceback
        st.error(traceback.format_exc())

def run_forecast_job():
    """Run forecast generation and oracle submission jobs."""
    try:
        # Load credentials from Streamlit secrets
        service_account_info = json.loads(GCLOUD)
        credentials = service_account.Credentials.from_service_account_info(service_account_info)

        # Initialize the Cloud Run Jobs client
        client = run_v2.JobsClient(credentials=credentials)
        
        project_id = "gridalert-c48ee"
        region = "europe-west6"
        
        # Run the forecast generation job
        st.write('Starting forecast generation...', pd.Timestamp.now('CET').strftime('%Y-%m-%d %H:%M:%S'))
        response = client.run_job(name=f"projects/{project_id}/locations/{region}/jobs/generate-forecast")
        st.write('Forecast job submitted', pd.Timestamp.now('CET').strftime('%Y-%m-%d %H:%M:%S'))
        
        # Wait for the job to complete
        time.sleep(5)
        
        # Run the oracle submission job
        st.write('Preparing to generate oracle...', pd.Timestamp.now('CET').strftime('%Y-%m-%d %H:%M:%S'))
        response = client.run_job(name=f"projects/{project_id}/locations/{region}/jobs/oraclepredictions")
        st.write('Oracle submitted', pd.Timestamp.now('CET').strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        st.error(f"Error running forecast job: {e}")
        import traceback
        st.error(traceback.format_exc())

def solar_view():
    """Display solar generation data and forecasts using the latest solar forecast for a selected date."""
    import datetime
    st.subheader("Solar View")

    try:
        # Add a date selector
        selected_date = st.date_input("Select date", pd.to_datetime("today"))
        
        if st.button("Clear Cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("Cache cleared!")
            time.sleep(1)
        
        # Convert selected date to datetime with UTC timezone for consistent comparison
        selected_date_utc = pd.Timestamp(selected_date).tz_localize('UTC')
        selected_date_end = selected_date_utc + pd.Timedelta(days=1)
        
        # Function to load forecast models from Google Cloud Storage
        @st.cache_data(ttl=3600, show_spinner=False)
        def load_solar_forecast_model(model_name, selected_date):
            try:
                # Create Google Cloud Storage client
                service_account_info = json.loads(GCLOUD)
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
                storage_client = storage.Client(credentials=credentials)
                bucket = storage_client.bucket('oracle_predictions')
                
                # List files for this model
                prefix = f'predico-elia/forecasts_solar/{model_name}'
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                # Filter for parquet files
                valid_files = [blob.name for blob in blobs if blob.name.endswith(".parquet")]
                
                if not valid_files:
                    return None
                
                # Extract date info and find relevant file
                date_file_pairs = []
                for file in valid_files:
                    try:
                        filename = file.split('/')[-1]
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            year, month, day = int(parts[0]), int(parts[1]), int(parts[2].split('.')[0])
                            file_date = pd.Timestamp(year=year, month=month, day=day)
                            date_file_pairs.append((file_date, file))
                    except Exception:
                        continue
                
                if not date_file_pairs:
                    return None
                
                # Find files for the selected date
                # Convert both to date objects to ensure proper comparison
                selected_date_obj = selected_date if isinstance(selected_date, datetime.date) and not isinstance(selected_date, datetime.datetime) else pd.Timestamp(selected_date).date()
                selected_date_files = [f for d, f in date_file_pairs if d.date() == selected_date_obj]
                
                # If we have files for the selected date, use the latest one
                if selected_date_files:
                    best_match = selected_date_files[-1]  # Use the last file (assuming it's the latest)
                else:
                    # If no files for the selected date, find the closest date
                    date_file_pairs.sort(key=lambda x: abs((x[0] - pd.Timestamp(selected_date)).total_seconds()))
                    best_match = date_file_pairs[0][1]
                
                # Download and load the file
                blob = bucket.blob(best_match)
                
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_parquet(temp_file.name)
                
                # Process the DataFrame
                if df.empty:
                    return None
                
                # Check for p50, p10, p90 columns (or convert equivalents)
                if all(col in df.columns for col in [0.1, 0.5, 0.9]):
                    # Rename columns to match expected format
                    df = df.rename(columns={0.1: 'p10', 0.5: 'p50', 0.9: 'p90'})
                elif all(col in df.columns for col in ['0.1', '0.5', '0.9']):
                    # Rename string columns
                    df = df.rename(columns={'0.1': 'p10', '0.5': 'p50', '0.9': 'p90'})
                
                # Check if we now have standard columns
                standard_cols = ['p50', 'p10', 'p90']
                available_cols = [col for col in standard_cols if col in df.columns]
                
                if not available_cols:
                    st.warning(f"No standard forecast columns found in {model_name} data")
                    return None
                
                # Prepare output DataFrame with renamed columns
                result = pd.DataFrame()
                for col in available_cols:
                    result[f"{model_name}_{col}"] = df[col]
                
                # Ensure datetime index
                if not isinstance(result.index, pd.DatetimeIndex):
                    if 'datetime' in df.columns:
                        result.index = pd.to_datetime(df['datetime'])
                    elif 'date' in df.columns:
                        result.index = pd.to_datetime(df['date'])
                    else:
                        # Create synthetic index based on selected date
                        result.index = pd.date_range(
                            start=selected_date_utc,
                            periods=len(result),
                            freq='H'
                        )
                
                # Make sure index is timezone-aware in UTC for consistent comparison
                if result.index.tz is None:
                    result.index = result.index.tz_localize('UTC')
                elif result.index.tz.zone != 'UTC':
                    result.index = result.index.tz_convert('UTC')
                
                return result
                
            except Exception as e:
                st.warning(f"Error loading {model_name} model: {str(e)}")
                return None
        
        # Function to load solar data with caching
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def load_solar_data():
            try:
                latest_pv = pd.read_csv('https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods087/exports/csv?lang=fr&timezone=Europe%2FBrussels&use_labels=true&delimiter=%3B',
                                     sep=';')
                latest_pv = latest_pv[['Datetime','Region','Monitored capacity',
                                   'Measured & upscaled','Day Ahead 11AM forecast',
                                   'Most recent forecast']].replace(0.0,np.nan)
                latest_pv['Datetime'] = pd.to_datetime(latest_pv['Datetime'], utc=True)
                return latest_pv
            except Exception as e:
                st.error(f"Error loading solar data: {e}")
                return pd.DataFrame()
        
        # Load the Elia data
        solar_data = load_solar_data()
        
        # Load forecast models
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        models = ['dmi_seamless', 'icon_d2', 'metno_seamless', 'meteofrance_seamless']
        forecasts = {}
        
        for i, model in enumerate(models):
            progress_text.text(f"Loading {model} forecast...")
            result = load_solar_forecast_model(model, selected_date)
            if result is not None:
                # Filter model data to match the selected date range
                result = result[(result.index >= selected_date_utc) & 
                               (result.index < selected_date_end)]
                forecasts[model] = result
            progress_bar.progress((i + 1) / len(models))
        
        progress_bar.empty()
        progress_text.empty()
        
        if solar_data.empty:
            st.error("No solar data available")
            return
            
        # Filter Elia data for the selected date
        filtered_data = solar_data[(solar_data['Datetime'] >= selected_date_utc) & 
                                (solar_data['Datetime'] < selected_date_end)]
        
        # Show data availability
        if filtered_data.empty:
            st.warning(f"No Elia data available for {selected_date}")
            return
        
        # Create country-wide totals by summing all regions
        total_data = filtered_data.loc[filtered_data.Region=='Belgium',:].groupby('Datetime').agg({
            'Monitored capacity': 'sum',
            'Measured & upscaled': 'sum',
            'Day Ahead 11AM forecast': 'sum',
            'Most recent forecast': 'sum'
        }).reset_index()
        
        # Combine forecast data with Elia data
        plot_data = total_data.copy()
        plot_data.set_index('Datetime', inplace=True)
        
        # Add model forecasts if available
        for model_name, model_data in forecasts.items():
            if model_data is not None and not model_data.empty:
                # Make sure the model data is properly filtered to the selected date range
                model_data = model_data[(model_data.index >= selected_date_utc) & 
                                      (model_data.index < selected_date_end)]
                
                # Resample model data to match Elia data frequency if needed
                target_freq = pd.infer_freq(plot_data.index)
                if target_freq and (pd.infer_freq(model_data.index) != target_freq):
                    model_data = model_data.resample(target_freq).interpolate()
                
                # Merge with plot data using the index
                model_data = model_data.reindex(plot_data.index, method='nearest')
                plot_data = pd.concat([plot_data, model_data], axis=1)
        
        # Reset index for plotting
        plot_data.reset_index(inplace=True)
        plot_data = plot_data.replace(0.0, np.nan)
        
        # Plot the country-wide totals
        fig = go.Figure()
        
        # Add measured data
        fig.add_trace(
            go.Scatter(
                x=plot_data['Datetime'],
                y=plot_data['Measured & upscaled'],
                name='Actual Generation',
                mode='lines',
                line_color='gold'
            )
        )
        
        # Add day ahead forecast
        fig.add_trace(
            go.Scatter(
                x=plot_data['Datetime'],
                y=plot_data['Day Ahead 11AM forecast'],
                name='Day Ahead Forecast (Elia)',
                mode='lines',
                line_color='orange'
            )
        )
        
        # Add model forecasts with a standard format
        model_colors = {
            'dmi_seamless': 'green',
            'icon_d2': 'purple',
            'metno_seamless': 'red',
            'meteofrance_seamless': 'blue'
        }
        
        for model_name, color in model_colors.items():
            p50_col = f'{model_name}_p50'
            p10_col = f'{model_name}_p10'
            p90_col = f'{model_name}_p90'
            
            if p50_col in plot_data.columns:
                # Add uncertainty band (p10-p90) if available
                if p10_col in plot_data.columns and p90_col in plot_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data['Datetime'],
                            y=plot_data[p90_col],
                            name=f'{model_name} p90',
                            mode='lines',
                            line_color='rgba(0,0,0,0)',
                            showlegend=False
                        )
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=plot_data['Datetime'],
                            y=plot_data[p10_col],
                            name=f'{model_name} range [p10-p90]',
                            mode='lines',
                            fill='tonexty',
                            fillcolor=f'rgba({",".join(map(str, [int(c) for c in color.lstrip("rgb(").rstrip(")").split(",")]))}, 0.2)' 
                                if color.startswith('rgb') else f'rgba({",".join(["0", "0", "0"])}, 0.2)',
                            line_color='rgba(0,0,0,0)',
                            showlegend=True
                        )
                    )
                
                # Add p50 forecast
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['Datetime'],
                        y=plot_data[p50_col],
                        name=f'{model_name} (p50)',
                        mode='lines',
                        line_color=color
                    )
                )
        
        # Update layout with dynamic y-axis range
        max_y_value = plot_data.select_dtypes(include=[np.number]).max().max()
        y_max = max_y_value * 1.1 if max_y_value > 0 else 100
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="MW",
            yaxis=dict(range=[0, y_max]),
            template="plotly_dark",
            height=500,
            title=f"Solar Generation in Belgium - {selected_date}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button for the data
        csv = plot_data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"solar_data_{selected_date}.csv",
            mime="text/csv",
        )
        
    except Exception as e:
        st.error(f"Error in solar view: {e}")
        import traceback
        st.error(traceback.format_exc())


def main():
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Generate forecast"):
        run_forecast_job()

    page_choice = st.sidebar.radio("Go to page:", ["Submission Viewer", "Overview", "Benchmark", "Solar View"])
    
    if page_choice == "Submission Viewer":
        submission_viewer()
    elif page_choice == "Overview":
        overview()
    elif page_choice == "Benchmark":
        benchmark()
    elif page_choice == "Solar View":
        solar_view()

if __name__ == "__main__":
    main()