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
from predicoclient import PredicoClient


def calculate_month_to_date_pnl(client, market_sessions, ressource_type):
    """
    Calculate the Month-to-Date PnL based on the current month's market sessions.
    
    Args:
        client: The PredicoClient instance
        market_sessions: List of market sessions
        
    Returns:
        float: The sum of daily payouts for the current month
    """
    try:

        # Get the current date and the first day of the current month
        current_date = pd.Timestamp.now().tz_localize('UTC')+pd.Timedelta(days=1)
        first_day_of_month = pd.Timestamp(year=current_date.year, month=current_date.month, day=1).tz_localize('UTC') - pd.Timedelta(days=1)
        
        # Filter market sessions for the current month
        current_month_sessions = []
        for session in market_sessions:
            session_date = pd.to_datetime(session["open_ts"])
            if session_date >= first_day_of_month:
                current_month_sessions.append(session)
        
        if not current_month_sessions:
            return 0.0  # No sessions in the current month
        
        # Collect all daily payouts for the current month
        all_payouts = []
        
        for session in current_month_sessions:
            try:
                session_id = session["id"]
                challenge_id = None
                
                # First, get challenges for this session (using Wind resource type as default)
                # You might need to check both resource types (Wind and Solar) for a complete calculation
                resource_id = ressource_type  # Wind
                challenges = client.get_challenges(session_id, resource_id)
                
                if challenges:
                    challenge_id = challenges[0]["id"]
                    
                    # Get scores for this challenge
                    url_sc = "https://predico-elia.inesctec.pt/api/v1/market/challenge/submission-scores"
                    sc_resp = requests.get(url_sc, params={"challenge": challenge_id}, headers=client.headers)
                    sc_data = sc_resp.json()["data"]["personal_metrics"]
                    
                    if sc_data:
                        df_scores = pd.DataFrame(sc_data)
                        
                        # Calculate forecast date (day after open_ts)
                        forecast_date = (pd.to_datetime(session["open_ts"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                        df_scores['market_date'] = forecast_date
                        
                        # Calculate daily payout
                        df_scores = add_daily_payout(df_scores)
                        
                        # Get the payout for this day (should be the same for all rows)
                        if 'daily_payout' in df_scores.columns and not df_scores.empty:
                            # Remove duplicate days and get the payout
                            payout_df = df_scores.drop_duplicates(subset=['market_date', 'daily_payout'])[['market_date', 'daily_payout']]
                            all_payouts.append(payout_df)
            except Exception as e:
                print(f"Error processing session {session.get('id', 'unknown')}: {e}")
                continue
        
        if all_payouts:
            combined_payouts = pd.concat(all_payouts, ignore_index=True)
            combined_payouts = combined_payouts.drop_duplicates(subset=['market_date'])
            mtd_pnl = combined_payouts['daily_payout'].sum()
            return mtd_pnl
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error calculating Month-to-Date PnL: {e}")
        return None

def submission_viewer():
    """Display submission details and visualizations with proper authentication."""
    st.title("Predico Submission Viewer")
    
    # Initialize session state variables
    if 'predico_client' not in st.session_state:
        st.session_state.predico_client = None
        st.session_state.is_authenticated = False
    
    # Authentication section
    with st.sidebar:
        st.header("Authentication")
        
        if not st.session_state.is_authenticated:
            # Login form
            with st.form("login_form"):
                email = st.text_input("Email", type="default")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")
                
                if submit_button and email and password:
                    with st.spinner("Authenticating..."):
                        client = PredicoClient(email, password)
                        if client.authenticate():
                            st.session_state.predico_client = client
                            st.session_state.is_authenticated = True
                            st.success("Logged in successfully!")
                            # Force a rerun to update the UI
                            st.rerun()
                        else:
                            st.error("Authentication failed. Please check your credentials.")
        else:
            # Show user info and logout button when logged in
            st.success(f"Logged in as: {st.session_state.predico_client.email}")
            #st.info(f"User ID: {st.session_state.predico_client.user_id}")
            
            if st.button("Logout"):
                st.session_state.predico_client = None
                st.session_state.is_authenticated = False
                st.success("Logged out successfully!")
                # Force a rerun to update the UI
                st.rerun()
    
    # Main content - only show when authenticated
    if st.session_state.is_authenticated and st.session_state.predico_client:
        client = st.session_state.predico_client
        
        try:
            # Step 1: Fetch market sessions
            with st.spinner("Fetching market sessions..."):
                market_sessions = client.get_market_sessions(status="finished")
                
                if not market_sessions:
                    st.error("No market sessions available.")
                    return
                    
                # Sort sessions by date (newest first)
                market_sessions = sorted(market_sessions, key=lambda x: x["open_ts"], reverse=True)
                
                # Create labels for the sessions
                session_labels = {}
                for session in market_sessions:
                    open_date = pd.to_datetime(session["open_ts"]).tz_convert('CET').strftime("%Y-%m-%d")
                    forecast_date = (pd.to_datetime(session["open_ts"]).tz_convert('CET') + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    label = f"Market {session['id']} - Date: {open_date} (Forecast: {forecast_date})"
                    session_labels[label] = session
                
                # Select market session from dropdown
                selected_session_label = st.selectbox(
                    "Select Market Session:", 
                    options=list(session_labels.keys()),
                    index=0
                )
                
                selected_session = session_labels[selected_session_label]
                session_id = selected_session["id"]
                
                # Display basic session info
                #st.info(f"Selected Market Session ID: {session_id}")
                
                # Resource selector (radio button)
                resource_options = {
                    "Solar": "5792ca63-2051-4186-8c5c-7167ee1c6c6f",
                    "Wind": "491949aa-8662-4010-8a29-75f4267a76c2"  # Using the same ID for both since it works
                }
                resource_type = st.radio("Select Resource Type:", list(resource_options.keys()), horizontal=True)
                resource_id = resource_options[resource_type]

            
                # Step 2: Fetch challenges for the selected market session and resource
                with st.spinner(f"Fetching challenges for market session {session_id}..."):
                    challenges = client.get_challenges(session_id, resource_id)
                    
                    if challenges:
                        # Display challenges info in expandable section
                        with st.expander("Challenge Information", expanded=False):
                            challenges_df = pd.DataFrame(challenges)
                            st.dataframe(challenges_df)
                        
                        # Get the challenge ID from the response
                        challenge_id = challenges[0]["id"]
                        target_day = challenges[0]["target_day"]
                        
                        # Step 3: Fetch forecasts for the selected challenge
                        with st.spinner(f"Fetching forecasts for challenge {challenge_id}..."):
                            # Use the user_id from the client (extracted from token)
                            forecasts_data = client.get_forecasts(challenge_id)
                            
                            if forecasts_data:
                            #    with st.expander("Raw Forecasts Data", expanded=False):
                            #        st.json(forecasts_data)
                                
                                try:
                                    st.write("Submission Scores")
                                    url_sc = "https://predico-elia.inesctec.pt/api/v1/market/challenge/submission-scores"
                                    sc_resp = requests.get(url_sc, params={"challenge": challenge_id}, headers=client.headers)
                                    sc_data = sc_resp.json()["data"]["personal_metrics"]
                                    df_scores = pd.DataFrame(sc_data)
                                    df_scores['market_date'] = forecast_date
                                    df_scores = add_daily_payout(df_scores)

                                    # Display scores
                                    if not df_scores.empty:
                                        st.dataframe(df_scores)
                                    else:
                                        st.info("No submission scores available for this challenge.")
                                except Exception as e:
                                    st.warning(f"Could not fetch submission scores: {str(e)}")
                                        

                                # Process the forecasts data
                                if "data" in forecasts_data and forecasts_data["data"]:
                                    try:
                                        # Parse forecast data
                                        raw_forecasts = forecasts_data["data"]
                                        
                                        # Create dictionaries to store values by timestamp for each variable
                                        q10_dict = {}
                                        q50_dict = {}
                                        q90_dict = {}
                                        
                                        # Process data points and organize by variable and timestamp
                                        for point in raw_forecasts:
                                            timestamp = point.get("datetime")
                                            variable = point.get("variable")
                                            value = point.get("value")
                                            
                                            # Store each value in the appropriate dictionary
                                            if variable == "q10":
                                                q10_dict[timestamp] = value
                                            elif variable == "q50":
                                                q50_dict[timestamp] = value
                                            elif variable == "q90":
                                                q90_dict[timestamp] = value
                                        
                                        # Get all unique timestamps
                                        all_timestamps = sorted(set(q10_dict.keys()) | set(q50_dict.keys()) | set(q90_dict.keys()))
                                        
                                        # Create a DataFrame with all data
                                        data_rows = []
                                        for ts in all_timestamps:
                                            data_rows.append({
                                                "timestamp": ts,
                                                "q10": q10_dict.get(ts, None),
                                                "q50": q50_dict.get(ts, None),
                                                "q90": q90_dict.get(ts, None)
                                            })
                                        
                                        # Create a DataFrame
                                        df = pd.DataFrame(data_rows)
                                        
                                        # Convert timestamp to datetime
                                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                                        
                                        # Set timestamp as index and sort
                                        df.set_index('timestamp', inplace=True)
                                        df.sort_index(inplace=True)
                                        
                                        # Display dataframe
                                       # with st.expander("Forecast Data Table", expanded=False):
                                       #     st.dataframe(df)
                                        
                                        # Extract the date from the first timestamp for fetching actual data
                                        selected_date_str = (df.index[0]+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                                        selected_date_utc = pd.to_datetime(selected_date_str)
                                        selected_date_end = selected_date_utc + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                                        
                                        # Helper functions for fetching actual data
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

                                        @st.cache_data(ttl=3600)  # Cache for 1 hour
                                        def get_cached_actual(date):
                                            try:
                                                return get_latest_wind_offshore(date)
                                            except Exception as e:
                                                st.error(f"Error getting latest offshore data: {e}")
                                                return pd.DataFrame()
                                        
                                        actual_data = None
                                        
                                        # Fetch actual data based on resource type
                                        if resource_type == "Solar":
                                            # Fetch actual solar data
                                            api_url = f'https://griddata.elia.be/eliabecontrols.prod/interface/solareforecasting/chartdataforzone?dateFrom={selected_date_utc.strftime("%Y-%m-%d")}&dateTo={selected_date_end.strftime("%Y-%m-%d")}&sourceID=1'
                                                    
                                            #st.info(f"Fetching actual solar data from: {api_url}")
                                            
                                            try:
                                                # Read the JSON data
                                                response = requests.get(api_url)
                                                
                                                if response.status_code == 200:
                                                    solar_data = response.json()
                                                    
                                                    # Convert the JSON data to a pandas DataFrame
                                                    actual_pv = pd.DataFrame(solar_data).iloc[:96]
                                                    
                                                    # Convert timestamp string to datetime
                                                    actual_pv['Datetime'] = pd.to_datetime(actual_pv['startsOn'])
                                                    
                                                    # Check if the index has timezone info
                                                    has_tz = False
                                                    if not df.index.empty:
                                                        has_tz = df.index[0].tzinfo is not None
                                                    
                                                    # Make the timestamps timezone-aware or timezone-naive to match the forecast data
                                                    if has_tz:
                                                        # If forecast data is timezone-aware, make actual data timezone-aware
                                                        if actual_pv['Datetime'].dt.tz is None:
                                                            actual_pv['Datetime'] = actual_pv['Datetime'].dt.tz_localize('CET')
                                                    else:
                                                        # If forecast data is timezone-naive, make actual data timezone-naive
                                                        if actual_pv['Datetime'].dt.tz is not None:
                                                            actual_pv['Datetime'] = actual_pv['Datetime'].dt.tz_localize(None)
                                                    
                                                    actual_pv = actual_pv.set_index('Datetime')
                                                    
                                                    # Create timezone-consistent comparison dates
                                                    if has_tz and selected_date_utc.tzinfo is None:
                                                        comparison_start = selected_date_utc.tz_localize('CET')
                                                        comparison_end = selected_date_end.tz_localize('CET')
                                                    elif not has_tz and selected_date_utc.tzinfo is not None:
                                                        comparison_start = selected_date_utc.tz_localize(None)
                                                        comparison_end = selected_date_end.tz_localize(None)
                                                    else:
                                                        comparison_start = selected_date_utc
                                                        comparison_end = selected_date_end
                                                    
                                                    # Filter to match our date range
                                                    actual_pv = actual_pv[(actual_pv.index >= comparison_start) & 
                                                                        (actual_pv.index <= comparison_end)]
                                                    
                                                    if not actual_pv.empty:
                                                        #st.success(f"Successfully loaded actual solar measurements for {len(actual_pv)} time points")
                                                        actual_data = actual_pv
                                                        
                                                        # Display actual data
                                                        with st.expander("Actual Solar Data", expanded=False):
                                                            st.dataframe(actual_data)
                                                else:
                                                    st.warning(f"Could not fetch actual solar data: HTTP {response.status_code}")

                                            except Exception as e:
                                                st.error(f"Error fetching actual solar data: {str(e)}")
                                                import traceback
                                                st.error(traceback.format_exc())
                                        else:  # Wind resource type
                                            #st.info(f"Fetching actual wind data for: {selected_date_utc.strftime('%Y-%m-%d')}")

                                            
                                            try:
                                                # Get cached wind data
                                                actual_data = get_cached_actual(selected_date_utc-pd.Timedelta(days=1))
                                                
                                                if not actual_data.empty:
                                                    #st.success(f"Successfully loaded actual wind measurements for {len(actual_data)} time points")
                                                    
                                                    # Display actual data
                                                    with st.expander("Actual Wind Data", expanded=False):
                                                        st.dataframe(actual_data)
                                                else:
                                                    st.warning("No actual wind data available for the selected date.")
                                                    
                                            except Exception as e:
                                                st.error(f"Error fetching actual wind data: {str(e)}")
                                                import traceback
                                                st.error(traceback.format_exc())
                                        
                                        # Create the plot
                                        st.subheader(f"{resource_type} Power Forecast for {target_day}")
                                        
                                        # Create a plotly figure
                                        fig = go.Figure()
                                        df = df.tz_convert('CET')
                                        # Add the uncertainty band (q10 - q90)
                                        fig.add_trace(
                                            go.Scatter(
                                                x=df.index,
                                                y=df['q90'],
                                                name="q90",
                                                mode="lines",
                                                line=dict(width=0),
                                                showlegend=False
                                            )
                                        )
                                        
                                        fig.add_trace(
                                            go.Scatter(
                                                x=df.index,
                                                y=df['q10'],
                                                name="Uncertainty Band (q10-q90)",
                                                mode="lines",
                                                fill='tonexty',
                                                fillcolor='rgba(0, 100, 80, 0.2)',
                                                line=dict(width=0),
                                                showlegend=True
                                            )
                                        )
                                        
                                        # Add median forecast line
                                        fig.add_trace(
                                            go.Scatter(
                                                x=df.index,
                                                y=df['q50'],
                                                name="Median Forecast (q50)",
                                                mode="lines",
                                                line=dict(color='rgb(0, 100, 80)', width=2)
                                            )
                                        )
                                        
                                        # Add actual data if available
                                        if actual_data is not None and not actual_data.empty:
                                            actual_data = actual_data.tz_convert('CET')
                                            if resource_type == "Solar":
                                                # For Solar data
                                                # Add Elia Day-Ahead forecast
                                                if 'dayAheadForecast' in actual_data.columns:
                                                    fig.add_trace(
                                                        go.Scatter(
                                                            x=actual_data.index,
                                                            y=actual_data['dayAheadForecast'],
                                                            name="ELIA DA",
                                                            mode="lines",
                                                            line=dict(color='darkorange', width=2)
                                                        )
                                                    )
                                                
                                                # Add actual measurements
                                                if 'realTime' in actual_data.columns:
                                                    fig.add_trace(
                                                        go.Scatter(
                                                            x=actual_data.index,
                                                            y=actual_data['realTime'],
                                                            name="Actual Measurements",
                                                            mode="lines",
                                                            line=dict(color='white', width=2, dash='solid')
                                                        )
                                                    )

                                                if 'mostRecentForecast' in actual_data.columns:
                                                    fig.add_trace(
                                                        go.Scatter(
                                                            x=actual_data.index,
                                                            y=actual_data['mostRecentForecast'],
                                                            name="ELIA latest",
                                                            mode="lines",
                                                            line=dict(color='purple', width=2, dash='solid')
                                                        )
                                                    )
                                            else:
                                                # For Wind data
                                                # Add Elia Day-Ahead forecast
                                                if 'DA elia (11AM)' in actual_data.columns:
                                                    fig.add_trace(
                                                        go.Scatter(
                                                            x=actual_data.index,
                                                            y=actual_data['DA elia (11AM)'],
                                                            name="ELIA DA",
                                                            mode="lines",
                                                            line=dict(color='darkorange', width=2)
                                                        )
                                                    )
                                                
                                                # Add actual measurements
                                                if 'actual' in actual_data.columns:
                                                    fig.add_trace(
                                                        go.Scatter(
                                                            x=actual_data.index,
                                                            y=actual_data['actual'],
                                                            name="Actual Measurements",
                                                            mode="lines",
                                                            line=dict(color='white', width=2, dash='solid')
                                                        )
                                                    )


                                                if 'latest elia forecast' in actual_data.columns:
                                                    fig.add_trace(
                                                        go.Scatter(
                                                            x=actual_data.index,
                                                            y=actual_data['latest elia forecast'],
                                                            name="ELIA latest",
                                                            mode="lines",
                                                            line=dict(color='purple', width=2, dash='solid')
                                                        )
                                                    )
                                        
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Calculate RMSE for actual vs forecasts
                                        if actual_data is not None and not actual_data.empty:
                                            st.subheader("Forecast Accuracy Metrics")
                                            
                                            # Prepare a merged dataframe for RMSE calculation
                                            if resource_type == "Solar":
                                                actual_col = 'realTime'
                                                elia_col = 'dayAheadForecast'
                                                latest_col = 'mostRecentForecast'
                                            else:  # Wind
                                                actual_col = 'actual'
                                                elia_col = 'DA elia (11AM)'
                                                latest_col = 'latest elia forecast'
                                            
                                            # Check if required columns exist
                                            if actual_col in actual_data.columns and elia_col in actual_data.columns:
                                                # Create a common timestamp index for comparison
                                                common_index = sorted(set(df.index).intersection(set(actual_data.index)))
                                                
                                                if common_index:
                                                    # Initialize metrics container
                                                    metrics = {}
                                                    
                                                    # Prepare data for comparison
                                                    actual_values = actual_data.loc[common_index, actual_col]
                                                    elia_da_values = actual_data.loc[common_index, elia_col]
                                                    elia_latest_values = actual_data.loc[common_index, latest_col]
                                                    

                                                    # Align forecast data with the same index
                                                    forecast_values = df.loc[common_index, 'q50']
                                                    
                                                    # Calculate RMSE
                                                    import numpy as np
                                                    
                                                    # RMSE for ELIA DA vs Actual
                                                    if not np.isnan(actual_values).all() and not np.isnan(elia_da_values).all():
                                                        elia_rmse = np.sqrt(np.nanmean((elia_da_values - actual_values)**2))
                                                        metrics["ELIA DA RMSE"] = elia_rmse
                                                    
                                                    # RMSE for Median Forecast vs Actual
                                                    if not np.isnan(actual_values).all() and not np.isnan(forecast_values).all():
                                                        forecast_rmse = np.sqrt(np.nanmean((forecast_values - actual_values)**2))
                                                        metrics["Median Forecast RMSE"] = forecast_rmse
                                                    
                                                    # RMSE for Median Forecast vs Actual
                                                    if not np.isnan(actual_values).all() and not np.isnan(elia_latest_values).all():
                                                        latest_rmse = np.sqrt(np.nanmean((elia_latest_values - actual_values)**2))
                                                        metrics["ELIA latest RMSE"] = latest_rmse
                                                    
                                                    # Display metrics
                                                    col1, col2, col3 = st.columns(3)
                                                    
                                                    with col1:
                                                        if "ELIA DA RMSE" in metrics:
                                                            st.metric("ELIA DA RMSE", f"{metrics['ELIA DA RMSE']:.2f}")
                                                        else:
                                                            st.info("Could not calculate ELIA DA RMSE (insufficient data)")
                                                            
                                                    with col2:
                                                        if "Median Forecast RMSE" in metrics:
                                                            st.metric("Median Forecast RMSE", f"{metrics['Median Forecast RMSE']:.2f}")
                                                        else:
                                                            st.info("Could not calculate Median Forecast RMSE (insufficient data)")
                                                
                                                    with col3:
                                                        if "ELIA latest RMSE" in metrics:
                                                            st.metric("ELIA latest RMSE", f"{metrics['ELIA latest RMSE']:.2f}")
                                                        else:
                                                            st.info("Could not calculate ELIA latest RMSE (insufficient data)")
                                                    


                                                    # Display improvement percentage if both metrics are available
                                                    if "ELIA DA RMSE" in metrics and "Median Forecast RMSE" in metrics:
                                                        improvement = (metrics["ELIA DA RMSE"] - metrics["Median Forecast RMSE"]) / metrics["ELIA DA RMSE"] * 100
                                                        if improvement > 0:
                                                            st.success(f"Your forecast is {improvement:.2f}% better than ELIA's day-ahead forecast")
                                                        else:
                                                            st.warning(f"Your forecast is {abs(improvement):.2f}% worse than ELIA's day-ahead forecast")
                                                
                                                else:
                                                    st.warning("No overlapping timestamps between forecast and actual data for RMSE calculation")
                                            else:
                                                st.warning(f"Missing required columns for RMSE calculation: {actual_col} and/or {elia_col}")
                                        
                                    except Exception as e:
                                        st.error(f"Error processing forecasts data: {e}")
                                        import traceback
                                        st.error(traceback.format_exc())
                                else:
                                    st.warning("No forecasts data available for this challenge and user.")

                    else:
                        st.warning(f"No challenges found for market session {session_id} and resource {resource_type}")
                
        except Exception as e:
            st.error(f"Error in submission viewer: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        # Show login message when not authenticated
        st.info("Please log in using the sidebar to access the submission viewer.")

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

                df = df.iloc[-96-4-3:-4].copy()
                
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
    """Display submission details and visualizations with proper authentication."""
    st.title("PnL Viewer")
    
    # Initialize session state variables
    if 'predico_client' not in st.session_state:
        st.session_state.predico_client = None
        st.session_state.is_authenticated = False
    
    # Authentication section
    with st.sidebar:
        st.header("Authentication")
        
        if not st.session_state.is_authenticated:
            # Login form
            with st.form("login_form"):
                email = st.text_input("Email", type="default")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login")
                
                if submit_button and email and password:
                    with st.spinner("Authenticating..."):
                        client = PredicoClient(email, password)
                        if client.authenticate():
                            st.session_state.predico_client = client
                            st.session_state.is_authenticated = True
                            st.success("Logged in successfully!")
                            # Force a rerun to update the UI
                            st.rerun()
                        else:
                            st.error("Authentication failed. Please check your credentials.")
        else:
            # Show user info and logout button when logged in
            st.success(f"Logged in as: {st.session_state.predico_client.email}")
            #st.info(f"User ID: {st.session_state.predico_client.user_id}")
            
            if st.button("Logout"):
                st.session_state.predico_client = None
                st.session_state.is_authenticated = False
                st.success("Logged out successfully!")
                # Force a rerun to update the UI
                st.rerun()
    
    # Main content - only show when authenticated
    if st.session_state.is_authenticated and st.session_state.predico_client:
        client = st.session_state.predico_client
        
        try:
            # Step 1: Fetch market sessions
            with st.spinner("Fetching market sessions..."):
                market_sessions = client.get_market_sessions(status="finished")
                
                if not market_sessions:
                    st.error("No market sessions available.")
                    return
                    
                # Sort sessions by date (newest first)
                market_sessions = sorted(market_sessions, key=lambda x: x["open_ts"], reverse=True)
                
                # Create labels for the sessions
                session_labels = {}
                for session in market_sessions:
                    open_date = pd.to_datetime(session["open_ts"]).tz_convert('CET').strftime("%Y-%m-%d")
                    forecast_date = (pd.to_datetime(session["open_ts"]).tz_convert('CET') + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    label = f"Market {session['id']} - Date: {open_date} (Forecast: {forecast_date})"
                    session_labels[label] = session
                
                # Resource selector (radio button)
                resource_options = {
                    "Solar": "5792ca63-2051-4186-8c5c-7167ee1c6c6f",
                    "Wind": "491949aa-8662-4010-8a29-75f4267a76c2"  # Using the same ID for both since it works
                }

                with st.spinner("Calculating Month-to-Date PnL ..."):
                    mtd_pnl = calculate_month_to_date_pnl(client, market_sessions, resource_options['Solar'])
                    
                    # Display Month-to-Date PnL in a prominent box
                    if mtd_pnl is not None:
                        st.success(f"SOLAR Month-to-Date PnL: €{mtd_pnl:.2f}")
                    else:
                        st.warning("Could not calculate Month-to-Date PnL")
                    
                    mtd_pnl = calculate_month_to_date_pnl(client, market_sessions, resource_options['Wind'])
                    
                    # Display Month-to-Date PnL in a prominent box
                    if mtd_pnl is not None:
                        st.success(f"WIND Month-to-Date PnL: €{mtd_pnl:.2f}")
                    else:
                        st.warning("Could not calculate Month-to-Date PnL")
        except Exception as e:
            st.error(f"Error PnL: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        # Show login message when not authenticated
        st.info("Please log in using the sidebar to access the Pnl.")


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
    """Display solar forecasts for all models with optimized groupby operations and forecast scoring."""
    import datetime
    import requests
    import numpy as np
    
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
        selected_date_utc = pd.Timestamp(selected_date).tz_localize('UTC') + pd.Timedelta(days=1)
        selected_date_end = selected_date_utc + pd.Timedelta(days=2)
        
        # Function to load and process solar forecast data
        @st.cache_data(ttl=3600, show_spinner=False)
        def load_solar_forecast_data(model_name, selected_date):
            try:
                # Create Google Cloud Storage client
                service_account_info = json.loads(GCLOUD)
                credentials = service_account.Credentials.from_service_account_info(service_account_info)
                storage_client = storage.Client(credentials=credentials)
                bucket = storage_client.bucket('oracle_predictions')
                
                # List files from the regional subdirectory
                prefix = f'predico-elia/forecasts_solar/v2/{model_name}'
                blobs = list(bucket.list_blobs(prefix=prefix))
                
                # Filter for parquet files
                valid_files = [blob.name for blob in blobs if blob.name.endswith(".parquet")]
                
                if not valid_files:
                    st.warning(f"No solar forecast files found for model: {model_name}")
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
                    st.warning(f"Could not parse dates from filenames for model: {model_name}")
                    return None
                
                # Find files for the selected date
                selected_date_obj = selected_date if isinstance(selected_date, datetime.date) and not isinstance(selected_date, datetime.datetime) else pd.Timestamp(selected_date).date()
                selected_date_files = [f for d, f in date_file_pairs if d.date() == selected_date_obj]
                
                # If we have files for the selected date, use the latest one
                if selected_date_files:
                    best_match = selected_date_files[-1]  # Use the last file (assuming it's the latest)
                else:
                    # If no files for the selected date, find the closest date
                    date_file_pairs.sort(key=lambda x: abs((x[0] - pd.Timestamp(selected_date)).total_seconds()))
                    best_match = date_file_pairs[0][1]
                    st.info(f"No data found for {selected_date_obj} in model {model_name}. Using closest available date: {date_file_pairs[0][0].date()}")
                
                # Download and load the file
                blob = bucket.blob(best_match)
                
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=True) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    df = pd.read_parquet(temp_file.name)
                
                # Process the DataFrame
                if df.empty:
                    st.warning(f"Empty DataFrame for model: {model_name}")
                    return None
                
                # Check for minimal required columns for plotting
                required_cols = ['Region', 'Day Ahead 11AM forecast', 'rec', 'rec_0.2', 'rec_0.8']
                missing_cols = [col for col in required_cols if col not in df.columns]

                
                if missing_cols:
                    st.warning(f"Missing required columns in {model_name} data: {missing_cols}")
                    st.write(f"Available columns: {df.columns.tolist()}")
                    #return None
                
                # Ensure datetime index
                if 'Datetime' in df.columns:
                    df.index = pd.to_datetime(df['Datetime'])
                elif not isinstance(df.index, pd.DatetimeIndex):
                    # Create synthetic index based on selected date
                    df.index = pd.date_range(
                        start=selected_date_utc,
                        periods=len(df),
                        freq='H'
                    )
                
                # Make sure index is timezone-aware in UTC for consistent comparison
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                elif df.index.tz.zone != 'UTC':
                    df.index = df.index.tz_convert('UTC')
                
                return df
                
            except Exception as e:
                st.error(f"Error loading {model_name} model: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return None

        # Function to calculate forecast scoring metrics
        def calculate_forecast_scores(df, actual_column='Measured & upscaled'):
            """
            Calculate forecast accuracy metrics for different forecast columns against actual measurements.
            
            Parameters:
            df (pd.DataFrame): DataFrame containing forecast and actual data
            actual_column (str): Column name containing actual measured values
            
            Returns:
            pd.DataFrame: DataFrame with scoring metrics for each forecast
            """
            if actual_column not in df.columns or df[actual_column].isna().all():
                st.warning(f"No actual measurement data ({actual_column}) available for scoring")
                return None
                
            # Filter to only rows where we have actual measurements
            score_df = df.dropna(subset=[actual_column]).copy()
            
            if score_df.empty:
                st.warning("No overlapping data points between forecasts and measurements")
                return None
                
            # Identify all forecast columns (excluding uncertainty bounds)
            forecast_columns = [col for col in score_df.columns 
                               if col not in [actual_column, 'rec_0.2', 'rec_0.8'] 
                               and not col.startswith('Region') 
                               and not col.startswith('Model')
                               and not col.startswith('Datetime')]
            
            # Create a DataFrame to store results
            results = []
            
            for forecast in forecast_columns:
                # Calculate metrics
                mae = np.mean(np.abs(score_df[actual_column] - score_df[forecast]))
                rmse = np.sqrt(np.mean(np.square(score_df[actual_column] - score_df[forecast])))
                bias = np.mean(score_df[forecast] - score_df[actual_column])
                
                results.append({
                    'Forecast': forecast,
                    'MAE (MW)': mae,
                    'RMSE (MW)': rmse,
                    'Bias (MW)': bias,
                })
            
            # Convert to DataFrame and sort by RMSE
            results_df = pd.DataFrame(results).sort_values('RMSE (MW)')
            return results_df

        # Define all available models
        available_models = ['meteofrance_seamless', 'dmi_seamless', 'icon_d2', 'metno_seamless']
        
        # Load data for all models and store them in a dictionary
        model_data = {}
        total_dfs = {}
        
        for model_name in available_models:
            with st.spinner(f"Loading {model_name} solar forecast data..."):
                df = load_solar_forecast_data(model_name, selected_date)
                
                if df is not None:
                    # Filter data to match the selected date range
                    df = df[(df.index >= selected_date_utc) & (df.index <= selected_date_end)]
                    
                    if not df.empty:
                        model_data[model_name] = df
                        
                        # Create a copy to avoid SettingWithCopyWarning
                        df_plot = df.copy()
                        
                        # Add Model column for better tracking
                        df_plot['Model'] = model_name
                        
                        # Metrics to be aggregated
                        metrics = ['Day Ahead 11AM forecast', 'rec', 'rec_0.2', 'rec_0.8']
                        
                        try:

                            # Use pandas groupby directly - much more efficient
                            regional_df = df_plot.groupby(['Region', df_plot.index]).agg({
                                metric: 'sum' for metric in metrics
                            }).reset_index()

                            
                            # Rename the datetime index column
                            regional_df.rename(columns={'level_1': 'Datetime'}, inplace=True)
                            
                            # Add Model column back
                            regional_df['Model'] = model_name

                            st.dataframe(regional_df)

                            
                            if not regional_df.empty:
                                # Group by datetime only and sum all regions together
                                total_df = regional_df.groupby('Datetime').agg({
                                    'Day Ahead 11AM forecast': 'sum',
                                    'rec': 'sum',
                                    'rec_0.2': 'sum',
                                    'rec_0.8': 'sum'
                                }).reset_index().set_index('Datetime')
                                
                                total_df = total_df.resample('15min').interpolate().iloc[:96]
                                total_dfs[model_name] = total_df
                        except:
                            total_df = df_plot.copy()
                            total_df['Model'] = model_name
                            print(total_df)
                            total_df.columns = ['Day Ahead 11AM forecast','rec','rec_0.2','rec_0.8']
                            total_dfs[model_name] = total_df
        
        if not total_dfs:
            st.error("No forecast data available for any model")
            return
        
        # Fetch actual measured data from Elia
        actual_data = None
        with st.spinner("Fetching actual measured PV data from Elia..."):
            try:
                # Using the new API endpoint with direct JSON response
                api_url = f'https://griddata.elia.be/eliabecontrols.prod/interface/solareforecasting/chartdataforzone?dateFrom={selected_date_utc.strftime("%Y-%m-%d")}&dateTo={selected_date_utc.strftime("%Y-%m-%d")}&sourceID=1'
                
                st.info(f"Fetching solar data from: {api_url}")
                
                # Read the JSON data directly instead of CSV
                response = requests.get(api_url)
                if response.status_code == 200:
                    solar_data = response.json()
                    
                    # Convert the JSON data to a pandas DataFrame
                    actual_pv = pd.DataFrame(solar_data)
                    
                    # Convert timestamp string to datetime
                    actual_pv['Datetime'] = pd.to_datetime(actual_pv['startsOn'])
                    actual_pv = actual_pv.set_index('Datetime')
                    
                    # Filter to match our date range
                    actual_pv = actual_pv[(actual_pv.index >= selected_date_utc) & 
                                         (actual_pv.index <= selected_date_end)]
                    
                    if not actual_pv.empty:
                        st.success(f"Successfully loaded actual measurements for {len(actual_pv)} time points")
                        actual_data = actual_pv
                        
                        # Add actual data to each model's total_df
                        for model_name, total_df in total_dfs.items():
                            # Keep only data points that match our total_df index
                            common_indices = actual_pv.index.intersection(total_df.index)
                            
                            if len(common_indices) > 0:
                                total_dfs[model_name].loc[common_indices, 'Measured & upscaled'] = actual_pv.loc[common_indices, 'realTime']
                                total_dfs[model_name].loc[common_indices, 'Most recent forecast'] = actual_pv.loc[common_indices, 'mostRecentForecast']
                                total_dfs[model_name].loc[common_indices, 'Week ahead forecast'] = actual_pv.loc[common_indices, 'weekAheadForecast']
                    else:
                        st.warning("No matching actual measurement data found for the selected date range")
                else:
                    st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error fetching actual data: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        
        # Create combined dataframe with data from all models
        combined_df = pd.DataFrame()
        
        for model_name, total_df in total_dfs.items():
            if combined_df.empty:
                combined_df = total_df.copy()
                # Rename columns to include model name
                combined_df.rename(columns={
                    'Day Ahead 11AM forecast': f'Day Ahead 11AM forecast ({model_name})',
                    'rec': f'p50 ({model_name})',
                    'rec_0.2': f'rec_0.2 ({model_name})',
                    'rec_0.8': f'rec_0.8 ({model_name})'
                }, inplace=True)
            else:
                # Add columns from this model
                combined_df[f'Day Ahead 11AM forecast ({model_name})'] = total_df['Day Ahead 11AM forecast']
                combined_df[f'p50 ({model_name})'] = total_df['rec']
                combined_df[f'rec_0.2 ({model_name})'] = total_df['rec_0.2']
                combined_df[f'rec_0.8 ({model_name})'] = total_df['rec_0.8']
        
        # Add the actual measurements (should be the same for all models)
        if actual_data is not None:
            for col in ['Measured & upscaled', 'Most recent forecast', 'Week ahead forecast']:
                if col in next(iter(total_dfs.values())):
                    combined_df[col] = next(iter(total_dfs.values()))[col]
        
        # Create figure for combined plot
        fig = go.Figure()
        
        # Define colors for different models
        model_colors = {
            'meteofrance_seamless': 'green',
            'dmi_seamless': 'blue',
            'icon_d2': 'red',
            'metno_seamless': 'yellow',
            'hyb1':'lightblue'
        }
        
        # Add traces for each model's p50 forecast
        for model_name in total_dfs.keys():
            fig.add_trace(
                go.Scatter(
                    x=combined_df.index,
                    y=combined_df[f'p50 ({model_name})'],
                    name=f"p50 ({model_name})",
                    mode='lines',
                    line_color=model_colors.get(model_name, 'gray')
                )
            )
        
        # Add ELIA Day Ahead 11AM forecast from the first model
        first_model = list(total_dfs.keys())[0]
        fig.add_trace(
            go.Scatter(
                x=combined_df.index,
                y=combined_df[f'Day Ahead 11AM forecast ({first_model})'],
                name=f"ELIA DA 11AM",
                mode='lines',
                line_color='orange'
            )
        )
        
        # Add actual measured data if available
        if 'Measured & upscaled' in combined_df.columns and not combined_df['Measured & upscaled'].isna().all():
            fig.add_trace(
                go.Scatter(
                    x=combined_df.index,
                    y=combined_df['Measured & upscaled'],
                    name=f"Actual Measured & Upscaled",
                    mode='lines',
                    line_color='white',
                    line_width=2
                )
            )
        
        # Update layout with dynamic y-axis range
        max_y_value = combined_df.max().max()
        y_max = max_y_value * 1.1 if max_y_value > 0 else 100
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="MW",
            yaxis=dict(range=[0, y_max]),
            template="plotly_dark",
            height=600,
            title=f"Total Solar Forecast - All Models - {selected_date}",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    
        
        # Calculate and display forecast scoring if actual measurements are available
        if actual_data is not None:
            st.subheader("Forecast Scoring")
            
            with st.spinner("Calculating forecast scores..."):
                # Prepare a DataFrame for scoring with data from all models
                score_data = combined_df.copy().rename(columns={'Day Ahead 11AM forecast (icon_d2)':'ELIA DA 11AM'})
                score_data[score_data<10] = 0
                score_data['avg icon+dmi'] = 0.5*(score_data['p50 (dmi_seamless)'] + score_data['p50 (icon_d2)'])
                score_data['avg ALL'] = 0.25*(score_data['p50 (dmi_seamless)']+score_data['p50 (metno_seamless)']+score_data['p50 (meteofrance_seamless)'] + score_data['p50 (icon_d2)'])
                
             
                score_data[score_data<10] = 0.0
                
                # Keep only forecast columns and actual data
                exclude_cols = [col for col in score_data.columns if 'rec_0.2' in col or 'rec_0.8' in col or 'Most recent forecast' in col or 'Week ahead forecast' in col
                                or 'Day Ahead 11AM forecast (meteofrance_seamless)' in col or 'Day Ahead 11AM forecast (metno_seamless)' in col 
                                or 'Day Ahead 11AM forecast (dmi_seamless)' in col 
                                ]
                score_data = score_data.drop(columns=exclude_cols)
                
                scores_df = calculate_forecast_scores(score_data)
                
                if scores_df is not None:
                    # Create a styled dataframe for better visualization
                    def highlight_min(s):
                        """Highlight the minimum in a Series green."""
                        is_min = s == s.min()
                        return ['background-color: #006400' if v else '' for v in is_min]
                    
                    # Format the dataframe for display
                    display_df = scores_df.copy()
                    for col in ['MAE (MW)', 'RMSE (MW)', 'Bias (MW)']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].round(2)
                    
                    # Apply styling
                    styled_df = display_df.style.apply(highlight_min, subset=['MAE (MW)', 'RMSE (MW)'])
                    
                    # Display the scores
                    st.write("Forecast performance metrics compared to actual measurements:")
                    st.dataframe(styled_df)
                    
                    # Also provide explanations of the metrics
                    with st.expander("Explanation of Scoring Metrics"):
                        st.markdown("""
                        - **MAE (Mean Absolute Error)**: Average absolute difference between forecasted and actual values. Lower is better.
                        - **RMSE (Root Mean Square Error)**: Square root of the average of squared differences. More sensitive to large errors. Lower is better.
                        - **Bias**: Average difference (forecast - actual). Positive values indicate over-forecasting, negative values indicate under-forecasting.
                        """)
                    
                    # Create a bar chart comparing RMSE values
                    fig_scores = go.Figure()
                    fig_scores.add_trace(go.Scatter(
                    x=score_data.index,
                    y=score_data['Measured & upscaled'] - score_data['avg icon+dmi'],
                    name=f"avg ALL",
                    mode='lines',
                    line_color='red',
                    showlegend=False
                    ))
                    
                    fig_scores.update_layout(
                        title='error profile ',
                        xaxis_title='',
                        yaxis_title='error (MW)',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig_scores, use_container_width=True)
                    
                    # Create a scatter plot of forecast vs actual for each model
                    
                    # Download button for scores
                    scores_csv = scores_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Forecast Scores as CSV",
                        data=scores_csv,
                        file_name=f"solar_forecast_scores_{selected_date}.csv",
                        mime="text/csv",
                    )
        
        # Show combined data table
        with st.expander("View Combined Data"):
            st.dataframe(combined_df)
        
        # Add download button for combined data
        combined_csv = combined_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Combined Solar Data as CSV",
            data=combined_csv,
            file_name=f"solar_forecasts_all_models_{selected_date}.csv",
            mime="text/csv",
        )
        
    except Exception as e:
        st.error(f"Error in solar view: {e}")
        import traceback
        st.error(traceback.format_exc())

def main():
    st.sidebar.title("Navigation")
    
    try:
        if st.session_state.is_authenticated and st.session_state.predico_client:
            if st.sidebar.button("Generate forecast"):
                run_forecast_job()
    except:
        pass

    page_choice = st.sidebar.radio("Go to page:", ["Submission Viewer", "PnL", "wind models", "solar models"])
    
    if page_choice == "Submission Viewer":
        submission_viewer()
    elif page_choice == "PnL":
        overview()
    elif page_choice == "wind models":
        benchmark()
    elif page_choice == "solar models":
        solar_view()


if __name__ == "__main__":
    main()