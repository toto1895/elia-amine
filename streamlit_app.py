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

# Cache configuration
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
@st.cache_resource(ttl=3600)
def get_api():
    """Create and authenticate API instance."""
    api = PredicoAPI(user_env, pwd_env)
    api.authenticate()
    return api

@st.cache_data(ttl=1800)
def list_submissions(_api):
    """Return submission info sorted by most recent submission time."""
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

@st.cache_data(ttl=1800)
def fetch_submission_data(_api, _challenge_id, _submission_time):
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

@st.cache_data(ttl=3600)
def add_daily_payout(df, daily_pool=225.8065):
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

@st.cache_data(ttl=3600)
def fetch_last_50_scores(_api):
    """
    Fetch the last 50 submissions and retrieve their scores.
    Returns a combined DataFrame of all personal_metrics found.
    """
    # Get list of submissions
    df_subs = list_submissions(_api)
    
    # Take the latest 50
    latest_50 = df_subs.head(60)  # top 50 rows if sorted by descending time
    
    all_scores = []
    for _, row in latest_50.iterrows():
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

@st.cache_data
def plot_rank_and_payout_separate(df):
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

# Error metrics functions
def calculate_rmse(actual, predicted):
    """Compute Root Mean Square Error (RMSE)."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((predicted - actual) ** 2))

def calculate_mase(actual, predicted, training_actual=None):
    """Compute Mean Absolute Scaled Error (MASE)."""
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

def mean_pinball_loss(actual, forecast, alpha=0.5):
    """Compute pinball loss."""
    return np.mean(np.maximum(alpha*(actual - forecast), (alpha-1)*(actual - forecast)))

# File handling functions
@st.cache_data(ttl=1800)
def get_latest_da_fcst_file(selected_date, files):
    """Get the latest day-ahead forecast file for the selected date."""
    selected_str = pd.to_datetime(selected_date).strftime("%Y_%m_%d")
    files_time = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        basename = f.split("/")[-1].split('_')
        date_part = basename[0]+'_'+basename[1]+'_'+basename[2]
        hour = basename[3] 
        
        if (date_part == selected_str) and (int(hour) < 10):
            files_time.append(f)

    if len(files_time) == 0:
        return None
    selected_file = sorted(files_time)
    return selected_file[-1]

@st.cache_data(ttl=1800)
def get_latest_wind_offshore(start) -> pd.DataFrame:
    """Get the latest wind offshore data."""
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

# Application pages
def submission_viewer():
    """Display submission details and visualizations."""
    st.subheader("Submission Viewer")

    # Authenticate & get submissions
    api = get_api()
    df_subs = list_submissions(api)

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
    forecasts.index = forecasts.index.tz_localize(None)

    # Display data
    st.subheader("Scores for This Submission")
    if scores.empty:
        st.warning("No scoring data for this submission. Showing forecasts only.")
    else:
        st.dataframe(scores)

    st.subheader("Forecast vs Actual")

    if scores.empty:
        data_slice = forecasts
        market_date = forecasts["q10"].dropna().index[0]
    else:
        market_date = scores["market_date"].iloc[0]
        
    data_slice = forecasts.dropna(subset='q10')
    df_sc = data_slice[['q50','DA elia (11AM)','actual elia']].copy().dropna()
    
    # Calculate metrics
    myrmse = round(calculate_rmse(df_sc['q50'], df_sc['actual elia']),1)
    eliarmse = round(calculate_rmse(df_sc['DA elia (11AM)'], df_sc['actual elia']),1)
    mymase = round(calculate_mase(df_sc['q50'], df_sc['actual elia']),1)
    eliamase = round(calculate_mase(df_sc['DA elia (11AM)'], df_sc['actual elia']),1)

    # Display metrics
    st.markdown(f"**RMSE (q50):** {myrmse} , MASE : {mymase}")
    st.markdown(f"**RMSE (DA elia):** {eliarmse} , MASE : {eliamase}")

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

def benchmark():
    """Benchmark different forecasting models."""
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.write("Cache cleared!")
        time.sleep(5)

    st.title("Benchmark Models")
    conn = st.connection('gcs', type=FilesConnection)

    selected_date = st.date_input("Submission date", pd.to_datetime("today"))
    latest_actual = get_latest_wind_offshore(selected_date)

    # Fetch model forecasts
    models = ['oracle','avg', 'metno', 'dmi_seamless', 'meteofrance', 'icon', 'knmi']
    forecasts = []
    
    for model in models:
        try:
            all_files = []
            token = None
            
            while True:
                res = conn._instance.ls(
                    f"oracle_predictions/predico-elia/forecasts/{model}",
                    max_results=100000,
                    page_token=token
                )
                
                if isinstance(res, tuple):
                    files = res[0]
                    token = res[1] if len(res) > 1 else None
                else:
                    files = res
                    token = None

                all_files.extend(files)
                if not token:
                    break

            sel = get_latest_da_fcst_file(selected_date, all_files)
            
            if sel:
                df = conn.read(sel, input_format="parquet")
                
                try:
                    df = df[[0.1, 0.5, 0.9]]
                except:
                    df = df[['0.1', '0.5', '0.9']]
                    
                df.columns = [0.1, 0.5, 0.9]
                
                if not df.empty:
                    forecasts.append(df.add_prefix(f'{model}_'))
        except Exception as e:
            st.error(f"Error loading {model}: {str(e)}")
            pass
            
    # Combine forecasts
    if forecasts:
        df = pd.concat(forecasts, axis=1)
        df.index = pd.to_datetime(df.index)
        
        try:
            df = pd.concat([latest_actual.drop(columns='Datetime'), df], axis=1)
            default_cols = ['actual', 'DA elia (11AM)', 'oracle_0.5','avg_0.5', 'icon_0.5', 'metno_0.5', 
                            'dmi_seamless_0.5', 'meteofrance_0.5', 'knmi_0.5']
        except:
            default_cols = ['DA elia (11AM)', 'avg_0.5', 'icon_0.5', 'metno_0.5', 
                           'dmi_seamless_0.5', 'meteofrance_0.5', 'knmi_0.5']

        df = df.iloc[-96:].copy()
        y_cols = df.columns

        # Color mapping for visualization
        color_map = {
            'actual': 'white',
            'DA elia (11AM)': 'orange',
            'avg_0.5': "rgb(5, 222, 255)",
            'metno_0.5': 'red',
            'dmi_seamless_0.5': 'green',
            'meteofrance_0.5': 'purple',
            'knmi_0.5': 'grey',
            'icon_0.5': 'yellow',
            'oracle_0.5':'blue',
        }
        
        # Create plot
        fig = go.Figure()

        for col in y_cols:
            try:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    visible=(col in default_cols),
                    line_color=color_map.get(col, 'blue'),
                    showlegend=False
                ))
            except:
                pass
                
        fig.update_layout(
            xaxis_title="Datetime",
            yaxis_title="MW",
            yaxis=dict(range=[0, 2300]),
            template="plotly_dark",
        )

        st.plotly_chart(fig)

        # Compute and display scores
        try:
            df = df.dropna()
            
            # Columns to evaluate
            cols = [
                'DA elia (11AM)', 'oracle_0.5', 'metno_0.5', 'meteofrance_0.5', 'avg_0.5',
                'icon_0.5', 'knmi_0.5', 'dmi_seamless_0.5',
            ]
            
            def compute_scores(group, col):
                error = group.actual - group[col]
                rmse = np.sqrt(np.mean(error**2))
                mae = np.mean(np.abs(error))
                return pd.Series({f'{col}_RMSE': rmse, f'{col}_MAE': mae})

            scores = df.groupby(df.index.date).apply(
                lambda grp: pd.concat([compute_scores(grp, col) for col in cols if col in grp.columns])
            )

            rmse = scores.loc[:, scores.columns.str.contains('RMSE')].dropna().T
            mae = scores.loc[:, scores.columns.str.contains('MAE')].dropna().T

            st.dataframe(rmse.T.tail(1))
            st.dataframe(mae.T.tail(1))
        except Exception as e:
            st.error(f"Error computing scores: {str(e)}")
    else:
        st.warning("No forecast data available for the selected date.")

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
    
    api = get_api()
    data = fetch_last_50_scores(api)
    st.plotly_chart(plot_rank_and_payout_separate(data))
    st.dataframe(data.sort_values(by='market_date', ascending=False).drop(columns='variable'))

def run_forecast_job():
    """Run forecast generation and oracle submission jobs."""
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

# Main application
def main():
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Generate forecast"):
        run_forecast_job()

    page_choice = st.sidebar.radio("Go to page:", ["Submission Viewer", "Overview", "Benchmark"])
    
    if page_choice == "Submission Viewer":
        submission_viewer()
    elif page_choice == "Overview":
        overview()
    elif page_choice == "Benchmark":
        benchmark()

if __name__ == "__main__":
    main()