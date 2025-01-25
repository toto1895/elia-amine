# -*- coding: utf-8 -*-

import pandas as pd
import requests as req
import tqdm
import time

# --------------------------------------------------------

def get_wind_park():
    wind_turbines = {
        "C-Power": {"lat": 51.33, "lon": 2.58, "capa": 325},
        "Belwind": {"lat": 51.40, "lon": 2.48, "capa": 171},
        "Northwind": {"lat": 51.37, "lon": 2.54, "capa": 216},
        "Nobelwind": {"lat": 51.3947, "lon": 2.5002, "capa": 165},
        "Rentel": {"lat": 51.3528, "lon": 2.5638, "capa": 307},
        "Norther": {"lat": 51.3141, "lon": 3.0155, "capa": 370},
        "Northwester 2": {"lat": 51.4115, "lon": 2.4456, "capa": 219},
        "Mermaid": {"lat": 51.3748, "lon": 2.5135, "capa": 235.2},
        "SeaStar": {"lat": 51.4311, "lon": 2.4424, "capa": 252}
    }
    wind_park = pd.DataFrame.from_records(wind_turbines).T
    return wind_park

def download_historical_meteo(
    params=[
        'relative_humidity_2m',
        'temperature_2m',
        'wind_direction_10m',
        'wind_speed_10m',
        'wind_speed_100m',
        'wind_gusts_10m'
    ],
    start='2024-01-01', 
    end='2024-12-07'
):
    wind_park = get_wind_park()
    params = ",".join(params)
    
    def fetch_data_with_retries(url, max_retries=5, initial_wait=3):
        retries = 0
        while retries < max_retries:
            try:
                response = req.get(url, timeout=10)
                response.raise_for_status()  # Raise an error for HTTP errors
                return response.json()
            except req.RequestException as e:
                print(f"Error fetching data: {e}. Retrying in {initial_wait} seconds...")
                time.sleep(initial_wait)
                retries += 1
                initial_wait *= 2  # Exponential backoff
        raise Exception(f"Failed to fetch data after {max_retries} retries.")

    for i in tqdm.tqdm(range(len(wind_park))):
        name = wind_park.index[i]
        lat = wind_park['lat'].iloc[i]
        lon = wind_park['lon'].iloc[i]
        url = (f'https://historical-forecast-api.open-meteo.com/v1/forecast?'
               f'latitude={lat}&longitude={lon}&'
               f'start_date={start}&end_date={end}&'
               f'hourly={params}&'
               f'models=icon_d2,meteofrance_arome_france_hd,metno_seamless,knmi_harmonie_arome_netherlands&'
               f'format=json&timeformat=unixtime')
        
        try:
            data = fetch_data_with_retries(url)
            df = pd.DataFrame(data['hourly']).set_index('time').add_prefix(name+'_')
            df.index = pd.to_datetime(df.index, unit='s')
            df = df.tz_localize('UTC')
            for p in params.split(','):
                df.loc[:, df.columns.str.startswith(f'{name}_{p}')].copy().to_parquet(
                    f'data/{name}_{p}.parquet', compression='gzip'
                )
        except Exception as e:
            print(f"Failed to process data for {name}: {e}")
        time.sleep(3)
          
        
def get_latest_meteo_forecast(params=[
        'relative_humidity_2m',
        'temperature_2m',
    
        'wind_direction_10m',
        'wind_speed_10m',
        'wind_speed_100m',
        'wind_gusts_10m'
        
        ]):
    
    wind_park = get_wind_park()
    
    l = []
    max_retries = 4  # Number of retries
    
    for i in tqdm.tqdm(range(len(wind_park))):
        for param in params:
            name = wind_park.index[i]
            lat = wind_park['lat'].iloc[i]
            lon = wind_park['lon'].iloc[i]
            url = (
                f'https://api.open-meteo.com/v1/forecast?'
                f'latitude={lat}&longitude={lon}&'
                f'hourly={param}&'
                f'models=icon_d2,meteofrance_arome_france_hd,metno_seamless,knmi_harmonie_arome_netherlands&'
                f'format=json&timeformat=unixtime&'
                f'past_days=3&forecast_days=4'
            )
    
            for attempt in range(max_retries):
                try:
                    response = req.get(url)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = response.json()['hourly']
                    df = pd.DataFrame(data).set_index('time').add_prefix(name + '_')
                    df.index = pd.to_datetime(df.index, unit='s')
                    df = df.tz_localize('CET')
                    df.to_parquet(f'data/latest/{name}_{param}.parquet', compression='gzip')
                    l.append(df.copy())
                    break  # Break the retry loop on success
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = 4 * (2 ** attempt)  # 4, 8, 16 seconds
                        time.sleep(delay)
                    else:
                        print(f"Failed to fetch data for {name}, param {param}: {e}")

# --------------------------------------------------------

def get_historical_wind_offshore(start_date='2024-01-01', end_date='2024-12-07'):
    url = (
        f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031/exports/csv?"
        f"lang=fr&refine=offshoreonshore%3A%22Offshore%22&refine=datetime%3A%222024%22&"
        f"qv1=(datetime%3A%5B{start_date}T23%3A00%3A00Z%20TO%20{end_date}T22%3A59%3A59Z%5D)&"
        f"timezone=Europe%2FBrussels&use_labels=true&delimiter=%3B")
    data = pd.read_csv(url, sep=';').rename(
        columns={'Measured & Upscaled':'actual',
                 'Monitored capacity':'capa',
                 'Load factor':'loadfactor',
                 'Datetime':'datetime'
                 })
    data.set_index('datetime', inplace=True)
    data[['actual','loadfactor','capa']].to_csv('data/target.csv')


def get_latest_wind_offshore() -> pd.DataFrame:
    start = pd.Timestamp.today()- pd.Timedelta(days=5)
    end = start + pd.Timedelta(days=6)
    start = start.strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    # url = (
    #     f"https://griddata.elia.be/eliabecontrols.prod/interface/fdn/download/"
    #     f"windweekly/currentselection?dtFrom={start}&dtTo={end}&regionId=1&"
    #     f"isOffshore=True&isEliaConnected=&forecast=wind"
    #     )
    url = (f'https://griddata.elia.be/eliabecontrols.prod/interface/windforecasting/'
    f'forecastdata?beginDate={start}&endDate={end}&region=1&'
    f'isEliaConnected=&isOffshore=True')
    # d =pd.read_excel(url, skiprows=range(5), engine='xlrd').rename(
    #     columns={'Measured & upscaled [MW]':'actual elia','Monitored Capacity [MW]':'capa',
    #              'DateTime':'Datetime','Day-ahead forecast (11h00) [MW]':'DA elia (11AM)',
    #              'Most recent forecast [MW]':'latest elia forecast'
    #              })
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
    # d = d.tz_localize('CET')
    return d


def get_actual_wind_offshore_from_date(start) -> pd.DataFrame:
    # start = pd.to_datetime(dat)
    end = pd.Timestamp.today() + pd.Timedelta(days=2)
    start = start.strftime('%Y-%m-%d')
    end = end.strftime('%Y-%m-%d')
    # url = (
    #     f"https://griddata.elia.be/eliabecontrols.prod/interface/fdn/download/"
    #     f"windweekly/currentselection?dtFrom={start}&dtTo={end}&regionId=1&"
    #     f"isOffshore=True&isEliaConnected=&forecast=wind"
    #     )
    url = (f'https://griddata.elia.be/eliabecontrols.prod/interface/windforecasting/'
    f'forecastdata?beginDate={start}&endDate={end}&region=1&'
    f'isEliaConnected=&isOffshore=True')
    # d =pd.read_excel(url, skiprows=range(5), engine='xlrd').rename(
    #     columns={'Measured & upscaled [MW]':'actual elia','Monitored Capacity [MW]':'capa',
    #              'DateTime':'Datetime','Day-ahead forecast (11h00) [MW]':'DA elia (11AM)',
    #              'Most recent forecast [MW]':'latest elia forecast'
    #              })
    d = pd.read_json(url).rename(
        columns={
            'dayAheadConfidence10':'DA elia (11AM) P10',
            'dayAheadConfidence90':'DA elia (11AM) P90',
            'dayAheadForecast':'DA elia (11AM)',
            'monitoredCapacity':'capa',
            'mostRecentForecast':'latest elia',
            'realtime':'actual elia',
            'startsOn':'Datetime',
            })[['DA elia (11AM)','actual elia','Datetime','latest elia']]
    d['Datetime'] = pd.to_datetime(d['Datetime'])
    d.index = d['Datetime']
    # d = d.tz_localize('CET')
    return d

# --------------------------------------------------------

def get_unavailability():
    url = ('https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods180/exports/csv?'
          'lang=fr&refine=startoutagetstime%3A%222024%22&timezone=Europe%2FBrussels&use_labels=true&'
           'delimiter=%3B')
    unavail = pd.read_csv(url, sep=';', index_col=0, parse_dates=True)[
        ['End','Unit','Technical Pmax', 'Pmax available during the outage']].reset_index()
    
    
    url = ('https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods179/exports/csv?lang=fr&'
           'refine=unittype%3A%22WOF%22&timezone=Europe%2FBrussels&use_labels=true&delimiter=%3B')
    units = pd.read_csv(url, sep=';', index_col=0, ).drop_duplicates('Technical Unit').reset_index(drop=True)
    
    
    filtered_df2 = unavail[unavail['Unit'].isin(units['Technical Unit'])]
    filtered_df2['curtailment'] = -filtered_df2.loc[:,'Pmax available during the outage']+filtered_df2.loc[:,'Technical Pmax']
    
    filtered_df2.loc[:,'Start'] = pd.to_datetime(filtered_df2.loc[:,'Start'], utc=True)
    filtered_df2.loc[:,'End'] = pd.to_datetime(filtered_df2.loc[:,'End'], utc=True)
    # filtered_df2 = filtered_df2.T
    # Initialize an empty DataFrame for the result
    all_ranges = []
    
    for _, row in filtered_df2.iterrows():
        time_range = pd.date_range(start=row['Start'], end=row['End'], freq='15min')
        temp_df = pd.DataFrame({'Timestamp': time_range, 'Curtailment': row['curtailment']})
        all_ranges.append(temp_df)
    
    # Concatenate all ranges and sum curtailments
    df2 = pd.concat(all_ranges).groupby('Timestamp', as_index=True)['Curtailment'].sum()
    
    # Convert to DataFrame
    curtail = df2.asfreq('15min').fillna(0)
    curtail.to_frame().to_csv('data/unavailibility.csv')







