import requests
import pandas as pd
import numpy as np
import datetime as dt

class PredicoAPI:
    BASE_URL = "https://predico-elia.inesctec.pt/api/v1"

    def __init__(self, email, password):
        self.email = email
        self.password = password
        self.access_token = None

    def authenticate(self):
        response = requests.post(f"{self.BASE_URL}/token", data={"email": self.email, "password": self.password})
        if response.status_code == 200:
            self.access_token = response.json().get("access")
            print("Authentication successful.")
        else:
            raise Exception("Authentication failed: " + response.text)

    def _headers(self):
        if not self.access_token:
            raise Exception("Access token is missing. Please authenticate first.")
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }

    def get_open_sessions(self):
        response = requests.get(f"{self.BASE_URL}/market/session", params={"status": "open"}, headers=self._headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to retrieve open sessions: " + response.text)

    def get_challenges(self, session_id):
        response = requests.get(f"{self.BASE_URL}/market/challenge", params={"market_session": session_id}, headers=self._headers())
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception("Failed to retrieve challenges: " + response.text)

    def download_raw_data(self, resource_id, start_date, end_date):
        params = {
            "resource": resource_id,
            "start_date": start_date,
            "end_date": end_date
        }
        next_url = f"{self.BASE_URL}/data/raw-measurements/"
        dataset = []
        while next_url:
            response = requests.get(next_url, params=params, headers=self._headers())
            if response.status_code == 200:
                data = response.json()
                dataset.extend(data["data"]["results"])
                next_url = data["data"]["next"]
            else:
                raise Exception("Failed to download raw data: " + response.text)
        return dataset

    def submit_forecast(self, challenge_id, forecasts):
        for forecast in forecasts:
            response = requests.post(f"{self.BASE_URL}/market/challenge/submission/{challenge_id}", json=forecast, headers=self._headers())
            if response.status_code == 201:
                print(f"Forecast submission successful for {forecast['variable']}.")
            else:
                raise Exception(f"Failed to submit forecast for {forecast['variable']}: " + response.text)

        # Submit the forecasts:
        for submission in forecasts:
            response = requests.post(url=f"{self.BASE_URL}/market/challenge/submission/{challenge_id}",
                                     json=submission,
                                     headers=self._headers())

            # Check if the request was successful
            if response.status_code == 201:
                print(f"Forecast submission successful for {submission['variable']} quantile.")
            else:
                print(f"Failed to submit forecast for {submission['variable']} quantile.")

    def submit_historical_forecast(self, resource_id, n_days):
        dt_now = pd.to_datetime(dt.datetime.utcnow()).round("15min")
        hist_datetime_range = pd.date_range(start=dt_now - pd.DateOffset(days=n_days),
                                            end=dt_now.replace(hour=23, minute=45, second=0),
                                            freq='15min')
        hist_datetime_range = [x.strftime("%Y-%m-%dT%H:%M:%SZ") for x in hist_datetime_range]

        hist_values = np.random.uniform(low=0.0, high=1.0, size=len(hist_datetime_range))
        hist_values = [round(x, 3) for x in hist_values]

        hist_submission_list = []
        for qt in ["q50", "q10", "q90"]:
            qt_forec = pd.DataFrame({
                'datetime': hist_datetime_range,
                'value': hist_values,
            })
            hist_submission_list.append({
                "resource": resource_id,
                "variable": qt,
                "launch_time": dt_now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "forecasts": qt_forec.to_dict(orient="records")
            })

        for submission in hist_submission_list:
            response = requests.post(f"{self.BASE_URL}/data/individual-forecasts/historical", json=submission, headers=self._headers())
            if response.status_code == 201:
                print(f"Historical forecast submission successful for {submission['variable']}.")
            else:
                raise Exception(f"Failed to submit historical forecast for {submission['variable']}: " + response.text)

# Usage Example
# api = PredicoAPI("your_email@example.com", "your_password")
# api.authenticate()
# sessions = api.get_open_sessions()
# challenges = api.get_challenges(sessions["data"][0]["id"])
# api.submit_historical_forecast("resource_id", 35)
