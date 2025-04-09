import requests
import pandas as pd
import json
import plotly.graph_objects as go
import streamlit as st
import base64

class PredicoClient:
    BASE_URL = "https://predico-elia.inesctec.pt/api/v1"

    def __init__(self, email=None, password=None):
        self.email = email
        self.password = password
        self.access_token = None
        self.headers = None
        self.user_id = None

    def decode_jwt(self, token):
        """
        Decode a JWT token to extract the payload including user_id
        
        Args:
            token (str): JWT token string
        
        Returns:
            dict: Decoded payload or None if decoding fails
        """
        try:
            # Split the JWT token into its three parts
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            # Get the payload (middle part)
            payload_base64 = parts[1]
            
            # Add padding if necessary
            payload_base64 += '=' * (4 - len(payload_base64) % 4) if len(payload_base64) % 4 != 0 else ''
            
            # Decode the base64 payload
            payload_bytes = base64.b64decode(payload_base64)
            payload_str = payload_bytes.decode('utf-8')
            
            # Parse the JSON payload
            payload = json.loads(payload_str)
            
            return payload
        except Exception as e:
            print(f"Error decoding JWT: {e}")
            return None

    def authenticate(self):
        """Authenticate with the Predico API and get an access token"""
        try:
            response = requests.post(
                f"{self.BASE_URL}/token", 
                data={"email": self.email, "password": self.password}
            )
            
            if response.status_code == 200:
                self.access_token = response.json().get("access")
                
                # Extract user_id from token
                decoded_token = self.decode_jwt(self.access_token)
                if decoded_token and "user_id" in decoded_token:
                    self.user_id = decoded_token["user_id"]
                
                # Set up headers with the new token
                self.headers = {
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Encoding": "gzip, deflate, br, zstd",
                    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7,de;q=0.6,da;q=0.5,it;q=0.4,la;q=0.3",
                    "Authorization": f"Bearer {self.access_token}",
                    "Connection": "keep-alive",
                    "Host": "predico-elia.inesctec.pt",
                    "Referer": "https://predico-elia.inesctec.pt/sessions",
                    "Sec-Fetch-Dest": "empty",
                    "Sec-Fetch-Mode": "cors",
                    "Sec-Fetch-Site": "same-origin",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
                    "sec-ch-ua": '"Chromium";v="134", "Not:A-Brand";v="24", "Google Chrome";v="134"',
                    "sec-ch-ua-mobile": "?0",
                    "sec-ch-ua-platform": '"Windows"'
                }
                return True
            else:
                st.error(f"Authentication failed: {response.text}")
                return False
                
        except Exception as e:
            st.error(f"Error during authentication: {e}")
            return False
    
    def get_market_sessions(self, status="finished"):
        """Fetch market sessions with given status"""
        if not self.headers:
            st.error("Please authenticate first")
            return None
        
        try:
            url = f"{self.BASE_URL}/market/session/?status={status}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                st.error(f"Failed to fetch market sessions: HTTP {response.status_code}")
                st.text(response.text)
                return None
                
        except Exception as e:
            st.error(f"Error fetching market sessions: {e}")
            return None
    
    def get_challenges(self, session_id, resource_id):
        """Fetch challenges for a given market session and resource"""
        if not self.headers:
            st.error("Please authenticate first")
            return None
            
        try:
            url = f"{self.BASE_URL}/market/challenge/?market_session={session_id}&resource={resource_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                st.error(f"Failed to fetch challenges: HTTP {response.status_code}")
                st.text(response.text)
                return None
                
        except Exception as e:
            st.error(f"Error fetching challenges: {e}")
            return None
    
    def get_forecasts(self, challenge_id, user_id=None):
        """Fetch forecasts for a given challenge and user"""
        if not self.headers:
            st.error("Please authenticate first")
            return None
        
        # Use the user_id from the token if not provided
        if user_id is None:
            user_id = self.user_id
            
        if not user_id:
            st.error("User ID is required but not found in token or provided explicitly")
            return None
            
        try:
            url = f"{self.BASE_URL}/market/challenge/submission/forecasts?challenge={challenge_id}&user={user_id}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to fetch forecasts: HTTP {response.status_code}")
                st.text(response.text)
                return None
                
        except Exception as e:
            st.error(f"Error fetching forecasts: {e}")
            return None