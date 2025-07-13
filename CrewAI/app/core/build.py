from langchain_community.utilities import OpenWeatherMapAPIWrapper

import os
from dotenv import load_dotenv
from pathlib import Path



class Temperature:

    def __init__(self):
        try:
            env_path = Path(__file__).resolve().parent.parent.parent / '.env'
            load_dotenv(dotenv_path=env_path)
            self.openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
            print(f"✅ Fetched OpenWeatherApiKey successfully: {self.openweather_api_key}")
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise   

    def get_open_weather_api(self):
        try:
           print(f"✅ Fetched OpenWeatherApiKey successfully: {self.openweather_api_key}")      
        except Exception as e:
            print(f"❌ Error while fetching OpenWeatherApiKey: {e}")
            raise  

if __name__ == '__main__':
    temp_instance = Temperature()
    temp_instance.get_open_weather_api()
