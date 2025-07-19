import os
from dotenv import load_dotenv
from pathlib import Path
from crewai import Agent, Crew, Task, LLM

class TemperatureCrewBuilder:

    def __init__(self, topic: str):
        self.topic = topic
        self._load_env()

    def _load_env(self):
        env_path = Path(__file__).resolve().parent.parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")
        os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")

    def _setup_llm(self):
        return LLM(model="groq/llama-3.3-70b-versatile")    