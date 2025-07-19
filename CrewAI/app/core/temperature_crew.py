import os
from dotenv import load_dotenv
from pathlib import Path
from crewai import Agent, Crew, Task, LLM
from langchain_community.utilities import OpenWeatherMapAPIWrapper


class TemperatureCrewBuilder:

    def __init__(self):
        self._load_env()
        self.llm = self._setup_llm()

    def _load_env(self):
        env_path = Path(__file__).resolve().parent.parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")
        os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")

    def _setup_llm(self):
        return LLM(model="groq/llama-3.3-70b-versatile")

    def extract_city_name(self, state):
        try:
            user_input = state.get('messages', [""])[0]
            prompt = (
                "Your task is to extract only the city name from the user's query. "
                f"User query: {user_input}"
            )
            response = self.llm.call(prompt)
            city_name = response.strip()

            print(f"🧪 Extracted City: {city_name}")
            state['messages'].append(city_name)
            return state, city_name
        except Exception as e:
            print(f"❌ Error while extracting city name: {e}")
            raise

    def _agent_weather_info(self, user_input, available_info):
        return Agent(
            role='Researcher',
            goal=(
                "Provide concise information based on the user query and available data.\n"
                f"User query: {user_input}\n"
                f"Available information: {available_info}"
            ),
            backstory='An expert weather analyst with 10 years of experience.',
            llm=self.llm
        )
    
    def _weather_info_task(self, user_input, weather_data):
        weather_info_extracter_agent = self._agent_weather_info(user_input, weather_data['messages'][-1])
        return Task(
            description="Analyze the weather information and provide a concise summary for the user query.",
            expected_output="A short summary (2-3 lines) describing the current weather conditions in the specified city.",
            agent=weather_info_extracter_agent
        )

    def _assemble_crew(self):
        weather_agent = self._agent_weather_info(user_input, weather_data)
        weather_task = self._weather_info_task(user_input, weather_data)
        return Crew(
            agents=[weather_agent],
            tasks=[weather_task],
            verbose=True
        )

    def _fetch_weather(self, state: dict):
        try:
            USE_REAL_API = False
            self.weather = OpenWeatherMapAPIWrapper()
            city = state.get("messages", [""])[0]

            if USE_REAL_API:
                data = self.weather.run(city)
                state["messages"].append(data)
                return state
            else:
                # Mocked weather response
                mock_report = (
                    f"In {city}, the current weather is as follows:\n"
                    "Detailed status: smoke\n"
                    "Wind speed: 1.03 m/s, direction: 160°\n"
                    "Humidity: 66%\n"
                    "Temperature:\n"
                    "  - Current: 31.99°C\n"
                    "  - High: 31.99°C\n"
                    "  - Low: 31.99°C\n"
                    "  - Feels like: 38.97°C\n"
                    "Rain: {}\n"
                    "Heat index: None\n"
                    "Cloud cover: 40%"
                )
                return {"messages": [city, mock_report]}
        except Exception as e:
            print(f"❌ Error while fetching weather: {e}")
            raise


if __name__ == "__main__":
    user_input = input("🔍 Enter your weather query (e.g., 'What's the temperature in Lahore?'): ")
    crew_runner = TemperatureCrewBuilder()

    # Step 1: Create initial state from user input
    initial_state = {"messages": [user_input]}

    # Step 2: Extract city
    updated_state, city = crew_runner.extract_city_name(initial_state)

    # Step 3: Fetch weather (mocked or real)
    weather_data = crew_runner._fetch_weather(updated_state)

   
    crew_result = crew_runner._assemble_crew()    
    final_result = crew_result.kickoff()
    # Output for now
    print("\n📦 Final Weather Info:\n", weather_data['messages'][-1])
    print("\n🤖 Weather Info Agent:\n", final_result)
