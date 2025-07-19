from langchain_community.utilities import OpenWeatherMapAPIWrapper
from crewai import Agent, Crew, Task, LLM
import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")


class Temperature:

    def __init__(self):
        try:
            self.openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
            self.weather = OpenWeatherMapAPIWrapper()
            self.llm = LLM(
                model="groq/llama-3.3-70b-versatile"
            )
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


    def extract_city_name(self, state):
        try:
            message = state.get('messages')
            user_input = message[0] if message else ""
            complete_query = (
                "Your task is to provide only the city name based on the user query. "
                "Nothing more, just the city name mentioned. Following is the user query: " + user_input
            )
            response = self.llm.call(complete_query)
            print(f"🧪 LLM Response Type: {type(response)}, Content: {response}")
            state['messages'].append(response.content)
            return state
        except Exception as e:        
            print(f"❌ Error while extractin city name: {e}")
            raise

    def fetch_weather(self, state: dict):
        try:
            USE_REAL_API = False
            city = state["messages"][0] if state.get("messages") else ""
            if USE_REAL_API:
                data = self.weather.run(city)
                state["messages"].append(data)
                return state
            else:
                # return mock
                mocked_state = {
                    'messages': [
                        city,
                        f'In {city}, the current weather is as follows:\n'
                        'Detailed status: smoke\n'
                        'Wind speed: 1.03 m/s, direction: 160°\n'
                        'Humidity: 66%\n'
                        'Temperature:\n'
                        '  - Current: 31.99°C\n'
                        '  - High: 31.99°C\n'
                        '  - Low: 31.99°C\n'
                        '  - Feels like: 38.97°C\n'
                        'Rain: {}\n'
                        'Heat index: None\n'
                        'Cloud cover: 40%'
                    ]
                }
                return mocked_state  
        except Exception as e:
            print(f"❌ Error while fetching state temperature: {e}")
            raise  

    def summarize_info(self, state: dict):
        try:
            message = state.get('messages')
            user_input = message[0] if message else ""
            available_info = message[-1] if len(message) > 1 else ""
            agent2_query = (
                "Your task is to provide info concisely based on the user query and the available information from the internet. "
                "Following is the user query: " + user_input + " Available information: " + available_info
            )
            response = self.llm.invoke(agent2_query)
            print(f"Response of summarize info: {response}")
            state['messages'].append(response.content)
            return state
        except Exception as e:
            print(f"❌ Error while generating summary: {e}")
            raise         

    

if __name__ == "__main__":
    # Step 1: Create Temperature instance
    temp_instance = Temperature()

    # Step 2: Define agents
    agent1 = Agent(
        role="City Extractor",
        goal="Extract only the city name from user query.",
        backstory="I specialize in extracting cities from text input.",
        verbose=True,
        function=temp_instance.extract_city_name
    )
    
    # state = {"messages": ["What's the temperature in Lahore?"]}

    # state = temp_instance.extract_city_name(state)
    # print("👉 Extracted city:", state['messages'][-1])  # Should be "Lahore"

    # state = temp_instance.fetch_weather(state)
    # print("🌤️ Weather data:", state['messages'][-1])  # Mock or real weather info

    # state = temp_instance.summarize_info(state)
    # print("📝 Summary:", state['messages'][-1]) 

    agent2 = Agent(
        role="Weather Fetcher",
        goal="Fetch weather details for the extracted city.",
        backstory="I use OpenWeatherMap API to provide weather information.",
        verbose=True,
        function=temp_instance.fetch_weather
    )

    agent3 = Agent(
        role="Summary Writer",
        goal="Generate a helpful summary of the weather based on input and data.",
        backstory="I create final user-friendly weather reports.",
        verbose=True,
        function=temp_instance.summarize_info
    )

    # # Step 3: Define tasks
    task1 = Task(
        description="Extract the city name from the user's question.",
        expected_output="Only the city name as a string.",
        agent=agent1
    )

    task2 = Task(
        description="Fetch the weather details for the extracted city.",
        expected_output="Weather report with temperature, humidity, wind speed, etc.",
        agent=agent2
    )

    task3 = Task(
        description="Generate a concise weather summary for the user.",
        expected_output="Short natural language summary of the weather.",
        agent=agent3
    )

    # # Step 4: Create Crew
    crew = Crew(
        agents=[agent1, agent2, agent3],
        tasks=[task1, task2, task3],
        verbose=True
    )

    # # Step 5: Run the Crew
    initial_state = {"messages": ["What's the temperature in Lahore?"]}
    final_state = crew.kickoff(initial_state)

    # # Step 6: Show result
    # print("\n✅ Final Summary:")
    print(final_state["messages"][-1])
