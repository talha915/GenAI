from crewai import Crew, Agent, Task
import os
from dotenv import load_dotenv
from pathlib import Path
from textwrap import dedent
from crewai.tools import tool
from crewai import Agent, Crew,Task,LLM



env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
os.environ["OPENWEATHERMAP_API_KEY"] = os.getenv("OPENWEATHERMAP_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

# Set up LLM
llm = LLM(model="groq/llama-3.3-70b-versatile")

# Define agents
researcher = Agent(
    role='Researcher',
    goal='Gather information on {{ topic }} market trends',
    backstory='An expert market analyst with 10 years of experience.',
    llm=llm
)

writer = Agent(
    role='Writer',
    goal='Write a detailed {{ topic }} market report',
    backstory='A professional writer specializing in economics.',
    llm=llm
)

# Define tasks using input variable {{ topic }}
extract_research = Task(
    description='Find current trends in the {{ topic }} market.',
    expected_output='A summary of 3-5 key trends in the {{ topic }} market, with examples or sources.',
    agent=researcher
)

write_report = Task(
    description='Write a report based on the findings from the researcher about {{ topic }}.',
    expected_output='A well-written market report in paragraph form, summarizing the {{ topic }} market trends.',
    agent=writer,
    context=[extract_research]
)

# Assemble the crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[extract_research, write_report]
)

topic_input = input("🔍 Enter the market topic: ")
inputs = {"topic": topic_input}

# Run the crew with dynamic input
print("\n🚀 Running Crew...\n")
# Run the crew with dynamic input
result = crew.kickoff(inputs=inputs)
print("\n📝 Final Report:\n")
print(result)