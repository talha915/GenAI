import os
from dotenv import load_dotenv
from pathlib import Path
from crewai import Agent, Crew, Task, LLM

class MarketCrewBuilder:
    def __init__(self, topic: str):
        self.topic = topic
        self._load_env()
        self.llm = self._setup_llm()
        self.researcher = self._create_researcher()
        self.writer = self._create_writer()
        self.tasks = self._create_tasks()
        self.crew = self._assemble_crew()

    def _load_env(self):
        env_path = Path(__file__).resolve().parent.parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

    def _setup_llm(self):
        return LLM(model="groq/llama-3.3-70b-versatile")

    def _create_researcher(self):
        return Agent(
            role='Researcher',
            goal='Gather information on {{ topic }} market trends',
            backstory='An expert market analyst with 10 years of experience.',
            llm=self.llm
        )

    def _create_writer(self):
        return Agent(
            role='Writer',
            goal='Write a detailed {{ topic }} market report',
            backstory='A professional writer specializing in economics.',
            llm=self.llm
        )

    def _create_tasks(self):
        extract_research = Task(
            description='Find current trends in the {{ topic }} market.',
            expected_output='A summary of 3-5 key trends in the {{ topic }} market, with examples or sources.',
            agent=self.researcher
        )

        write_report = Task(
            description='Write a report based on the findings from the researcher about {{ topic }}.',
            expected_output='A well-written market report in paragraph form, summarizing the {{ topic }} market trends.',
            agent=self.writer,
            context=[extract_research]
        )

        return [extract_research, write_report]

    def _assemble_crew(self):
        return Crew(
            agents=[self.researcher, self.writer],
            tasks=self.tasks
        )

    def run(self):
        print("\n🚀 Running Crew...\n")
        result = self.crew.kickoff(inputs={"topic": self.topic})
        print("\n📝 Final Report:\n")
        print(result)

if __name__ == "__main__":
    topic_input = input("🔍 Enter the market topic: ")
    crew_runner = MarketCrewBuilder(topic_input)
    crew_runner.run()