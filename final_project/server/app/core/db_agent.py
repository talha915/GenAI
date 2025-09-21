import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import text, inspect
from langgraph.graph import StateGraph, END
from sqlalchemy import create_engine
from langchain_core.runnables.config import RunnableConfig
from pathlib import Path
import sqlite3
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.models import GraphState, CheckRelevance


class DatabaseAgent:
    def __init__(self):
        try:
            # Load .env variables
            env_path = Path(__file__).resolve().parents[2] / '.env'
            load_dotenv(dotenv_path=env_path)

            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

            self.groq_llm = ChatGroq(
                model="llama-3.3-70b-versatile",  
                temperature=0.0
            )

            # DATABASE_URL = "sqlite:///:memory:"
            db_path = Path(__file__).resolve().parents[1] / "db" / "used_cars.db"
            DATABASE_URL = f"sqlite:///{db_path}"
            

            # Create engine
            self.engine = create_engine(
                DATABASE_URL, connect_args={"check_same_thread": False}
            )

            # Create SessionLocal factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
        except Exception as e:
            print(f"Error: {e}")
            return e    


    def get_database_schema(self):
        inspector = inspect(self.engine)
        schema = ""
        for table_name in inspector.get_table_names():
            schema += f"Table: {table_name}\n"
            for column in inspector.get_columns(table_name):
                col_name = column["name"]
                col_type = str(column["type"])
                if column.get("primary_key"):
                    col_type += ", Primary Key"
                if column.get("foreign_keys"):
                    fk = list(column["foreign_keys"])[0]
                    col_type += f", Foreign Key to {fk.column.table.name}.{fk.column.name}"
                schema += f"- {col_name}: {col_type}\n"
            schema += "\n"
            
        print("Retrieved database schema.")
        return schema

    def check_relevance(self, state: GraphState):
        question = state["query"] 
        schema = self.get_database_schema()
        print(f"Checking relevance of the question: {question}")
        system = """You are an assistant that determines whether a given question is related to the following database schema.

        Schema:
        {schema}

        Respond with only "relevant" or "not_relevant".
        """.format(schema=schema)
        human = f"Question: {question}"
        check_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )
        structured_llm = self.groq_llm.with_structured_output(CheckRelevance)
        relevance_checker = check_prompt | structured_llm
        relevance = relevance_checker.invoke({})
        state["relevance"] = relevance.relevance
        print(f"Relevance determined: {state['relevance']}")
        if relevance.relevance == "relevant":
            return "database"
        else:
            return "knowledge_base"   


# --- Run interactively ---
if __name__ == '__main__':
    agent = DatabaseAgent()
    schema = agent.get_database_schema()
    print(schema)