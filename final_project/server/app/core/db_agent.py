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
import sqlite3, yaml
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.models import GraphState, CheckRelevance, ConvertToSQL, RewrittenQuestion


class DatabaseAgent:
    def __init__(self):
        try:
            # Load .env variables
            env_path = Path(__file__).resolve().parents[2] / '.env'
            load_dotenv(dotenv_path=env_path)

            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

            prompt_path = Path(__file__).resolve().parent / "prompts.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f)

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
        return state
        # if relevance.relevance == "relevant":
        #     return "convert_to_sql"
        # else:
        #     return "knowledge_base"   


    def convert_nl_to_sql(self, state: GraphState):
        question = state["query"]
        system_prompt = self.prompts["convert_to_sql"]["system"]
        convert_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", f"Question: {question}")
            ]
        )
        
        structured_llm = self.groq_llm.with_structured_output(ConvertToSQL)
        sql_generator = convert_prompt | structured_llm
        result = sql_generator.invoke({"question": question})
        
        state["sql_query"] = result.sql_query
        print(f"Generated SQL query: {state['sql_query']}")
        return state


    def execute_sql(self, state: GraphState):
        sql_query = state["sql_query"].strip()
        session = self.SessionLocal()
        print(f"Executing SQL query: {sql_query}")
        try:
            result = session.execute(text(sql_query))
            if sql_query.lower().startswith("select"):
                rows = result.fetchall()
                columns = result.keys()
                if rows:
                    header = ", ".join(columns)
                    state["query_rows"] = [dict(zip(columns, row)) for row in rows]
                    print(f"Raw SQL Query Result: {state['query_rows']}")
                    # Format the result for readability
                    data = "; ".join([
                        ", ".join([f"{key}: {value}" for key, value in row.items()])
                        for row in state["query_rows"]
                    ])
                    formatted_result = f"{header}\n{data}"
                else:
                    state["query_rows"] = []
                    formatted_result = "No results found."
                state["query_result"] = formatted_result
                state["sql_error"] = False
                print("SQL SELECT query executed successfully.")
            else:
                session.commit()
                state["query_result"] = "The action has been successfully completed."
                state["sql_error"] = False
                print("SQL command executed successfully.")
        except Exception as e:
            state["query_result"] = f"Error executing SQL query: {str(e)}"
            state["sql_error"] = True
            print(f"Error executing SQL query: {str(e)}")
        finally:
            session.close()
        return state
    
    def generate_human_readable_answer(self, state: GraphState):
        sql = state.get("sql_query", "")
        result = state.get("query_result", "")
        query_rows = state.get("query_rows", [])
        sql_error = state.get("sql_error", False)

        print("Generating a human-readable answer.")

        system_prompt = self.prompts["generate_human_readable_answer"]["system"]

        if sql_error:
            human_text = f"""SQL Query:
            {sql}

        Result/Error:
        {result}

        Formulate a single, clear, human-readable error message."""
        elif sql.lower().startswith("select") and not query_rows:
            human_text = f"""SQL Query:
        {sql}

        Result:
        {result}

    Generate a concise message informing the user that no matching records were found."""
        else:
            human_text = f"""SQL Query:
    {sql}

    Result:
    {result}

        Generate a concise, executive-friendly summary highlighting the key car listings and relevant metrics."""

        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_text)
            ]
        )

        human_response = generate_prompt | self.groq_llm | StrOutputParser()
        answer = human_response.invoke({})

        state["query_result"] = answer.strip()
        print("Generated human-readable answer.")
        return state


    def regenerate_query(self, state: GraphState):
        question = state["query"]
        print("Regenerating the SQL query by rewriting the question.")
        system = self.prompts["regenerate_query"]["system"]
        rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    f"Original Question: {question}\nReformulate the question to enable more precise SQL queries, ensuring all necessary details are preserved.",
                ),
            ]
        )
    
        structured_llm = self.groq_llm.with_structured_output(RewrittenQuestion)
        rewriter = rewrite_prompt | structured_llm
        rewritten = rewriter.invoke({})
        state["query"] = rewritten.question
        state["attempts"] += 1
        print(f"Rewritten question: {state['question']}")
        return state

    def generate_fallback_response(self, state: GraphState):
        print("LLM could not find an answer. Returning fallback response.")
        state["query_result"] = "Sorry, I don't know the answer to that."
        return state
    

    def end_max_iterations(self, state: GraphState):
        state["query_result"] = "Please try again."
        print("Maximum attempts reached. Ending the workflow.")
        return state
    
    def relevance_router(self, state: GraphState):
        if state["relevance"].lower() == "relevant":
            return "convert_to_sql"
        else:
            return "knowledge_base"
        
    def check_attempts_router(self, state: GraphState):
        if state["attempts"] < 3:
            return "convert_to_sql"
        else:
            return "end_max_iterations"  

    def execute_sql_router(self, state: GraphState):
        if not state.get("sql_error", False):
            return "generate_human_readable_answer"
        else:
            return "regenerate_query"      

# --- Run interactively ---
if __name__ == '__main__':
    agent = DatabaseAgent()
    schema = agent.get_database_schema()
    print(schema)