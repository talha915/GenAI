from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# --- Graph state ---
class GraphState(TypedDict):
    query: str
    relevance: str
    answer: str 
    sql_query: str
    query_rows: list
    attempts: int
    sql_error: bool

class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'."
    )    

class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The SQL query corresponding to the user's natural language question."
    )    

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")    