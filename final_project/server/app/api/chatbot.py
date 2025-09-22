from fastapi import Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph, END

from app.core.kb_query import RAGQueryEngine
from app.core.db_agent import DatabaseAgent
from app.core.models import GraphState

router = APIRouter()
db_agent = DatabaseAgent()

# --- Knowledge Base Node ---
def knowledge_base_agent(state: GraphState) -> dict:
    engine = RAGQueryEngine()
    result = engine.ask(state["query"])
    return {"answer": result}

# --- Database Node ---
def database_agent(state: GraphState) -> dict:
    # Placeholder database logic
    print("Database agent called")
    return {"answer": f"Database agent received query: {state['query']}"}

# --- Router Node (normal node) ---
def router_node(state: GraphState) -> dict:
    """
    This node can set metadata in the state if needed.
    Returns dict because LangGraph requires normal nodes to return dict.
    """
    return state

# --- Conditional router function ---
def relevance_router(state: GraphState) -> str:
    """
    Determines which node to go next.
    Must return string key matching conditional edges mapping.
    """
    if "sales" in state["query"].lower():
        state["relevance"] = "relevant"
        return "database"
    else:
        state["relevance"] = "irrelevant"
        return "knowledge_base"
    

# --- Build Graph ---
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("check_relevance", db_agent.check_relevance)
graph.add_node("convert_to_sql", db_agent.convert_nl_to_sql)
graph.add_node("execute_sql", db_agent.execute_sql)
graph.add_node("generate_human_readable_answer", db_agent.generate_human_readable_answer)
graph.add_node("regenerate_query", db_agent.regenerate_query)
graph.add_node("generate_fallback_response", db_agent.generate_fallback_response)
graph.add_node("end_max_iterations", db_agent.end_max_iterations)
graph.add_node("knowledge_base", knowledge_base_agent) 

# Add edges
graph.add_conditional_edges(
    "check_relevance",
    lambda state: db_agent.relevance_router(state),
    {
        "knowledge_base": "knowledge_base",               
        "convert_to_sql": "convert_to_sql",
        "generate_fallback_response": "generate_fallback_response",
    }
)

graph.add_edge("convert_to_sql", "execute_sql")

graph.add_conditional_edges(
    "execute_sql",
    lambda state: db_agent.execute_sql_router(state),
    {
        "generate_human_readable_answer": "generate_human_readable_answer",
        "regenerate_query": "regenerate_query",
    }
)

graph.add_conditional_edges(
    "regenerate_query",
    lambda state: db_agent.check_attempts_router(state),
    {
        "convert_to_sql": "convert_to_sql",
        "max_iterations": "end_max_iterations",
    }
)

# Terminal paths
graph.add_edge("generate_human_readable_answer", END)
graph.add_edge("generate_fallback_response", END)
graph.add_edge("end_max_iterations", END)
graph.add_edge("knowledge_base", END)

# Entry
graph.set_entry_point("check_relevance")


# Compile graph
app_graph = graph.compile()

# --- POST endpoint ---
@router.post("/chatbot")
async def query_vectorstore(query: str = Form(...)):
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    
    state: GraphState = {"query": query, "relevance": ""}
    
    try:
        answer = app_graph.invoke(state)
        return JSONResponse({
            "results": answer,
            "status_code": 200
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
