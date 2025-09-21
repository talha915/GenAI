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
graph.add_node("router", router_node)
graph.add_node("database", database_agent)
graph.add_node("knowledge_base", knowledge_base_agent)

# Conditional routing from router node
graph.add_conditional_edges(
    "router",                 # router node
    lambda state: db_agent.check_relevance(state),      
    {
        "database": "database",
        "knowledge_base": "knowledge_base"
    }
)

# Connect normal nodes to END
graph.add_edge("database", END)
graph.add_edge("knowledge_base", END)

# Entry point
graph.set_entry_point("router")

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
            "results": answer.get("answer"),
            "status_code": 200
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
