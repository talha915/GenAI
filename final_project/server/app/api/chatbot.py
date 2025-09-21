from fastapi import Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# Importing KnowledgeBase
from app.core.query import RAGQueryEngine

# Router
router = APIRouter()

class GraphState(TypedDict):
    query: str


def knowledge_base_agent(state):
    engine = RAGQueryEngine()
    result = engine.ask(state["query"])
    return {"answer": result}

# --- Database Agent Node ---
def database_agent(state):
    return {"answer": "Database agent not implemented yet."}

# --- Router Node ---
def router_agent(state):
    query = state["query"].lower()
    if "sales" in query or "employee" in query or "product" in query:
        return {"next": "database"}  
    else:
        return {"next": "knowledge_base"}  
    

# --- Build LangGraph ---
graph = StateGraph(GraphState)

# Add nodes
graph.add_node("router", router_agent)
graph.add_node("knowledge_base", knowledge_base_agent)
graph.add_node("database", database_agent)

# Add conditional edges from router
graph.add_conditional_edges(
    "router",
    router_agent,
    {"knowledge_base": "knowledge_base", "database": "database"},
)

# Set entry and finish points
graph.set_entry_point("router")
graph.set_finish_point("knowledge_base")
graph.set_finish_point("database")

app_graph = graph.compile()


# POST endpoint to query Chroma (form-data version)
@router.post("/chatbot")
async def query_vectorstore(query: str = Form(...)):
    try:
        if not query:
            raise HTTPException(
                status_code=400, detail="Missing 'query' in request body"
            )

        # # Set up QA chain
        # engine = RAGQueryEngine()
        # answer = engine.ask(query) 
        answer = app_graph.invoke({"query": query})

        return JSONResponse(content={
            "results": answer,
            "status_code": 200
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    