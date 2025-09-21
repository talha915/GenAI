from fastapi import Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from langgraph.graph import StateGraph

# Importing KnowledgeBase
from app.core.query import RAGQueryEngine

# Router
router = APIRouter()

def knowledge_base_agent(state):
    engine = RAGQueryEngine()
    result = engine.ask(state["query"])
    return {"answer": result}

# --- Database Agent Node ---
def database_agent(state):
    return
    db = DatabaseAgent()
    result = db.run_query(state["query"])
    return {"answer": result}

# --- Router Node ---
def router_agent(state):
    query = state["query"].lower()
    if "sales" in query or "employee" in query or "product" in query:
        return "database"
    else:
        return "knowledge_base"

# --- Build LangGraph ---
graph = StateGraph()
graph.add_node("knowledge_base", knowledge_base_agent)
graph.add_node("database", database_agent)

graph.add_conditional_edges(
    "router",
    router_agent,
    {
        "knowledge_base": "knowledge_base",
        "database": "database",
    },
)

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