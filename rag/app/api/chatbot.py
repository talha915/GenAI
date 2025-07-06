from fastapi import Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse

# Importing classes
from app.core.query import RAGQueryEngine

# Router
router = APIRouter()

# POST endpoint to query Pinecone (form-data version)
@router.post("/chatbot")
async def query_vectorstore(query: str = Form(...)):
    try:
        if not query:
            raise HTTPException(
                status_code=400, detail="Missing 'query' in request body"
            )
        
        print(f"[DEBUG] Received query: {query!r}")

        # Set up QA chain
        engine = RAGQueryEngine()
        answer = engine.ask(query) 

        return JSONResponse(content={
            "results": answer,
            "status_code": 200
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    