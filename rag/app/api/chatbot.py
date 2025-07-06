from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import os

# Importing classes
from app.core.query import RAGQueryEngine

router = APIRouter()

# POST endpoint to query Pinecone (form-data version)
@router.post("/chatbot")
async def query_vectorstore(query: str = Form(...)):
    try:
        if not query:
            raise HTTPException(
                status_code=400, detail="Missing 'query' in request body"
            )

        # Set up QA chain
        engine = RAGQueryEngine()
        answer = engine.ask(query) 

        return JSONResponse(content={
            "results": answer,
            "status_code": 200
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    