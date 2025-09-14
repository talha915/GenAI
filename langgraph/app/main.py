from fastapi import FastAPI
from app.api import ingestion

app = FastAPI()

app.include_router(ingestion.router)

@app.get("/test")
def testing():
    return {"message": "✅ Deployment Pipeline Working"}
