from fastapi import FastAPI
from app.api import ingestion, chatbot

app = FastAPI()

app.include_router(ingestion.router)
app.include_router(chatbot.router)

@app.get("/test")
def testing():
    return {"message": "✅ Deployment Pipeline Working"}
