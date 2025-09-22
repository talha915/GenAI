from fastapi import FastAPI, File, UploadFile, Form, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import os

# Importing classes
from app.core.knowledge_base import BuildRag

router = APIRouter()

upload_directory = "app/documents"

# Create the folder if it doesn't exist
def create_folder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as e:
        print(f"Error while creating folder {e}")
        raise e    


@router.post("/ingestion-pipeline")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Check file type
        file_type = file.content_type
        if file_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
        # Generating / checking folder
        create_folder(folder_name=upload_directory)

        # File path
        file_path = os.path.join(upload_directory, file.filename)

        # Save uploaded file into Documents folder
        with open(file_path, "wb") as f:
            f.write(await file.read())

        build_instance = BuildRag()
        texts = build_instance.load_documents(upload_directory)
        build_instance.embedding_vector_store(texts)    

        return JSONResponse(content={
            "message": "✅ File uploaded and ingested in vector store successfully.",
            "filename": file.filename,
            "path": file_path,
            "status_code": 200
        })    

    except Exception as e:
        print(f"Error while ingestion {e}")
        raise e     