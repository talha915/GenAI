from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import uuid

app = FastAPI()
upload_directory = "documents"

@app.get("/test")
def testing():
    msg = "Deployment Pipeline"
    return JSONResponse(content={"message": msg})


# Create the folder if it doesn't exist
def create_folder(folder_name):
    try:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    except Exception as e:
        print(f"Error while creating folder {e}")
        raise e    


@app.post("/ingestion-pipeline")
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

        # Generate unique filename BEFORE saving
        # unique_filename = f"temp_pdf_{uuid.uuid4().hex}.pdf"
        # file_path = os.path.join(upload_directory, unique_filename)

        file_path = os.path.join(upload_directory, file.filename)

        # Save uploaded file into Documents folder
        with open(file_path, "wb") as f:
            f.write(await file.read())

        return JSONResponse(content={
            "message": "✅ File uploaded successfully.",
            "filename": file.filename,
            "path": file_path
        })    

    except Exception as e:
        print(f"Error while ingestion {e}")
        raise e        