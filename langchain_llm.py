from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
import os

# Get parent directory 
env_path = Path(__file__).resolve().parent / '.env'

# Load .env from parent directory
# load_dotenv(dotenv_path=env_path)
# print(grok_api_token)

# file_path = "filename.pdf"
# loader = PyPDFLoader(file_path)

def load_files(folder_path):
    try:
        loader = DirectoryLoader(path=folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error while loading files: {e}")
        raise    
    
file_content = load_files('./documents')
print(f"Here is collective file content: {file_content}")    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=9, separators=["\n\n", "\n", " ", ""])
texts = text_splitter.split_documents(file_content)
print(f"Here is splitted text : {texts}")