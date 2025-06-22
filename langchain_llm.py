from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

# file_path = "filename.pdf"
# loader = PyPDFLoader(file_path)

def load_files(folder_path):
    try:
        loader = DirectoryLoader(path=folder_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents
    except Exception as e:
        return e    
    
file_content = load_files('./documents')
print(f"Here is collective file content: {file_content}")    