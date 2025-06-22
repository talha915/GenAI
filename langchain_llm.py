from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader(path="./documents", glob="*.pdf", loader_cls=PyPDFLoader)
# file_path = "filename.pdf"
# loader = PyPDFLoader(file_path)

documents = loader.load()
print(documents)