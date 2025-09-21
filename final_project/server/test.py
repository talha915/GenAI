from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = '/path/to/persist/directory'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory=PERSIST_DIRECTORY,
)

# Check that documents are there 

print("Results: ", vector_store.get()['documents'])