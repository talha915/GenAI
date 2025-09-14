from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name="my_docs",
    embedding_function=embedding_model,
    persist_directory="../../chroma_store",
)

# Check that documents are there 
print("Listed Collections:", vector_store._client.list_collections())
# print(f"Results: {vector_store.get()['documents']}")
all_data = vector_store._collection.get(include=["documents", "metadatas"])
print(f"Total documents: {len(all_data['documents'])}")
print("Sample documents:", all_data['documents'][:3])
