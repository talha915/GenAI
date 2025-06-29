from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = DirectoryLoader('./documents', glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(texts, embedding_model)
vectorstore.save_local("faiss_index")
print("✅ FAISS index saved.")
