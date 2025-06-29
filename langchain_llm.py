from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA



from dotenv import load_dotenv
from pathlib import Path
import os

# Get parent directory 
env_path = Path(__file__).resolve().parent / '.env'

# Load .env from parent directory
load_dotenv(dotenv_path=env_path)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20, separators=["\n\n", "\n", " ", ""])
texts = text_splitter.split_documents(file_content)
# print(f"Here is splitted text : {texts}")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(texts, embedding_model)

vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # top 5 relevant docs

groqllm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192"
)

qa_chain = RetrievalQA.from_chain_type(llm=groqllm, retriever=retriever, chain_type="stuff")

query = "What are key aspects of generative AI?"
answer = qa_chain.invoke(query)
print("Answer:", answer)
