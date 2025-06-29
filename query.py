from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

groqllm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192"
)

qa_chain = RetrievalQA.from_chain_type(llm=groqllm, retriever=retriever, chain_type="stuff")

query = input("❓ Ask a question: ")
answer = qa_chain.invoke(query)
print("Answer:", answer)
