from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path


class RAGQueryEngine:
    def __init__(self, index_path="faiss_index"):
        try:
            # Load .env variables
            env_path = Path(__file__).resolve().parent / '.env'
            load_dotenv(dotenv_path=env_path)

            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

            # Load embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load FAISS vectorstore
            self.vectorstore = FAISS.load_local(
                index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            print("✅ FAISS index loaded.")

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

            # Initialize LLM
            self.llm = ChatGroq(
                temperature=0,
                model_name="llama3-70b-8192"
            )

            # Build the QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                chain_type="stuff"
            )
            print("✅ RAG Query Engine initialized.\n")

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            raise

    def ask(self, query: str):
        try:
            response = self.qa_chain.invoke(query)
            return response
        except Exception as e:
            print(f"❌ Error during query: {e}")
            return None


# --- Run interactively ---
if __name__ == '__main__':
    engine = RAGQueryEngine()
    while True:
        user_input = input("❓ Ask a question (or type 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        answer = engine.ask(user_input)
        print("💬 Answer:", answer, "\n")
