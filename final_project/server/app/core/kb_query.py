from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path


class RAGQueryEngine:
    def __init__(self, index_path="../../chroma_store"):
        try:
            # Load .env variables
            env_path = Path(__file__).resolve().parent.parent.parent / '.env'
            load_dotenv(dotenv_path=env_path)

            os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            os.environ["GROQ_API_KEY"] = os.getenv("grok_api_key")

            # Load embedding model
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            # Load Chroma vectorstore
            self.vectorstore = Chroma(
                collection_name="my_docs",
                embedding_function=self.embedding_model,
                persist_directory=index_path
            )
            print("✅ Chroma index loaded.")

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

            # Initialize LLM
            self.llm = ChatGroq(
                temperature=0,
                model_name="llama-3.3-70b-versatile"
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
