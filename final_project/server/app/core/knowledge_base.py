from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4

class BuildRag:

    def load_documents(self, path):
        try:
            # Loading documents
            loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
            docs = loader.load()

            # Splitting text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            texts = text_splitter.split_documents(docs)
            return texts
        except Exception as e:
            print(f"❌ Error while loading documents: {e}")
            raise e        

    def embedding_vector_store(self, texts):
        try:
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            # vectorstore = FAISS.from_documents(texts, embedding_model)
            # vectorstore.save_local("faiss_index")
            persist_directory = "./chroma_store"  # Local folder for persistence

            vector_store = Chroma(
                collection_name="my_docs",
                embedding_function=embedding_model,
                persist_directory=persist_directory
            )
            ids = [str(uuid4()) for _ in range(len(texts))]
            vector_store.add_documents(documents=texts, ids=ids)
            vector_store.persist()

            print(f"✅ {len(texts)} documents stored in Chroma.")

        except Exception as e:
            print(f"❌ Error while creating embeddings and vector store {e}")
            raise e
        
if __name__ == '__main__':
    build_instance = BuildRag()
    texts = build_instance.load_documents('./documents')
    build_instance.embedding_vector_store(texts)
