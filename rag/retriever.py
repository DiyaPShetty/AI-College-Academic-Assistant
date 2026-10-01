from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def get_retriever(k=3):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )