from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, RETRIEVAL_K


def get_vectorstore():
    """
    Initialize and return a ChromaDB vector store instance.

    Returns:
        Chroma: Configured ChromaDB instance connected to the local
            chroma_db directory with college_knowledge collection.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings
    )


def get_retriever(k=5):
    """
    Get a retriever from the vector store.

    Args:
        k (int, optional): Number of documents to retrieve. Defaults to 5.

    Returns:
        Retriever: LangChain retriever instance.
    """
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )


def retrieve_with_scores(query, k=5):
    """
    Retrieve documents with relevance scores.

    Args:
        query (str): Search query.
        k (int, optional): Number of documents to retrieve. Defaults to 5.

    Returns:
        tuple: (documents, scores) where:
            - documents (list): List of retrieved documents.
            - scores (list): List of relevance scores for each document.
    """
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    documents = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]

    return documents, scores
