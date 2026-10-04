from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


VECTOR_DB_DIR = "chroma_db"


def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        collection_name="college_knowledge",
        embedding_function=embeddings
    )


def get_retriever(k=5):

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )


def retrieve_with_scores(query, k=5):

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_relevance_scores(
        query,
        k=k
    )

    documents = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]

    return documents, scores
