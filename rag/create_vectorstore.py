import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


DATA_DIR = "data"


def load_all_documents():
    """
    Load all PDF documents from the data directories.

    Returns:
        list: List of loaded documents with metadata indicating document type.
    """
    documents = []

    for folder in [
        "academic_regulations",
        "department_syllabus",
    ]:
        folder_path = os.path.join(DATA_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"Skipping missing folder: {folder_path}")
            continue

        loader = PyPDFDirectoryLoader(folder_path)
        loaded = loader.load()

        for doc in loaded:
            doc.metadata["document_type"] = folder

            if folder == "academic_regulations":
                doc.metadata["document_type"] = "academic_regulations"
            elif folder == "department_syllabus":
                doc.metadata["document_type"] = "department_syllabus"

        documents.extend(loaded)

        print(
            f"{folder}: {len(loaded)} pages loaded"
        )

    return documents


def create_vectorstore():
    """
    Create the vector database from PDF documents.

    This function loads all PDF documents from the data directories,
    splits them into chunks, creates embeddings, and stores them in
    ChromaDB.

    Raises:
        RuntimeError: If no PDF documents are found in the data folders.
    """
    print("\n===== LOADING DOCUMENTS =====")

    documents = load_all_documents()

    print(
        f"\nTotal pages loaded: {len(documents)}"
    )

    if not documents:
        raise RuntimeError(
            "No PDF documents were found in the data folders."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=250,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(
        f"Total chunks created: {len(chunks)}"
    )

    # Remove old database so old/incomplete indexing
    # does not remain mixed with the new one.
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

    print("\n===== CREATING EMBEDDINGS =====")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("\n===== BUILDING CHROMA =====")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAME
    )

    print("\nVector database created successfully.")
    print(f"Location: {CHROMA_PERSIST_DIR}/")


if __name__ == "__main__":
    create_vectorstore()
