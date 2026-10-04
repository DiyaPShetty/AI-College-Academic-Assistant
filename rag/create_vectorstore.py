import os
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


DATA_DIR = "data"
VECTOR_DB_DIR = "chroma_db"


def load_all_documents():
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
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)

    print("\n===== CREATING EMBEDDINGS =====")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\n===== BUILDING CHROMA =====")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
        collection_name="college_knowledge"
    )

    print("\nVector database created successfully.")
    print(f"Location: {VECTOR_DB_DIR}/")


if __name__ == "__main__":
    create_vectorstore()
