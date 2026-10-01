from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_college_documents():
    documents = []

    # Load academic regulations
    regulations_loader = PyPDFDirectoryLoader(
        "data/academic_regulations"
    )
    documents.extend(regulations_loader.load())

    # Load department syllabus
    syllabus_loader = PyPDFDirectoryLoader(
        "data/department_syllabus"
    )
    documents.extend(syllabus_loader.load())

    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(documents)


def create_vectorstore(chunks):
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating ChromaDB vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectorstore


if __name__ == "__main__":
    print("Loading college documents...")

    documents = load_college_documents()
    print(f"Pages loaded: {len(documents)}")

    print("Creating text chunks...")

    chunks = create_chunks(documents)
    print(f"Total chunks created: {len(chunks)}")

    vectorstore = create_vectorstore(chunks)

    print("\nVector store created successfully!")
    print("Location: chroma_db/")