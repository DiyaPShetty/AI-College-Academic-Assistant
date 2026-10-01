from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------
# Load syllabus documents
# ---------------------------------------------------------

loader = PyPDFDirectoryLoader(
    "data/department_syllabus"
)

documents = loader.load()

print(f"Total pages loaded: {len(documents)}")


# ---------------------------------------------------------
# Create larger chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=400
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# Create Chroma vector database
# ---------------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector store created successfully!")
print("Location: chroma_db/")