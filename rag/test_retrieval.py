from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

question = "DATABASE MANAGEMENT SYSTEMS CS2102-1"

results = vectorstore.similarity_search(
    question,
    k=5
)

print("\n===== RETRIEVED DOCUMENTS =====\n")

for document in results:
    print(
        f"Source: {document.metadata.get('source')}"
    )
    print(
        f"Page: {document.metadata.get('page')}"
    )
    print("\nContent:")
    print(document.page_content)
    print("\n" + "=" * 80)