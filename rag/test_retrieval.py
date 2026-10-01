from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Load the same embedding model used when creating ChromaDB
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Test question
query = "What is the minimum attendance requirement for students?"

print("\nQuestion:")
print(query)

print("\nSearching the college documents...\n")

results = retriever.invoke(query)

print(f"Retrieved {len(results)} chunks.\n")

for i, doc in enumerate(results, start=1):
    print("=" * 70)
    print(f"RESULT {i}")
    print("=" * 70)

    print("\nSource:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))

    print("\nContent:")
    print(doc.page_content[:1000])

print("\n" + "=" * 70)