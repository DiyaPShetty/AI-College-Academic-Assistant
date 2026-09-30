from langchain_community.document_loaders import PyPDFDirectoryLoader

print("Starting PDF loading...")

loader = PyPDFDirectoryLoader(
    "data/academic_regulations"
)

print("Loader created.")
print("Loading regulations PDF...")

documents = loader.load()

print(f"Finished loading!")
print(f"Total pages loaded: {len(documents)}")

for i, doc in enumerate(documents[:3]):
    print(f"\n--- Page {i + 1} ---")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Text:")
    print(doc.page_content[:500])