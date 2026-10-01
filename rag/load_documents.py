from langchain_community.document_loaders import PyPDFDirectoryLoader


def load_college_documents():
    documents = []

    # Load academic regulations
    regulations_loader = PyPDFDirectoryLoader(
        "data/academic_regulations"
    )
    regulations = regulations_loader.load()
    documents.extend(regulations)

    # Load department syllabus
    syllabus_loader = PyPDFDirectoryLoader(
        "data/department_syllabus"
    )
    syllabus = syllabus_loader.load()
    documents.extend(syllabus)

    return documents


if __name__ == "__main__":
    documents = load_college_documents()

    print(f"Total pages loaded: {len(documents)}")

    print("\nPages from each source:")

    regulations_count = sum(
        1 for doc in documents
        if "academic_regulations" in doc.metadata.get("source", "")
    )

    syllabus_count = sum(
        1 for doc in documents
        if "department_syllabus" in doc.metadata.get("source", "")
    )

    print(f"Regulations: {regulations_count}")
    print(f"Syllabus: {syllabus_count}")