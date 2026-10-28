from pathlib import Path

from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents


def main():
    file_path = Path(
        "data/sample_kb/company_it_handbook.md"
    )

    # Load the IT handbook.
    docs = load_file(file_path)
    print(f"Loaded documents: {len(docs)}")

    # Split documents into chunks.
    chunks = chunk_documents(docs)
    print(f"Created chunks: {len(chunks)}")

    # Embed and store the chunks in Pinecone.
    add_documents(chunks)

    print("Document ingestion completed successfully!")


if __name__ == "__main__":
    main()

