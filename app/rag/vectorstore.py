import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


settings = get_settings()

_embeddings = None
_vectorstore = None


# Embedding dimensions for supported models.
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
    "sentence-transformers/all-minilm-l6-v2": 384,
}


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Use a separate index for 384-dimensional MiniLM embeddings.
DEFAULT_INDEX_NAME = "fde-it-support-rag-384"


def get_embedding_dimension(model_name: str | None = None) -> int:
    """Return the dimension of the configured embedding model."""

    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        name = DEFAULT_EMBEDDING_MODEL

    normalized = name.lower()

    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]

    if "text-embedding-3-small" in normalized:
        return 1536

    if "text-embedding-3-large" in normalized:
        return 3072

    if "text-embedding-ada-002" in normalized:
        return 1536

    if "all-minilm-l6-v2" in normalized:
        return 384

    raise ValueError(
        f"Unsupported embedding model '{name}'. "
        "Configure sentence-transformers/all-MiniLM-L6-v2."
    )


def get_embeddings():
    """Load the local MiniLM embedding model."""

    global _embeddings

    if _embeddings is None:
        model_name = (
            settings.embedding_model or DEFAULT_EMBEDDING_MODEL
        ).strip()

        if "all-minilm-l6-v2" not in model_name.lower():
            raise ValueError(
                f"Unsupported embedding model: {model_name}. "
                "Set EMBEDDING_MODEL=sentence-transformers/"
                "all-MiniLM-L6-v2 in your .env file."
            )

        _embeddings = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def ensure_index():
    """Validate the Pinecone index or create it if necessary."""

    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing from your configuration."
        )

    # Prefer the index name configured in .env.
    # Use the new index name only when no name is configured.
    index_name = (
        settings.pinecone_index_name or DEFAULT_INDEX_NAME
    ).strip()

    desired_dimension = get_embedding_dimension(
        DEFAULT_EMBEDDING_MODEL
    )

    pc = Pinecone(api_key=settings.pinecone_api_key)

    existing_indexes = pc.list_indexes()
    index_names = [
        item["name"] for item in existing_indexes
    ]

    if index_name in index_names:
        index_info = pc.describe_index(index_name)

        current_dimension = getattr(
            index_info, "dimension", None
        )

        if current_dimension is None and isinstance(
            index_info, dict
        ):
            current_dimension = index_info.get("dimension")

        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Pinecone index '{index_name}' has dimension "
                f"{current_dimension}, but the embedding model "
                f"requires {desired_dimension}.\n"
                f"Update PINECONE_INDEX_NAME in your .env file to "
                f"'{DEFAULT_INDEX_NAME}' to use a separate index. "
                "Your existing index will not be deleted."
            )

        print(
            f"Using existing Pinecone index: {index_name} "
            f"({current_dimension} dimensions)"
        )

    else:
        print(
            f"Creating Pinecone index: {index_name} "
            f"({desired_dimension} dimensions)"
        )

        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # Wait for Pinecone to finish creating the index.
        while True:
            index_info = pc.describe_index(index_name)
            status = getattr(index_info, "status", None)

            if isinstance(status, dict):
                ready = status.get("ready", False)
            else:
                ready = getattr(status, "ready", False)

            if ready:
                break

            time.sleep(1)

        print(f"Pinecone index '{index_name}' is ready.")

    return pc.Index(index_name)


def get_vectorstore():
    """Return the cached Pinecone vector store."""

    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    """Return a retriever for relevant document chunks."""

    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    """Embed and store document chunks in Pinecone."""

    if not chunks:
        print("No document chunks to add.")
        return []

    store = get_vectorstore()

    result = store.add_documents(chunks)

    print(f"Successfully added {len(chunks)} document chunks.")

    return result
