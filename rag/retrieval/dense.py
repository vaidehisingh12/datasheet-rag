"""
Dense retrieval baseline: embed chunks with a small sentence-transformer and
search them in Qdrant.

Qdrant runs in its embedded "local" mode here (just a folder on disk), so
there's no server or Docker to set up yet. The same client code works against
a real Qdrant server later by changing how the client is created.
"""

import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from rag.ingest.chunk import Chunk
from rag.settings import COLLECTION, EMBED_MODEL, INDEX_DIR

# bge models were trained with this instruction in front of queries (but not
# in front of the passages being searched). Leaving it off costs a bit of
# retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(INDEX_DIR))


def _point_id(chunk_id: str) -> str:
    # Qdrant wants an int or a UUID. Deriving the UUID from the chunk ID means
    # re-indexing a document overwrites its old points instead of duplicating.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def reset_index() -> None:
    client = get_client()
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(
            size=get_model().get_sentence_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )


def index_chunks(chunks: list[Chunk], batch_size: int = 64) -> None:
    model = get_model()
    vectors = model.encode(
        [c.embed_text() for c in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    get_client().upsert(
        COLLECTION,
        points=[
            PointStruct(id=_point_id(c.chunk_id), vector=v.tolist(), payload=c.payload())
            for c, v in zip(chunks, vectors)
        ],
    )


def search(query: str, k: int = 5, component_class: str | None = None) -> list[dict]:
    """Top-k chunks for a question, optionally limited to one component class."""
    vector = get_model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
    flt = None
    if component_class:
        flt = Filter(must=[FieldCondition(key="component_class",
                                          match=MatchValue(value=component_class))])
    hits = get_client().query_points(
        COLLECTION, query=vector.tolist(), limit=k, query_filter=flt
    ).points
    return [{**h.payload, "score": h.score} for h in hits]
