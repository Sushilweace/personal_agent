"""Local retrieval-augmented generation: ingestion, indexing and querying.

Documents (``.txt``, ``.md``, ``.pdf``) are read from ``settings.DOCUMENTS_DIR``,
split with ``RecursiveCharacterTextSplitter``, embedded with Ollama
(``nomic-embed-text``) and persisted in ChromaDB under ``settings.CHROMA_DIR``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".pdf"})
MAX_QUERY_RESULTS: int = 10

ProgressCallback = Callable[[int, int], None]


# --------------------------------------------------------------------------- #
# Embeddings / vector store
# --------------------------------------------------------------------------- #
class NomicEmbeddings(OllamaEmbeddings):
    """``OllamaEmbeddings`` that adds nomic-embed-text's task prefixes.

    nomic-embed-text is trained with ``search_document:`` / ``search_query:``
    prefixes; using them noticeably improves retrieval quality.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"search_document: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


@lru_cache(maxsize=1)
def get_embeddings() -> NomicEmbeddings:
    """Return the shared embedding client."""
    return NomicEmbeddings(
        model=settings.EMBED_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        keep_alive=settings.LLM_KEEP_ALIVE,
        client_kwargs={"timeout": settings.LLM_TIMEOUT_SECONDS},
    )


@lru_cache(maxsize=1)
def _get_client() -> chromadb.ClientAPI:
    """Return the shared persistent Chroma client (telemetry disabled)."""
    settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Return the shared Chroma vector store (cosine distance)."""
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        client=_get_client(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_vectorstore() -> None:
    """Delete the collection and drop cached handles so it is rebuilt fresh."""
    try:
        _get_client().delete_collection(settings.CHROMA_COLLECTION)
    except Exception as exc:  # collection may simply not exist yet
        logger.debug("delete_collection ignored: %s", exc)
    get_vectorstore.cache_clear()


def indexed_chunk_count() -> int:
    """Number of chunks currently stored in the vector store."""
    return int(get_vectorstore()._collection.count())  # noqa: SLF001 - no public count API


def is_index_empty() -> bool:
    """True if nothing has been indexed yet."""
    return indexed_chunk_count() == 0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path, source: str) -> list[Document]:
    """One ``Document`` per non-empty PDF page."""
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty user password.
        if not reader.decrypt(""):
            raise ValueError("PDF is password protected")
    docs: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source, "file_name": path.name, "page": page_number},
                )
            )
    if not docs:
        raise ValueError("no extractable text (scanned PDF?)")
    return docs


def load_documents(directory: Path | None = None) -> tuple[list[Document], list[str]]:
    """Load every supported file below ``directory``.

    Returns:
        ``(documents, problems)`` where ``problems`` lists human-readable
        messages for files that could not be read. One bad file never aborts
        the whole load.
    """
    root = (directory or settings.DOCUMENTS_DIR).resolve()
    documents: list[Document] = []
    problems: list[str] = []

    if not root.is_dir():
        return documents, [f"Documents directory not found: {root}"]

    files = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    )

    for path in files:
        source = path.relative_to(root).as_posix()
        try:
            if path.suffix.lower() == ".pdf":
                documents.extend(_load_pdf(path, source))
            else:
                text = _read_text_file(path).strip()
                if not text:
                    raise ValueError("file is empty")
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": source, "file_name": path.name},
                    )
                )
        except Exception as exc:
            logger.warning("Skipping %s: %s", source, exc)
            problems.append(f"{source}: {exc}")

    return documents, problems


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents with ``RecursiveCharacterTextSplitter``."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _chunk_ids(chunks: list[Document]) -> list[str]:
    """Deterministic IDs so re-indexing the same content upserts, not duplicates."""
    counters: dict[str, int] = defaultdict(int)
    ids: list[str] = []
    for chunk in chunks:
        key = f"{chunk.metadata.get('source')}|{chunk.metadata.get('page', 0)}"
        index = counters[key]
        counters[key] += 1
        digest = hashlib.sha1(f"{key}|{index}|{chunk.page_content}".encode("utf-8")).hexdigest()
        chunk.metadata["chunk"] = index
        ids.append(digest)
    return ids


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
@dataclass
class IndexStats:
    """Summary of an indexing run."""

    files: int = 0
    chunks: int = 0
    problems: list[str] = field(default_factory=list)


def index_documents(
    reset: bool = False,
    progress: ProgressCallback | None = None,
) -> IndexStats:
    """Read, chunk, embed and store all documents.

    Args:
        reset: Wipe the existing collection first (full re-index).
        progress: Optional ``callback(done_chunks, total_chunks)``.

    Raises:
        RuntimeError: If embedding fails (e.g. Ollama is down or the embedding
            model is missing). Partial progress remains in the store.
    """
    if reset:
        reset_vectorstore()

    documents, problems = load_documents()
    stats = IndexStats(files=len({d.metadata["source"] for d in documents}), problems=problems)
    if not documents:
        return stats

    chunks = split_documents(documents)
    ids = _chunk_ids(chunks)
    stats.chunks = len(chunks)

    store = get_vectorstore()
    batch = settings.EMBED_BATCH_SIZE
    try:
        for start in range(0, len(chunks), batch):
            store.add_documents(chunks[start : start + batch], ids=ids[start : start + batch])
            if progress:
                progress(min(start + batch, len(chunks)), len(chunks))
    except Exception as exc:
        raise RuntimeError(
            f"Embedding failed ({type(exc).__name__}: {exc}). "
            f"Is Ollama running and is '{settings.EMBED_MODEL}' pulled?"
        ) from exc

    # Drop chunks of edited/deleted files (only after every batch succeeded).
    if not reset:
        keep = set(ids)
        stale = [i for i in store.get(include=[])["ids"] if i not in keep]
        for start in range(0, len(stale), 500):
            store.delete(ids=stale[start : start + 500])
    return stats


# --------------------------------------------------------------------------- #
# Querying
# --------------------------------------------------------------------------- #
def query_knowledge_base(query: str, n_results: int | None = None) -> str:
    """Semantic search over the indexed personal documents.

    Never raises: every failure is returned as an ``"Error: ..."`` string so
    that it can be handed straight back to the LLM.

    Args:
        query: Natural-language search query.
        n_results: Number of chunks to return (clamped to 1..10). Defaults to
            ``settings.RETRIEVAL_TOP_K``.

    Returns:
        Formatted passages with their source and relevance score.
    """
    query = (query or "").strip()
    if not query:
        return "Error: the search query is empty."
    n_results = max(1, min(int(n_results or settings.RETRIEVAL_TOP_K), MAX_QUERY_RESULTS))

    try:
        if is_index_empty():
            return (
                "The knowledge base is empty. Add .txt/.md/.pdf files to "
                f"'{settings.DOCUMENTS_DIR}' and run the 'reindex' command."
            )
        results = get_vectorstore().similarity_search_with_score(query, k=n_results)
    except Exception as exc:
        logger.exception("Knowledge base query failed")
        return f"Error: knowledge base query failed ({type(exc).__name__}: {exc})"

    if not results:
        return "No relevant passages found in the personal notes."

    parts: list[str] = []
    for rank, (doc, distance) in enumerate(results, start=1):
        meta = doc.metadata
        location = str(meta.get("source", "unknown"))
        if "page" in meta:
            location += f" (page {meta['page']})"
        relevance = max(0.0, 1.0 - float(distance))  # cosine distance -> similarity
        parts.append(f"[{rank}] Source: {location} | relevance: {relevance:.2f}\n{doc.page_content.strip()}")
    return "\n\n".join(parts)
