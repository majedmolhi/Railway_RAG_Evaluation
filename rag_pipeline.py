"""
urban_metro_rag/rag_pipeline.py
================================
Core RAG pipeline for the Urban Metro Railway Technical Knowledge Retrieval project.

Pipeline stages:
  1. Document ingestion   — load and parse section-structured .txt files
  2. Chunking             — hierarchical, section-aware splitting
  3. Embedding            — dense vector generation
  4. Vector store         — ChromaDB with rich metadata
  5. Retrieval            — hybrid BM25 + dense search with optional reranking

Usage:
    python rag_pipeline.py --mode index   # build the index from documents
    python rag_pipeline.py --mode query --q "What is the standard track gage?"
"""

from __future__ import annotations
import os, re, json, csv, argparse, logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Paths -----------------------------------------------------------------------
DOCS_DIR        = Path("data/sections")          # place all *_sections.txt here
METADATA_CSV    = Path("data/metadata.csv")
CHROMA_DIR      = Path("chroma_db")
COLLECTION_NAME = "urban_metro_rag"

# Chunking parameters ---------------------------------------------------------
MAX_CHUNK_TOKENS     = 400   # hard ceiling per chunk (approx words * 1.3)
MIN_CHUNK_WORDS      = 5     # discard fragments smaller than this
OVERLAP_SENTENCES    = 2     # sentences of overlap between sub-section chunks

# Retrieval parameters --------------------------------------------------------
TOP_K_DENSE          = 10    # initial dense retrieval pool
TOP_K_BM25           = 10    # BM25 candidate pool
TOP_K_FINAL          = 5     # final returned chunks (after optional reranking)
ALPHA_HYBRID         = 0.6   # weight for dense score in hybrid fusion (0=BM25 only, 1=dense only)

# Embedding model (sentence-transformers, runs locally) -----------------------
# Recommended for scientific/technical text:
#   "BAAI/bge-large-en-v1.5"   ← strong on technical retrieval benchmarks
#   "intfloat/e5-large-v2"     ← pair queries with "query: " prefix
#   "all-mpnet-base-v2"        ← solid general baseline
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DocumentMetadata:
    document_id  : str
    title        : str
    domain       : str
    organization : str
    year         : int
    pages        : int
    source_url   : str


@dataclass
class Chunk:
    chunk_id     : str
    document_id  : str
    section      : str           # e.g. "Section 3 — Emergency protection"
    text         : str
    word_count   : int = field(init=False)
    # propagated from DocumentMetadata
    domain       : str = ""
    organization : str = ""
    year         : int = 0

    def __post_init__(self):
        self.word_count = len(self.text.split())


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Document ingestion
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(csv_path: Path) -> dict[str, DocumentMetadata]:
    """Load document-level metadata from CSV."""
    meta = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            doc_id = row["document_id"].strip()
            meta[doc_id] = DocumentMetadata(
                document_id  = doc_id,
                title        = row["title"].strip(),
                domain       = row["domain"].strip(),
                organization = row["organization"].strip(),
                year         = int(row["year"].strip()),
                pages        = int(row["pages"].strip()),
                source_url   = row["source_url"].strip(),
            )
    log.info(f"Loaded metadata for {len(meta)} documents")
    return meta


def load_document_text(path: Path) -> str:
    """Read a sections .txt file, normalising line endings."""
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Chunking
# ─────────────────────────────────────────────────────────────────────────────

# Section delimiter patterns observed across the corpus:
#   RSSB    : "===...===" lines followed by "SECTION N — Title"
#   SCRRA   : numeric headers "3.3.4 Gage Limits", "9.12.1 Heat Inspections"
#   ERA     : "3.1.2.1. Maximum distance ..."
#   FRA_LERT: "1.4 Extrication Experiments", "1.6 Locomotive Experiments"
#   JICA    : topic header lines (no number, mixed caps)
#   NR_ESG  : short title lines before indented bullet content
#
# FIX: Single-number list items like "1. Ambient temperature..." must NOT be
# treated as section headers. Only multi-level numbers (X.Y or X.Y.Z) qualify.
# Before fix: (?:\d{1,2}\.[\d\.]*\s+...)  matched "1. text" ← wrong
# After fix:  (?:\d{1,2}\.\d+[\d\.]*\s+...) requires at least X.Y format

SECTION_HEADER_RE = re.compile(
    r"^(?:"
    r"={5,}"                                      # RSSB separator lines
    r"|(?:\d{1,2}\.\d+[\d\.]*\s+\S.{2,60}$)"     # multi-level: 3.3.4 or 9.12.1 (NOT "1.")
    r"|(?:SECTION\s+\d+\s*[—\-–].+$)"            # RSSB "SECTION N — ..."
    r"|(?:[A-Z][A-Z\s/\(\)&,-]{8,60}$)"          # ALL-CAPS topic headers (JICA / NR)
    r")",
    re.MULTILINE,
)

# Section headers must NOT have leading whitespace.
# Indented lines like "   2.1. Air temperature..." are list sub-items, not headers.
def is_section_header(line: str) -> bool:
    if line != line.lstrip():   # indented → never a header
        return False
    return bool(SECTION_HEADER_RE.match(line.strip()) and line.strip())

# Patterns that identify a numbered or bulleted list item line
LIST_ITEM_RE = re.compile(
    r"^(?:"
    r"\d+[\.\)]"          # "1." or "1)"
    r"|\([a-zA-Z0-9]+\)"  # "(a)" or "(1)"
    r"|[•\-\*]"           # bullet symbols
    r"|\([ivxlIVXL]+\)"   # roman numerals "(i)" "(ii)"
    r")"
)


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split a document into (section_header, section_body) tuples.
    Returns a flat list; the first item may have an empty header
    for any preamble before the first recognised section.
    """
    sections: list[tuple[str, str]] = []
    lines = text.split("\n")
    current_header = ""
    current_body: list[str] = []

    for line in lines:
        if is_section_header(line) and line.strip():
            if current_body:
                body = "\n".join(current_body).strip()
                if body:
                    sections.append((current_header, body))
            current_header = line.strip()
            current_body = []
        else:
            current_body.append(line)

    # flush last section
    if current_body:
        body = "\n".join(current_body).strip()
        if body:
            sections.append((current_header, body))

    return sections


def approx_token_count(text: str) -> int:
    """Rough token estimate: words * 1.35 (accounts for sub-word tokenisation)."""
    return int(len(text.split()) * 1.35)


def split_into_blocks(text: str) -> list[str]:
    """
    Split section body into semantic blocks for chunking.

    A block is one of:
      - A prose sentence or paragraph
      - A LIST BLOCK: an introductory sentence ending with ":" PLUS all
        following list items — kept together as one atomic unit so that
        "Heat inspections must be performed when:" stays with items 1-4.

    This fixes the core chunking bug where numbered list items were being
    split from their introductory sentence.
    """
    blocks: list[str] = []
    lines  = [l.rstrip() for l in text.split("\n")]
    i      = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # ── Detect intro sentence ending with ":" ────────────────────────────
        # Collect all consecutive list items that follow as one block
        if line.endswith(":") and not LIST_ITEM_RE.match(line):
            block_lines = [line]
            j = i + 1
            # skip blank lines immediately after the colon line
            while j < len(lines) and not lines[j].strip():
                j += 1
            # collect list items
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    # blank line — peek ahead to see if list continues
                    peek = j + 1
                    while peek < len(lines) and not lines[peek].strip():
                        peek += 1
                    if peek < len(lines) and LIST_ITEM_RE.match(lines[peek].strip()):
                        j = peek   # continue to next list item
                        continue
                    else:
                        break      # end of list
                if LIST_ITEM_RE.match(next_line):
                    block_lines.append(next_line)
                    j += 1
                else:
                    break          # non-list line ends the list

            blocks.append("\n".join(block_lines))
            i = j

        # ── Regular list item not preceded by detected intro ──────────────────
        elif LIST_ITEM_RE.match(line):
            blocks.append(line)
            i += 1

        # ── Prose line ────────────────────────────────────────────────────────
        else:
            # Split long prose on sentence boundaries
            parts = re.split(r'(?<=[a-z]{3})\.\s+', line)
            for p in parts:
                p = p.strip()
                if p:
                    blocks.append(p)
            i += 1

    return blocks


def sub_chunk_section(header: str, body: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Split a section body into token-limited chunks.

    Uses split_into_blocks() so that list blocks (intro + items) are treated
    as atomic units and never split mid-list.

    Each chunk gets the section header prepended for context.
    """
    if approx_token_count(body) <= max_tokens:
        return [f"{header}\n\n{body}".strip()] if header else [body]

    blocks        : list[str] = split_into_blocks(body)
    chunks        : list[str] = []
    current       : list[str] = []
    current_tokens: int       = 0

    for block in blocks:
        block_tokens = approx_token_count(block)

        # If a single block exceeds max_tokens on its own (very long list),
        # flush current and add the oversized block as its own chunk
        if block_tokens > max_tokens:
            if current:
                chunk_text = (f"{header}\n\n" if header else "") + "\n\n".join(current)
                chunks.append(chunk_text.strip())
                current        = []
                current_tokens = 0
            chunk_text = (f"{header}\n\n" if header else "") + block
            chunks.append(chunk_text.strip())
            continue

        if current_tokens + block_tokens > max_tokens and current:
            chunk_text = (f"{header}\n\n" if header else "") + "\n\n".join(current)
            chunks.append(chunk_text.strip())
            # keep last `overlap` blocks for continuity
            current        = current[-overlap:] if overlap > 0 else []
            current_tokens = sum(approx_token_count(b) for b in current)

        current.append(block)
        current_tokens += block_tokens

    if current:
        chunk_text = (f"{header}\n\n" if header else "") + "\n\n".join(current)
        chunks.append(chunk_text.strip())

    return chunks


def chunk_document(
    doc_id   : str,
    text     : str,
    meta     : DocumentMetadata,
) -> list[Chunk]:
    """
    Full chunking pipeline for one document.
    Returns a list of Chunk objects with all metadata populated.
    """
    sections = split_into_sections(text)
    all_chunks: list[Chunk] = []
    chunk_index = 0

    for header, body in sections:
        sub_chunks = sub_chunk_section(header, body, MAX_CHUNK_TOKENS, OVERLAP_SENTENCES)
        for sub in sub_chunks:
            if len(sub.split()) < MIN_CHUNK_WORDS:
                continue   # discard micro-fragments
            chunk_id = f"{doc_id}_c{chunk_index:04d}"
            c = Chunk(
                chunk_id     = chunk_id,
                document_id  = doc_id,
                section      = header or "Preamble",
                text         = sub,
                domain       = meta.domain,
                organization = meta.organization,
                year         = meta.year,
            )
            all_chunks.append(c)
            chunk_index += 1

    log.info(f"  {doc_id}: {len(sections)} sections → {len(all_chunks)} chunks")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 & 4 — Embedding + Vector Store (ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

def get_embedding_function():
    """
    Returns a ChromaDB-compatible embedding function.
    Uses sentence-transformers locally — no API key required.

    For BGE models, add instruction prefix "Represent this document for retrieval: "
    to passage text; queries use "query: " prefix (handled in retrieval stage).
    """
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cpu",       # change to "cuda" if GPU is available
    )


def build_index(chunks: list[Chunk]) -> chromadb.Collection:
    """
    Insert all chunks into ChromaDB. Stores rich metadata for filtered retrieval.
    Index is persisted to CHROMA_DIR.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop and recreate for clean indexing
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    ef = get_embedding_function()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 128
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        collection.add(
            ids        = [c.chunk_id for c in batch],
            documents  = [c.text for c in batch],
            metadatas  = [
                {
                    "document_id" : c.document_id,
                    "section"     : c.section,
                    "domain"      : c.domain,
                    "organization": c.organization,
                    "year"        : c.year,
                    "word_count"  : c.word_count,
                }
                for c in batch
            ],
        )
        log.info(f"Indexed batch {i//BATCH + 1}/{(len(chunks)-1)//BATCH + 1}")

    log.info(f"Index built: {collection.count()} chunks in '{COLLECTION_NAME}'")
    return collection


def load_index() -> chromadb.Collection:
    """Load an existing persistent index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = get_embedding_function()
    return client.get_collection(COLLECTION_NAME, embedding_function=ef)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Retrieval: Hybrid BM25 + Dense with Reciprocal Rank Fusion
# ─────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Combines BM25 sparse retrieval with dense vector search using
    Reciprocal Rank Fusion (RRF) for score combination.

    Design rationale for this corpus:
    - BM25 excels on exact technical terms (e.g., "0.05 Ohm", "56-1/2 inches",
      "detonators", "PRLT") which dense embeddings may miss.
    - Dense retrieval handles paraphrased questions and semantic similarity.
    - RRF is robust to score scale differences between the two systems.
    """

    def __init__(self, collection: chromadb.Collection, chunks: list[Chunk]):
        self.collection = collection
        self.chunks     = chunks
        self.chunk_map  = {c.chunk_id: c for c in chunks}

        # Build BM25 index over tokenised chunk texts
        tokenised = [c.text.lower().split() for c in chunks]
        self.bm25      = BM25Okapi(tokenised)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def _dense_retrieve(
        self,
        query  : str,
        top_k  : int,
        where  : Optional[dict] = None,
    ) -> list[tuple[str, float]]:
        """
        Dense retrieval from ChromaDB.
        Returns [(chunk_id, cosine_similarity), ...] sorted by score desc.

        For BGE-family models, prepend the retrieval instruction to queries:
        "Represent this sentence for searching relevant passages: <query>"
        """
        query_text = f"Represent this sentence for searching relevant passages: {query}"
        kwargs = dict(query_texts=[query_text], n_results=min(top_k, self.collection.count()))
        if where:
            kwargs["where"] = where
        results = self.collection.query(**kwargs)
        ids        = results["ids"][0]
        distances  = results["distances"][0]   # cosine distance in [0,2]; convert to similarity
        scores     = [1.0 - d / 2.0 for d in distances]
        return list(zip(ids, scores))

    def _bm25_retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """BM25 retrieval; returns [(chunk_id, bm25_score), ...]."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices]

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[str, float]]],
        k: int = 60,
    ) -> list[tuple[str, float]]:
        """
        RRF score = Σ 1 / (k + rank_i) across all ranked lists.
        k=60 is the standard constant from Cormack et al. (2009).
        """
        rrf_scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_id, _) in enumerate(ranked, start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def retrieve(
        self,
        query        : str,
        top_k        : int = TOP_K_FINAL,
        filter_domain: Optional[str] = None,
    ) -> list[dict]:
        """
        Main retrieval method. Returns top_k chunks as dicts with all metadata.

        Args:
            query:         Natural language question.
            top_k:         Number of chunks to return.
            filter_domain: Optional metadata filter (e.g., "emergency", "signaling").
        """
        where = {"domain": filter_domain} if filter_domain else None

        dense_results = self._dense_retrieve(query, TOP_K_DENSE, where)
        bm25_results  = self._bm25_retrieve(query, TOP_K_BM25)
        fused         = self._reciprocal_rank_fusion([dense_results, bm25_results])

        output = []
        for chunk_id, rrf_score in fused[:top_k]:
            if chunk_id not in self.chunk_map:
                continue
            c = self.chunk_map[chunk_id]
            output.append({
                "chunk_id"    : chunk_id,
                "document_id" : c.document_id,
                "section"     : c.section,
                "domain"      : c.domain,
                "organization": c.organization,
                "year"        : c.year,
                "text"        : c.text,
                "rrf_score"   : round(rrf_score, 6),
            })
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Optional: Cross-encoder reranker (improves precision, adds latency)
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Uses a cross-encoder model to rerank retrieved candidates.
    Recommended model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    This is a lightweight model trained specifically for passage reranking.
    Install: pip install sentence-transformers
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return candidates
        pairs  = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_index():
    """Load all documents, chunk them, and build the vector index."""
    metadata = load_metadata(METADATA_CSV)
    all_chunks: list[Chunk] = []

    for doc_id, meta in metadata.items():
        # Try all known filename patterns
        candidates = list(DOCS_DIR.glob(f"{doc_id}*sections*.txt"))
        if not candidates:
            log.warning(f"No file found for {doc_id}, skipping")
            continue
        text   = load_document_text(candidates[0])
        chunks = chunk_document(doc_id, text, meta)
        all_chunks.extend(chunks)

    log.info(f"Total chunks across corpus: {len(all_chunks)}")

    # Persist chunk data for BM25 re-use
    chunk_cache = [
        {"chunk_id": c.chunk_id, "document_id": c.document_id, "section": c.section,
         "domain": c.domain, "organization": c.organization, "year": c.year,
         "text": c.text}
        for c in all_chunks
    ]
    Path("chunk_cache.json").write_text(json.dumps(chunk_cache, indent=2))
    log.info("Chunk cache saved to chunk_cache.json")

    build_index(all_chunks)
    log.info("Indexing complete.")


def run_query(query: str, top_k: int = TOP_K_FINAL):
    """Query the built index and print top results."""
    if not Path("chunk_cache.json").exists():
        log.error("chunk_cache.json not found. Run --mode index first.")
        return

    chunk_data = json.loads(Path("chunk_cache.json").read_text())
    chunks = [
        Chunk(chunk_id=d["chunk_id"], document_id=d["document_id"],
              section=d["section"], text=d["text"],
              domain=d["domain"], organization=d["organization"], year=d["year"])
        for d in chunk_data
    ]

    collection = load_index()
    retriever  = HybridRetriever(collection, chunks)
    results    = retriever.retrieve(query, top_k=top_k)

    print(f"\nQuery: {query}")
    print(f"Top {top_k} retrieved chunks:\n{'─'*60}")
    for i, r in enumerate(results, 1):
        print(f"[{i}] chunk_id={r['chunk_id']} | doc={r['document_id']} | rrf={r['rrf_score']:.4f}")
        print(f"    section: {r['section']}")
        print(f"    text: {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urban Metro RAG Pipeline")
    parser.add_argument("--mode", choices=["index", "query"], required=True)
    parser.add_argument("--q", type=str, default="", help="Query string (--mode query)")
    parser.add_argument("--k", type=int, default=TOP_K_FINAL, help="Number of results")
    args = parser.parse_args()

    if args.mode == "index":
        run_index()
    elif args.mode == "query":
        if not args.q:
            parser.error("--q is required for query mode")
        run_query(args.q, args.k)