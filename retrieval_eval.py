"""
retrieval_eval.py
==================
Comprehensive retrieval ablation study for the Urban Metro RAG research paper.

4 retrieval strategies compared:
  1. Dense Only          — BGE-large-en-v1.5 embedding similarity
  2. BM25 Only           — Okapi BM25 keyword matching
  3. Hybrid RRF          — Dense + BM25 fused via Reciprocal Rank Fusion
  4. Hybrid + Reranker   — Hybrid results reranked by cross-encoder

Evaluation metrics (two levels):

  LEVEL 1 — Document Retrieval
    Doc Hit@K       : Was the correct document in top-K? (K=1,3,5,10)
    Doc MRR         : Mean Reciprocal Rank of correct document
    Doc NDCG@10     : Normalised Discounted Cumulative Gain

  LEVEL 2 — Answer in Context  ← main metric for the paper
    Answer@K        : Does any top-K chunk CONTAIN the gold answer text?
    Answer MRR      : 1/rank of first chunk containing the answer
    Answer NDCG@10  : NDCG weighted by answer presence

  Additional metrics
    Precision@K     : Fraction of top-K chunks containing the answer
    Rank Variance   : Consistency of correct retrieval across questions

Output files:
  results/retrieval_ablation.csv        — main comparison table (for paper)
  results/retrieval_per_document.csv    — per-document breakdown
  results/retrieval_failures.json       — questions where Answer@5 failed
  results/retrieval_raw_scores.json     — all individual scores
  results/retrieval_ablation_summary.txt — human-readable summary

Usage:
    python retrieval_eval.py                  # full run (all 4 strategies)
    python retrieval_eval.py --no-reranker    # skip reranker (faster)
    python retrieval_eval.py --limit 20       # debug: first 20 questions only
    python retrieval_eval.py --top-k 10       # retrieve top-10 per strategy
"""

from __future__ import annotations
import sys, json, csv, re, argparse, logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import HybridRetriever, load_index, Chunk

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

GOLD_QA_CSV    = Path("data/gold_QA.csv")
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
K_VALUES       = [1, 3, 5, 10]
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL    = 20   # how many hybrid results to feed the reranker

STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","must","can","of","in","on","at","to","for","with","by","from",
    "that","this","these","those","and","or","but","if","you","your","it","its"
}


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldQA:
    question_id : str
    document_id : str
    question    : str
    answer      : str
    section     : str


@dataclass
class RetrievalScore:
    question_id       : str
    document_id       : str
    strategy          : str
    # Level 1 — Document
    doc_hit_at_1      : int   = 0
    doc_hit_at_3      : int   = 0
    doc_hit_at_5      : int   = 0
    doc_hit_at_10     : int   = 0
    doc_mrr           : float = 0.0
    doc_ndcg_at_10    : float = 0.0
    # Level 2 — Answer in Context
    answer_in_top_1   : int   = 0
    answer_in_top_3   : int   = 0
    answer_in_top_5   : int   = 0
    answer_in_top_10  : int   = 0
    answer_mrr        : float = 0.0
    answer_ndcg_at_10 : float = 0.0
    # Additional
    answer_precision_at_5  : float = 0.0   # fraction of top-5 chunks containing answer
    answer_precision_at_10 : float = 0.0
    answer_rank            : int   = 0     # rank of first chunk with answer (0=not found)
    top1_doc               : str   = ""
    top1_section           : str   = ""


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gold_qa() -> list[GoldQA]:
    items = []
    with open(GOLD_QA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            qid = row.get("question_id", "").strip()
            if not qid:
                continue
            items.append(GoldQA(
                question_id = qid,
                document_id = row["document_id"].strip(),
                question    = row["question"].strip(),
                answer      = row["answer"].strip(),
                section     = row["section"].strip(),
            ))
    log.info(f"Loaded {len(items)} gold Q&A pairs")
    return items


def load_chunks() -> list[Chunk]:
    data = json.loads(Path("chunk_cache.json").read_text())
    return [
        Chunk(chunk_id=d["chunk_id"], document_id=d["document_id"],
              section=d["section"], text=d["text"],
              domain=d["domain"], organization=d["organization"], year=d["year"])
        for d in data
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Answer matching
# ─────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_in_chunk(gold_answer: str, chunk_text: str) -> bool:
    """
    True if the gold answer is present in the chunk.

    Two-pass strategy:
      1. Hard match  : normalised answer is a substring of normalised chunk
      2. Soft match  : ≥60% of content words in the answer appear in the chunk
                       (handles minor formatting differences between PDF extraction
                        and the gold answer text)
    """
    norm_chunk  = normalise(chunk_text)
    norm_answer = normalise(gold_answer)

    if norm_answer in norm_chunk:
        return True

    content = set(norm_answer.split()) - STOPWORDS
    if not content:
        return False
    coverage = len(content & set(norm_chunk.split())) / len(content)
    return coverage >= 0.60


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def ndcg(relevance: list[int], k: int = 10) -> float:
    """Binary NDCG@k. relevance[i]=1 if rank i+1 is relevant."""
    k    = min(k, len(relevance))
    dcg  = sum(relevance[i] / np.log2(i + 2) for i in range(k))
    idcg = 1.0 / np.log2(2)   # ideal: relevant at rank 1
    return dcg / idcg if idcg > 0 else 0.0


def compute_scores(
    strategy    : str,
    question_id : str,
    document_id : str,
    gold_answer : str,
    retrieved   : list[dict],
) -> RetrievalScore:
    doc_ids  = [r["document_id"] for r in retrieved]
    texts    = [r["text"]        for r in retrieved]
    sections = [r.get("section", "") for r in retrieved]

    sc = RetrievalScore(
        question_id  = question_id,
        document_id  = document_id,
        strategy     = strategy,
        top1_doc     = doc_ids[0]  if doc_ids  else "",
        top1_section = sections[0] if sections else "",
    )

    # ── Level 1: Document ─────────────────────────────────────────────────────
    doc_rank = next((i+1 for i, d in enumerate(doc_ids) if d == document_id), None)
    for k in K_VALUES:
        setattr(sc, f"doc_hit_at_{k}", int(document_id in doc_ids[:k]))
    sc.doc_mrr        = 1.0 / doc_rank if doc_rank else 0.0
    sc.doc_ndcg_at_10 = ndcg([int(d == document_id) for d in doc_ids])

    # ── Level 2: Answer in Context ────────────────────────────────────────────
    hit_flags  = [int(answer_in_chunk(gold_answer, t)) for t in texts]
    answer_rank = next((i+1 for i, h in enumerate(hit_flags) if h), None)

    for k in K_VALUES:
        setattr(sc, f"answer_in_top_{k}", int(any(hit_flags[:k])))

    sc.answer_mrr        = 1.0 / answer_rank if answer_rank else 0.0
    sc.answer_ndcg_at_10 = ndcg(hit_flags)
    sc.answer_rank       = answer_rank if answer_rank else 0

    # Precision@K — fraction of retrieved chunks containing the answer
    sc.answer_precision_at_5  = sum(hit_flags[:5])  / min(5,  len(hit_flags)) if hit_flags else 0.0
    sc.answer_precision_at_10 = sum(hit_flags[:10]) / min(10, len(hit_flags)) if hit_flags else 0.0

    return sc


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval strategies
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_dense(r: HybridRetriever, query: str, top_k: int) -> list[dict]:
    results = []
    for chunk_id, score in r._dense_retrieve(query, top_k):
        c = r.chunk_map.get(chunk_id)
        if c:
            results.append({"chunk_id": chunk_id, "document_id": c.document_id,
                            "section": c.section, "text": c.text, "score": score})
    return results


def retrieve_bm25(r: HybridRetriever, query: str, top_k: int) -> list[dict]:
    results = []
    for chunk_id, score in r._bm25_retrieve(query, top_k):
        c = r.chunk_map.get(chunk_id)
        if c:
            results.append({"chunk_id": chunk_id, "document_id": c.document_id,
                            "section": c.section, "text": c.text, "score": score})
    return results


def retrieve_hybrid(r: HybridRetriever, query: str, top_k: int) -> list[dict]:
    return r.retrieve(query, top_k=top_k)


def retrieve_reranked(r: HybridRetriever, reranker, query: str, top_k: int) -> list[dict]:
    candidates = r.retrieve(query, top_k=RERANK_POOL)
    return reranker.rerank(query, candidates, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder reranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        log.info(f"Loading cross-encoder: {model_name}")
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        log.info("Cross-encoder ready.")

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return candidates
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(scores: list[RetrievalScore]) -> dict:
    def mean(vals): return round(float(np.mean(vals)), 4) if vals else 0.0
    def std(vals):  return round(float(np.std(vals)),  4) if vals else 0.0

    ranks = [s.answer_rank for s in scores if s.answer_rank > 0]
    return {
        "n"                        : len(scores),
        # Level 1
        "doc_hit_at_1"             : mean([s.doc_hit_at_1      for s in scores]),
        "doc_hit_at_3"             : mean([s.doc_hit_at_3      for s in scores]),
        "doc_hit_at_5"             : mean([s.doc_hit_at_5      for s in scores]),
        "doc_hit_at_10"            : mean([s.doc_hit_at_10     for s in scores]),
        "doc_mrr"                  : mean([s.doc_mrr           for s in scores]),
        "doc_ndcg_at_10"           : mean([s.doc_ndcg_at_10    for s in scores]),
        # Level 2
        "answer_in_top_1"          : mean([s.answer_in_top_1   for s in scores]),
        "answer_in_top_3"          : mean([s.answer_in_top_3   for s in scores]),
        "answer_in_top_5"          : mean([s.answer_in_top_5   for s in scores]),
        "answer_in_top_10"         : mean([s.answer_in_top_10  for s in scores]),
        "answer_mrr"               : mean([s.answer_mrr        for s in scores]),
        "answer_ndcg_at_10"        : mean([s.answer_ndcg_at_10 for s in scores]),
        # Additional
        "answer_precision_at_5"    : mean([s.answer_precision_at_5  for s in scores]),
        "answer_precision_at_10"   : mean([s.answer_precision_at_10 for s in scores]),
        "answer_rank_mean"         : mean(ranks),
        "answer_rank_std"          : std(ranks),
        "answer_not_found_rate"    : mean([int(s.answer_in_top_10 == 0) for s in scores]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def save_ablation_csv(results: dict[str, dict]):
    path = RESULTS_DIR / "retrieval_ablation.csv"
    rows = [{"strategy": s, **m} for s, m in results.items()]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Ablation table → {path}")


def save_per_document_csv(all_scores: list[RetrievalScore]):
    path       = RESULTS_DIR / "retrieval_per_document.csv"
    strategies = sorted(set(s.strategy     for s in all_scores))
    doc_ids    = sorted(set(s.document_id  for s in all_scores))
    rows = []
    for doc_id in doc_ids:
        for strategy in strategies:
            subset = [s for s in all_scores
                      if s.document_id == doc_id and s.strategy == strategy]
            if subset:
                rows.append({"document_id": doc_id, "strategy": strategy,
                             **aggregate(subset)})
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    log.info(f"Per-document table → {path}")


def save_failure_cases(all_scores: list[RetrievalScore], gold_qa: list[GoldQA]) -> list:
    qa_map     = {q.question_id: q for q in gold_qa}
    strategies = sorted(set(s.strategy for s in all_scores))
    q_ids      = sorted(set(s.question_id for s in all_scores))
    failures   = []

    for qid in q_ids:
        q_scores = {s.strategy: s for s in all_scores if s.question_id == qid}
        failed   = [st for st in strategies
                    if q_scores.get(st) and q_scores[st].answer_in_top_5 == 0]
        if failed:
            qa = qa_map.get(qid)
            failures.append({
                "question_id"       : qid,
                "document_id"       : qa.document_id if qa else "",
                "question"          : qa.question    if qa else "",
                "gold_answer"       : qa.answer[:300] if qa else "",
                "failed_strategies" : failed,
                "passed_strategies" : [st for st in strategies if st not in failed],
                "top1_per_strategy" : {
                    st: f"{q_scores[st].top1_doc} | {q_scores[st].top1_section}"
                    for st in strategies if st in q_scores
                },
            })

    (RESULTS_DIR / "retrieval_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    log.info(f"Failure cases → results/retrieval_failures.json  ({len(failures)} questions)")
    return failures


def print_and_save_summary(results: dict[str, dict], n_failures: int):
    strategies = list(results.keys())
    lines = []

    def add(line=""):
        lines.append(line)
        print(line)

    add("\n" + "=" * 78)
    add("  URBAN METRO RAG — RETRIEVAL ABLATION STUDY")
    add("=" * 78)
    add(f"  Questions: {results[strategies[0]]['n']}   Strategies: {len(strategies)}")

    # ── Level 1 ──────────────────────────────────────────────────────────────
    add("\n  ── LEVEL 1: Document Hit@K ─────────────────────────────────────────────")
    hdr = f"  {'Strategy':<28} {'@1':>6} {'@3':>6} {'@5':>6} {'@10':>6}  {'MRR':>6}  {'NDCG@10':>8}"
    add(hdr)
    add("  " + "─"*74)
    for s in strategies:
        m = results[s]
        add(f"  {s:<28} {m['doc_hit_at_1']:>6.3f} {m['doc_hit_at_3']:>6.3f} "
            f"{m['doc_hit_at_5']:>6.3f} {m['doc_hit_at_10']:>6.3f}  "
            f"{m['doc_mrr']:>6.3f}  {m['doc_ndcg_at_10']:>8.3f}")

    # ── Level 2 ──────────────────────────────────────────────────────────────
    add("\n  ── LEVEL 2: Answer in Context@K  ◄ main metric ────────────────────────")
    add(hdr)
    add("  " + "─"*74)
    for s in strategies:
        m = results[s]
        add(f"  {s:<28} {m['answer_in_top_1']:>6.3f} {m['answer_in_top_3']:>6.3f} "
            f"{m['answer_in_top_5']:>6.3f} {m['answer_in_top_10']:>6.3f}  "
            f"{m['answer_mrr']:>6.3f}  {m['answer_ndcg_at_10']:>8.3f}")

    # ── Additional ────────────────────────────────────────────────────────────
    add("\n  ── Additional Metrics ──────────────────────────────────────────────────")
    add(f"  {'Strategy':<28} {'Prec@5':>8} {'Prec@10':>8}  {'Rank μ':>7}  {'Rank σ':>7}  {'NotFound':>9}")
    add("  " + "─"*74)
    for s in strategies:
        m = results[s]
        add(f"  {s:<28} {m['answer_precision_at_5']:>8.3f} {m['answer_precision_at_10']:>8.3f}  "
            f"{m['answer_rank_mean']:>7.2f}  {m['answer_rank_std']:>7.2f}  "
            f"{m['answer_not_found_rate']:>9.3f}")

    # ── Best ──────────────────────────────────────────────────────────────────
    best_a5  = max(strategies, key=lambda s: results[s]["answer_in_top_5"])
    best_mrr = max(strategies, key=lambda s: results[s]["answer_mrr"])
    add("\n  ── Best Strategy ───────────────────────────────────────────────────────")
    add(f"  Answer@5 : {best_a5:<25} ({results[best_a5]['answer_in_top_5']:.3f})")
    add(f"  MRR      : {best_mrr:<25} ({results[best_mrr]['answer_mrr']:.3f})")
    add(f"  Questions with Answer@5 failure (any strategy): {n_failures}")
    add("=" * 78 + "\n")

    (RESULTS_DIR / "retrieval_ablation_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log.info("Summary → results/retrieval_ablation_summary.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",       type=int,  default=None,
                        help="Evaluate first N questions only (debug)")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Skip cross-encoder reranker")
    parser.add_argument("--top-k",       type=int,  default=10,
                        help="Chunks to retrieve per strategy (default: 10)")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────────
    log.info("Loading index and chunks...")
    chunks     = load_chunks()
    collection = load_index()
    retriever  = HybridRetriever(collection, chunks)
    log.info(f"Index ready — {len(chunks)} chunks")

    reranker = None
    if not args.no_reranker:
        try:
            reranker = CrossEncoderReranker()
        except Exception as e:
            log.warning(f"Reranker not available ({e}) — skipping")

    gold_qa = load_gold_qa()
    if args.limit:
        gold_qa = gold_qa[:args.limit]

    strategies = ["Dense", "BM25", "Hybrid_RRF"]
    if reranker:
        strategies.append("Hybrid_RRF_Reranked")

    log.info(f"Running: {strategies}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_scores: list[RetrievalScore] = []
    total = len(gold_qa) * len(strategies)
    done  = 0

    for qa in gold_qa:
        for strategy in strategies:
            done += 1
            if done % 50 == 0 or done == total:
                log.info(f"[{done}/{total}] {strategy} — {qa.question_id}")

            if strategy == "Dense":
                retrieved = retrieve_dense(retriever, qa.question, args.top_k)
            elif strategy == "BM25":
                retrieved = retrieve_bm25(retriever, qa.question, args.top_k)
            elif strategy == "Hybrid_RRF":
                retrieved = retrieve_hybrid(retriever, qa.question, args.top_k)
            elif strategy == "Hybrid_RRF_Reranked":
                retrieved = retrieve_reranked(retriever, reranker, qa.question, args.top_k)
            else:
                continue

            sc = compute_scores(
                strategy    = strategy,
                question_id = qa.question_id,
                document_id = qa.document_id,
                gold_answer = qa.answer,
                retrieved   = retrieved,
            )
            all_scores.append(sc)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg_results = {
        s: aggregate([sc for sc in all_scores if sc.strategy == s])
        for s in strategies
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    save_ablation_csv(agg_results)
    save_per_document_csv(all_scores)
    failures = save_failure_cases(all_scores, gold_qa)
    (RESULTS_DIR / "retrieval_raw_scores.json").write_text(
        json.dumps([asdict(s) for s in all_scores], indent=2), encoding="utf-8"
    )

    print_and_save_summary(agg_results, len(failures))
    log.info("All results saved in results/")


if __name__ == "__main__":
    main()