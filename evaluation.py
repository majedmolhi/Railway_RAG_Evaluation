"""
evaluation.py
==============
Evaluation framework for the Urban Metro RAG system.

Retrieval is evaluated at TWO levels:
  Level 1 — Document Hit@K   : Was the correct document in top-K?
  Level 2 — Answer in Context: Does any retrieved chunk CONTAIN the gold answer?
             This is the most rigorous metric for a research paper because a system
             can retrieve the right document but the wrong chunk.

Answer correctness (requires API):
  - Exact Match, Token F1, ROUGE-L, BERTScore

Hallucination (requires API):
  - LLM-as-judge faithfulness scoring

Usage:
    python evaluation.py --no-generation --no-faithfulness   # retrieval only, free
    python evaluation.py --no-faithfulness                   # retrieval + answers
    python evaluation.py                                     # full evaluation
"""

from __future__ import annotations
import os, json, csv, re, time, logging, argparse
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import numpy as np

try:
    from rouge_score import rouge_scorer as rouge_lib
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score_fn
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

GOLD_QA_CSV  = Path("data/gold_QA.csv")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

LLM_MODEL      = "gpt-4o-mini"
MAX_TOKENS_GEN = 512
TOP_K_EVAL     = [1, 3, 5, 10]


def make_openai_client():
    """Initialise OpenAI client from OPENAI_API_KEY environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set.\n"
            "Add OPENAI_API_KEY=sk-proj-... to your .env file"
        )
    return OpenAI(api_key=api_key)


def groq_call(client, system: str, user: str) -> str:
    """Call OpenAI and return text response — with rate limit retry."""
    import time as _time
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS_GEN,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 30 * (attempt + 1)
                log.warning(f"Rate limit hit — waiting {wait}s before retry {attempt+1}/5")
                _time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenAI rate limit exceeded after 5 retries")

RAG_SYSTEM_PROMPT = """You are a precise railway safety and operations assistant.
Answer the question based ONLY on the provided context passages.
If the answer is not found in the context, state "Not found in context."
Be factual and concise. Do not add information beyond what is in the context."""

LLM_SYSTEM_PROMPT = """You are a knowledgeable railway safety and operations assistant.
Answer the question as accurately as possible based on your training knowledge.
Be factual and concise."""


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
class EvalScores:
    question_id      : str
    document_id      : str
    gold_section     : str = ""

    # ── Retrieval Level 1 — Document ──────────────────────────────────────────
    # Was the correct document retrieved in top-K?
    doc_hit_at_1     : int   = 0
    doc_hit_at_3     : int   = 0
    doc_hit_at_5     : int   = 0
    doc_hit_at_10    : int   = 0
    doc_mrr          : float = 0.0

    # ── Retrieval Level 2 — Answer in Context ─────────────────────────────────
    # Does any retrieved chunk CONTAIN the gold answer text?
    # This is the most rigorous retrieval metric for a scientific paper.
    # A system may retrieve the correct document but the wrong chunk —
    # answer_in_context detects this failure explicitly.
    answer_in_top_1  : int   = 0
    answer_in_top_3  : int   = 0
    answer_in_top_5  : int   = 0
    answer_in_top_10 : int   = 0
    answer_mrr       : float = 0.0  # 1/rank of first chunk containing the answer

    # ── Answer quality — RAG ──────────────────────────────────────────────────
    rag_exact_match  : int   = 0
    rag_f1_token     : float = 0.0
    rag_rouge_l      : float = 0.0
    rag_bert_f1      : float = 0.0

    # ── Answer quality — LLM baseline ─────────────────────────────────────────
    llm_exact_match  : int   = 0
    llm_f1_token     : float = 0.0
    llm_rouge_l      : float = 0.0
    llm_bert_f1      : float = 0.0

    # ── Hallucination ─────────────────────────────────────────────────────────
    rag_faithfulness : float = 0.0
    llm_faithfulness : float = 0.0
    rag_not_found    : int   = 0


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_gold_qa(path: Path) -> list[GoldQA]:
    items = []
    with open(path, newline="", encoding="utf-8") as f:
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


# ─────────────────────────────────────────────────────────────────────────────
# A. Retrieval Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, remove punctuation, normalise whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_in_chunk(gold_answer: str, chunk_text: str) -> bool:
    """
    Check whether the gold answer is contained in a retrieved chunk.

    Uses normalised substring matching:
    - Lowercased + punctuation stripped on both sides
    - Handles multi-line gold answers by checking all key phrases

    For multi-sentence gold answers, checks if at least 60% of the
    content words from the gold answer appear in the chunk (soft match),
    which handles minor formatting differences between the extracted
    sections and the original source text.
    """
    norm_chunk  = normalise(chunk_text)
    norm_answer = normalise(gold_answer)

    # --- Hard match: full answer is a substring of chunk ---
    if norm_answer in norm_chunk:
        return True

    # --- Soft match: key content words overlap ---
    # Useful for long multi-sentence answers where minor formatting differs
    answer_words = set(norm_answer.split())
    chunk_words  = set(norm_chunk.split())

    # Remove stopwords for content-word matching
    stopwords = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","must","can","of","in","on",
        "at","to","for","with","by","from","that","this","these",
        "those","and","or","but","if","you","must","your","it","its"
    }
    content_answer = answer_words - stopwords
    if not content_answer:
        return norm_answer in norm_chunk

    overlap  = content_answer & chunk_words
    coverage = len(overlap) / len(content_answer)

    return coverage >= 0.60


def compute_retrieval_metrics(
    gold_doc_id      : str,
    gold_answer      : str,
    retrieved        : list[dict],
    k_values         : list[int] = TOP_K_EVAL,
) -> dict:
    """
    Compute retrieval metrics at two levels:

    Level 1 — Document Hit@K:
        Was the correct document_id present in the top-K results?

    Level 2 — Answer in Context Hit@K:
        Did any of the top-K chunks actually contain the gold answer text?
        This is the ground truth for whether the LLM *could* answer correctly.

    Both levels use MRR (Mean Reciprocal Rank).
    """
    results = {}

    retrieved_doc_ids = [r["document_id"] for r in retrieved]
    retrieved_texts   = [r["text"]        for r in retrieved]

    # ── Level 1: Document Hit@K ───────────────────────────────────────────────
    doc_rank = None
    for i, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            doc_rank = i
            break

    for k in k_values:
        results[f"doc_hit_at_{k}"] = int(gold_doc_id in retrieved_doc_ids[:k])

    results["doc_mrr"] = (1.0 / doc_rank) if doc_rank else 0.0

    # ── Level 2: Answer in Context Hit@K ─────────────────────────────────────
    answer_rank = None
    for i, chunk_text in enumerate(retrieved_texts, start=1):
        if answer_in_chunk(gold_answer, chunk_text):
            answer_rank = i
            break

    for k in k_values:
        top_k_texts = retrieved_texts[:k]
        results[f"answer_in_top_{k}"] = int(
            any(answer_in_chunk(gold_answer, t) for t in top_k_texts)
        )

    results["answer_mrr"] = (1.0 / answer_rank) if answer_rank else 0.0

    # ── NDCG@10 (document level) ──────────────────────────────────────────────
    k_ndcg = min(10, len(retrieved_doc_ids))
    dcg  = sum(
        int(retrieved_doc_ids[i] == gold_doc_id) / np.log2(i + 2)
        for i in range(k_ndcg)
    )
    idcg = 1.0 / np.log2(2)
    results["ndcg_at_10"] = dcg / idcg if idcg > 0 else 0.0

    return results


# ─────────────────────────────────────────────────────────────────────────────
# B. Answer correctness metrics
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> int:
    return int(normalise(prediction) == normalise(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = set(normalise(prediction).split())
    gold_tokens = set(normalise(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction: str, gold: str) -> float:
    if not HAS_ROUGE:
        return 0.0
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(gold, prediction)["rougeL"].fmeasure


def compute_bert_score(predictions: list[str], references: list[str]) -> list[float]:
    if not HAS_BERTSCORE:
        return [0.0] * len(predictions)
    P, R, F = bert_score_fn(
        predictions, references,
        model_type="distilbert-base-uncased",
        lang="en", verbose=False,
    )
    return F.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# C. Hallucination / Faithfulness
# ─────────────────────────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are evaluating whether an AI answer is faithful to the source context.

Context passages:
{context}

AI Answer:
{answer}

Determine what fraction of the factual claims in the AI answer are supported by the context.

Output ONLY a JSON object:
{{
  "supported_claims": <integer>,
  "total_claims": <integer>,
  "faithfulness": <float 0.0-1.0>,
  "explanation": "<brief note on unsupported claims if any>"
}}"""


def compute_faithfulness(answer: str, context: list[str], client) -> float:
    if not context or not answer.strip():
        return 0.0
    context_text = "\n\n---\n\n".join(context[:5])
    prompt = FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
    try:
        raw  = groq_call(client, "", prompt)
        raw  = re.sub(r"```(?:json)?|```", "", raw).strip()
        data = json.loads(raw)
        return float(data.get("faithfulness", 0.0))
    except Exception as e:
        log.warning(f"Faithfulness scoring failed: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LLM generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_rag_answer(question: str, context: list[str], client) -> str:
    context_block = "\n\n".join(
        f"[Passage {i+1}]\n{c}" for i, c in enumerate(context[:5])
    )
    user_message = f"Context:\n{context_block}\n\nQuestion: {question}"
    return groq_call(client, RAG_SYSTEM_PROMPT, user_message)


def generate_llm_answer(question: str, client) -> str:
    return groq_call(client, LLM_SYSTEM_PROMPT, question)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    retriever,
    top_k            : int  = 5,
    run_generation   : bool = True,
    run_faithfulness : bool = True,
    limit            : Optional[int] = None,
):
    gold_qa = load_gold_qa(GOLD_QA_CSV)
    if limit:
        gold_qa = gold_qa[:limit]

    client = None
    if run_generation:
        if not HAS_OPENAI:
            raise RuntimeError("openai not installed: pip install openai")
        client = make_openai_client()

    all_scores         : list[EvalScores] = []
    answer_records     : list[dict]       = []
    hallucination_cases: list[dict]       = []
    rag_predictions, llm_predictions, gold_answers = [], [], []

    # ── Resume support ───────────────────────────────────────────────────────
    resume_path = RESULTS_DIR / "eval_progress.json"
    completed   = set()
    if resume_path.exists():
        try:
            prev = json.loads(resume_path.read_text(encoding="utf-8"))
            completed = set(s["question_id"] for s in prev.get("scores", []))
            all_scores = [EvalScores(**s) for s in prev.get("scores", [])]
            answer_records = prev.get("answers", [])
            log.info(f"Resuming from checkpoint — {len(completed)} questions already done")
        except Exception:
            pass

    for i, qa in enumerate(gold_qa):
        if qa.question_id in completed:
            continue
        log.info(f"[{i+1}/{len(gold_qa)}] {qa.question_id}: {qa.question[:70]}...")

        # ── Retrieve ──────────────────────────────────────────────────────────
        retrieved       = retriever.retrieve(qa.question, top_k=max(top_k, 10))
        retrieved_texts = [r["text"] for r in retrieved]

        # ── Compute retrieval metrics ─────────────────────────────────────────
        ret = compute_retrieval_metrics(
            gold_doc_id  = qa.document_id,
            gold_answer  = qa.answer,
            retrieved    = retrieved,
        )

        scores = EvalScores(
            question_id     = qa.question_id,
            document_id     = qa.document_id,
            gold_section    = qa.section,
            doc_hit_at_1    = ret["doc_hit_at_1"],
            doc_hit_at_3    = ret["doc_hit_at_3"],
            doc_hit_at_5    = ret["doc_hit_at_5"],
            doc_hit_at_10   = ret["doc_hit_at_10"],
            doc_mrr         = ret["doc_mrr"],
            answer_in_top_1 = ret["answer_in_top_1"],
            answer_in_top_3 = ret["answer_in_top_3"],
            answer_in_top_5 = ret["answer_in_top_5"],
            answer_in_top_10= ret["answer_in_top_10"],
            answer_mrr      = ret["answer_mrr"],
        )

        # ── Generation ────────────────────────────────────────────────────────
        rag_ans = llm_ans = ""
        if run_generation and client:
            context_for_llm = retrieved_texts[:top_k]
            rag_ans = generate_rag_answer(qa.question, context_for_llm, client)
            llm_ans = generate_llm_answer(qa.question, client)
            time.sleep(0.5)

        rag_predictions.append(rag_ans)
        llm_predictions.append(llm_ans)
        gold_answers.append(qa.answer)

        # ── Lexical metrics ────────────────────────────────────────────────────
        scores.rag_exact_match = exact_match(rag_ans, qa.answer)
        scores.rag_f1_token    = token_f1(rag_ans, qa.answer)
        scores.rag_rouge_l     = rouge_l_score(rag_ans, qa.answer)
        scores.llm_exact_match = exact_match(llm_ans, qa.answer)
        scores.llm_f1_token    = token_f1(llm_ans, qa.answer)
        scores.llm_rouge_l     = rouge_l_score(llm_ans, qa.answer)
        scores.rag_not_found   = int("not found in context" in rag_ans.lower())

        # ── Faithfulness ──────────────────────────────────────────────────────
        if run_faithfulness and client and retrieved_texts:
            scores.rag_faithfulness = compute_faithfulness(
                rag_ans, retrieved_texts[:top_k], client
            )

        all_scores.append(scores)

        # Save checkpoint after every question
        resume_path.write_text(
            json.dumps({"scores": [asdict(s) for s in all_scores],
                        "answers": answer_records}, indent=2),
            encoding="utf-8"
        )

        if scores.rag_faithfulness < 0.5 and rag_ans:
            hallucination_cases.append({
                "question_id"    : qa.question_id,
                "question"       : qa.question,
                "gold_answer"    : qa.answer,
                "rag_answer"     : rag_ans,
                "faithfulness"   : scores.rag_faithfulness,
                "retrieved_texts": retrieved_texts[:2],
            })

        answer_records.append({
            "question_id"      : qa.question_id,
            "document_id"      : qa.document_id,
            "question"         : qa.question,
            "gold_answer"      : qa.answer,
            "rag_answer"       : rag_ans,
            "llm_answer"       : llm_ans,
            "answer_in_top_5"  : bool(scores.answer_in_top_5),
            "doc_hit_at_5"     : bool(scores.doc_hit_at_5),
        })

    # ── BERTScore batched ──────────────────────────────────────────────────────
    if run_generation and HAS_BERTSCORE and rag_predictions:
        log.info("Computing BERTScore...")
        rag_bert = compute_bert_score(rag_predictions, gold_answers)
        llm_bert = compute_bert_score(llm_predictions, gold_answers)
        new_scores = [s for s in all_scores if s.question_id not in completed]
        for i, sc in enumerate(new_scores):
            if i < len(rag_bert):
                sc.rag_bert_f1 = rag_bert[i]
            if i < len(llm_bert):
                sc.llm_bert_f1 = llm_bert[i]

    # ── Save ───────────────────────────────────────────────────────────────────
    (RESULTS_DIR / "eval_results.json").write_text(
        json.dumps({"scores": [asdict(s) for s in all_scores],
                    "answers": answer_records}, indent=2)
    )
    (RESULTS_DIR / "hallucination_cases.json").write_text(
        json.dumps(hallucination_cases, indent=2)
    )

    summary = aggregate_results(all_scores)
    save_summary_csv(summary)
    print_summary(summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(all_scores: list[EvalScores]) -> dict:
    def mean(vals): return float(np.mean(vals)) if vals else 0.0

    def agg(subset: list[EvalScores]) -> dict:
        return {
            "n"                  : len(subset),
            # Document Hit@K
            "doc_hit_at_1"       : mean([s.doc_hit_at_1      for s in subset]),
            "doc_hit_at_3"       : mean([s.doc_hit_at_3      for s in subset]),
            "doc_hit_at_5"       : mean([s.doc_hit_at_5      for s in subset]),
            "doc_hit_at_10"      : mean([s.doc_hit_at_10     for s in subset]),
            "doc_mrr"            : mean([s.doc_mrr           for s in subset]),
            # Answer in Context Hit@K  ← main metric
            "answer_in_top_1"    : mean([s.answer_in_top_1   for s in subset]),
            "answer_in_top_3"    : mean([s.answer_in_top_3   for s in subset]),
            "answer_in_top_5"    : mean([s.answer_in_top_5   for s in subset]),
            "answer_in_top_10"   : mean([s.answer_in_top_10  for s in subset]),
            "answer_mrr"         : mean([s.answer_mrr        for s in subset]),
            # RAG answer quality
            "rag_exact_match"    : mean([s.rag_exact_match   for s in subset]),
            "rag_f1"             : mean([s.rag_f1_token      for s in subset]),
            "rag_rouge_l"        : mean([s.rag_rouge_l       for s in subset]),
            "rag_bert_f1"        : mean([s.rag_bert_f1       for s in subset]),
            "rag_faithfulness"   : mean([s.rag_faithfulness  for s in subset]),
            "rag_not_found_rate" : mean([s.rag_not_found     for s in subset]),
            # LLM baseline
            "llm_exact_match"    : mean([s.llm_exact_match   for s in subset]),
            "llm_f1"             : mean([s.llm_f1_token      for s in subset]),
            "llm_rouge_l"        : mean([s.llm_rouge_l       for s in subset]),
            "llm_bert_f1"        : mean([s.llm_bert_f1       for s in subset]),
        }

    result = {"overall": agg(all_scores)}
    doc_groups: dict[str, list[EvalScores]] = {}
    for s in all_scores:
        doc_groups.setdefault(s.document_id, []).append(s)
    for doc_id, group in sorted(doc_groups.items()):
        result[doc_id] = agg(group)
    return result


def save_summary_csv(summary: dict):
    rows = []
    for group, metrics in summary.items():
        row = {"group": group}
        row.update(metrics)
        rows.append(row)
    if not rows:
        return
    path = RESULTS_DIR / "eval_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Summary saved to {path}")


def print_summary(summary: dict):
    o = summary.get("overall", {})
    print("\n" + "=" * 65)
    print("  URBAN METRO RAG — EVALUATION SUMMARY")
    print("=" * 65)
    print(f"  Questions evaluated : {o.get('n', 0)}")
    print()
    print("  ── RETRIEVAL LEVEL 1 — Document Hit@K ────────────────")
    print(f"  Doc Hit@1  : {o.get('doc_hit_at_1',  0):.3f}")
    print(f"  Doc Hit@3  : {o.get('doc_hit_at_3',  0):.3f}")
    print(f"  Doc Hit@5  : {o.get('doc_hit_at_5',  0):.3f}")
    print(f"  Doc Hit@10 : {o.get('doc_hit_at_10', 0):.3f}")
    print(f"  Doc MRR    : {o.get('doc_mrr',        0):.3f}")
    print()
    print("  ── RETRIEVAL LEVEL 2 — Answer in Context ─────────────")
    print("  (Does the retrieved chunk CONTAIN the gold answer?)")
    print(f"  Answer@1   : {o.get('answer_in_top_1',  0):.3f}")
    print(f"  Answer@3   : {o.get('answer_in_top_3',  0):.3f}")
    print(f"  Answer@5   : {o.get('answer_in_top_5',  0):.3f}")
    print(f"  Answer@10  : {o.get('answer_in_top_10', 0):.3f}")
    print(f"  Answer MRR : {o.get('answer_mrr',        0):.3f}")
    print()
    if o.get("rag_f1", 0) > 0:
        print("  ── ANSWER QUALITY ────────────────────────────────────")
        print(f"  {'Metric':<16} {'RAG':>8} {'LLM-only':>10}  {'Delta':>8}")
        print(f"  {'─'*16} {'─'*8} {'─'*10}  {'─'*8}")
        for key, label in [
            ("exact_match", "Exact Match"),
            ("f1",          "Token F1"),
            ("rouge_l",     "ROUGE-L"),
            ("bert_f1",     "BERTScore F1"),
        ]:
            r = o.get(f"rag_{key}", 0)
            l = o.get(f"llm_{key}", 0)
            print(f"  {label:<16} {r:>8.3f} {l:>10.3f}  {r-l:>+8.3f}")
        print()
        print("  ── HALLUCINATION ─────────────────────────────────────")
        print(f"  RAG Faithfulness    : {o.get('rag_faithfulness',   0):.3f}")
        print(f"  RAG Not-Found rate  : {o.get('rag_not_found_rate', 0):.3f}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Urban Metro RAG Evaluation")
    parser.add_argument("--top-k",           type=int,  default=5)
    parser.add_argument("--no-generation",   action="store_true")
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--limit",           type=int,  default=None)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_pipeline import HybridRetriever, load_index, Chunk

    chunk_data = json.loads(Path("chunk_cache.json").read_text())
    chunks = [
        Chunk(chunk_id=d["chunk_id"], document_id=d["document_id"],
              section=d["section"], text=d["text"],
              domain=d["domain"], organization=d["organization"], year=d["year"])
        for d in chunk_data
    ]
    collection = load_index()
    retriever  = HybridRetriever(collection, chunks)

    run_evaluation(
        retriever         = retriever,
        top_k             = args.top_k,
        run_generation    = not args.no_generation,
        run_faithfulness  = not args.no_faithfulness,
        limit             = args.limit,
    )