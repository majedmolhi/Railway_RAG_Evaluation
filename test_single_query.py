"""
test_single_query.py
=====================
Ask a question and get a RAG answer with proper document citations.

Usage:
    python test_single_query.py --q "What is the standard track gage on SCRRA?"
    python test_single_query.py --q "..." --top-k 3
"""

import sys, json, csv, argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from rag_pipeline import HybridRetriever, load_index, Chunk

RAG_SYSTEM_PROMPT = """You are a precise railway safety and operations assistant.
Answer the question based ONLY on the provided context passages.
If the answer is not found in the context, state "Not found in context."
Be factual and concise. Do not add information beyond what is in the context."""


def load_chunks():
    data = json.loads(Path("chunk_cache.json").read_text(encoding="utf-8"))
    return [
        Chunk(chunk_id=d["chunk_id"], document_id=d["document_id"],
              section=d["section"], text=d["text"],
              domain=d["domain"], organization=d["organization"], year=d["year"])
        for d in data
    ]


def load_metadata():
    meta = {}
    with open("data/Metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["document_id"].strip()] = {
                "title"       : row["title"].strip(),
                "organization": row["organization"].strip(),
                "year"        : row["year"].strip(),
                "source_url"  : row["source_url"].strip(),
            }
    return meta


def make_client():
    import os
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(client, system, user):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def run(question: str, top_k: int = 5):
    print(f"\n{'='*65}")
    print(f"  QUESTION")
    print(f"{'='*65}")
    print(f"  {question}\n")

    chunks     = load_chunks()
    collection = load_index()
    retriever  = HybridRetriever(collection, chunks)
    metadata   = load_metadata()

    results = retriever.retrieve(question, top_k=top_k)

    context_block = "\n\n".join(
        f"[Passage {i+1}]\n{r['text']}" for i, r in enumerate(results)
    )
    rag_user = f"Context:\n{context_block}\n\nQuestion: {question}"

    client  = make_client()
    rag_ans = call_llm(client, RAG_SYSTEM_PROMPT, rag_user)

    print(f"{'='*65}")
    print(f"  ANSWER")
    print(f"{'='*65}")
    print(f"  {rag_ans}\n")

    # Deduplicate by document_id — show each document once
    seen = {}
    for r in results:
        doc_id = r["document_id"]
        if doc_id not in seen:
            seen[doc_id] = r["section"]

    print(f"{'='*65}")
    print(f"  REFERENCES")
    print(f"{'='*65}")
    for i, (doc_id, section) in enumerate(seen.items(), 1):
        m = metadata.get(doc_id, {})
        title = m.get("title", doc_id)
        org   = m.get("organization", "")
        year  = m.get("year", "")
        url   = m.get("source_url", "")
        print(f"\n  [{i}] {title}")
        print(f"      {org}, {year}")
        print(f"      Section : {section}")
        print(f"      URL     : {url}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q",     type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(args.q, args.top_k)