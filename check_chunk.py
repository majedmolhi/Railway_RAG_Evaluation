"""
check_chunk.py
===============
Search for a keyword inside chunk_cache.json and show matching chunks.

Usage:
    python check_chunk.py "keyword"
    python check_chunk.py "keyword" --show-neighbors   # show chunk before and after
    python check_chunk.py "c0349"                      # search by chunk_id
"""

import json, sys, re
from pathlib import Path

keyword = sys.argv[1] if len(sys.argv) > 1 else ""
show_neighbors = "--show-neighbors" in sys.argv

chunks = json.loads(Path("chunk_cache.json").read_text(encoding="utf-8"))

# Index by chunk_id for neighbor lookup
chunk_list = chunks
chunk_by_id = {c["chunk_id"]: i for i, c in enumerate(chunks)}

results = [
    (i, c) for i, c in enumerate(chunks)
    if keyword.lower() in c["chunk_id"].lower()
    or keyword.lower() in c["section"].lower()
    or keyword.lower() in c["text"].lower()
]

print(f"\nSearching for: '{keyword}'")
print(f"Found: {len(results)} chunk(s)\n")
print("=" * 65)

for idx, c in results:
    print(f"chunk_id : {c['chunk_id']}")
    print(f"section  : {c['section']}")
    print(f"words    : {len(c['text'].split())}")
    print(f"text     :\n{c['text']}")
    print("-" * 65)

    if show_neighbors:
        # Show next chunk
        if idx + 1 < len(chunk_list):
            nxt = chunk_list[idx + 1]
            print(f"  ▼ NEXT chunk: {nxt['chunk_id']} | {nxt['section']}")
            print(f"  {nxt['text'][:300]}")
            print("-" * 65)