#!/usr/bin/env python3
"""
Run a set of test questions against the RAG pipeline and print results.

Usage:
    python scripts/eval.py
    python scripts/eval.py --questions questions.txt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline import get_vectorstore, build_rag_graph, ask

DEFAULT_QUESTIONS = [
    "What are the pet rules in the lease?",
    "What is the notice period for ending the tenancy?",
    "Who is responsible for maintenance repairs?",
    "What are the quiet hours in this building?",
    "How do I submit a maintenance request?",
]


def main():
    parser = argparse.ArgumentParser(description="Batch eval for turbovec RAG")
    parser.add_argument("--questions", help="Path to file with one question per line")
    args = parser.parse_args()

    if args.questions:
        questions = Path(args.questions).read_text().splitlines()
        questions = [q.strip() for q in questions if q.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    print("⚙️  Loading pipeline …\n")
    vectorstore = get_vectorstore()
    rag = build_rag_graph(vectorstore)

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] ❓ {q}")
        t0 = time.perf_counter()
        answer = ask(q, rag)
        elapsed = time.perf_counter() - t0
        print(f"💬 {answer}")
        print(f"⏱  {elapsed:.2f}s\n{'─'*60}\n")


if __name__ == "__main__":
    main()