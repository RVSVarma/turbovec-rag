#!/usr/bin/env python3
"""
Standalone script to (re)build the turbovec index from your documents.

Usage:
    python scripts/build_index.py                  # index data/sample_docs/
    python scripts/build_index.py --docs path/to/  # custom docs folder
    python scripts/build_index.py --rebuild        # force rebuild even if index exists
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_pipeline import get_vectorstore, INDEX_PATH


def main():
    parser = argparse.ArgumentParser(description="Build turbovec index")

    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--rebuild", action="store_true")

    args = parser.parse_args()

    if args.rebuild:
        print("🔄 Rebuilding index...")
        get_vectorstore(force_rebuild=True)
    else:
        print("📦 Building/loading index...")
        get_vectorstore(force_rebuild=False)

    print("✅ Done")


if __name__ == "__main__":
    main()