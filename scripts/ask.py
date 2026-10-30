"""
Show what retrieval returns for a question. No answer generation yet - this
is for checking the retriever by eye while building the golden set.

    python scripts/ask.py "max collector current of the BC337"
    python scripts/ask.py "input offset voltage" --class opamp -k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.retrieval import dense


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--class", dest="component_class", default=None,
                        choices=["transistor", "opamp", "connector"])
    args = parser.parse_args()

    # Datasheets are full of θ, µ, Ω and °. The Windows console defaults to
    # cp1252 and crashes on them.
    sys.stdout.reconfigure(encoding="utf-8")

    for rank, hit in enumerate(dense.search(args.question, args.k, args.component_class), 1):
        where = f"{hit['doc_id']} p.{','.join(map(str, hit['pages'])) or '?'}"
        print(f"\n#{rank}  {hit['score']:.3f}  {where}  [{hit['kind']}] {hit['section']}")
        if hit["caption"]:
            print(f"    caption: {hit['caption']}")
        print("    " + hit["text"][:400].replace("\n", "\n    "))

    dense.close()


if __name__ == "__main__":
    main()
