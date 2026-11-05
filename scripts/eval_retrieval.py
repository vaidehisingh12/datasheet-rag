"""
Score retrieval against the golden set, or compare two scored runs.

    python scripts/eval_retrieval.py run dense                          # dense is the default method
    python scripts/eval_retrieval.py run hybrid --method hybrid
    python scripts/eval_retrieval.py compare dense hybrid               # paired comparison of two saved runs

Methods: dense, bm25, hybrid (dense + BM25 fused), rerank (hybrid, then a
cross-encoder over the top 30).

Only answerable questions count towards hit rate / MRR / nDCG. Unanswerable
ones have no right document to find, and including them would cap every
metric below 100% for reasons that have nothing to do with retrieval. They
get scored separately later, on whether the system refuses to answer.

Each run saves per-question results to eval/results/<name>.json so later
setups can be compared question by question, which is what the paired
statistics need.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.eval.metrics import (
    bootstrap_ci,
    first_relevant_rank,
    hit_at_k,
    mcnemar_p,
    ndcg_at_k,
    paired_bootstrap_delta,
    reciprocal_rank,
)

GOLDEN = ROOT / "eval" / "golden.csv"
TEMPLATE = ROOT / "eval" / "golden_template.csv"
RESULTS = ROOT / "eval" / "results"
DEPTH = 10  # retrieve this many so ranks beyond k still show up in the results


def load_golden() -> list[dict]:
    path = GOLDEN
    if not path.exists():
        print(f"no {GOLDEN.name} yet, using the template's example questions\n")
        path = TEMPLATE
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["answerable"] = r["answerable"].strip().lower() == "yes"
        r["pages"] = [int(p) for p in r["pages"].replace(",", ";").split(";") if p.strip()]
    return rows


def get_searcher(method: str):
    # Imported here because loading the models is slow and `compare` doesn't need them.
    from rag.retrieval import bm25, dense, hybrid
    return {
        "dense": dense.search,
        "bm25": bm25.search,
        "hybrid": hybrid.search,
        "rerank": hybrid.search_reranked,
    }[method]


def run(name: str, method: str, use_class_filter: bool) -> None:
    from rag.retrieval import dense

    search = get_searcher(method)
    golden = [g for g in load_golden() if g["answerable"]]

    # One throwaway query first, so model loading doesn't land in the first
    # question's latency and skew p95.
    search("warm up", k=DEPTH)

    per_question = []
    for g in golden:
        cls = g["component_class"] if use_class_filter else None
        start = time.perf_counter()
        hits = search(g["question"], k=DEPTH, component_class=cls)
        elapsed_ms = (time.perf_counter() - start) * 1000
        rank = first_relevant_rank(hits, g)
        per_question.append({
            "id": g["id"],
            "question": g["question"],
            "rank": rank,
            "hit@1": hit_at_k(rank, 1),
            "hit@3": hit_at_k(rank, 3),
            "hit@5": hit_at_k(rank, 5),
            "rr": reciprocal_rank(rank),
            "ndcg@5": ndcg_at_k(hits, g, 5),
            "ms": round(elapsed_ms, 1),
            "top5": [f"{h['doc_id']} p.{h['pages']}" for h in hits[:5]],
        })
    dense.close()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(per_question, f, indent=2)

    print(f"{name}: {len(per_question)} answerable questions\n")
    for metric in ("hit@1", "hit@3", "hit@5", "rr", "ndcg@5"):
        mean, low, high = bootstrap_ci([q[metric] for q in per_question])
        label = "MRR" if metric == "rr" else metric
        print(f"  {label:7} {mean:.3f}   95% CI [{low:.3f}, {high:.3f}]")

    times = sorted(q["ms"] for q in per_question)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    print(f"\n  latency  p50 {statistics.median(times):.0f} ms   p95 {p95:.0f} ms")

    misses = [q for q in per_question if q["rank"] is None or q["rank"] > 5]
    if misses:
        print(f"\nmissed at k=5 ({len(misses)}):")
        for q in misses:
            print(f"  {q['id']}: {q['question']}")


def compare(a_name: str, b_name: str) -> None:
    a = {q["id"]: q for q in json.load(open(RESULTS / f"{a_name}.json", encoding="utf-8"))}
    b = {q["id"]: q for q in json.load(open(RESULTS / f"{b_name}.json", encoding="utf-8"))}
    ids = sorted(a.keys() & b.keys())
    if len(ids) < len(a) or len(ids) < len(b):
        print(f"warning: only {len(ids)} questions are in both runs\n")

    print(f"{b_name} minus {a_name}, {len(ids)} paired questions\n")
    for metric in ("hit@5", "rr", "ndcg@5"):
        delta, low, high = paired_bootstrap_delta([a[i][metric] for i in ids],
                                                  [b[i][metric] for i in ids])
        verdict = "" if low > 0 or high < 0 else "  (CI includes zero)"
        label = "MRR" if metric == "rr" else metric
        print(f"  {label:7} {delta:+.3f}   95% CI [{low:+.3f}, {high:+.3f}]{verdict}")

    p = mcnemar_p([a[i]["hit@5"] for i in ids], [b[i]["hit@5"] for i in ids])
    fixed = sum(1 for i in ids if b[i]["hit@5"] and not a[i]["hit@5"])
    broke = sum(1 for i in ids if a[i]["hit@5"] and not b[i]["hit@5"])
    print(f"\n  hit@5: {fixed} questions fixed, {broke} broken, McNemar p = {p:.3f}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--method", default="dense", choices=["dense", "bm25", "hybrid", "rerank"])
    r.add_argument("--class-filter", action="store_true",
                   help="restrict search to the question's component class")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args()

    if args.cmd == "run":
        run(args.name, args.method, args.class_filter)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
