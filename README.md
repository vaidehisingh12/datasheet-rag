# Datasheet RAG

A question-answering system over electronic component datasheets (transistors,
op-amps and circular connectors), built so that every design choice is backed
by a measurement rather than a guess.

The chatbot part is the easy bit. The point of this project is the evaluation
around it: a hand-written set of test questions, retrieval and answer-quality
metrics with confidence intervals, a count of how and why answers go wrong, and
an ablation table comparing several setups on the same questions, including a
small quantised model running locally on a laptop CPU.

I picked datasheets because I can read them. If I can't tell whether an answer
about a MOSFET's gate threshold voltage is right, I can't write a test set for
it, and without a test set none of the numbers mean anything.

**Status:** week 1 - corpus download, project setup. Nothing to demo yet.

---

## Plan

| Week | Work |
|------|------|
| 1-2  | Corpus manifest, PDF parsing with Docling, layout-aware chunking, dense retrieval baseline |
| 3    | Golden set (about 60 answerable + 15 unanswerable questions), first evaluation run |
| 4    | BM25 + dense hybrid, cross-encoder reranker, ablation runs |
| 5    | Failure analysis and write-up |
| 6    | FastAPI service, Docker, CI regression gate |

### Ablation I'm aiming for

All rows scored on the same golden set.

| Setup | Faithfulness | Hit@5 | MRR (95% CI) | p95 latency | Peak RAM | Failure rate | Cost / query |
|-------|---|---|---|---|---|---|---|
| API model, no retrieval | | | | | | | |
| API model, dense retrieval only | | | | | | | |
| API model, hybrid + reranker | | | | | | | |
| Local 3B Q4_K_M, hybrid + reranker | | | | | | | |
| Local 3B fp16, hybrid + reranker | | | | | | | |
| Local 3B Q4_K_M, hybrid + reranker + query rewriting | | | | | | | |

Empty on purpose. Numbers go in when they've been measured.

### Hardware

Everything local runs on one laptop, and the latency numbers only make sense
with that stated:

- Intel Core i5-1335U (10 cores, 15 W), 16 GB RAM
- No discrete GPU, so all inference is on the CPU
- Windows 11, Python 3.13

---

## Corpus

**Tier 1** is a sample from the Zenodo dataset published alongside
*"Creating Hardware Component Knowledge Bases with Training Data Generation and
Multi-task Learning"* (ACM TECS):
[doi:10.5281/zenodo.4558344](https://doi.org/10.5281/zenodo.4558344).
I take roughly 30 datasheets from each of the three component classes, about 90
in total.

**Tier 2** will be 3-5 documents that look nothing like a datasheet (a standard
with clause numbering, a long manual, a vendor selection guide). Their job is
to break retrieval in ways a pile of near-identical datasheets never would.

**Licensing.** The Zenodo data is CC BY-NC-SA 4.0. This repo does **not**
include any PDFs or text extracted from them. It contains a manifest listing
which documents are used, and a script that downloads them from the original
source. The code is MIT licensed.

---

## Setup

The repo lives in OneDrive, but the data doesn't: the archives alone are about
3.7 GB. Large files go in `~/datasheet-rag-local` by default. Set
`DATASHEET_RAG_DATA` to put them somewhere else.

```bash
python -m venv ~/datasheet-rag-local/venv
~/datasheet-rag-local/venv/Scripts/activate      # Windows, Git Bash
pip install -r requirements.txt

python scripts/download_corpus.py                 # ~3.7 GB, resumable
```

## Layout

```
rag/
  settings.py              paths and model names
scripts/
  download_corpus.py       fetch + verify the Zenodo archives
corpus/
  manifest.csv             which documents are in the corpus (added in week 1)
```

---

## Progress log

Newest first. Any change to the original plan is marked **Plan change** and
says why.

### Week 1

- Set up the repo and the download script. Zenodo publishes MD5s for each
  archive, so the script checks them and resumes interrupted downloads.
- **Plan change: smaller corpus.** The first draft said 150-250 datasheets.
  Cut to about 90. I have to know every document well enough to write test
  questions without the system's help, and 200 isn't realistic for that. A
  smaller corpus I actually understand is worth more than a bigger one I don't.
- **Plan change: 3B local model instead of 7-8B.** I want to measure how much
  quality a 4-bit (Q4_K_M) model loses compared with full precision (fp16).
  A 7B model in fp16 needs about 14 GB, which doesn't fit in 16 GB of RAM next
  to everything else. A 3B model fits in both forms, so I can measure the
  difference myself instead of quoting someone else's number.
- **Plan change: merged ablation table.** My original notes had two versions
  of the table. I kept the dense-only row from the first, because without it
  there's no way to tell how much the reranker actually adds. I also kept the
  cost column, and added an fp16 row for the quantisation comparison.
- Skipped the `mouser_opamps` archive from the Zenodo record. It's a smaller
  scrape that overlaps with `opamp_dataset`, and I'm only sampling about 30
  op-amps anyway.
