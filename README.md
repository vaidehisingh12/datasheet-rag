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

**Status:** end of week 1. Corpus picked (90 datasheets), the full parse is
running, and retrieval scoring is ready. Writing the golden set comes next. No
answer generation yet.

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
python scripts/build_manifest.py                  # pick 30 per class, extract them
python scripts/ingest.py                          # parse, chunk, index (~2 min/doc on CPU, resumable)
python scripts/ask.py "max collector current of the BC337"
pytest
```

## Layout

```
rag/
  settings.py              paths and model names
  ingest/parse.py          PDF -> structured document (Docling), cached as JSON
  ingest/chunk.py          section-aware chunking, tables kept whole
  retrieval/dense.py       bge-small embeddings + local Qdrant
  eval/metrics.py          hit@k, MRR, nDCG, bootstrap CIs, McNemar
eval/
  golden_template.csv      format + three worked examples
  golden.csv               the real question set (not started yet)
  results/                 per-question results of each scored run
scripts/
  eval_retrieval.py        score retrieval against the golden set, compare runs
  download_corpus.py       fetch + verify the Zenodo archives
  build_manifest.py        seeded sample of datasheets -> corpus/manifest.csv
  ingest.py                build the index from the manifest
  ask.py                   print what retrieval returns for a question
corpus/
  manifest.csv             the 90 documents in the corpus, with source and licence
tests/
```

## Golden set

Hand-written questions live in `eval/golden.csv`, one row per question.
`eval/golden_template.csv` has three worked examples showing the format:

| Column | Meaning |
|--------|---------|
| `answerable` | `yes` or `no`. Unanswerable questions are scored on whether the system refuses, not on retrieval. |
| `answer` | The correct answer, written from the datasheet, not from the system's output. |
| `doc_id`, `pages` | Where the answer is (pages separated by `;`). A retrieved chunk counts as a hit if it's from that document and covers one of those pages. |
| `question_type` | `lookup`, `comparison`, `multi-hop` or `unanswerable`, so results can be broken down later. |
| `notes` | What makes the question tricky. Useful when sorting failures. |

```bash
python scripts/eval_retrieval.py run dense              # score, save eval/results/dense.json
python scripts/eval_retrieval.py compare dense hybrid   # paired deltas + McNemar
```

Matching on (document, page) rather than chunk IDs means the golden set
doesn't have to be rewritten every time the chunker changes.

How I'm writing it:

- Questions come from reading the PDFs, never from looking at what the system
  retrieves. Otherwise the test set ends up measuring things the system is
  already good at.
- About 20 answerable and 5 unanswerable questions per component class, so no
  class dominates the averages.
- A mix of types: mostly lookups, some that need a specific variant or test
  condition (BC337-25 vs BC337-40, hFE at 100 mA vs 300 mA), some comparisons
  across two datasheets, and a few that need information from two places in
  one document.
- Unanswerable questions should be plausible: right component, wrong package,
  a parameter the datasheet doesn't specify, or a part number that isn't in
  the corpus. They test whether the system refuses rather than confidently
  quoting the nearest number.

## How chunking works

Fixed-size token windows are the usual default, and they're bad for
datasheets. A window boundary in the middle of an electrical characteristics
table leaves rows of numbers with no column headers, so you can't tell which
number is the minimum, which is the maximum, or what the unit is. Instead:

- Docling parses each PDF into headings, paragraphs and tables with their
  cells, in reading order. OCR is off because these PDFs are born-digital, and
  turning it on roughly triples the parse time.
- Text is grouped by section and never crosses a heading.
- Each table is its own chunk. Long tables are split by rows and the header
  row is repeated in every piece.
- Page headers and footers are dropped. They repeat on every page and add
  nothing.
- What gets embedded is `section | caption` followed by the chunk text, so a
  table full of numbers still carries a hint of what it's about.

## Things I've noticed so far

Notes for the failure analysis. These get counted properly once the golden set
exists.

- **Heading swallowed by a table.** On a Central Semiconductor datasheet the
  "MAXIMUM RATINGS" heading is inside the table's first header cell, so Docling
  never sees it as a heading, and the table gets filed under the previous
  section ("FEATURES"). Retrieval still found it this time, but section
  filtering would miss it.
- **Parser segfault.** The first full ingest died with a segfault on document
  8 after parsing fine for a while. The same PDF parsed without trouble on its
  own, so it's something going wrong inside Docling's native code over a long
  run, not a bad file. Parsing now runs in a worker process that gets
  restarted if it dies. A document that crashes it twice is skipped and
  listed in `ingest_failures.json`.
- **Part numbers get lost in dense search.** With 3 documents indexed, "max
  VCEO of the BC337" put the right table at rank 1. With 10 documents
  indexed, it dropped to rank 4: every transistor has a near-identical
  absolute maximum ratings table, and the embedding model doesn't weight the
  exact string "BC337". This is the main reason to add BM25 keyword search.
- **Boilerplate tables.** Some vendors finish with a page-long trademark table
  that becomes chunks full of brand names. They haven't caused a wrong hit
  yet, but they're worth filtering if they start to.
- **Merged cells repeated.** Spanned cells come out once per column they span
  (`Parameter | Parameter`). Harmless for retrieval, but it makes tables
  harder to read in answers.

---

## Progress log

Newest first. Any change to the original plan is marked **Plan change** and
says why.

### Week 1

- The second full parse was killed partway through (18/90) when the laptop
  ran low on memory. Each Docling worker uses over 1 GB on top of everything
  else running. Because parsed files are cached, the rerun picked up at
  document 19 instead of starting again. Running it in its own terminal with
  other heavy apps closed keeps it going.
- Added the golden set template and `eval_retrieval.py`. A quick check on 10
  indexed documents showed dense search dropping the right BC337 table from
  rank 1 to rank 4 once more transistors were in the index (see "Things I've
  noticed so far").
- The first full parse segfaulted inside Docling on document 8. Moved parsing
  into a restartable worker process (details in "Things I've noticed so far").
- Built the corpus: 30 datasheets each of transistors, op-amps and circular
  connectors, sampled from the dataset's dev/test splits with a fixed seed.
  I skipped PDFs over 3 MB (mostly 100-page family catalogues).
- Parsing, chunking and dense retrieval run end to end. First query ("max
  collector-emitter voltage of the BC337") brings back the right absolute
  maximum ratings table at rank 1.
- Parsing is the bottleneck: about 2 minutes per datasheet on this CPU, so
  about 3 hours for the corpus. It only runs once because output is cached.
- **Plan change: Qdrant in embedded mode, not pgvector.** Docker isn't on this
  machine and I don't need it until week 6. Qdrant's client can run as a local
  folder with metadata filtering, and switching to a Qdrant server later is a
  one-line change. pgvector would have meant installing Postgres now.
- **Plan change: stayed on Python 3.13.** I was going to use a 3.12 virtual
  environment in case libraries lagged behind, but everything (Docling, torch,
  sentence-transformers, qdrant-client) installs and runs on 3.13.
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
