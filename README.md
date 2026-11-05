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

**Status:** week 2. Corpus indexed (90 datasheets, 2,961 chunks), 85-question
golden set done, and retrieval is measured: hybrid search finds the right
page in the top 5 for 67 of 70 answerable questions. Next: answer generation
with citations, and scoring answers (faithfulness, refusals on the 15
unanswerable questions).

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
| API model, hybrid | | | | | | | |
| API model, hybrid + reranker | | | | | | | |
| Local 3B Q4_K_M, hybrid | | | | | | | |
| Local 3B fp16, hybrid | | | | | | | |
| Local 3B Q4_K_M, hybrid + query rewriting | | | | | | | |

Empty on purpose. Numbers go in when they've been measured. Retrieval on its
own has been measured, see below.

### Retrieval results so far

70 answerable golden-set questions, retrieval only (no answer generation).
Latency is per query on the laptop CPU, after the models are loaded.

| Setup | Hit@1 | Hit@5 | MRR | p50 | p95 |
|-------|-------|-------|-----|-----|-----|
| Dense, chunk text only | 0.371 | 0.686 | 0.503 | 29 ms | 34 ms |
| BM25, chunk text only | 0.500 | 0.743 | 0.605 | 7 ms | 9 ms |
| BM25 + document name | 0.586 | 0.857 | 0.700 | 6 ms | 8 ms |
| Dense + document name | 0.557 | 0.914 | 0.713 | 59 ms | 65 ms |
| **Hybrid (dense + BM25, RRF)** | 0.600 | **0.957** | 0.748 | 47 ms | 88 ms |
| Hybrid + MiniLM reranker | 0.714 | 0.914 | 0.804 | 1,839 ms | 3,105 ms |

What the paired comparisons say (bootstrap 95% CIs, McNemar on hit@5):

- **The biggest win was the least glamorous one.** Putting the datasheet's
  name in front of every chunk before embedding took dense from 0.686 to 0.914
  hit@5 (17 questions fixed, 1 broken, p < 0.001). Spec tables almost never
  name their own part. The part number is in the page title, which ends up
  in a different chunk.
- **Hybrid beats BM25 alone** (+0.100 hit@5, p = 0.039). Against dense with
  document names, its +0.043 is within noise (CI includes zero).
- **The reranker isn't worth it here.** On top of hybrid it improves MRR by
  0.055 and hit@1 a little, but hit@5 drops by 0.043, and neither change is
  significant (p = 0.45). It also takes p50 latency from 47 ms to 1.8 s. So
  hybrid without reranking is the default, and the reranker stays available
  as an option.
- Before the document name was added, BM25 alone (0.857) beat the hybrid
  (0.757). Checking why led to the fix above: BM25 had the part number and
  dense didn't, so it wasn't a fair fight.

Run files for every row are in `eval/results/`, so any of these can be
re-checked with `eval_retrieval.py compare`.

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
  retrieval/bm25.py        keyword search, tokenised to match part numbers
  retrieval/hybrid.py      rank fusion of dense + BM25, optional reranker
  eval/metrics.py          hit@k, MRR, nDCG, bootstrap CIs, McNemar
eval/
  golden_template.csv      format + three worked examples
  golden.csv               the question set: 70 answerable, 15 unanswerable
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
python scripts/eval_retrieval.py run hybrid --method hybrid   # dense | bm25 | hybrid | rerank
python scripts/eval_retrieval.py compare dense hybrid         # paired deltas + McNemar
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
- What gets embedded is `document | section | caption` followed by the
  chunk text. A table full of numbers usually doesn't name its own part,
  and without the document name dense search can't tell one transistor's
  ratings table from another's.

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
  *Later:* the real cause was that the ratings table chunk never contains
  "BC337" at all. Adding the document name to each chunk fixed most of it
  (see "Retrieval results so far").
- **Drawings aren't datasheets.** Three of the sampled connectors produced no
  chunks at all. Two Amphenol files are mechanical drawings: Docling read
  each page as one big picture, and the only real text was the title block
  (tolerances, signature boxes, a copyright notice). One TE file is a scanned
  drawing with no text layer. OCR could read it but produced things like
  "VY_ON" for NYLON. I excluded all three and drew replacements (see the
  progress log).
- **Some chunks were too long for the embedder.** 10 of 2,867 chunks went over
  bge-small's 512-token limit, so their ends were silently cut off. Most were
  revision-history tables, which nobody will ask about. One was a
  700-token paragraph of PCB layout advice in a TI op-amp datasheet, which the
  chunker had kept whole. Long paragraphs are now split at sentence ends.
  Nine table chunks are still over the limit. Each is a table where a single
  row is bigger than the budget: eight TI revision-history tables (Docling
  put the whole change list in one row) and a transistor cross-reference
  list. Splitting inside a row would take more work than those tables are
  worth.
- **Text inside figures was being dropped.** A golden-set question about a
  CUI connector note ("SD-130 has solid solder pins") could never be
  answered, because the note wasn't in any chunk. Docling had kept it, but
  attached it to a drawing, and the chunker's default walk over the
  document skips everything inside pictures. That was 18.6% of all
  extracted text, in 76 of 90 documents. Most of it is graph axis labels
  ("100k", "f - Frequency - Hz"), so including all of it would add noise.
  The chunker now keeps figure captions, footnotes, list items, and any
  figure text that is a full sentence. That added 38 chunks and fixed the
  question without breaking any others.
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

### Week 2

- Added BM25 (`rank-bm25`), reciprocal rank fusion and a cross-encoder
  reranker (`ms-marco-MiniLM-L-6-v2`), and scored all of them. Full table in
  "Retrieval results so far".
- **Plan change: every chunk now carries its document name.** It started as
  a fairness check: BM25 had the document ID in its searchable text and
  dense didn't. Giving dense the same information was the largest
  improvement of anything tried so far (+0.229 hit@5), so it's now part of
  every chunk's embedded text, not just an experiment.
- **Plan change: no reranker by default.** The original plan assumed hybrid +
  reranker as the production setup. Measured on this corpus, the reranker
  doesn't significantly change hit@5 or MRR and costs about 40x the latency.
  It also makes a new kind of mistake: it dropped the BC141-16 package
  question entirely in favour of op-amp chunks. That's probably because a
  web-search-trained model reads datasheet tables poorly. A bigger reranker
  (e.g. bge-reranker-base) might change this, and it's on the list to try. The
  ablation table now has hybrid as the base and the reranker as its own row.
- Remaining hybrid misses (3 of 70):
  - **Sanken 2SC4886:** the one-page layout separates symbols from values, so
    "IC" and "14 A" never end up in the same piece of text. A parsing
    problem.
  - **OP07D supply current and GBW:** the ±5 V and ±15 V spec tables are
    nearly identical. Retrieval finds the datasheet but picks the wrong
    supply condition.
- Golden set checked and committed. Dense baseline recorded before any of
  the changes above (hit@5 0.686, MRR 0.503).
- Drafted the golden set: 85 questions, 70 answerable and 15 unanswerable,
  roughly a third per component class (59 lookups, 7 comparisons, 4 multi-hop,
  15 unanswerable). Answers and page numbers come from the PDFs' own text,
  not from what the retriever returns. It's being checked against each PDF
  before it's committed, so no scores are published yet.
- A trial scoring run turned up a chunker bug before any retrieval tuning:
  text inside figures was being skipped (see "Things I've noticed so far").
  Checked the other misses as well. Every page number in the golden set
  matches where that text actually ended up in the index, so the rest are
  real retrieval misses, not labelling mistakes.

### Week 1

- Index rebuilt with the replacements: 90 documents, 2,923 chunks (1,004
  tables), every document contributing at least a few chunks. The parser
  crashed once on a replacement (Windows access violation), and the worker
  restart picked it up on the second try, as intended.
- Full parse finished: 90 documents, 2,867 chunks, 976 of them tables.
  Parse times varied a lot by class: connectors took 2-80 s each (mostly one
  or two pages), op-amps 45-460 s (long TI datasheets full of graphs and
  tables).
- **Plan change: three connector drawings swapped out.** Checking the output
  turned up three documents with nothing to retrieve (see "Drawings aren't
  datasheets"). Rather than keep them and inflate the failure count with
  documents nobody could write a question about, I excluded them in
  `build_manifest.py` with the reason written next to each one. Replacements
  come from a separately seeded draw and must have at least 2,000 characters
  of real text. The other 87 documents didn't change, so they didn't need
  re-parsing.
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
