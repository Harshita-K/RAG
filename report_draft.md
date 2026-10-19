# RAG Assignment Report: Baseline vs. Re-Ranking — DRAFT NOTES

> **Draft notes, not the final report.** The brief requires the report to be in your own words and not
> AI-generated. Keep the tables (they are your measured results) and rewrite every "Points" list as a few
> sentences of your own. Delete this box before submitting.

**GitHub:** `<add repo link here>`

## Setup

- **Document:** GDPR, Regulation (EU) 2016/679, official 88-page PDF. Chosen for its exceptions ("shall not
  apply where…"), prohibitions and heavily repeated terms (consent, controller, processing).
- **Chunking:** 478 chunks of ≤1,000 characters (~220 tokens); never crossing an article boundary;
  article/recital heading prepended; footnotes and page headers removed.
- **Bi-Encoder:** `sentence-transformers/all-MiniLM-L6-v2`, FAISS `IndexFlatIP` on normalized vectors (= cosine).
- **Cross-Encoder:** `cross-encoder/ms-marco-MiniLM-L6-v2`.
- **LLM:** `Qwen2.5-3B-Instruct` run locally, greedy decoding, identical prompt for Answer 1 and Answer 2.
- **Hardware:** Apple M1, 16 GB. Retrieval models on CPU, LLM on GPU (MPS).

## 1. Architecture Comparison

|  | Bi-Encoder | Cross-Encoder |
|---|---|---|
| Input | Query and chunk encoded **separately** | Query and chunk fed in **together** as one sequence |
| Output | One vector each (384 dimensions here) | One relevance score per pair |
| Relevance measure | Cosine similarity between two vectors | Learned score from joint self-attention across all tokens |
| Pre-computable? | Yes, chunk vectors computed once, offline | No, every pair computed at query time |
| Cost per query | 1 model pass + vector search | 1 model pass **per candidate chunk** |
| Weakness | Compresses meaning into one vector: blurs negation ("not allowed" ≈ "allowed"), rewards shared keywords | Too slow to run over a whole corpus |

**Points (write in your own words):**

- The Bi-Encoder never sees the query and the chunk at the same time; it can only judge whether two texts are
  about similar things.
- The Cross-Encoder's attention lets each query word look at each chunk word, so it can judge whether the
  chunk actually *answers* the question. Example: matching "fired" with "shall not be dismissed" (Q6).
- This motivates the two-stage design: Bi-Encoder for recall (10 plausible candidates, fast),
  Cross-Encoder for precision (best 3).

## 2. Qualitative & Rank Shift Analysis

The chunk containing the answer reached the LLM in **1 of 8** queries with the baseline and **6 of 8** after
re-ranking.

### Rank changes

| Query | Failure type | Critical chunk | Rank shift |
|---|---|---|---|
| Q1 Does the GDPR apply to someone keeping a personal address book? | Keyword trap | Chunk 16, Recital 18 (household exemption) | #8 → #1 |
| Q2 Can a decision about me be made only by a computer algorithm? | Synonym ("algorithm" vs "automated processing") | Chunk 276, Art. 22(1) | #4 → #1 |
| Q3 When does a company not need to tell affected users about a data breach? | Exception | Chunk 305, Art. 34(3) | #5 → #3 |
| Q4 Is a controller exempt from informing people when it got their data from someone else and contacting them would take disproportionate effort? | Exception | Chunk 258, Art. 14(5) | #4 → #2 |
| Q5 Which public bodies are not regarded as recipients of personal data? | Negation | Chunk 217, Art. 4(9) | #10 → #2 |
| Q6 Can the data protection officer be fired for doing their job? | Negation / role limit | Chunk 320, Art. 38(3) | #6 → #1 |
| Q7 Is an employer prohibited from processing health data about employees? | Counter-example | Chunk 237, Art. 9(2)(b) | #4 → #9 |
| Q8 Is processing allowed to save someone's life? | Counter-example | Chunk 228, Art. 6(1)(d) | #2 → #5 |

### Answer 1 vs Answer 2

| Query | Answer 1 (baseline) | Answer 2 (re-ranked) | Verdict |
|---|---|---|---|
| Q1 | "we cannot determine whether the GDPR applies" | Not applicable to "purely personal or household activities" (Recital 18) | Baseline **omitted** the exemption; fixed |
| Q2 | Correct: another Art. 22 chunk was already in its top 3 | Correct, cites Art. 22(1) directly | Rank improved, answer already fine |
| Q3 | Uses "unlikely to result in a risk", the test for notifying the *authority* (Art. 33), not users | Gives the encryption exemption, Art. 34(3)(a) | Baseline **misstated the rule**; Answer 2 correct but omits 34(3)(b) and (c) |
| Q4 | Claims the controller "may charge a reasonable fee or refuse to act" (rules for excessive requests) | Answers from Recital 57 instead of the Art. 14(5) chunk it was given | Baseline **hallucinated**; Answer 2 still wrong: an LLM failure, not retrieval |
| Q5 | "Not found in the provided context" | Authorities receiving data for "a particular inquiry" are not recipients | Baseline **omitted**; fixed |
| Q6 | "no explicit mention of whether the DPO can be fired" | "shall not be dismissed or penalised … for performing his tasks" | Baseline **omitted** the clause; fixed |
| Q7 | "no explicit prohibitions" (wrong: Art. 9(1) prohibits by default) | Same claim; the employment exception left the context | Both **misstate**; re-ranking made the context worse |
| Q8 | Correct via Recital 46 | Correct via Recital 46 | Rank dropped but no harm: a backup passage carried the answer |

**Points (write in your own words):**

- Q1, Q5, Q6: the baseline ranked chunks that share words with the question (information duties, DPO tasks)
  above the chunk that answers it.
- Q3, Q4: the general rule ranks high; the exception sits in a separate chunk lower down (the "except when…"
  failure from the brief).
- Q4: correct retrieval does not guarantee a correct answer; a 3B model can still read the wrong passage.
- Q7, Q8: the Cross-Encoder (trained on MS MARCO web search) preferred explanatory recitals over terse
  article text. This is a likely explanation, not tested. Re-ranking is not uniformly better.

## 3. Latency Profiling

| Stage | Pipeline 1 (ms) | Pipeline 2 (ms) |
|---|---|---|
| Query embedding (Bi-Encoder) | 5.9 ± 0.4 | 5.9 ± 0.4 |
| Vector search (FAISS, top 10) | 0.04 ± 0.02 | 0.04 ± 0.02 |
| Cross-Encoder re-rank (10 pairs) | – | 117.6 ± 23.1 |
| **Retrieval total** | **5.9** | **123.5** |
| LLM generation (Qwen2.5-3B) | 38,271 ± 9,986 | 34,858 ± 6,937 |
| **End-to-end** | **38,277** | **34,982** |

Method: mean ± std of 40 runs (8 queries × 5 repeats, first cold run per query dropped); retrieval on CPU,
Apple M1. LLM time is one run per query per pipeline.

**Points (write in your own words):**

- Re-ranking makes retrieval ~20× slower (5.9 → 123.5 ms) but is only ~0.3% of end-to-end time, because
  generation takes ~35 s.
- The Cross-Encoder costs ~12 ms per (query, chunk) pair, so its cost grows with the number of candidates.
- Pipeline 2's slightly lower LLM time is noise from answer length, not an effect of re-ranking.

## 4. Scaling Question

| Corpus size (chunks) | Cross-Encoder over everything, per query | Two-stage (Bi-Encoder + re-rank 10), per query |
|---|---|---|
| 478 (this PDF, measured) | 7.5 s | 0.12 s |
| 10,000 | 2.6 min | 0.12 s |
| 1,000,000 | 4.4 hours | 0.12 s |
| 100,000,000 | 18.1 days | 0.12 s |

Measured 15.7 ms per pair on CPU, extrapolated linearly.

**Points (write in your own words):**

- A Cross-Encoder must run once per (query, chunk) pair at query time and nothing can be pre-computed, so its
  cost grows with the whole corpus on every query.
- Bi-Encoder cost is paid once: chunk vectors are computed offline; a query costs one encoding pass plus a
  vector search (0.04 ms here; ANN indexes like HNSW keep this fast at millions of vectors).
- Two-stage keeps most of the benefit: the Bi-Encoder's top 10 already contained the right chunk in all 8
  queries; only its ordering was wrong.
- A full Cross-Encoder pass over all 478 chunks did not rank better here: Q3 and Q4 landed at #4 and #3
  instead of #3 and #2, and Q7 at #22.
