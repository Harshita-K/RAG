# RAG: Baseline Retrieval vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked copy of the **EU GDPR** (Regulation (EU) 2016/679, 88-page
official PDF in `data/gdpr.pdf`):

| | Pipeline 1 (baseline) | Pipeline 2 (re-ranking) |
|---|---|---|
| Stage 1 | `all-MiniLM-L6-v2` Bi-Encoder → FAISS `IndexFlatIP` (cosine) → top 10 | same top 10 |
| Stage 2 | take ranks 1–3 | `cross-encoder/ms-marco-MiniLM-L6-v2` scores all 10 pairs → new top 3 |
| Generation | `Qwen2.5-3B-Instruct` (local) with a fixed prompt, greedy decoding | identical prompt and settings |

Everything is in one notebook: **[`rag_reranking.ipynb`](rag_reranking.ipynb)** (executed, outputs included).
Result tables are also saved in [`results/`](results/).

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook rag_reranking.ipynb
```

- Models download from Hugging Face on first run (~6 GB for the 3B LLM). Set `LOCAL_LLM_ID=Qwen/Qwen2.5-1.5B-Instruct`
  for a smaller model.
- If `GEMINI_API_KEY` is set, the notebook uses `gemini-2.5-flash` instead of the local LLM.
- Tested on macOS (Apple M1, 16 GB), Python 3.13, torch 2.7. The LLM runs on MPS/CUDA if available, otherwise CPU.

## Results at a glance

Rank of the passage that actually answers each stress-test query (only ranks 1–3 reach the LLM):

| Query | Failure type | Baseline → Re-ranked |
|---|---|---|
| Q1 Does the GDPR apply to someone keeping a personal address book? | Keyword trap | #8 → #1 |
| Q2 Can a decision about me be made only by a computer algorithm? | Synonym | #4 → #1 |
| Q3 When does a company not need to tell affected users about a data breach? | Exception | #5 → #3 |
| Q4 Is a controller exempt from informing people … disproportionate effort? | Exception | #4 → #2 |
| Q5 Which public bodies are not regarded as recipients of personal data? | Negation | #10 → #2 |
| Q6 Can the data protection officer be fired for doing their job? | Negation / role limit | #6 → #1 |
| Q7 Is an employer prohibited from processing health data about employees? | Counter-example | #4 → #9 |
| Q8 Is processing allowed to save someone's life? | Counter-example | #2 → #5 |

Latency (CPU for retrieval, mean of 40 runs): baseline retrieval **5.9 ms**, re-ranking adds **118 ms** for 10 pairs;
LLM generation is ~35 s, so re-ranking is ~0.3 % of end-to-end time. Scoring all 478 chunks with the
Cross-Encoder takes 7.5 s per query, which extrapolates to ~4.4 hours per query at 1 M chunks.

See the notebook for full top-10 tables, both LLM answers per query, and the scaling analysis.
