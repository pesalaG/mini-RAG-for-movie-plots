# mini-rag — Movie-Plot RAG

A small retrieval-augmented generation (RAG) demo that indexes movie plot summaries and answers user queries by retrieving relevant plot chunks from a FAISS vector store and generating concise, grounded answers with a Hugging Face LLM.

---

## Repository layout

```
mini-rag/
│── data/
│   └── wiki_movie_plots_deduped.csv
│
│── rag/
│   ├── __init__.py
│   ├── load_data.py
│   ├── build_index.py
│   ├── rag_pipeline.py
│
│── app.py
│── requirements.txt
│── README.md
│── .env
```

## Prerequisites

* Python 3.10+ recommended
* pip
* (Optional) GPU with CUDA to accelerate embeddings/LLM
* A Hugging Face account and a `HUGGINGFACEHUB_API_TOKEN` (create one at your Hugging Face account settings)

---

## Quick setup

1. Clone the repo and open a terminal in the project root.

2. Create and activate a virtual environment:

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Upgrade packaging tools and install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


4. Add your Hugging Face token to a `.env` file in the project root:

```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

The code (rag/rag_pipeline.py) reads this value via `python-dotenv`.

---

## Build the FAISS index

The `rag/build_index.py` script reads `data/wiki_movie_plots_deduped.csv`, chunks plots, embeds them, and writes a FAISS index to `faiss_index/` by default.

Run either:

```bash
python -m rag.build_index
# or
python rag/build_index.py
```

Notes:

* The script uses `RecursiveCharacterTextSplitter` with `chunk_size` and `chunk_overlap` parameters (default in the script). Adjust those in `build_index.py` to tune retrieval granularity.
* Embeddings are computed using `HuggingFaceEmbeddings("BAAI/bge-base-en-v1.5")` by default; swap model names if needed.

---

## Run the demo app

Start the simple interactive CLI:

```bash
python app.py
```

Type a question and press Enter. Type `exit` to quit.

Output includes a JSON-like response with `answer`, `contexts`, and `reasoning`, followed by a list of source titles and similarity scores.

---

## Files explained

* `rag/load_data.py` — CSV loading and conversion to record list.
* `rag/build_index.py` — chunking, embedding, and FAISS index build/save routine.
* `rag/rag_pipeline.py` — loads FAISS, sets up retriever, prompt template, and Hugging Face LLM pipeline; returns a `RetrievalQA` chain.
* `app.py` — small interactive loop to run queries against the pipeline.

---


## Troubleshooting

* `HUGGINGFACEHUB_API_TOKEN` missing: check `.env` location and that you restarted the shell/IDE.
* FAISS import or build errors: install `faiss-cpu` for CPU or `faiss-gpu` for GPU, and ensure your Python and wheel specifications match your platform.
* If generation returns empty or overly long answers: reduce `max_new_tokens` or adjust the prompt and `temperature`.



Tell me which you'd like next.
