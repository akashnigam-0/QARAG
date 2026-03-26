# QA RAG Assistant

A local, free, AI-powered debugging assistant for SDETs. Ask plain-English questions about your Selenium logs, job failure logs, and Jira history — and get specific, cited answers in seconds.

No cloud API needed. Runs entirely on your machine.

---

## What it does

Instead of manually reading through hundreds of lines of logs after a test failure, you type a question:

```
Ask: why did the checkout test fail?
Ask: jira: has this error appeared before?
Ask: job: what caused the pipeline to crash?
```

The tool finds the most relevant sections of your actual log files and generates a structured answer with the exact evidence quoted and cited.

---

## How it works

```
Your log files
      ↓
Split into chunks → Convert to vectors → Store in ChromaDB
                                               ↓
Your question → Convert to vector → Find closest chunks
                                               ↓
                              Send chunks + question to Ollama
                                               ↓
                         ROOT CAUSE + EVIDENCE + DEBUGGING STEPS
```

This technique is called RAG — Retrieval-Augmented Generation. The AI reads your actual files before answering, not its training data.

---

## Tech stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Embedding model | `all-MiniLM-L6-v2` | Converts text to vectors locally |
| Vector database | ChromaDB | Stores and searches embedded chunks |
| LLM | Ollama `llama3.2` | Generates answers from retrieved context |
| Text splitting | LangChain | Cuts log files into searchable chunks |

---

## Installation

**1. Install Python 3.11+**
Download from https://python.org — check "Add to PATH" during install.

**2. Install Ollama**
Download from https://ollama.com and install. Then pull the model:

```powershell
ollama pull llama3.2
```

**3. Install Python packages**

```powershell
pip install chromadb sentence-transformers langchain-text-splitters ollama
```

---

## Project structure

```
your-project/
├── qa_rag_tool.py       ← main script
├── my_files/            ← drop YOUR files here
│   ├── jira.txt         ← Jira export (plain text or CSV)
│   ├── selenium.log     ← Selenium CI execution log
│   └── job_failures.log ← any job / pipeline failure log
├── qa_vectordb/         ← auto-created by ChromaDB (do not edit)
└── qa_session.json      ← auto-created, saves your Q&A session
```

---

## Adding your files

Open `qa_rag_tool.py` and edit the `MY_FILES` list at the top — this is the only section you need to change:

```python
MY_FILES = [
    ("my_files/jira.txt",         "jira"),
    ("my_files/selenium.log",     "selenium"),
    ("my_files/job_failures.log", "job"),

    # Add more files by adding lines here:
    # ("my_files/dataproc.log",   "dataproc"),
    # ("my_files/k8s_errors.log", "kubernetes"),
    # ("my_files/runbook.txt",    "runbook"),
]
```

The second value (`"jira"`, `"selenium"`, `"job"`) is the label you use when filtering questions. You can name it anything you want.

**To add a Jira export:** Go to your Jira project → Issues → Export → Export as CSV or plain text. Save as `my_files/jira.txt`.

---

## Running the tool

**First run** (ingests your files):

```powershell
cd D:\rag
python qa_rag_tool.py
```

**After adding new files** (re-ingests everything):

```powershell
Remove-Item -Recurse -Force qa_vectordb
python qa_rag_tool.py
```

---

## Asking questions

Once the tool is running you get a live prompt. You can ask in two ways:

**Search all files at once:**
```
Ask: why did the checkout test fail?
Ask: what errors appeared after the last deployment?
Ask: which tests are failing and what is the common cause?
```

**Search one specific file only** (prefix with the label):
```
Ask: jira: has this StaleElement error appeared before?
Ask: selenium: what was the exact exception on AddToCartTest?
Ask: job: why did the pipeline fail on stage 3?
```

**Other commands:**
```
list    → shows all loaded files, labels, and file sizes
quit    → exits and saves the session to qa_session.json
```

---

## Example output

```
Ask: why did the checkout test fail?

ROOT CAUSE:
  The AddToCartTest failed because React re-renders the cart component
  on every add-to-cart action, invalidating the previously located
  WebElement reference.

EVIDENCE:
  "WARN Page reloading due to cart update animation" followed by
  "ERROR stale element reference: element not attached to page document"
  [selenium | selenium_log | score:0.89]

DEBUGGING STEPS:
  1. Replace driver.findElement(...).click() with WebDriverWait using
     ExpectedConditions.refreshed(By.cssSelector(...))
  2. Wrap the click action in a retry loop that re-locates the element
     on each attempt
  3. Check if React batches state updates to reduce re-renders

RELATED ISSUES:
  BUG-101 (Resolved 2024-11-10) — identical StaleElement issue in cart
  tests, fixed in CartPage.java line 87 [jira]
```

---

## Tuning for better answers

If answers are poor, adjust these settings at the top of `qa_rag_tool.py`:

| Setting | Default | When to change |
|---------|---------|----------------|
| `CHUNK_SIZE` | `256` | Lower to 128 if answers miss specific lines |
| `CHUNK_OVERLAP` | `32` | Raise to 64 if related log lines are split apart |
| `TOP_K` | `8` | Raise to 12 if answers are missing context |

After changing any setting, delete `qa_vectordb/` and re-run to re-ingest.

---

## Evaluation

After getting an answer, the quality can be measured across 4 metrics using Ollama as the judge. Add this to the prompt loop in `qa_rag_tool.py` — or just review the `qa_session.json` file saved after each session to see which questions had high or low similarity scores.

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Faithfulness | Answer grounded in your logs only | > 0.85 |
| Answer relevancy | Answer addresses the question asked | > 0.80 |
| Context precision | Retrieved chunks were actually useful | > 0.75 |
| Context recall | All needed facts were retrieved | > 0.80 |

---

## Troubleshooting

**`ollama` not recognized**
Ollama is not installed or not in PATH. Re-install from https://ollama.com and restart PowerShell.

**`ModuleNotFoundError`**
Run `pip install chromadb sentence-transformers langchain-text-splitters ollama` again.

**`[SKIP] File not found`**
The file path in `MY_FILES` does not exist. Check spelling and make sure the file is inside `my_files/`.

**Answers are vague or wrong**
Lower `CHUNK_SIZE` to 128, raise `TOP_K` to 12, delete `qa_vectordb/`, and re-ingest.

**Low similarity scores (below 0.2)**
The question wording is very different from the log wording. Try more specific language — use exact error names like `StaleElementReferenceException` instead of `test failed`.

**`collection.query` error**
Delete the `qa_vectordb/` folder and re-run. The database may be from an older version.

---

## Author

Built by Akash Nigam — SDET  
Stack: Python · ChromaDB · Sentence Transformers · Ollama · LangChain
