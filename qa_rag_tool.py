"""
qa_rag_tool.py
==============
Generic QA assistant — ask questions about ANY logs, Jira exports,
or failure reports. No hardcoded log types. Just drop your files in.

HOW TO USE:
───────────
1. Drop your files into the 'my_files' folder (create it next to this script)
2. List them in MY_FILES below with a short label you choose
3. Delete qa_vectordb/ folder if it exists
4. Run: python qa_rag_tool.py
5. Type your question at the prompt

INSTALL (one time):
───────────────────
pip install chromadb sentence-transformers langchain-text-splitters ollama
Install Ollama from https://ollama.com then run: ollama pull llama3.2
"""

import os
import json
import chromadb
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════
#  YOUR FILES — edit this list only
#  Format: ("path/to/file", "label")
#  Label = short name YOU choose, used when filtering questions
# ═══════════════════════════════════════════════════════════════

MY_FILES = [

    # ── Paste your Jira export here ─────────────────────────────
    ("my_files/jira.txt",           "jira"),

    # ── Paste your Selenium / UI test logs here ──────────────────
    ("my_files/selenium.log",       "selenium"),

    # ── Paste any job failure logs here (Spark, Dataproc, CI...) ─
    ("my_files/job_failures.log",   "job"),

    # ── Add as many more as you need ────────────────────────────
    # ("my_files/any_other_file.txt", "your_label"),
    # ("my_files/k8s_errors.log",     "kubernetes"),
    # ("my_files/pipeline.log",       "pipeline"),
]

# ═══════════════════════════════════════════════════════════════
#  SETTINGS — tweak if answers are poor
# ═══════════════════════════════════════════════════════════════

CHUNK_SIZE    = 256    # smaller = more focused retrieval
CHUNK_OVERLAP = 32     # overlap between chunks (catches cross-boundary events)
TOP_K         = 8      # chunks retrieved per question (raise if answers miss details)
LLM_MODEL     = "llama3.2"

# ═══════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════

print("Loading model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client   = chromadb.PersistentClient(path="./qa_vectordb")
collection      = chroma_client.get_or_create_collection(name="qa_knowledge")
splitter        = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "]
)

# ═══════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def embed(text: str) -> list:
    return embedding_model.encode(text).tolist()


def ingest_file(path: str, label: str) -> int:
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        print(f"         Create the file and paste your content into it.")
        return 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    if not text:
        print(f"  [SKIP] File is empty: {path}")
        return 0

    chunks = splitter.split_text(text)
    doc_id = os.path.basename(path).replace(".", "_")
    print(f"  [{label}] {path} → {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        collection.add(
            ids        = [f"{doc_id}_{i}"],
            embeddings = [embed(chunk)],
            documents  = [chunk],
            metadatas  = [{"label": label, "doc_id": doc_id, "file": path}]
        )
    return len(chunks)


def search(query: str, label: str = None, top_k: int = TOP_K) -> list:
    total = collection.count()
    if total == 0:
        return []

    where = {"label": label} if label else None
    k     = min(top_k, total)

    results = collection.query(
        query_embeddings = [embed(query)],
        n_results        = k,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    return [
        {
            "text":  doc,
            "label": meta.get("label", "?"),
            "doc":   meta.get("doc_id", "?"),
            "score": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


SYSTEM_PROMPT = """You are a senior QA engineer and automation debugging expert.
You are given context extracted from logs, Jira tickets, and failure reports.

Rules:
1. Answer ONLY from the provided context. Never add outside knowledge.
2. Always quote the exact line or text that supports your answer.
3. Cite which file or label the evidence came from, e.g. [jira] or [selenium].
4. If the context does not contain enough detail, say "not found in provided files".
5. Do NOT guess or infer beyond what is written.

Always respond in this format:
ROOT CAUSE:
  <one clear sentence based only on the context>

EVIDENCE:
  <exact quote from the context> [label]

DEBUGGING STEPS:
  1. <step>
  2. <step>
  3. <step>

RELATED ISSUES:
  <any related entries from jira or other files, or "none found">
"""


def ask(question: str, label: str = None) -> dict:
    chunks = search(question, label=label)

    if not chunks:
        return {"question": question, "answer": "No data found. Ingest files first.", "chunks": []}

    context = "\n\n---\n\n".join(
        f"[{c['label']} | {c['doc']} | score:{c['score']}]\n{c['text']}"
        for c in chunks
    )

    print(f"  Searching {'[' + label + ']' if label else 'all files'}...")
    print(f"  Best match: score={chunks[0]['score']} from [{chunks[0]['label']}]")
    print(f"  Generating answer...")

    response = ollama.chat(
        model    = LLM_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content":
                f"CONTEXT FROM YOUR FILES:\n{context}"
                f"\n\nQUESTION:\n{question}"}
        ]
    )

    return {
        "question": question,
        "answer":   response["message"]["content"],
        "chunks":   chunks
    }


def get_labels() -> list:
    try:
        meta = collection.get(include=["metadatas"])["metadatas"]
        return sorted(set(m["label"] for m in meta))
    except Exception:
        return []

# ═══════════════════════════════════════════════════════════════
#  MAIN — interactive prompt
# ═══════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 58)
    print("  QA Assistant — ask anything about your logs & Jira")
    print("=" * 58)

    # ── Create my_files folder if missing ──────────────────────
    os.makedirs("my_files", exist_ok=True)

    # ── Ingestion ───────────────────────────────────────────────
    print("\n[1] Loading your files...")
    count = collection.count()

    if count > 0:
        labels = get_labels()
        print(f"  Already loaded: {count} chunks from labels: {labels}")
        print("  To reload fresh: delete qa_vectordb/ folder and re-run")
    else:
        total = sum(ingest_file(p, l) for p, l in MY_FILES)
        if total == 0:
            print("\n  No files were loaded. Check the steps below:")
            print("  1. Create a folder called 'my_files' next to this script")
            print("  2. Add your files to MY_FILES list at the top of this script")
            print("  3. Make sure the files are not empty")
            return
        print(f"\n  Total: {total} chunks loaded and ready.")

    # ── Question prompt ─────────────────────────────────────────
    labels = get_labels()
    print(f"\n[2] Ask your questions")
    print("─" * 58)
    print(f"  Loaded labels: {labels}")
    print()
    print("  How to ask:")
    print("  → Just type your question  (searches all files)")
    print("  → Type  label: question    (searches only that file)")
    print()

    for lbl in labels:
        print(f"     {lbl}: your question here")

    print()
    print("  Type 'list' to see what's loaded")
    print("  Type 'quit' to exit")
    print("─" * 58)

    session = []

    while True:
        try:
            raw = input("\n  Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit"):
            break

        # ── list command ────────────────────────────────────────
        if raw.lower() == "list":
            print(f"\n  Loaded labels: {get_labels()}")
            print(f"  Total chunks : {collection.count()}")
            for p, l in MY_FILES:
                exists = os.path.exists(p)
                size   = os.path.getsize(p) if exists else 0
                status = f"{size} bytes" if exists else "FILE NOT FOUND"
                print(f"  [{l}] {p} — {status}")
            continue

        # ── parse optional label prefix ─────────────────────────
        # e.g. "jira: has this bug appeared before?"
        label    = None
        question = raw

        for lbl in labels:
            if raw.lower().startswith(f"{lbl}:"):
                label    = lbl
                question = raw[len(lbl)+1:].strip()
                break

        # ── run RAG ─────────────────────────────────────────────
        result = ask(question, label=label)

        print(f"\n{'─'*56}")
        print(result["answer"])
        print(f"{'─'*56}")

        used = list(dict.fromkeys(c['label'] for c in result["chunks"]))
        print(f"  Evidence from: {used}")

        session.append({
            "question": question,
            "label_filter": label,
            "answer": result["answer"],
            "sources": used
        })

    # ── save session ────────────────────────────────────────────
    if session:
        with open("qa_session.json", "w") as f:
            json.dump(session, f, indent=2)
        print(f"\n  {len(session)} questions saved to qa_session.json")

    print("\n  Goodbye.")


if __name__ == "__main__":
    main()