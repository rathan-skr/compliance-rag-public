# RAG Chunking Strategies — With Real Examples

When building a RAG system over EU regulatory documents (GDPR, DORA, EU AI Act), how you split text determines whether the entire system works or fails. Here's every major chunking strategy with real output on the same input.

---

## The Input

Five sentences from GDPR Articles 5 and 6 — two different topics back to back:

```
S1: Personal data must be collected for specific purposes.
S2: Data should not be kept longer than necessary.
S3: Controllers must ensure data accuracy at all times.
S4: Processing is lawful only with user consent.
S5: A contract obligation also makes processing lawful.
```

---

## 1. Fixed-size Chunking

Split every N tokens with a fixed overlap. Ignores document structure entirely.

```python
def fixed_size_chunk(text, chunk_size=40, overlap=10):
    tokens = text.split()
    chunks, start = [], 0
    while start < len(tokens):
        chunks.append(" ".join(tokens[start:start + chunk_size]))
        start += chunk_size - overlap
    return chunks
```

**Output:**
```
Chunk 1: "Article 5 - Principles relating to processing of personal data
          Personal data shall be processed lawfully ... collected for specified,
          explicit and legitimate purposes and"

Chunk 4: "kept in a form which permits identification of data subjects for no
          longer than is necessary ... Article 6 - Lawfulness of processing
          Processing shall be lawful only if and to"
```

**Problem:** Chunk 4 merges the end of Article 5 with the start of Article 6 — two different legal concepts in one chunk.

---

## 2. Recursive Character Chunking

Try splitting on `\n\n`, then `\n`, then `. `, then ` ` — progressively smaller separators until the chunk fits the target size.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=200,
    chunk_overlap=20,
)
chunks = splitter.split_text(gdpr_text)
```

**Output (8 chunks):**
```
Chunk 1 (174 chars): "Article 5 - Principles ... processed lawfully, fairly ..."
Chunk 2 (160 chars): "Personal data shall be collected for specified, explicit ..."
Chunk 3 (132 chars): "Personal data shall be adequate, relevant and limited ..."
Chunk 4  (70 chars): "Personal data shall be accurate and, where necessary, kept up to date."
Chunk 5 (175 chars): "Personal data shall be kept in a form which permits identification ..."
Chunk 6 (134 chars): "Article 6 - Lawfulness of processing ..."
```

**Better** — each paragraph becomes its own chunk. But it doesn't *know* it's splitting on articles; it got lucky that paragraphs align with legal principles here.

---

## 3. Structure-aware Chunking

Parse the document hierarchy (Article → Paragraph) and split on legal boundaries explicitly.

```python
import re

def structure_aware_chunk(text):
    pattern = re.compile(r"(Article\s+\d+\s*-[^\n]*)", re.IGNORECASE)
    parts = pattern.split(text)
    chunks = []
    for i in range(1, len(parts) - 1, 2):
        header = parts[i].strip()
        body   = parts[i + 1].strip()
        article_num = re.search(r"\d+", header).group()
        chunks.append({
            "article":    article_num,
            "header":     header,
            "text":       f"{header}\n{body}",
            "paragraphs": [p.strip() for p in body.split("\n\n") if p.strip()]
        })
    return chunks
```

**Output (2 articles, 9 paragraphs):**
```
[Article 5] Article 5 - Principles relating to processing of personal data
  Para 1: Personal data shall be processed lawfully, fairly ...
  Para 2: Personal data shall be collected for specified, explicit ...
  Para 3: Personal data shall be adequate, relevant and limited ...
  Para 4: Personal data shall be accurate and, where necessary, kept up to date.
  Para 5: Personal data shall be kept in a form which permits identification ...

[Article 6] Article 6 - Lawfulness of processing
  Para 1: Processing shall be lawful only if and to the extent that ...
  Para 2: The data subject has given consent ...
  Para 3: Processing is necessary for the performance of a contract ...
  Para 4: Processing is necessary for compliance with a legal obligation ...
```

**Best for regulatory text.** Every chunk has metadata (`article`, `source`) — this is what makes citations trustworthy later. The answer `[GDPR Art. 5]` comes from the chunk's metadata, not hallucination.

---

## 4. Semantic Chunking

Embed every sentence, then split where cosine similarity between adjacent sentences drops — meaning the topic shifted.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

THRESHOLD = 0.5
chunks, current = [], [sentences[0]]
for i in range(len(sentences) - 1):
    sim = cosine(embeddings[i], embeddings[i+1])
    if sim < THRESHOLD:
        chunks.append(current)
        current = []
    current.append(sentences[i+1])
chunks.append(current)
```

**Real similarity scores on our 5 sentences:**
```
S1 -> S2: 0.517   (same topic: data principles)       stay together
S2 -> S3: 0.424   <-- SPLIT  (accuracy is different)
S3 -> S4: 0.171   <-- SPLIT  (lawfulness = new article)
S4 -> S5: 0.645   (same topic: lawful basis)           stay together
```

**Output (3 chunks):**
```
Chunk 1: S1 + S2  →  collection purposes + retention limit
Chunk 2: S3       →  data accuracy
Chunk 3: S4 + S5  →  consent + contract obligation
```

No rules — the model felt the topic shift at `S3→S4` (similarity `0.171`) and split there. Works without knowing what "Article 6" means.

---

## 5. Proposition Chunking

Use an LLM to rewrite each paragraph as atomic, self-contained facts. Each proposition is independently searchable.

```python
import anthropic

client = anthropic.Anthropic()

def proposition_chunk(paragraph):
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""
Convert this paragraph into atomic, self-contained factual propositions.
Each must make sense on its own. One per line.

Paragraph: {paragraph}"""}]
    )
    return response.content[0].text.strip().split("\n")
```

**Input (1 paragraph):**
```
Personal data shall be kept in a form which permits identification of data
subjects for no longer than is necessary for the purposes for which the
personal data are processed.
```

**Output (3 propositions):**
```
[1] GDPR Art 5(1)(e) requires data to be kept in identifiable form only as long as necessary.
[2] Data controllers must delete or anonymise personal data once its processing purpose is fulfilled.
[3] The storage limitation principle under GDPR applies to the form of data, not just its content.
```

1 paragraph → 3 independently searchable facts. A query about "data deletion" and "storage limits" now both hit relevant propositions. Expensive (LLM call per paragraph) but highest retrieval precision.

---

## 6. Parent-document / Hierarchical Chunking

Store two levels: small chunks for precise retrieval, large parent chunks for full context sent to the LLM.

```python
# Small chunks -> vector DB (retrieved by similarity)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=150)

# Large chunks -> docstore (sent to LLM as context)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=600)
```

**Output:**
```
[Parent p0] (542 chars — sent to LLM)
  "Article 5 - Principles ... Personal data shall be processed lawfully ..."
    |_ [Child p0_c1] (110 chars — in vector DB)
       "Personal data shall be processed lawfully, fairly and in a transparent manner ..."
    |_ [Child p0_c2] (150 chars — in vector DB)
       "Personal data shall be collected for specified, explicit and legitimate purposes ..."

[Parent p1] (Article 6 block)
    |_ [Child p1_c1] ...
    |_ [Child p1_c2] ...
```

**Why this matters:** the small child chunk matches the query precisely. But when passed to the LLM, the full parent (the entire article) is used — so the answer has complete context, not just the sentence that matched.

---

## 7. Late Chunking

Embed the full document first (preserving cross-sentence context in token embeddings), then chunk the embeddings — not the text.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model     = AutoModel.from_pretrained("bert-base-uncased")

# Step 1: embed the FULL document
inputs = tokenizer(full_doc, return_tensors="pt")
token_embeddings = model(**inputs).last_hidden_state[0]  # [n_tokens, 768]

# Step 2: slice the embeddings (not the text)
for start in range(0, len(tokens), CHUNK_SIZE):
    chunk_embedding = token_embeddings[start:end].mean(dim=0)  # context-aware
```

**Why it's different:**

The word `"necessary"` appears in both Article 5 and Article 6:

```
Art 5: "Data should not be kept longer than necessary."     → storage duration
Art 6: "Processing is lawful only with necessary consent."  → legal requirement
```

Normal chunking: each sentence embedded in isolation — `"necessary"` looks the same in both.

Late chunking: full document embedded first — `"necessary"` in Art 5 carries Art 5's context, `"necessary"` in Art 6 carries Art 6's context. The retriever can distinguish them.

---

## 8. Sliding Window Chunking

Fixed-size chunks with large overlap (50%). Every important sentence appears in at least two chunks — no content is ever lost at a seam.

```python
def sliding_window_chunk(text, chunk_size=40, step=20):
    tokens = text.split()
    for start in range(0, len(tokens), step):
        yield " ".join(tokens[start:start + chunk_size])
```

**Output (9 chunks, 50% overlap):**
```
Chunk 1: "Article 5 ... Personal data shall be processed lawfully, fairly and
          in a transparent manner ... collected for specified, explicit and
          legitimate purposes and"

Chunk 2: "transparent manner in relation to the data subject. Personal data
          shall be collected for specified, explicit ... not further processed
          in a manner that is incompatible ..."
```

Chunk 1 and Chunk 2 share ~20 tokens. The sentence at the boundary appears in both — retrieval never misses it. Downside: 9 chunks vs 7 (fixed-size) for the same text. Index grows fast at scale.

---

## Strategy Comparison

| # | Strategy | Chunks | Knows structure | Needs model | Best for |
|---|----------|--------|-----------------|-------------|----------|
| 1 | Fixed-size | 7 | No | No | Quick baseline |
| 2 | Recursive char | 8 | Partially | No | General text |
| 3 | Structure-aware | 2 articles / 9 para | Yes | No | **Regulatory docs** |
| 4 | Semantic | 3 | No | Embedding model | Any domain-shift text |
| 5 | Proposition | N×3 | Yes (via LLM) | LLM | High-precision QA |
| 6 | Parent-document | 3 parents / 12 children | Partially | No | **Production RAG** |
| 7 | Late chunking | N | Yes (implicit) | Full encoder | Context-sensitive terms |
| 8 | Sliding window | 9 | No | No | No-miss retrieval |

---

## What this project uses

**Structure-aware (3) → Recursive fallback (2) → Parent-document (6)**

- Split on Article/Section boundaries first
- Recursively split any article too long for the embedding model
- Store children in the vector DB, return parents to the LLM

This combination gives clean citation metadata, precise retrieval, and full legal context in every answer.
