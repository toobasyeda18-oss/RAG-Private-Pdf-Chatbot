import os
import time

import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "llama3.2:3b").strip()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
USE_REWRITE = os.environ.get("USE_REWRITE", "true").lower() == "true"
MAX_ANSWER_TOKENS = int(os.environ.get("MAX_ANSWER_TOKENS", "500"))
NUM_CTX = int(os.environ.get("NUM_CTX", "4096"))
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0"))
COLLECTION = "my_pdf"
TOP_K = 5
REFUSAL_TEXT = "I could not find that in the document."

CPU_ONLY = {"on": os.environ.get("CPU_ONLY", "false").lower() == "true"}

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_URL, timeout=60)
llm = ollama.Client(host=OLLAMA_URL, timeout=300)

SYSTEM_PROMPT = """You are an expert document analyst. You answer using ONLY the numbered sources you are given.

Rules:
1. If the user asks you to explain, describe, compare or summarize something, give a clear, well-organised explanation. Start with a one-sentence overview, then explain the key ideas in short bullet points or short paragraphs. Use a simple everyday analogy if it makes the idea easier to understand.
2. If the user asks a simple factual question, answer directly in one to three sentences.
3. Cite the sources inline, like [1] or [2], right after the statements they support.
4. Never add facts that are not in the sources. If the sources do not contain the answer, say exactly: I could not find that in the document. If the sources only partly answer the question, say what is covered and what is missing.
5. Write in plain, friendly English."""


def model_options(**extra):
    options = {"num_ctx": NUM_CTX}
    options.update(extra)
    if CPU_ONLY["on"]:
        options["num_gpu"] = 0
    return options


def document_hint():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not points:
        return ""
    payload = points[0].payload
    return payload.get("source", "") + ": " + payload["text"][:300]


def rewrite_query(question):
    if len(question.split()) <= 5:
        hint_text = (
            "The document's file name and opening text: " + document_hint() + "\n"
            "If the question is vague, such as 'what is this about' or "
            "'summarize it', use the main topic of the document in the query.\n"
        )
    else:
        hint_text = ""

    prompt = (
        "Rewrite the user's question as a short search query (under 15 words) "
        "for finding the relevant passages in a document. "
        "Keep all the important words from the question. Fix typos, and only "
        "add words when the question is vague.\n"
        + hint_text
        + "Reply with ONLY the search query and nothing else.\n\n"
        "Question: " + question
    )

    try:
        response = llm.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options=model_options(temperature=0, num_predict=40),
            keep_alive="30m",
        )
    except Exception:
        return question

    rewritten = response["message"]["content"].strip().strip('"')
    if rewritten == "" or len(rewritten) > 300:
        return question
    return rewritten


def retrieve(queries, top_k=TOP_K):
    best = {}

    for query in queries:
        query_vector = embed_model.encode(query).tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=top_k,
        )
        for point in results.points:
            if point.id not in best or point.score > best[point.id]["raw_score"]:
                best[point.id] = {
                    "raw_score": point.score,
                    "page": point.payload.get("page"),
                    "text": point.payload["text"],
                }

    ranked = sorted(best.values(), key=lambda item: item["raw_score"], reverse=True)

    sources = []
    for number, item in enumerate(ranked[:top_k], start=1):
        sources.append(
            {
                "id": number,
                "page": item["page"],
                "score": round(item["raw_score"] * 100, 1),
                "text": item["text"],
            }
        )
    return sources


def prepare(question):
    started = time.time()

    queries = [question]
    search_query = question
    if USE_REWRITE:
        rewritten = rewrite_query(question)
        if rewritten.lower() != question.lower():
            queries.append(rewritten)
            search_query = rewritten

    sources = retrieve(queries)

    top_score = sources[0]["score"] if len(sources) > 0 else 0
    refuse = len(sources) == 0 or top_score < MIN_SIMILARITY

    context = ""
    for source in sources:
        context = context + f"[Source {source['id']} | page {source['page']}]\n"
        context = context + source["text"] + "\n\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{context}\nQuestion: {question}"},
    ]
    return search_query, sources, messages, started, refuse


def stream_with_fallback(messages):
    last_error = None

    for attempt in range(2):
        started_streaming = False
        try:
            stream = llm.chat(
                model=MODEL_NAME,
                messages=messages,
                options=model_options(temperature=0.3, num_predict=MAX_ANSWER_TOKENS),
                keep_alive="30m",
                stream=True,
            )
            for part in stream:
                piece = part["message"]["content"]
                if piece:
                    started_streaming = True
                    yield piece
            return
        except Exception as error:
            last_error = error
            print("Model run failed:", str(error)[:200], flush=True)
            if started_streaming or CPU_ONLY["on"]:
                break
            print("Retrying on the CPU only...", flush=True)
            CPU_ONLY["on"] = True

    raise last_error


def ask(question):
    search_query, sources, messages, started, refuse = prepare(question)

    if refuse:
        answer = REFUSAL_TEXT
    else:
        answer = ""
        for piece in stream_with_fallback(messages):
            answer = answer + piece

    return {
        "answer": answer,
        "sources": sources,
        "search_query": search_query,
        "refused": refuse,
        "seconds": round(time.time() - started, 1),
    }


def ask_stream(question):
    search_query, sources, messages, started, refuse = prepare(question)

    yield {"type": "meta", "search_query": search_query, "sources": sources}

    if refuse:
        yield {"type": "token", "text": REFUSAL_TEXT}
    else:
        for piece in stream_with_fallback(messages):
            yield {"type": "token", "text": piece}

    yield {"type": "done", "seconds": round(time.time() - started, 1)}