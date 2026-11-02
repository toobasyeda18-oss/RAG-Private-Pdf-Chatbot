import json
import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Private PDF Chatbot", page_icon="📄", layout="wide")

CSS = """
<style>
:root {
    --bg: #0B0F17;
    --panel: #111827;
    --border: #1E293B;
    --ink: #F8FAFC;
    --muted: #94A3B8;
}
.stApp { background: var(--bg); color: var(--ink); }
.block-container { zoom: __ZOOM__; max-width: 1500px; padding-top: 1.5rem; }
footer { visibility: hidden; }
h1 { font-weight: 600; letter-spacing: -0.02em; }
[data-testid="stExpander"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
}
[data-testid="stChatMessage"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
}
[data-testid="stFileUploader"] section {
    background: var(--bg);
    border: 1px dashed var(--border);
    border-radius: 8px;
}
[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
button { border-radius: 8px !important; }
.panel-label {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
</style>
"""

if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc" not in st.session_state:
    st.session_state.doc = None
    try:
        first_reply = requests.get(API_URL + "/chunks", timeout=30)
        if first_reply.ok:
            st.session_state.doc = first_reply.json()
    except requests.exceptions.RequestException:
        pass


def refresh_document_info():
    try:
        reply = requests.get(API_URL + "/chunks", timeout=30)
        st.session_state.doc = reply.json() if reply.ok else None
    except requests.exceptions.RequestException:
        st.session_state.doc = None


def percent(value):
    if value is None:
        return "n/a"
    return f"{value * 100:.0f}%"


def load_metrics():
    try:
        reply = requests.get(API_URL + "/metrics", timeout=30)
        if reply.ok:
            return reply.json().get("runs", [])
    except requests.exceptions.RequestException:
        pass
    return []


def build_markdown(doc, messages):
    name = doc["document"] if doc and doc.get("document") else "unknown"
    lines = [
        "# Q&A session",
        "",
        "- Document: " + str(name),
        "- Exported: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
        "",
    ]
    for message in messages:
        if message["role"] == "user":
            lines.append("## Q: " + message["content"])
            lines.append("")
        else:
            lines.append(message["content"])
            lines.append("")
            sources = message.get("sources") or []
            if sources:
                lines.append("**Sources**")
                lines.append("")
                for source in sources:
                    preview = source["text"][:160].replace("\n", " ")
                    lines.append(
                        f"- [{source['id']}] page {source['page']}, "
                        f"match {source['score']}%: {preview}..."
                    )
                lines.append("")
    return "\n".join(lines)


def build_json(doc, messages):
    name = doc["document"] if doc and doc.get("document") else None
    data = {
        "document": name,
        "exported": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": messages,
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def show_sources(message):
    sources = message.get("sources") or []
    if not sources:
        return

    note = "Searched for: " + str(message.get("search_query", ""))
    if message.get("seconds"):
        note = note + "  |  " + str(message["seconds"]) + " s"
    st.caption(note)

    for source in sources:
        title = f"Source {source['id']} · page {source['page']} · match {source['score']}%"
        with st.expander(title):
            st.progress(max(0, min(100, int(source["score"]))))
            st.write(source["text"])


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message)


def stream_answer(question, placeholder):
    message = {
        "role": "assistant",
        "content": "",
        "sources": [],
        "search_query": question,
        "seconds": None,
    }
    placeholder.markdown("_Searching your document..._")

    try:
        with requests.post(
            API_URL + "/ask_stream",
            json={"question": question},
            stream=True,
            timeout=600,
        ) as reply:
            if not reply.ok:
                try:
                    detail = reply.json().get("detail", "Something went wrong.")
                except ValueError:
                    detail = "Server error. Check the API logs."
                message["content"] = detail
                placeholder.markdown(detail)
                return message

            for line in reply.iter_lines():
                if not line:
                    continue
                event = json.loads(line)

                if event["type"] == "meta":
                    message["sources"] = event["sources"]
                    message["search_query"] = event["search_query"]
                    placeholder.markdown("_Writing the answer..._")
                elif event["type"] == "token":
                    message["content"] = message["content"] + event["text"]
                    placeholder.markdown(message["content"] + " ▌")
                elif event["type"] == "done":
                    message["seconds"] = event["seconds"]
                elif event["type"] == "error":
                    message["content"] = message["content"] + "\n\nError: " + event["text"]
    except requests.exceptions.RequestException:
        message["content"] = "I could not reach the API. Is it running?"

    placeholder.markdown(message["content"])
    return message


top_left, top_right = st.columns([5, 2])
with top_left:
    st.title("Private PDF Chatbot")
    st.caption("Runs fully on your computer. Your documents never leave it.")
with top_right:
    zoom = st.select_slider(
        "Zoom",
        options=[75, 90, 100, 110, 125, 150],
        value=100,
        format_func=lambda value: f"{value}%",
    )

st.markdown(CSS.replace("__ZOOM__", str(zoom / 100)), unsafe_allow_html=True)

chat_tab, metrics_tab = st.tabs(["Chat", "System Metrics"])

with chat_tab:
    left_col, right_col = st.columns([1, 2.2], gap="medium")

    with left_col:
        with st.container(border=True):
            st.markdown('<div class="panel-label">Document</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload a PDF", type="pdf", label_visibility="collapsed"
            )

            if uploaded is not None and st.button("Process PDF"):
                with st.spinner("Reading, chunking and storing your PDF..."):
                    try:
                        reply = requests.post(
                            API_URL + "/upload",
                            files={
                                "file": (uploaded.name, uploaded.getvalue(), "application/pdf")
                            },
                            timeout=600,
                        )
                        if reply.ok:
                            st.session_state.messages = []
                            refresh_document_info()
                            st.success("Done. Ask a question on the right.")
                        else:
                            try:
                                detail = reply.json().get("detail", "Something went wrong.")
                            except ValueError:
                                detail = "Server error while processing the PDF. Check the API logs."
                            st.error(detail)
                    except requests.exceptions.RequestException:
                        st.error("I could not reach the API. Is it running?")

            doc = st.session_state.doc
            if doc and doc["count"] > 0:
                st.markdown(f"**{doc['document']}**")
                st.caption(f"{doc['count']} chunks stored")

                with st.expander("Chunk inspector"):
                    word = st.text_input("Filter by word", key="chunk_filter")
                    shown = 0
                    with st.container(height=320, border=False):
                        for chunk in doc["chunks"]:
                            if word and word.lower() not in chunk["text"].lower():
                                continue
                            shown += 1
                            st.markdown(f"**Chunk {chunk['id']} · page {chunk['page']}**")
                            st.caption(chunk["text"])
                    st.caption(f"Showing {shown} of {doc['count']} chunks")
            else:
                st.caption("No document yet. Upload a PDF to begin.")

        export_area = st.container()

    with right_col:
        chat_box = st.container(height=int(460 / (zoom / 100)), border=False)
        has_document = bool(st.session_state.doc and st.session_state.doc["count"] > 0)
        question = st.chat_input(
            "Ask something about your document...",
            disabled=not has_document,
        )

    with chat_box:
        for message in st.session_state.messages:
            render_message(message)

        if question:
            user_message = {"role": "user", "content": question}
            st.session_state.messages.append(user_message)
            render_message(user_message)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                assistant_message = stream_answer(question, placeholder)
                show_sources(assistant_message)
            st.session_state.messages.append(assistant_message)

    with export_area:
        if st.session_state.messages:
            with st.container(border=True):
                st.markdown('<div class="panel-label">Export this chat</div>', unsafe_allow_html=True)
                st.download_button(
                    "Download as Markdown",
                    data=build_markdown(st.session_state.doc, st.session_state.messages),
                    file_name="qa-session.md",
                    mime="text/markdown",
                )
                st.download_button(
                    "Download as JSON",
                    data=build_json(st.session_state.doc, st.session_state.messages),
                    file_name="qa-session.json",
                    mime="application/json",
                )

with metrics_tab:
    runs = load_metrics()

    if not runs:
        st.info("No evaluation has been run yet.")
        st.write("Run this in a second terminal (inside the project folder), then come back:")
        st.code("docker-compose exec api python evaluate.py --fast", language="bash")
    else:
        latest = runs[-1]
        rewrite_text = "on" if latest["use_rewrite"] else "off"
        st.markdown(
            f"**Latest run:** {latest['time']} · document `{latest.get('document', '')}` · "
            f"model `{latest['model']}` · query rewriting {rewrite_text} · "
            f"min similarity {latest.get('min_similarity', 0)} · "
            f"{latest['questions']} questions"
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Retrieval hit rate", percent(latest["retrieval_hit_rate"]))
        col2.metric("Answer keyword score", percent(latest["answer_keyword_score"]))
        col3.metric("Faithfulness", percent(latest["faithfulness"]))
        col4.metric("Refusal accuracy", percent(latest["refusal_accuracy"]))
        if latest["avg_seconds"] is None:
            col5.metric("Avg time per question", "n/a")
        else:
            col5.metric("Avg time per question", f"{latest['avg_seconds']} s")

        with st.expander("What do these numbers mean?"):
            st.markdown(
                "- **Retrieval hit rate:** how often the retrieved chunks contained all the "
                "expected keywords.\n"
                "- **Answer keyword score:** the average share of expected keywords found in the answer.\n"
                "- **Faithfulness:** how often a judge model says every claim in the answer is "
                "supported by the sources. The judge is the same small local model, so treat "
                "this as a rough guide. It is empty when the run used `--fast`.\n"
                "- **Refusal accuracy:** how often the app correctly said it could not find an "
                "answer to a question that is not in the document.\n"
                "- **Min similarity:** the cut-off below which the app refuses to answer "
                "without asking the model (0 means the cut-off is off)."
            )

        st.markdown("#### All runs")
        table = []
        for run in reversed(runs):
            table.append(
                {
                    "Time": run["time"],
                    "Model": run["model"],
                    "Rewrite": "on" if run["use_rewrite"] else "off",
                    "Min similarity": run.get("min_similarity", 0),
                    "Hit rate": percent(run["retrieval_hit_rate"]),
                    "Keyword score": percent(run["answer_keyword_score"]),
                    "Faithfulness": percent(run["faithfulness"]),
                    "Refusal": percent(run["refusal_accuracy"]),
                    "Avg seconds": run["avg_seconds"],
                }
            )
        st.dataframe(table)

        st.markdown("#### Per-question results (latest run)")
        for row in latest["details"]:
            if row["in_document"]:
                icon = "✅" if row.get("retrieval_hit") else "❌"
                title = f"{icon} {row['question']}"
            else:
                icon = "✅" if row.get("refused_correctly") else "❌"
                title = f"{icon} (not in document) {row['question']}"

            with st.expander(title):
                details_text = f"Searched for: {row['search_query']}  |  {row['seconds']} s"
                if "top_score" in row:
                    details_text = details_text + f"  |  top match {row['top_score']}%"
                if "faithful" in row:
                    details_text = details_text + "  |  faithful: " + ("yes" if row["faithful"] else "no")
                st.caption(details_text)
                st.write(row["answer"])