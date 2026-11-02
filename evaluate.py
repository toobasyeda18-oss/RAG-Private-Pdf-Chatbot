import json
import os
import sys
from datetime import datetime

import rag_core
from rag_core import ask, llm, MODEL_NAME

QA_FILE = "eval/qa_pairs.json"
RESULTS_FILE = "eval/results.json"
REFUSAL_PHRASES = [
    "could not find",
    "couldn't find",
    "cannot find",
    "can't find",
    "not mentioned",
    "does not mention",
    "doesn't mention",
    "not in the document",
    "no information",
]


def all_keywords_present(text, keywords):
    lowered = text.lower()
    for keyword in keywords:
        if keyword.lower() not in lowered:
            return False
    return True


def share_of_keywords(text, keywords):
    if len(keywords) == 0:
        return 1.0
    lowered = text.lower()
    found = 0
    for keyword in keywords:
        if keyword.lower() in lowered:
            found += 1
    return found / len(keywords)


def looks_like_refusal(answer):
    lowered = answer.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in lowered:
            return True
    return False


def judge_faithfulness(answer, sources):
    context = ""
    for source in sources:
        context = context + source["text"] + "\n\n"

    prompt = (
        "You are a strict fact checker. Below are SOURCES and an ANSWER. "
        "Reply with exactly one word: YES if every claim in the ANSWER is "
        "supported by the SOURCES, or NO if any claim is not supported.\n\n"
        "SOURCES:\n" + context +
        "ANSWER:\n" + answer + "\n\nYour one-word verdict:"
    )
    response = llm.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=rag_core.model_options(temperature=0, num_predict=5),
        keep_alive="30m",
    )
    verdict = response["message"]["content"].strip().upper()
    return verdict.startswith("YES")


def average(values):
    if len(values) == 0:
        return None
    return round(sum(values) / len(values), 3)


def main():
    use_judge = "--fast" not in sys.argv

    if not rag_core.qdrant.collection_exists(rag_core.COLLECTION):
        print("No document is loaded. Upload a PDF in the app first.")
        return

    with open(QA_FILE, encoding="utf-8") as qa_file:
        pairs = json.load(qa_file)

    document = rag_core.document_hint().split(": ")[0]
    print("Document under test:", document, flush=True)
    print("Model:", MODEL_NAME, "| query rewriting:", rag_core.USE_REWRITE, flush=True)
    print("Minimum similarity:", rag_core.MIN_SIMILARITY, flush=True)
    print("Faithfulness judge:", "on" if use_judge else "off (--fast)", flush=True)

    details = []
    hits = []
    answer_scores = []
    faithful = []
    refusals = []
    times = []

    for number, pair in enumerate(pairs, start=1):
        print(f"[{number}/{len(pairs)}] {pair['question']}", flush=True)
        result = ask(pair["question"])
        answer = result["answer"]

        retrieved_text = ""
        for source in result["sources"]:
            retrieved_text = retrieved_text + " " + source["text"]

        top_score = result["sources"][0]["score"] if len(result["sources"]) > 0 else 0

        row = {
            "question": pair["question"],
            "in_document": pair.get("in_document", True),
            "search_query": result["search_query"],
            "top_score": top_score,
            "seconds": result["seconds"],
            "answer": answer,
        }
        times.append(result["seconds"])

        if row["in_document"]:
            keywords = pair.get("keywords", [])
            hit = all_keywords_present(retrieved_text, keywords)
            score = share_of_keywords(answer, keywords)
            hits.append(1 if hit else 0)
            answer_scores.append(score)
            row["retrieval_hit"] = hit
            row["answer_keyword_score"] = round(score, 2)

            if use_judge and not result["refused"]:
                verdict = judge_faithfulness(answer, result["sources"])
                faithful.append(1 if verdict else 0)
                row["faithful"] = verdict
        else:
            refused = looks_like_refusal(answer)
            refusals.append(1 if refused else 0)
            row["refused_correctly"] = refused

        details.append(row)

    run = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "document": document,
        "model": MODEL_NAME,
        "use_rewrite": rag_core.USE_REWRITE,
        "min_similarity": rag_core.MIN_SIMILARITY,
        "questions": len(pairs),
        "retrieval_hit_rate": average(hits),
        "answer_keyword_score": average(answer_scores),
        "faithfulness": average(faithful),
        "refusal_accuracy": average(refusals),
        "avg_seconds": average(times),
        "details": details,
    }

    history = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as old_file:
            history = json.load(old_file)
    history.append(run)
    with open(RESULTS_FILE, "w", encoding="utf-8") as results_file:
        json.dump(history, results_file, indent=2)

    print("\n--- Results ---", flush=True)
    print("Retrieval hit rate:   ", run["retrieval_hit_rate"], flush=True)
    print("Answer keyword score: ", run["answer_keyword_score"], flush=True)
    print("Faithfulness:         ", run["faithfulness"], flush=True)
    print("Refusal accuracy:     ", run["refusal_accuracy"], flush=True)
    print("Average seconds:      ", run["avg_seconds"], flush=True)
    print("Saved to", RESULTS_FILE, flush=True)

    print("\n--- Top similarity per question (to choose MIN_SIMILARITY) ---", flush=True)
    in_scores = []
    out_scores = []
    for row in details:
        if row["in_document"]:
            label = "in document  "
            in_scores.append(row["top_score"])
        else:
            label = "NOT in doc   "
            out_scores.append(row["top_score"])
        print(f"{label}{row['top_score']:>6}%   {row['question']}", flush=True)

    if len(in_scores) > 0 and len(out_scores) > 0:
        highest_out = max(out_scores)
        lowest_in = min(in_scores)
        if highest_out < lowest_in:
            suggestion = round((highest_out + lowest_in) / 2, 1)
            print(f"\nThe scores separate cleanly. Suggested MIN_SIMILARITY={suggestion}", flush=True)
        else:
            print(
                f"\nThe scores overlap: the highest out-of-document score ({highest_out}%) "
                f"is not below the lowest in-document score ({lowest_in}%). "
                "A threshold alone cannot separate them perfectly.",
                flush=True,
            )


if __name__ == "__main__":
    main()