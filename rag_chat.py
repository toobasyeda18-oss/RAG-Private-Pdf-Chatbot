from rag_core import ask

while True:
    user_question = input("\nAsk about your PDF (or type quit): ")
    if user_question.lower() == "quit":
        break

    print("\nThinking...\n")
    result = ask(user_question)

    print("Search query used:", result["search_query"])
    print("\n" + result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"  [{source['id']}] page {source['page']} - match {source['score']}%")