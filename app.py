from rag.rag_pipeline import load_rag_pipeline
import json

def main():
    print("🚀 Movie RAG System")
    qa = load_rag_pipeline()

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        # Get answer
        result = qa.invoke(query)

        # Get sources with similarity scores
        retriever = qa.retriever
        docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=3)

        contexts = [doc.page_content for doc in result['source_documents']]
        source_titles = [doc.metadata.get("title", "Unknown Title") for doc in result['source_documents']]

        # Create the structured response
        response = {
            "answer": result['result'].strip(),
            "contexts": contexts,
            "reasoning": (
                f"The question asked about '{query}'. I searched the movie plots database and "
                f"based the answer on the following sources: {', '.join(source_titles)}."
            )
        }
        
        # Print as formatted JSON-like output
        print("\n" + json.dumps(response, indent=2))

        print("Sources:")
        # sort by score descending
        for i, (doc, score) in enumerate(sorted(docs_with_scores, key=lambda x: x[1], reverse=True), start=1):
            print(f"  {i}. {doc.metadata.get('title', 'Unknown Title')} (score: {score:.4f})")


if __name__ == "__main__":
    main()
