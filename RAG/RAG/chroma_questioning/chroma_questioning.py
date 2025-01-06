import chromadb
from sentence_transformers import SentenceTransformer

def ask_question(question, collection, model):
    # Generate embedding for the question
    question_embedding = model.encode(question).tolist()
    
    # Query the collection
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3  # Get top 3 most relevant chunks
    )
    
    print("\nQuestion:", question)
    print("\nRelevant passages:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\nPassage {i+1}:")
        print(doc)
        print("-" * 50)

def main():
    # Initialize ChromaDB client
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )

    try:
        # Get the collection
        collection = client.get_collection("pdf_chunks_collection")
        
        # Initialize the same model used for document embeddings
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        print("Successfully connected to ChromaDB!")
        print("Ready to answer questions! Type 'quit' to exit.")
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'quit':
                break
            
            ask_question(question, collection, model)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()