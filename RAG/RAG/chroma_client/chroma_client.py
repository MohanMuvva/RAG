import chromadb

# Initialize ChromaDB client
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    ssl=False
)

try:
    # Test the connection by listing collections
    collections = client.list_collections()
    print("Successfully connected to ChromaDB!")
    print("Available collections:", collections)
    print(f"Number of collections: {len(collections)}")
except Exception as e:
    print(f"Error connecting to ChromaDB: {e}")
