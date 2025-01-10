import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text_pages.append(page.get_text())
    return text_pages

# Step 2: Chunk the Text
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap for context preservation
    return chunks

# Step 3: Generate Embeddings for Chunks
def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings


# Step 4: Store Chunks and Embeddings in ChromaDB
def store_in_chromadb(chunks, embeddings):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )
    
    # Create or get collection
    collection = client.get_or_create_collection("pdf_chunks_collection")

    for idx, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{idx}"],
            metadatas=[{"chunk_index": idx}],
            documents=[chunk],
            embeddings=[embeddings[idx].tolist()],
        )

    print(f"{len(chunks)} chunks successfully added to ChromaDB!")


# Main Workflow
def main():
    pdf_path = "420 Medical Terminology Certificate.pdf"

    # Step 1: Extract text
    text_pages = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk text
    all_chunks = []
    for page_text in text_pages:
        all_chunks.extend(chunk_text(page_text))

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(all_chunks)

    # Step 4: Store in ChromaDB
    store_in_chromadb(all_chunks, embeddings)

if __name__ == "__main__":
    main()
