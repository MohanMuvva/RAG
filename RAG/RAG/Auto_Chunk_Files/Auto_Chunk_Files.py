import os
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


# Step 4: Check if File Data Exists in ChromaDB
def is_file_processed_in_chromadb(pdf_name):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )
    collection = client.get_or_create_collection("pdf_chunks_collection")
    results = collection.get(where={"source_file": pdf_name})
    return len(results["documents"]) > 0


# Step 5: Store Chunks and Embeddings in ChromaDB
def store_in_chromadb(pdf_name, chunks, embeddings):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )
    collection = client.get_or_create_collection("pdf_chunks_collection")

    for idx, chunk in enumerate(chunks):
        unique_id = f"{pdf_name}_chunk_{idx}"
        collection.add(
            ids=[unique_id],
            metadatas=[{"chunk_index": idx, "source_file": pdf_name}],
            documents=[chunk],
            embeddings=[embeddings[idx].tolist()],
        )

    print(f"{len(chunks)} chunks from {pdf_name} successfully added to ChromaDB!")


# Step 6: Main Workflow
def main():
    pdf_folder = "C:\\Users\\saich\\Favorites\\Mohan\\RAG\\RAG"  # Update this to your folder path
    processed_files_log = "processed_files.log"

    # Read the log of already processed files
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r") as log_file:
            processed_files = set(log_file.read().splitlines())
    else:
        processed_files = set()

    # Process each new PDF file in the folder
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)

            # Skip if already processed
            if pdf_file in processed_files or is_file_processed_in_chromadb(pdf_file):
                print(f"Skipping already processed file: {pdf_file}")
                continue

            # Step 1: Extract text
            text_pages = extract_text_from_pdf(pdf_path)

            # Step 2: Chunk text
            all_chunks = []
            for page_text in text_pages:
                all_chunks.extend(chunk_text(page_text))

            # Step 3: Generate embeddings
            embeddings = generate_embeddings(all_chunks)

            # Step 4: Store in ChromaDB
            store_in_chromadb(pdf_file, all_chunks, embeddings)

            # Update the log of processed files
            with open(processed_files_log, "a") as log_file:
                log_file.write(f"{pdf_file}\n")


if __name__ == "__main__":
    main()
