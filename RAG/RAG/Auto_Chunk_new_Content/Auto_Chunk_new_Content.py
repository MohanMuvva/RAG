import os
import fitz  # PyMuPDF for PDF
import docx  # For DOCX files
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import hashlib
from typing import Dict, Set
import time
from datetime import datetime


# Step 1: Extract Text from Files
def extract_text_from_file(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        text = []
        for page in doc:
            text.append(page.get_text())
        return " ".join(text)
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    else:
        raise ValueError("Unsupported file format. Only .pdf and .docx are supported.")


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


# Step 4: Check for Existing Chunks in ChromaDB
def get_existing_chunks(file_name):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )
    collection = client.get_or_create_collection("document_chunks")
    results = collection.get(where={"source_file": file_name})
    return {result["chunk_index"] for result in results["metadatas"]}


# Step 5: Store New Chunks in ChromaDB
def store_in_chromadb(file_name, chunks, embeddings, start_chunk_idx):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )
    collection = client.get_or_create_collection("document_chunks")

    # Get existing chunks if any
    try:
        old_results = collection.get(where={"source_file": file_name})
        old_chunks = old_results["documents"] if old_results["documents"] else []
    except:
        old_chunks = []

    # Find changes
    added_chunks, removed_chunks = get_chunk_changes(old_chunks, chunks)

    # Print changes
    if removed_chunks:
        print("\nRemoved content:")
        for chunk in removed_chunks[:3]:  # Show first 3 removed chunks
            print(f"- {chunk[:100]}...")
        if len(removed_chunks) > 3:
            print(f"... and {len(removed_chunks) - 3} more removals")

    if added_chunks:
        print("\nAdded content:")
        for chunk in added_chunks[:3]:  # Show first 3 added chunks
            print(f"+ {chunk[:100]}...")
        if len(added_chunks) > 3:
            print(f"... and {len(added_chunks) - 3} more additions")

    # Store new chunks
    for idx, chunk in enumerate(chunks):
        unique_id = f"{file_name}_chunk_{start_chunk_idx + idx}"
        collection.add(
            ids=[unique_id],
            metadatas=[{"chunk_index": start_chunk_idx + idx, "source_file": file_name}],
            documents=[chunk],
            embeddings=[embeddings[idx].tolist()],
        )

    print(f"\nProcessed {len(chunks)} total chunks")


# Add these functions at the top after imports
def get_content_hash(file_path: str) -> str:
    """Generate a hash of file content to detect changes."""
    try:
        if file_path.endswith('.pdf'):
            doc = fitz.open(file_path)
            content = ' '.join(page.get_text() for page in doc)
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            content = ' '.join(paragraph.text for paragraph in doc.paragraphs)
        return hashlib.md5(content.encode()).hexdigest()
    except Exception as e:
        print(f"Error generating hash for {file_path}: {str(e)}")
        return ""

def load_content_hashes() -> Dict[str, str]:
    """Load saved content hashes from file."""
    hash_file = "content_hashes.log"
    content_hashes = {}
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            for line in f:
                if line.strip():
                    filename, hash_value = line.strip().split(",")
                    content_hashes[filename] = hash_value
    return content_hashes

def save_content_hashes(content_hashes: Dict[str, str]):
    """Save content hashes to file."""
    hash_file = "content_hashes.log"
    with open(hash_file, "w") as f:
        for filename, hash_value in content_hashes.items():
            f.write(f"{filename},{hash_value}\n")


# Main Workflow
def process_files():
    folder_path = "C:\\Users\\saich\\Favorites\\Mohan\\RAG\\RAG"
    processed_files_log = "processed_files.log"
    
    # Load existing content hashes
    content_hashes = load_content_hashes()

    # Track already processed files
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r") as log_file:
            processed_files = set(log_file.read().splitlines())
    else:
        processed_files = set()

    # Get current files in directory
    current_files = {
        f for f in os.listdir(folder_path) 
        if f.endswith((".pdf", ".docx")) 
        and not f.startswith(('~', '.'))
    }

    # Check for deleted files
    deleted_files = processed_files - current_files
    if deleted_files:
        print("\nDetected deleted files:")
        for deleted_file in deleted_files:
            print(f"Removing chunks for: {deleted_file}")
            remove_file_chunks(deleted_file)
            processed_files.remove(deleted_file)
            if deleted_file in content_hashes:
                del content_hashes[deleted_file]

    # Process current files
    for file_name in current_files:
        file_path = os.path.join(folder_path, file_name)
        
        if not os.access(file_path, os.R_OK):
            print(f"Skipping inaccessible file: {file_name}")
            continue

        # Check if content has changed
        current_hash = get_content_hash(file_path)
        if current_hash:
            if file_name in content_hashes:
                if content_hashes[file_name] == current_hash:
                    print(f"\nNo changes detected in: {file_name}")
                    continue
                else:
                    print(f"\nContent changes detected in: {file_name}")
                    # Remove old chunks before processing new content
                    remove_file_chunks(file_name)
            
            content_hashes[file_name] = current_hash

        print(f"\nProcessing file: {file_name}")
        try:
            file_text = extract_text_from_file(file_path)
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue

        # Process the chunks
        all_chunks = chunk_text(file_text)
        print(f"\nAnalyzing changes in {file_name}...")

        # Generate embeddings for all chunks
        embeddings = generate_embeddings(all_chunks)

        # Store in ChromaDB with change detection
        store_in_chromadb(file_name, all_chunks, embeddings, 0)

        # Update processed files log
        if file_name not in processed_files:
            with open(processed_files_log, "a") as log_file:
                log_file.write(f"{file_name}\n")
            processed_files.add(file_name)

    # Save updated content hashes
    save_content_hashes(content_hashes)


# Add this new function to remove chunks for deleted files
def remove_file_chunks(file_name):
    client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        ssl=False
    )
    collection = client.get_or_create_collection("document_chunks")
    
    try:
        # Get all chunks for the file
        results = collection.get(where={"source_file": file_name})
        if results["ids"]:
            # Delete all chunks for this file
            collection.delete(ids=results["ids"])
            print(f"Successfully removed {len(results['ids'])} chunks for {file_name}")
    except Exception as e:
        print(f"Error removing chunks for {file_name}: {str(e)}")


# Add this new function to detect changes between old and new chunks
def get_chunk_changes(old_chunks: list, new_chunks: list) -> tuple:
    """Compare old and new chunks to identify changes."""
    old_set = set(old_chunks)
    new_set = set(new_chunks)
    
    added_chunks = new_set - old_set
    removed_chunks = old_set - new_set
    
    return list(added_chunks), list(removed_chunks)


# Add this function to track changes
def monitor_files(interval=10, max_unchanged_time=60):
    """
    Monitor files for changes every {interval} seconds.
    Stop if no changes detected for {max_unchanged_time} seconds.
    """
    print(f"\nStarting file monitor at {datetime.now().strftime('%H:%M:%S')}")
    print("Checking for changes every", interval, "seconds")
    print("Will stop after", max_unchanged_time, "seconds without changes")
    
    last_change_time = time.time()
    last_hashes = {}
    
    while True:
        current_time = time.time()
        
        # Get current state of all files
        folder_path = "C:\\Users\\saich\\Favorites\\Mohan\\RAG\\RAG"
        current_files = {
            f: get_content_hash(os.path.join(folder_path, f))
            for f in os.listdir(folder_path)
            if f.endswith((".pdf", ".docx")) and not f.startswith(('~', '.'))
        }
        
        # Check for changes
        changes_detected = False
        if last_hashes:
            # Check for new or modified files
            for file, hash_value in current_files.items():
                if file not in last_hashes or last_hashes[file] != hash_value:
                    changes_detected = True
                    break
            
            # Check for deleted files
            if not changes_detected:
                for file in last_hashes:
                    if file not in current_files:
                        changes_detected = True
                        break
        
        if changes_detected:
            print(f"\nChanges detected at {datetime.now().strftime('%H:%M:%S')}")
            process_files()
            last_change_time = current_time
        else:
            time_since_last_change = current_time - last_change_time
            if time_since_last_change >= max_unchanged_time:
                print(f"\nNo changes detected for {max_unchanged_time} seconds")
                print(f"Monitor stopping at {datetime.now().strftime('%H:%M:%S')}")
                break
            
        last_hashes = current_files
        time.sleep(interval)


# Update the main section
if __name__ == "__main__":
    try:
        monitor_files()
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    except Exception as e:
        print(f"\nError in monitor: {str(e)}")
