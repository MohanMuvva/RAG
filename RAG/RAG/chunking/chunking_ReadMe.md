# chunking.py

## Purpose
Processes a single PDF file by extracting text, chunking it, generating embeddings, and storing the data in ChromaDB.

## Key Functions
- **extract_text_from_pdf**: Extracts text from a PDF file.
- **chunk_text**: Splits text into smaller, overlapping chunks.
- **generate_embeddings**: Creates embeddings for chunks using SentenceTransformer.
- **store_in_chromadb**: Stores chunks and their embeddings in a ChromaDB collection.

## Usage
- Specify the path to a PDF file.
- Run the script to process the file and store results in ChromaDB.
