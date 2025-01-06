# Auto_Chunk_Files.py

## Purpose
This script automates the process of:
1. Extracting text from PDF files.
2. Chunking the extracted text into smaller parts.
3. Generating embeddings for the chunks.
4. Storing the embeddings and chunks in a ChromaDB collection.

## Key Functions
- **extract_text_from_pdf**: Extracts text from each page of a PDF file.
- **chunk_text**: Splits the extracted text into chunks of a specified size with an overlap for context preservation.
- **generate_embeddings**: Uses SentenceTransformer to create embeddings for the text chunks.
- **is_file_processed_in_chromadb**: Checks if a file has already been processed and stored in ChromaDB.
- **store_in_chromadb**: Saves the text chunks and their embeddings to ChromaDB.

## Usage
- Place PDF files in the specified folder (`pdf_folder`).
- Run the script to process new PDFs and store the results in ChromaDB.
