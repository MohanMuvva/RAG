# Auto_Chunk_new_Content.py

## Purpose
Enhanced version of `Auto_Chunk_Files.py`:
1. Supports both PDF and DOCX file formats.
2. Monitors a folder for new or modified files.
3. Detects content changes and updates ChromaDB accordingly.

## Key Features
- **extract_text_from_file**: Handles both PDF and DOCX formats.
- **monitor_files**: Continuously monitors the folder for changes and triggers processing.
- **get_chunk_changes**: Detects added and removed content between old and new chunks.
- **remove_file_chunks**: Removes obsolete chunks from ChromaDB for deleted files.

## Usage
- Specify the folder containing files.
- Start the monitoring process using `monitor_files()` in the main section.
- Automatically processes changes and updates ChromaDB.
