# main.py

from ocr_processor import extract_text
from chunker import chunk_text
from embedder import embed_chunks, create_faiss_index, search_faiss_index

# 1. OCR: Get text from image
text = extract_text("sample_note.jpg")  # Replace with your image file

# 2. Chunk the text
chunks = chunk_text(text)

# 3. Embed the chunks
embeddings = embed_chunks(chunks)

# 4. Store in FAISS index
index = create_faiss_index(embeddings)

# 5. Ask a question
while True:
    query = input("\nAsk a question based on your notes (type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    answer = search_faiss_index(query, chunks, index)
    print("🔍 Answer:", answer)
