# embedder.py

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Loaded the pre-trained MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Converting list of text chunks into embeddings
def embed_chunks(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

# Creating a FAISS index from embeddings
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Searching the index using a user question
def search_faiss_index(question, chunks, index):
    question_embedding = model.encode([question], convert_to_numpy=True)
    _, I = index.search(question_embedding, k=1)  # top 1 result
    return chunks[I[0][0]]