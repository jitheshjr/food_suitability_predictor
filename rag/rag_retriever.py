import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
vector_dir = os.path.join(BASE_DIR, "vector_store")

index_path = os.path.join(vector_dir, "faiss_index")
chunks_path = os.path.join(vector_dir, "chunks.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(index_path)

with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

def retrieve_context(query, k=3):

    query_vector = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_vector, k)

    results = [chunks[i] for i in indices[0] if i != -1]

    return "\n".join(results)