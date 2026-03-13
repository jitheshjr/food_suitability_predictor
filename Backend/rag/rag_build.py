import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_FOLDER = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_FOLDER = os.path.join(BASE_DIR, "vector_store")

os.makedirs(VECTOR_FOLDER, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(PDF_FOLDER, file))

        for page in reader.pages:
            text = page.extract_text()

            if text:
                documents.append(text)

print(f"Loaded {len(documents)} pages")

chunks = []
chunk_size = 500

for doc in documents:
    for i in range(0, len(doc), chunk_size):
        chunks.append(doc[i:i+chunk_size])

print(f"Created {len(chunks)} chunks")

embeddings = model.encode(chunks,
                          show_progress_bar=True,
                          convert_to_numpy=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

index_path = os.path.join(VECTOR_FOLDER, "faiss_index")
chunks_path = os.path.join(VECTOR_FOLDER, "chunks.pkl")

faiss.write_index(index, index_path)

with open(chunks_path, "wb") as f:
    pickle.dump(chunks, f)

print("Vector database created successfully!")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

index_path = os.path.join(VECTOR_FOLDER, "faiss_index")
chunks_path = os.path.join(VECTOR_FOLDER, "chunks.pkl")

faiss.write_index(index, index_path)

with open(chunks_path, "wb") as f:
    pickle.dump(chunks, f)

print("Vector database created successfully!")