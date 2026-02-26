import pandas as pd
import re
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load the model you already have
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Create a "Wrapper" so LangChain can talk to your model
class LocalEmbedder:
    def embed_documents(self, texts):
        return model.encode(texts).tolist()
    def embed_query(self, text):
        return model.encode([text])[0].tolist()

# 3. Use your text and clean it (KEEPING newlines this time!)
with open("data/textDoc.txt", "r", encoding="utf-8") as f:
    policy_text = f.read()

# Cleaning symbols but preserving structure for semantic logic
policy_text = re.sub(r'[^\w\s.,;?#$§%()-]', '', policy_text)
policy_text = re.sub(r'\s\s+', ' ', policy_text)

# 4. Semantic Chunking with the wrapper
embedder = LocalEmbedder()
semantic_splitter = SemanticChunker(embedder, breakpoint_threshold_type="percentile")

# Create chunks
docs = semantic_splitter.create_documents([policy_text])
semantic_chunks = [doc.page_content for doc in docs]

# 5. The rest of FAISS logic
embeddings = model.encode(semantic_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

query = "Who are the legal experts that curated the Atticus Project data?"
query_emb = model.encode([query])
D, I = index.search(np.array(query_emb), k=2)

print(f"Query: {query}\n")
for i, idx in enumerate(I[0]):
    # In FAISS IndexFlatL2, D is the squared L2 distance. 
    # To make it readable, we can show the distance score.
    distance = D[0][i]
    
    # Optional: Convert distance to a rough "Similarity %" 
    # (Note: This is an estimation, as L2 distance isn't bounded like Cosine)
    similarity = 1 / (1 + distance) 

    print(f"--- Retrieved Chunk (Index {idx}) ---")
    print(f"Distance Score: {distance:.4f} (Similarity: {similarity:.2%})")
    print(f"Content: {semantic_chunks[idx]}")
    print("-" * 30)
