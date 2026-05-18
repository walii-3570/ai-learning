from sentence_transformers import SentenceTransformer
from groq import Groq
import numpy as np
from config import GROQ_API_KEY

model = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(
    api_key=GROQ_API_KEY
)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

with open("ai_notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split by blank lines (paragraph chunking)
chunks = text.split("\n\n")

# Remove empty chunks
chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

print(f"Total Chunks: {len(chunks)}")

# Create embeddings once
embeddings = model.encode(chunks)

while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Embed user question
    query_embedding = model.encode(query)

    scores = []

    # Compare query with every chunk
    for i, chunk_embedding in enumerate(embeddings):

        score = cosine_similarity(query_embedding, chunk_embedding)

        scores.append((score, chunks[i]))

    # Sort highest similarity first
    scores = sorted(scores, key=lambda x: x[0], reverse=True)

    # Take top 3 chunks
    top_chunks = scores[:3]

    context = ""

    print("\nTop Retrieved Chunks:\n")

    for score, chunk in top_chunks:

        print(f"Score: {score:.4f}")
        print(chunk)
        print("-" * 50)

        context += chunk + "\n\n"

    prompt = f"""
You are a RAG assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You answer only from given context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    reply = response.choices[0].message.content

    print("\nBot:", reply)