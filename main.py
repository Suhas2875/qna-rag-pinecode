# --- Import Libraries ---
import requests
from tqdm.auto import tqdm
from pinecone import Pinecone
import os

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
pinecone_environment = ""

gemini_endpoint = f""

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)
index_name = ""  # Replace with your Pinecone index name

# --- Sample Data ---
data = [
    {"id": "doc1", "text": "Our company offers premium quality coffee beans."},
    {"id": "doc2", "text": "We source our coffee from sustainable farms in Ethiopia."},
    {"id": "doc3", "text": "Our flagship product is the Ethiopian Yirgacheffe."},
    {"id": "doc4", "text": "Customer service is available 24/7 via email and phone."},
    {"id": "doc5", "text": "Our mission is to provide the best coffee experience to our customers."},
]

# --- Embedding Generation ---
def get_embedding(text):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": text}]}]}
    response = requests.post(gemini_endpoint, json=payload, headers=headers)

    # Log raw response for debugging
    print(f"API Response for text: {text}")
    print(response.json())

    if response.status_code != 200:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")

    # Check and extract embedding from response
    response_data = response.json()
    # Update this field based on the actual API response structure
    embedding = response_data.get("embedding")  # Adjust key based on actual API response
    if not embedding or not isinstance(embedding, list):
        raise ValueError("Invalid or missing embedding in API response")
    
    return embedding

# --- Pinecone Indexing ---
index = pinecone.Index(index_name)

for entry in tqdm(data):
    try:
        embedding = get_embedding(entry["text"])  # Ensure this returns a numerical list
        index.upsert(vectors=[(entry["id"], embedding)])
    except Exception as e:
        print(f"Error processing entry {entry['id']}: {e}")

# --- Query Function ---
def query_pinecone(query, top_k=2):
    query_embedding = get_embedding(query)
    results = index.query(query_embedding, top_k=top_k, include_metadata=True)
    return results

# --- RAG Pipeline ---
def rag_pipeline(query):
    results = query_pinecone(query)
    context = "\n".join([match.metadata["text"] for match in results.matches])

    headers = {"Content-Type": "application/json"}
    prompt = f"""Answer the question based on the context below.
    Context: {context}
    Question: {query}
    Answer:"""
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(gemini_endpoint, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Gemini API Error: {response.status_code} - {response.text}")

    return response.json()["contents"][0]["parts"][0]["text"].strip()

# --- Example Query ---
query = "What is our flagship coffee product?"
try:
    answer = rag_pipeline(query)
    print(f"Question: {query}")
    print(f"Answer: {answer}")
except Exception as e:
    print(f"Error: {e}")

query = "What is our customer service like?"
try:
    answer = rag_pipeline(query)
    print(f"Question: {query}")
    print(f"Answer: {answer}")
except Exception as e:
    print(f"Error: {e}")
