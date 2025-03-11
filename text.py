import os
import requests
from dotenv import load_dotenv

# 1. Get your free API key from https://huggingface.co/settings/tokens
# 2. Create .env file with: HF_API_KEY="your_token_here"

load_dotenv()

def get_embeddings(texts):
    """Get embeddings without local model using Hugging Face API"""
    API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Authorization": f"Bearer {os.environ['HF_API_KEY']}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": texts})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"Error: {err.response.text}")
        return None

# Test with sample text
if __name__ == "__main__":
    sample_texts = ["COVID-19 symptoms include", "Polio vaccination schedule is"]
    
    embeddings = get_embeddings(sample_texts)
    
    if embeddings:
        print(f"Success! Embedding dimension: {len(embeddings[0])}")
        print("First embedding values:", embeddings[0][:5])  # Show first 5 values


# end