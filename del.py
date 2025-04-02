import os
import pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "miniproject"  # Change to your actual index name

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if INDEX_NAME in pc.list_indexes().names():
    index = pc.Index(INDEX_NAME)
    
    # Delete all vectors from the index
    index.delete(delete_all=True)
    
    print(f"All contents from index '{INDEX_NAME}' have been deleted.")
else:
    print(f"Index '{INDEX_NAME}' does not exist.")
