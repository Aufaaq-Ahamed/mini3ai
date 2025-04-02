import streamlit as st
import requests
from bs4 import BeautifulSoup
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = "miniproject"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, metric="cosine", dimension=384)
index = pc.Index(INDEX_NAME)

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Function to scrape content from the provided URL
def scrape_website(url):
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text from <p> and <h> tags
        paragraphs = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        text = " ".join([p.get_text() for p in paragraphs])
        
        # Store the scraped text in Pinecone
        store_in_pinecone(text, url, "url")
        
        return text
    except Exception as e:
        st.error(f"Error scraping the URL: {e}")
        return ""

# Function to store text in Pinecone
def store_in_pinecone(text, source, data_type):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = []
    
    for chunk in chunks:
        vector = model.encode(chunk).tolist()
        if any(np.isnan(vector)) or any(np.isinf(vector)):
            st.error("Error: Vector contains NaN or Inf values. Skipping this chunk.")
            continue
        vectors.append((source + str(hash(chunk)), vector, {"source": source, "text": chunk, "type": data_type}))
    
    if vectors:
        index.upsert(vectors)

# Streamlit UI
st.title("Web & PDF Search with Pinecone and Gemini")

# Input options
option = st.radio("Select Input Type:", ("URL", "PDF"))

if option == "URL":
    url = st.text_input("Enter Website URL:")
    if st.button("Process URL"):
        scraped_text = scrape_website(url)
        if scraped_text:
            st.success("Website data stored in Pinecone!")
            st.text_area("Scraped Text", scraped_text, height=300)

elif option == "PDF":
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if pdf_files and st.button("Process PDFs"):
        for pdf_file in pdf_files:
            text = process_pdf(pdf_file)
            store_in_pinecone(text, pdf_file.name, "pdf")
        st.success("PDF data stored in Pinecone!")

# Query handling
query = st.text_input("Ask a question:")
if query and st.button("Search"):
    query_vector = model.encode(query).tolist()
    
    index_stats = index.describe_index_stats()
    expected_dim = index_stats["dimension"]
    
    if len(query_vector) != expected_dim:
        st.error(f"Error: Query vector has incorrect dimensions. Expected {expected_dim}, got {len(query_vector)}.")
    else:
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        
        if results["matches"]:
            filtered_results = [match for match in results["matches"] if match["metadata"].get("type") == option.lower()]
            
            if filtered_results:
                retrieved_text = "\n".join([match["metadata"].get("text", "No text found") for match in filtered_results])
                sources = [match["metadata"].get("source", "Unknown source") for match in filtered_results]
                
                st.subheader("Relevant Links:")
                for source in set(sources):
                    st.write(source)
                
                response = send_to_gemini(query, retrieved_text)
                st.subheader("AI Response:")
                st.write(response)
            else:
                st.write("No relevant results found in the selected category.")
        else:
            st.write("No relevant results found.")
