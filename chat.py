import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from dotenv import load_dotenv
import numpy as np
import google.generativeai as genai
import threading
import time
import random

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

def send_to_gemini(query, retrieved_text):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Given the following query and relevant extracted text, generate a concise and relevant response.
        
        Query: {query}
        Extracted Text:
        {retrieved_text}
        
        Provide a response based on the extracted text.
        """
        response = model.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        st.error(f"Error generating content with Gemini: {e}")
        return f"Error generating content with Gemini: {e}"

# Function to extract sub-URL
def get_sub_url(url):
    parsed_url = urlparse(url)
    return parsed_url.path if parsed_url.path else "/"

# Function to extract all links from a webpage
def extract_links(soup, base_url):
    links = {}
    for tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, tag["href"])
        parsed_url = urlparse(full_url)
        if parsed_url.netloc == urlparse(base_url).netloc:
            links[full_url] = get_sub_url(full_url)
    return links

# Multi-threaded crawler function
def crawl_website(url, depth=2, visited=set(), lock=threading.Lock()):
    if depth == 0 or url in visited:
        return
    
    try:
        time.sleep(random.uniform(1, 3))
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        store_in_pinecone(text, get_sub_url(url), "url")
        
        links = extract_links(soup, url)
        
        threads = []
        with lock:
            visited.add(url)
        
        for link in links:
            thread = threading.Thread(target=crawl_website, args=(link, depth-1, visited, lock))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    except Exception as e:
        print(f"Error crawling {url}: {e}")

# Function to process PDFs
def process_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to store text in Pinecone
def store_in_pinecone(text, source, data_type):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = []
    
    for chunk in chunks:
        vector = model.encode(chunk).tolist()
        if any(np.isnan(vector)) or any(np.isinf(vector)):
            st.error("Error: Vector contains NaN or Inf values. Skipping this chunk.")
            continue
        vectors.append((f"{source}-{hash(chunk)}", vector, {"source": source, "text": chunk, "type": data_type}))
    
    if vectors:
        index.upsert(vectors)

# Streamlit UI
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)

st.sidebar.title("Options")
option = st.sidebar.radio("Select Input Type:", ("URL", "PDF"))

if option == "URL":
    url = st.sidebar.text_input("Enter Website URL:")
    if st.sidebar.button("Process URL"):
        crawl_website(url, depth=2)
        st.sidebar.success("Website data stored in Pinecone!")

elif option == "PDF":
    pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if pdf_files and st.sidebar.button("Process PDFs"):
        for pdf_file in pdf_files:
            text = process_pdf(pdf_file)
            store_in_pinecone(text, pdf_file.name, "pdf")
        st.sidebar.success("PDF data stored in Pinecone!")

# Chat logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Type your message here...")
if query and st.button("Send"):
    query_vector = model.encode(query).tolist()
    
    index_stats = index.describe_index_stats()
    expected_dim = index_stats["dimension"]
    
    if len(query_vector) != expected_dim:
        st.error(f"Error: Query vector has incorrect dimensions. Expected {expected_dim}, got {len(query_vector)}.")
    else:
        results = index.query(vector=query_vector, top_k=5, include_metadata=True)
        
        if results["matches"]:
            retrieved_text = "\n".join([match["metadata"].get("text", "No text found") for match in results["matches"]])
            response = send_to_gemini(query, retrieved_text)
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", response))
        else:
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", "No relevant results found."))

# Display chat history
st.subheader("Chat History")
for sender, message in st.session_state.chat_history:
    with st.chat_message("assistant" if sender == "Bot" else "user"):
        st.write(f"**{sender}:** {message}")
