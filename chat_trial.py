import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pinecone import Pinecone, ServerlessSpec
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

def send_to_gemini(query, retrieved_text):
    try:
        model_llm = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"""
        Given the following query and relevant extracted text, generate a concise and relevant response.
        
        Query: {query}
        Extracted Text:
        {retrieved_text}
        
        Provide a response based on the extracted text.
        """
        response = model_llm.generate_content(prompt)
        return response.text if response else "No response generated."
    except Exception as e:
        st.error(f"Error generating content with Gemini: {e}")
        return f"Error generating content with Gemini: {e}"

# Helper function to extract the base URL
def get_base_url(url):
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"

# Function to extract links from a webpage (only same-domain links)
def extract_links(soup, base_url):
    links = []
    for tag in soup.find_all("a", href=True):
        full_url = urljoin(base_url, tag["href"])
        parsed = urlparse(full_url)
        if parsed.netloc == urlparse(base_url).netloc:
            links.append(full_url)
    return links

# Process the URL provided by the user:
# 1. Scrape the homepage and store its text in Pinecone.
# 2. Extract the list of same-domain links.
# 3. Maintain the base URL and the links in session_state.
def process_url(url):
    base_url = get_base_url(url)
    response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])
    store_in_pinecone(text, base_url, "url")
    links = extract_links(soup, base_url)
    st.session_state["urls"] = links
    st.session_state["base_url"] = base_url
    # Maintain a set of already scraped URLs (starting with homepage)
    st.session_state["scraped_urls"] = {url}
    st.success("Homepage data stored and links extracted.")

# Process PDFs (unchanged)
def process_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Store text in Pinecone using the base URL as the source.
def store_in_pinecone(text, source, data_type):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    vectors = []
    
    for chunk in chunks:
        vector = model.encode(chunk).tolist()
        if any(np.isnan(vector)) or any(np.isinf(vector)):
            st.error("Error: Vector contains NaN or Inf values. Skipping this chunk.")
            continue
        # Use the base URL as the source
        vectors.append((f"{source}-{hash(chunk)}", vector, {"source": source, "text": chunk, "type": data_type}))
    
    if vectors:
        index.upsert(vectors)

# Use Gemini to select relevant URLs from the extracted list based on the query.
import re

def get_relevant_urls(query, urls):
    try:
        model_llm = genai.GenerativeModel("gemini-2.0-flash")
        prompt = f"Given the query: {query}\nand the following list of URLs:\n{', '.join(urls)}\nReturn a comma-separated list of URLs that are most likely to contain relevant information to answer the query. If none are relevant, return 'none'."
        response = model_llm.generate_content(prompt)
        if response and response.text.lower().strip() != "none":
            # Use regex to extract valid URLs
            found_urls = re.findall(r"https?://[^\s,]+", response.text)
            return found_urls
        else:
            return []
    except Exception as e:
        st.error(f"Error in Gemini filtering URLs: {e}")
        return []

# Scrape a specific URL and store its data (if not already scraped)
def scrape_and_store(url):
    base_url = get_base_url(url)
    if "scraped_urls" not in st.session_state:
        st.session_state["scraped_urls"] = set()
    if url in st.session_state["scraped_urls"]:
        return
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        store_in_pinecone(text, base_url, "url")
        st.session_state["scraped_urls"].add(url)
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")

# Streamlit UI remains largely the same.
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot</h1>", unsafe_allow_html=True)

st.sidebar.title("Options")
option = st.sidebar.radio("Select Input Type:", ("URL", "PDF"))

if option == "URL":
    url = st.sidebar.text_input("Enter Website URL:")
    if url:
        new_base = get_base_url(url)
        # If a new base URL is entered, clear the chat history
        if "base_url" in st.session_state and st.session_state["base_url"] != new_base:
            st.session_state.chat_history = []
    if st.sidebar.button("Process URL"):
        process_url(url)

elif option == "PDF":
    pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if pdf_files and st.sidebar.button("Process PDFs"):
        for pdf_file in pdf_files:
            text = process_pdf(pdf_file)
            store_in_pinecone(text, pdf_file.name, "pdf")
        st.sidebar.success("PDF data stored in Pinecone!")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat logic
query = st.text_input("Type your message here...")
if query and st.button("Send"):
    # For URL option, use the extracted URLs and LLM filtering logic.
    if option == "URL":
        if "urls" in st.session_state and st.session_state["urls"]:
            relevant_urls = get_relevant_urls(query, st.session_state["urls"])
            # For each relevant URL, scrape and store the content if not done already.
            for r_url in relevant_urls:
                scrape_and_store(r_url)
    
    query_vector = model.encode(query).tolist()
    
    # Filter results so that only chunks from the current base URL are used
    filter_dict = {}
    if option == "URL" and "base_url" in st.session_state:
        filter_dict = {"source": st.session_state["base_url"]}
    
    index_stats = index.describe_index_stats()
    expected_dim = index_stats["dimension"]
    
    if len(query_vector) != expected_dim:
        st.error(f"Error: Query vector has incorrect dimensions. Expected {expected_dim}, got {len(query_vector)}.")
    else:
        results = index.query(vector=query_vector, top_k=5, include_metadata=True, filter=filter_dict)
        
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
