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

# Function to scrape maximum content from a webpage
def scrape_page(url):
    try:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract content
        headings = "\n".join([tag.get_text() for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])])
        paragraphs = "\n".join([tag.get_text() for tag in soup.find_all("p")])
        lists = "\n".join([tag.get_text() for tag in soup.find_all(["ul", "ol", "li"])])
        tables = "\n".join([" ".join([cell.get_text() for cell in row.find_all(["td", "th"])]) for row in soup.find_all("tr")])
        divs = "\n".join([tag.get_text() for tag in soup.find_all("div") if len(tag.get_text().strip()) > 50])

        # Combine all extracted text
        full_text = f"{headings}\n\n{paragraphs}\n\n{lists}\n\n{tables}\n\n{divs}"

        # Store in Pinecone with the full URL as the source
        store_in_pinecone(full_text, url, "url")

        return True
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return False

# Function to process PDFs
def process_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text
    except Exception as e:
        st.error(f"Error processing PDF {pdf_file.name}: {e}")
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
        if scrape_page(url):
            st.success(f"Website data from {url} stored in Pinecone!")

elif option == "PDF":
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if pdf_files and st.button("Process PDFs"):
        for pdf_file in pdf_files:
            text = process_pdf(pdf_file)
            if text:
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
            retrieved_text = "\n".join([match["metadata"].get("text", "No text found") for match in results["matches"]])
            sources = [match["metadata"].get("source", "Unknown source") for match in results["matches"]]

            st.subheader("Relevant Links:")
            for source in set(sources):
                st.write(source)

            response = send_to_gemini(query, retrieved_text)
            st.subheader("AI Response:")
            st.write(response)
        else:
            st.write("No relevant results found.")
