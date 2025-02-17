import os
import requests
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up the page configuration
st.set_page_config(page_title="⚖️ LegalAssist - Your BNS and BSA Guide", layout="wide")

# Sidebar with branding and description
st.sidebar.image("lady_justice.png", width=100)
st.sidebar.title("⚖️ LegalAssist")
st.sidebar.write("AI-powered legal query assistant to retrieve relevant legal sections and punishments.")
st.sidebar.markdown("---")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to reset the conversation
def reset_conversation():
    st.session_state.messages = []

# Load summarization model
@st.cache_resource
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

summarizer = load_summarizer()

# Load FAISS vector database
@st.cache_resource
def load_vector_db():
    db_path = "bns_bsa_vector_db_st"

    # Download from GitHub if not found locally
    if not os.path.exists(db_path):
        url = "bns_bsa_vector_db_st"
        response = requests.get(url)
        with open(db_path, "wb") as f:
            f.write(response.content)

    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

db_retriever = load_vector_db()

# Function to extract punishments mentioned in the text
def extract_punishment(text):
    punishments = ["death", "imprisonment", "fine", "community service", "forfeiture of property"]
    found_punishments = [p for p in punishments if p in text.lower()]
    return ', '.join(found_punishments) if found_punishments else "No specific punishment mentioned."

# Generate legal response based on the query
def generate_legal_response(query):
    retrieved_docs = db_retriever.get_relevant_documents(query)
    relevant_info = "\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant legal information found."
    
    punishment = extract_punishment(relevant_info)
    
    if len(relevant_info) > 1000:
        summary = summarizer(relevant_info[:1000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    else:
        summary = relevant_info

    return f"""
### 📜 **Query:**
{query}

### 📖 **Relevant Legal Section:**
{summary}

### ⚖️ **Punishment (if any):**
{punishment}
"""

# Main UI Components
st.title("⚖️ LegalAssist - AI-Powered Legal Query Resolver")
st.image("doodles.png", width=200)
st.markdown("---")

# Input prompt for legal query
input_prompt = st.text_area("**Enter your legal query:**", height=100)
if st.button("🔍 Get Legal Information") and input_prompt:
    response = generate_legal_response(input_prompt)
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.markdown("## 📜 **Legal Response**")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Reset chat button
st.button("🔄 Reset Chat", on_click=reset_conversation)
