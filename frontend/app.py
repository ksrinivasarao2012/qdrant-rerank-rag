import streamlit as st 
import requests
import json
import os
import subprocess
import sys

# Start FastAPI backend process in background on port 8000 if not already running
if "FASTAPI_STARTED" not in os.environ:
    os.environ["FASTAPI_STARTED"] = "1"
    subprocess.Popen([
        sys.executable, "-m", "uvicorn", "main:app",
        "--app-dir", "backend",
        "--host", "127.0.0.1",
        "--port", "8000"
    ])

# Point this to your FastAPI server (default to local, check environment variable for production)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")


st.set_page_config(page_title="Portfolio RAG", page_icon="📚", layout="wide")
st.title("📚 Portfolio RAG Assistant")

# 1. Initialize chat history and state in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch available documents from backend
available_docs = ["🔍 All Documents"]
try:
    doc_res = requests.get(f"{API_URL}/documents")
    if doc_res.status_code == 200:
        available_docs.extend(doc_res.json().get("documents", []))
except Exception:
    pass

# 2. Sidebar: Document Ingestion & Search Configuration
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf', 'PDF'])

    if st.button('Process File') and uploaded_file:
        with st.spinner("Chunking and generating embeddings..."):
            # Send file as multipart/form-data
            files = {'file' : (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            response = requests.post(f'{API_URL}/upload', files=files)
            
            if response.status_code == 200:
                st.success(f"Success! {response.json()['filename']} stored in vector DB.")
                st.rerun()  # Rerun to update the selectbox options
            else:
                st.error(f"Error: {response.text}")
                
    st.markdown("---")
    st.header("🔍 Search Configuration")
    
    # Initialize document filter states
    if "selected_doc" not in st.session_state:
        st.session_state.selected_doc = "🔍 All Documents"
        
    if st.session_state.selected_doc not in available_docs:
        st.session_state.selected_doc = "🔍 All Documents"
        
    # Render document selectbox
    selected_doc_widget = st.selectbox(
        "Search Source Filter",
        options=available_docs,
        index=available_docs.index(st.session_state.selected_doc),
        key="selected_doc_widget"
    )
    st.session_state.selected_doc = selected_doc_widget

# 3. Main UI: Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 4. Main UI: Query Input & API Call (Handle retries)
retry_prompt = None
if "retry_query" in st.session_state and st.session_state.retry_query:
    retry_prompt = st.session_state.retry_query
    st.session_state.retry_query = None

prompt = st.chat_input("Ask a question about your documents...")
if retry_prompt:
    prompt = retry_prompt

if prompt:
    # Add user message to session state & display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Fetch and render AI streaming response
    with st.chat_message('assistant'):
        # Determine if we should filter by a specific source file
        filter_source = st.session_state.selected_doc
        api_filter_value = None if filter_source == "🔍 All Documents" else filter_source
        
        payload = {
            "query": prompt,
            "top_k": 3,
            "chat_history": st.session_state.messages[:-1],  # Exclude current prompt
            "source_file": api_filter_value
        }
        
        try:
            response = requests.post(f'{API_URL}/query', json=payload, stream=True)
            
            if response.status_code == 200:
                citations = []
                full_text = ""
                response_placeholder = st.empty()
                
                # Read NDJSON lines as they stream in
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        chunk_type = chunk.get("type")
                        data = chunk.get("data")
                        
                        if chunk_type == "citations":
                            citations = data
                        elif chunk_type == "token":
                            full_text += data
                            response_placeholder.markdown(full_text + "▌")
                
                # Check if search returned no results/meaningful answers
                not_found = "do not have enough information" in full_text.lower() or not citations
                
                # Append citations footer only if the LLM found relevant information
                final_display = full_text
                if citations and not not_found:
                    final_display += "\n\n### Relevant Context\n"
                    for i, cite in enumerate(citations, 1):
                        final_display += f"**{i}. {cite['source_file']} (Page {cite['page_number']})**\n> {cite['text_snippet']}\n\n"
                
                response_placeholder.markdown(final_display)
                
                # Save assistant response to session state
                st.session_state.messages.append({'role': 'assistant', 'content': final_display})
                
                # Option A Fallback: If filtered search failed, present option to search all
                if not_found and api_filter_value:
                    st.warning(f"No matches found in your selected document: **{api_filter_value}**.")
                    if st.button("🔍 Search all documents instead"):
                        st.session_state.selected_doc = "🔍 All Documents"
                        st.session_state.retry_query = prompt
                        st.rerun()
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to communicate with backend: {e}")
