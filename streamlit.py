from config import RETRIEVAL_CONFIG, DOCUMENT_CONFIG
import streamlit as st
from langchain.chains import RetrievalQA
from local_llm import llm
from extraction_embeddings import get_vectorstore
import os

# Initialize vectorstore
with st.spinner("Loading documents..."):
    try:
        vectorstore = get_vectorstore()
        st.success(f"Loaded {len(vectorstore.index_to_docstore_id)} document chunks")
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize QA chain with updated vectorstore
@st.cache_resource
def get_qa_chain(_vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=RETRIEVAL_CONFIG["chain_type"],
        retriever=_vectorstore.as_retriever(search_kwargs=RETRIEVAL_CONFIG["search_kwargs"])
    )

qa = get_qa_chain(vectorstore)

st.title("DocsBase Chat")
st.caption("Ask questions about your documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa.run(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
