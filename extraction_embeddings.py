from config import DOCUMENT_CONFIG
import os
from typing import Optional
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_vectorstore(pdf_path: Optional[str] = None, recreate: bool = False) -> FAISS:
    """Create or load FAISS vectorstore from PDF documents"""
    os.makedirs(DOCUMENT_CONFIG["vectorstore_dir"], exist_ok=True)
    index_path = os.path.join(DOCUMENT_CONFIG["vectorstore_dir"], "faiss_index")
    
    if not recreate and os.path.exists(index_path):
        # Load existing vectorstore
        embedding_model = SentenceTransformer(DOCUMENT_CONFIG["embedding_model"])
        return FAISS.load_local(index_path, embedding_model.encode, allow_dangerous_deserialization=True)
    
    # Create new vectorstore from all documents in directory
    embedding_model = SentenceTransformer(DOCUMENT_CONFIG["embedding_model"])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DOCUMENT_CONFIG["chunk_size"],
        chunk_overlap=DOCUMENT_CONFIG["chunk_overlap"]
    )
    
    all_texts = []
    all_metadatas = []
    
    # Process all PDFs in documents directory
    documents_dir = DOCUMENT_CONFIG["documents_directory"]
    for filename in os.listdir(documents_dir):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(documents_dir, filename)
            try:
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                texts = text_splitter.split_documents(documents)
                all_texts.extend([t.page_content for t in texts])
                all_metadatas.extend([{**t.metadata, "source": filename} for t in texts])
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    if not all_texts:
        raise ValueError("No valid PDF documents found in the documents directory")
    
    embeddings = embedding_model.encode(all_texts)
    
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, embeddings)),
        embedding=embedding_model.encode,
        metadatas=all_metadatas
    )
    vectorstore.save_local(index_path)
    return vectorstore