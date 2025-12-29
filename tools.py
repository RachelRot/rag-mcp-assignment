"""
MCP (Model Context Protocol) Tools for Document Processing.

Contains 3 tools:
1. read_file - Read text files
2. chunk_and_store - Split text into chunks and store in database
3. search_document - Search for relevant chunks
"""

from langchain_core.tools import tool
from embeddings import LocalEmbeddingModel
import chromadb
import uuid
import logging

logger = logging.getLogger(__name__)

embedding_model = LocalEmbeddingModel()

chroma_client = chromadb.Client(
    chromadb.config.Settings(persist_directory="./db/chroma")
)

collection = chroma_client.get_or_create_collection("documents")

@tool
def read_file(path: str) -> str:
    """Read text file from disk"""
    logger.info(f"\nTOOL CALLED: read_file with path={path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"File read successfully, length={len(content)} characters\n")
    return content

@tool
def chunk_and_store(text: str) -> str:
    """Split text into chunks and store them in the vector database automatically."""
    logger.info(f"\nTOOL CALLED: chunk_and_store with text length={len(text)}")
    
    # Split into chunks
    max_chars = 500
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end + 1
    
    logger.info(f"   Created {len(chunks)} chunks")
    
    # Store automatically
    embeddings = embedding_model.embed(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
    )
    
    logger.info(f"Stored {len(chunks)} chunks successfully in ChromaDB\n")
    return f"Successfully split text into {len(chunks)} chunks and stored them in the database."

@tool
def search_document(question: str) -> str:
    """Search for relevant information in stored document chunks to answer a question."""
    logger.info(f"\nTOOL CALLED: search_document with question='{question}'")
    query_embedding = embedding_model.embed([question])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    
    if results["documents"] and len(results["documents"]) > 0:
        logger.info(f"Found {len(results['documents'][0])} relevant chunks\n")
        return "\n\n".join(results["documents"][0])
    logger.info("No relevant information found\n")
    return "No relevant information found."

