"""
Text to Vector Embeddings Module using SentenceTransformers.

Uses the all-MiniLM-L6-v2 model which is lightweight and fast for English text processing.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class LocalEmbeddingModel:
    """Class for managing local embedding model."""
    
    def __init__(self):
        """Initialize the model - downloads automatically on first run."""
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert list of texts to vectors.
        
        Args:
            texts: List of strings to convert
            
        Returns:
            List of vectors (each vector is a list of floats)
        """
        return self.model.encode(texts, show_progress_bar=False).tolist()