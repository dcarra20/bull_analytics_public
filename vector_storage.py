"""
Vector Database Module
Stores and searches vector embeddings with multiple similarity algorithms.
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional
import time


class VectorDatabase:
    """
    In-memory vector database supporting multiple search algorithms.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector database.
        
        Args:
            dimension: Dimension of vectors to store
        """
        self.dimension = dimension
        self.vectors = None  # numpy array of shape (n, dimension)
        self.metadata = []   # list of metadata dicts
        self.index_built = False
        
    def add_vectors(self, chunks: List[Dict]):
        """
        Add vectors from embedded chunks to database.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' field
        """
        print(f"Adding {len(chunks)} vectors to database...")
        
        # Extract vectors and metadata
        vectors_list = []
        metadata_list = []
        
        for chunk in chunks:
            if 'embedding' in chunk:
                # Convert to numpy if needed
                vector = chunk['embedding']
                if isinstance(vector, list):
                    vector = np.array(vector)
                vectors_list.append(vector)
                
                # Store metadata (everything except embedding)
                meta = {k: v for k, v in chunk.items() if k != 'embedding'}
                metadata_list.append(meta)
        
        # Stack into matrix
        self.vectors = np.vstack(vectors_list)
        self.metadata = metadata_list
        
        print(f"Added {len(self.metadata)} vectors")
        print(f"Vector matrix shape: {self.vectors.shape}")
        print(f"Memory usage: {self.vectors.nbytes / 1024 / 1024:.2f} MB")
        
    def build_index(self):
        """
        Build search index (for future optimization).
        Currently uses brute-force search.
        """
        if self.vectors is None:
            raise ValueError("No vectors added to database")
        
        print("Building index...")
        # For brute-force, no index needed
        # Future: could implement IVF, HNSW, etc.
        self.index_built = True
        print("Index ready (brute-force search)")
        
    def cosine_similarity(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search using cosine similarity.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        # Normalize query vector
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        
        # Compute cosine similarities (dot product since vectors are normalized)
        similarities = np.dot(self.vectors, query_norm)
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Return results
        results = []
        for idx in top_k_indices:
            results.append((self.metadata[idx], float(similarities[idx])))
        
        return results
    
    def euclidean_distance(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search using Euclidean distance (L2).
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (metadata, distance) tuples
        """
        # Compute L2 distances
        distances = np.linalg.norm(self.vectors - query_vector, axis=1)
        
        # Get top k indices (smallest distances)
        top_k_indices = np.argsort(distances)[:k]
        
        # Return results
        results = []
        for idx in top_k_indices:
            results.append((self.metadata[idx], float(distances[idx])))
        
        return results
    
    def dot_product(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search using dot product similarity.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (metadata, dot_product_score) tuples
        """
        # Compute dot products
        scores = np.dot(self.vectors, query_vector)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Return results
        results = []
        for idx in top_k_indices:
            results.append((self.metadata[idx], float(scores[idx])))
        
        return results
    
    def search(self, query_vector: np.ndarray, k: int = 5, method: str = "cosine") -> Tuple[List[Tuple[Dict, float]], float]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            method: Search method ("cosine", "euclidean", "dot_product")
            
        Returns:
            Tuple of (results, search_time)
        """
        if self.vectors is None:
            raise ValueError("No vectors in database")
        
        if not self.index_built:
            self.build_index()
        
        # Convert to numpy if needed
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        
        # Search using specified method
        start_time = time.time()
        
        if method == "cosine":
            results = self.cosine_similarity(query_vector, k)
        elif method == "euclidean":
            results = self.euclidean_distance(query_vector, k)
        elif method == "dot_product":
            results = self.dot_product(query_vector, k)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        search_time = time.time() - start_time
        
        return results, search_time
    
    def save(self, filepath: str):
        """
        Save vector database to disk.
        
        Args:
            filepath: Path to save to
        """
        if self.vectors is None:
            raise ValueError("No vectors to save")
        
        data = {
            'vectors': self.vectors.tolist(),
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved vector database to {filepath}")
        
    def load(self, filepath: str):
        """
        Load vector database from disk.
        
        Args:
            filepath: Path to load from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = np.array(data['vectors'])
        self.metadata = data['metadata']
        self.dimension = data['dimension']
        
        print(f"Loaded vector database from {filepath}")
        print(f"{len(self.metadata)} vectors, dimension={self.dimension}")
    
    def get_stats(self) -> Dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.vectors is None:
            return {"status": "empty"}
        
        return {
            "num_vectors": len(self.metadata),
            "dimension": self.dimension,
            "memory_mb": self.vectors.nbytes / 1024 / 1024,
            "index_built": self.index_built
        }


def main():
    """
    Demonstrate vector database functionality.
    """
    from embedder import Embedder
    
    print("="*70)
    print("STEP 3: Vectors → VectorDB")
    print("="*70)
    
    # Load embeddings
    print("\nLoading embeddings...")
    embedder = Embedder(method="tfidf")
    chunks = embedder.load_embeddings('embeddings_tfidf.json')
    
    # Create and populate vector database
    print("\nCreating vector database...")
    vector_db = VectorDatabase(dimension=384)
    vector_db.add_vectors(chunks)
    vector_db.build_index()
    
    # Save database
    print(f"\nSaving vector database...")
    vector_db.save('vector_database.json')
    
    # Show stats
    print(f"\n{'='*70}")
    print("DATABASE STATISTICS:")
    print(f"{'='*70}")
    stats = vector_db.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nVector database implementation complete!")


if __name__ == "__main__":
    main()