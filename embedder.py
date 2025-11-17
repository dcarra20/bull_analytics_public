"""
Embedding Module
Converts text chunks into vector embeddings using various methods.
"""

import json
import numpy as np
from typing import List, Dict, Optional
import hashlib


class Embedder:
    """
    Generates embeddings for text chunks.
    Supports multiple embedding methods.
    """
    
    def __init__(self, method: str = "tfidf", dimension: int = 384):
        """
        Initialize the embedder.
        
        Args:
            method: Embedding method to use ("tfidf", "api", "simple_bow")
            dimension: Dimension of output vectors
        """
        self.method = method
        self.dimension = dimension
        self.vocabulary = {}
        self.idf_scores = {}
        
    def fit_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from texts (for TF-IDF method).
        
        Args:
            texts: List of text strings to build vocabulary from
        """
        if self.method != "tfidf":
            return
        
        # Build vocabulary
        word_doc_count = {}
        total_docs = len(texts)
        
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        # Create vocabulary (top N words by document frequency)
        sorted_words = sorted(word_doc_count.items(), key=lambda x: x[1], reverse=True)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_words[:self.dimension])}
        
        # Calculate IDF scores
        for word, count in word_doc_count.items():
            if word in self.vocabulary:
                self.idf_scores[word] = np.log(total_docs / (1 + count))
        
        print(f"Built vocabulary with {len(self.vocabulary)} words")
    
    def embed_text_tfidf(self, text: str) -> np.ndarray:
        """
        Create TF-IDF embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        vector = np.zeros(len(self.vocabulary))
        words = text.lower().split()
        word_counts = {}
        
        # Count word frequencies
        for word in words:
            if word in self.vocabulary:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Calculate TF-IDF
        total_words = len(words)
        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / total_words
                idf = self.idf_scores.get(word, 0)
                idx = self.vocabulary[word]
                vector[idx] = tf * idf
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_text_simple_bow(self, text: str) -> np.ndarray:
        """
        Create simple bag-of-words embedding using hashing.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        vector = np.zeros(self.dimension)
        words = text.lower().split()
        
        for word in words:
            # Hash word to dimension index
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            vector[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using selected method.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array embedding
        """
        if self.method == "tfidf":
            return self.embed_text_tfidf(text)
        elif self.method == "simple_bow":
            return self.embed_text_simple_bow(text)
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")
    
    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed all chunks and add embeddings to chunk objects.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Chunks with added 'embedding' field
        """
        # Build vocabulary if using TF-IDF
        if self.method == "tfidf" and not self.vocabulary:
            print("Building vocabulary from chunks...")
            texts = [chunk['text'] for chunk in chunks]
            self.fit_vocabulary(texts)
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        embedded_chunks = []
        
        for i, chunk in enumerate(chunks):
            embedding = self.embed_text(chunk['text'])
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            embedded_chunks.append(chunk_with_embedding)
            
            if (i + 1) % 100 == 0:
                print(f"  Embedded {i + 1}/{len(chunks)} chunks...")
        
        print(f"Embedding complete!")
        return embedded_chunks
    
    def save_embeddings(self, chunks: List[Dict], filepath: str):
        """
        Save chunks with embeddings to file.
        
        Args:
            chunks: List of embedded chunks
            filepath: Path to save to
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_chunks = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            if 'embedding' in chunk_copy:
                chunk_copy['embedding'] = chunk_copy['embedding'].tolist()
            serializable_chunks.append(chunk_copy)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_chunks, f)
        
        print(f"Saved {len(chunks)} embedded chunks to {filepath}")
    
    def load_embeddings(self, filepath: str) -> List[Dict]:
        """
        Load chunks with embeddings from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            List of chunks with embeddings as numpy arrays
        """
        with open(filepath, 'r') as f:
            chunks = json.load(f)
        
        # Convert lists back to numpy arrays
        for chunk in chunks:
            if 'embedding' in chunk:
                chunk['embedding'] = np.array(chunk['embedding'])
        
        print(f"Loaded {len(chunks)} embedded chunks from {filepath}")
        return chunks


def main():
    """
    Demonstrate embedding generation.
    """
    from data_loader import load_companies
    from chunk_maker import TextChunker
    
    print("="*70)
    print("Chunks → Vectors (Embeddings)")
    print("="*70)
    
    # Load and chunk data
    print("\nLoading and chunking companies...")
    companies = load_companies('sp500_companies.csv')
    
    print("\nCreating text chunks...")
    chunker = TextChunker(chunk_size=500, overlap=50, include_stock_data=True)
    chunks = chunker.chunk_companies(companies)
    print(f"   Total chunks: {len(chunks)}")
    
    # Test different embedding methods
    methods = ["simple_bow", "tfidf"]
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing {method.upper()} embedding method")
        print(f"{'='*70}")
        
        embedder = Embedder(method=method, dimension=384)
        embedded_chunks = embedder.embed_chunks(chunks)
        
        # Show example
        example = embedded_chunks[0]
        print(f"\nExample embedding:")
        print(f"  Chunk ID: {example['chunk_id']}")
        print(f"  Company: {example['company_symbol']} - {example['company_name']}")
        print(f"  Text length: {len(example['text'])} chars")
        print(f"  Embedding shape: {example['embedding'].shape}")
        print(f"  Embedding norm: {np.linalg.norm(example['embedding']):.4f}")
        print(f"  Non-zero dimensions: {np.count_nonzero(example['embedding'])}")
        print(f"  Sample values: {example['embedding'][:10]}")
        
        # Save embeddings
        output_file = f'embeddings_{method}.json'
        embedder.save_embeddings(embedded_chunks, output_file)


if __name__ == "__main__":
    main()