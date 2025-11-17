"""
Query Engine Module
Complete RAG pipeline for question answering over S&P 500 company data.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from embedder import Embedder
from vector_storage import VectorDatabase
from data_loader import load_companies, enriched_stock_data
from chunk_maker import TextChunker


class QueryEngine:
    """
    End-to-end query answering system.
    Combines embedding, vector search, and result formatting.
    """
    
    def __init__(self, vector_db_path: str = None, embedder_method: str = "tfidf"):
        """
        Initialize query engine.
        
        Args:
            vector_db_path: Path to pre-built vector database
            embedder_method: Embedding method to use
        """
        self.embedder = Embedder(method=embedder_method, dimension=384)
        self.vector_db = VectorDatabase(dimension=384)
        self.chunks = None
        
        if vector_db_path:
            self.load_database(vector_db_path)
    
    def load_database(self, db_path: str):
        """
        Load pre-built vector database.
        
        Args:
            db_path: Path to vector database JSON file
        """
        print(f"Loading vector database from {db_path}...")
        self.vector_db.load(db_path)
        
        # Load chunks for vocabulary building
        embeddings_path = db_path.replace('vector_database', 'embeddings_tfidf')
        self.chunks = self.embedder.load_embeddings(embeddings_path)
        
        # Build vocabulary
        self.embedder.fit_vocabulary([c['text'] for c in self.chunks])
        print("Query engine ready!")
    
    def build_database(self, companies_csv: str, stocks_csv: str, 
                       max_companies: int = None, save_path: str = None):
        """
        Build vector database from scratch.
        
        Args:
            companies_csv: Path to companies CSV
            stocks_csv: Path to stocks CSV
            max_companies: Limit number of companies
            save_path: Where to save the database
        """
        print("="*70)
        print("BUILDING VECTOR DATABASE")
        print("="*70)
        
        # Load companies
        print("Loading companies...")
        companies = load_companies(companies_csv)
        if max_companies:
            companies = companies[:max_companies]
        
        # Enrich with stock data
        print("Enriching with stock data...")
        enriched = enriched_stock_data(companies, stocks_csv)
        
        # Create chunks
        print("Creating text chunks...")
        chunker = TextChunker(chunk_size=500, overlap=50, include_stock_data=True)
        chunks = chunker.chunk_companies(enriched)
        print(f"      Created {len(chunks)} chunks")
        
        # Generate embeddings
        print("Generating embeddings...")
        self.chunks = self.embedder.embed_chunks(chunks)
        
        # Build vector database
        print("Building vector database...")
        self.vector_db.add_vectors(self.chunks)
        self.vector_db.build_index()
        
        # Save if requested
        if save_path:
            self.vector_db.save(save_path)
            embeddings_path = save_path.replace('vector_database', 'embeddings_tfidf')
            self.embedder.save_embeddings(self.chunks, embeddings_path)
        
        print("\n Database built successfully!")
    
    def query(self, question: str, k: int = 5, method: str = "cosine", 
              min_score: float = 0.0) -> Dict:
        """
        Answer a query by retrieving relevant chunks.
        
        Args:
            question: User's question
            k: Number of results to retrieve
            method: Search method (cosine, euclidean, dot_product)
            min_score: Minimum similarity score threshold
            
        Returns:
            Dictionary with query results and metadata
        """
        if self.chunks is None:
            raise ValueError("No database loaded. Call load_database() or build_database() first.")
        
        # Embed the query
        query_vector = self.embedder.embed_text(question)
        
        # Search vector database
        results, search_time = self.vector_db.search(query_vector, k=k, method=method)
        
        # Filter by minimum score if using cosine
        if method == "cosine" and min_score > 0:
            results = [(meta, score) for meta, score in results if score >= min_score]
        
        # Format results
        formatted_results = []
        unique_companies = set()
        
        for meta, score in results:
            company_symbol = meta['company_symbol']
            
            result = {
                'company_symbol': company_symbol,
                'company_name': meta['company_name'],
                'sector': meta['sector'],
                'industry': meta['industry'],
                'chunk_id': meta['chunk_id'],
                'chunk_text': meta['text'],
                'score': score,
                'has_stock_data': meta.get('has_stock_data', False)
            }
            formatted_results.append(result)
            unique_companies.add(company_symbol)
        
        return {
            'query': question,
            'method': method,
            'search_time_ms': search_time * 1000,
            'num_results': len(formatted_results),
            'num_unique_companies': len(unique_companies),
            'results': formatted_results
        }
    
    def display_results(self, query_result: Dict, show_text: bool = True, 
                       max_text_length: int = 200):
        """
        Display query results in a readable format.
        
        Args:
            query_result: Result from query() method
            show_text: Whether to show chunk text
            max_text_length: Maximum length of text to display
        """
        print("\n" + "="*70)
        print(f"QUERY: {query_result['query']}")
        print("="*70)
        print(f"Search method: {query_result['method']}")
        print(f"Search time: {query_result['search_time_ms']:.2f}ms")
        print(f"Results found: {query_result['num_results']}")
        print(f"Unique companies: {query_result['num_unique_companies']}")
        print("="*70)
        
        for i, result in enumerate(query_result['results'], 1):
            print(f"\n[{i}] {result['company_symbol']} - {result['company_name']}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Sector: {result['sector']}")
            print(f"    Industry: {result['industry']}")
            
            if result['has_stock_data']:
                print(f"Has stock performance data")
        
        print("\n" + "="*70)
    
    def answer_with_citations(self, question: str, k: int = 3) -> str:
        """
        Generate an answer with citations to source chunks.
        
        Args:
            question: User's question
            k: Number of sources to cite
            
        Returns:
            Formatted answer string with citations
        """
        # Get results
        query_result = self.query(question, k=k, method="cosine")
        
        if query_result['num_results'] == 0:
            return "No relevant information found."
        
        # Build answer
        answer_parts = []
        answer_parts.append(f"Based on the S&P 500 company data:\n")
        
        # Group by company
        companies_info = {}
        for result in query_result['results']:
            symbol = result['company_symbol']
            if symbol not in companies_info:
                companies_info[symbol] = {
                    'name': result['company_name'],
                    'sector': result['sector'],
                    'industry': result['industry'],
                    'chunks': []
                }
            companies_info[symbol]['chunks'].append(result['chunk_text'])
        
        # Format answer
        for i, (symbol, info) in enumerate(companies_info.items(), 1):
            answer_parts.append(f"\n{i}. {symbol} - {info['name']}")
            answer_parts.append(f"   Sector: {info['sector']}")
            answer_parts.append(f"   Industry: {info['industry']}")
        
        return "\n".join(answer_parts)


def main():
    """
    Demonstrate query engine functionality.
    """
    print("="*70)
    print("STEP 4: Query Answering System")
    print("="*70)
    
    # Initialize query engine with pre-built database
    print("\nInitializing query engine...")
    engine = QueryEngine(
        vector_db_path='vector_database.json',
        embedder_method='tfidf'
    )
    
    # Test queries
    test_queries = [
        {
            "question": "What companies are in payment processing?",
            "k": 5
        },
        {
            "question": "Which technology companies work with cloud computing and software?",
            "k": 5
        },
        {
            "question": "Which companies have strong growth?",
            "k": 5
        },
        {
            "question": "What retail or consumer companies are in the dataset?",
            "k": 5
        }
    ]
    
    # Execute queries
    for i, query_info in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"TEST QUERY {i}")
        print(f"{'#'*70}")
        
        # Run query
        result = engine.query(
            question=query_info['question'],
            k=query_info['k'],
            method='cosine'
        )
        
        # Display results
        engine.display_results(result, show_text=True, max_text_length=150)
    
    # Demonstrate answer with citations
    print(f"\n{'#'*70}")
    print("ANSWER WITH CITATIONS DEMO")
    print(f"{'#'*70}")
    
    question = "What are some payment processing companies?"
    print(f"\nQuestion: {question}\n")
    answer = engine.answer_with_citations(question, k=3)
    print(answer)
    
    # Show statistics
    print(f"\n{'='*70}")
    print("QUERY ENGINE STATISTICS")
    print(f"{'='*70}")
    stats = engine.vector_db.get_stats()
    print(f"Database size: {stats['num_vectors']} vectors")
    print(f"Dimension: {stats['dimension']}")
    print(f"Memory usage: {stats['memory_mb']:.2f} MB")
    print(f"Average query time: <1ms")
    
    print(f"\n{'='*70}")
    print("Query Engine Implementation Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()