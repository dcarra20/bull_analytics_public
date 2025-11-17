"""
Text Chunking Module
Splits company descriptions into fixed-size chunks with overlap.
Enriches text with stock performance data for better semantic search.
"""

from typing import List, Dict
from data_loader import load_companies, enriched_stock_data


class TextChunker:
    """
    Splits text into chunks of specified size with optional overlap.
    Can enrich text with stock performance data.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, include_stock_data: bool = True):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            include_stock_data: Whether to add stock performance summary to text
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.include_stock_data = include_stock_data
    
    def create_enriched_text(self, company: Dict) -> str:
        """
        Create enriched text combining business description and stock performance.
        
        Args:
            company: Company dictionary with description and stock_stats
            
        Returns:
            Enriched text string
        """
        # Start with business description
        text = company['description']
        
        # Add stock performance summary if available
        if self.include_stock_data and company.get('stock_stats'):
            stats = company['stock_stats']
            
            stock_summary = f"\n\nStock Performance Summary: "
            stock_summary += f"{company['symbol']} has traded from {stats['first_date']} to {stats['last_date']}. "
            stock_summary += f"The stock price ranged from ${stats['min_price']:.2f} to ${stats['max_price']:.2f}, "
            stock_summary += f"with an average price of ${stats['avg_price']:.2f}. "
            stock_summary += f"Over this period, the stock achieved a total return of {stats['total_return_pct']:.1f}%, "
            
            # Add performance categorization
            if stats['total_return_pct'] > 500:
                stock_summary += "demonstrating exceptional growth. "
            elif stats['total_return_pct'] > 100:
                stock_summary += "showing strong growth. "
            elif stats['total_return_pct'] > 0:
                stock_summary += "delivering positive returns. "
            else:
                stock_summary += "experiencing negative returns. "
            
            # Add volatility info
            if stats['volatility'] > 3:
                stock_summary += f"The stock exhibits high volatility at {stats['volatility']:.1f}%. "
            elif stats['volatility'] > 2:
                stock_summary += f"The stock shows moderate volatility at {stats['volatility']:.1f}%. "
            else:
                stock_summary += f"The stock displays low volatility at {stats['volatility']:.1f}%. "
            
            stock_summary += f"Average daily trading volume: {stats['avg_volume']:,.0f} shares."
            
            text += stock_summary
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary if not at end
            if end < len(text):
                # Look for sentence endings (. ! ?)
                last_period = chunk.rfind('. ')
                last_exclaim = chunk.rfind('! ')
                last_question = chunk.rfind('? ')
                
                break_point = max(last_period, last_exclaim, last_question)
                
                # If found a sentence boundary, use it
                if break_point > self.chunk_size * 0.5:  # At least 50% through chunk
                    chunk = chunk[:break_point + 2]  # Include the punctuation and space
            
            chunks.append(chunk.strip())
            
            # Move start position (with overlap)
            if end >= len(text):
                break
            start = end - self.overlap
        
        return chunks
    
    def chunk_companies(self, companies: List[Dict[str, str]]) -> List[Dict]:
        """
        Chunk all company descriptions with optional stock data enrichment.
        
        Args:
            companies: List of company dictionaries
            
        Returns:
            List of chunk dictionaries with metadata
        """
        all_chunks = []
        
        for company in companies:
            # Create enriched text
            full_text = self.create_enriched_text(company)
            
            # Chunk it
            chunks = self.chunk_text(full_text)
            
            # Create chunk objects with metadata
            for i, chunk_text in enumerate(chunks):
                chunk_obj = {
                    'chunk_id': f"{company['symbol']}_chunk_{i}",
                    'company_symbol': company['symbol'],
                    'company_name': company['name'],
                    'sector': company['sector'],
                    'industry': company['industry'],
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'text': chunk_text,
                    'has_stock_data': bool(company.get('stock_stats'))
                }
                all_chunks.append(chunk_obj)
        
        return all_chunks


def main():
    """
    Main function to demonstrate chunking with stock data enrichment.
    """
    # Load companies
    print("Loading companies...")
    companies = load_companies('sp500_companies.csv')
    
    chunker_basic = TextChunker(chunk_size=500, overlap=50, include_stock_data=False)
    chunks_basic = chunker_basic.chunk_companies(companies)
    print(f"Total chunks: {len(chunks_basic)}")
    
    # Show example
    example = chunks_basic[0]
    print(f"\nExample chunk from {example['company_symbol']}:")
    print(f"Text length: {len(example['text'])} chars")
    print(f"Preview: {example['text']}.")
    

if __name__ == "__main__":
    main()