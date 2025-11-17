# Bull Analytics

## CS480 Database Systems - Fall 2025

## Phase 3: Vector Pipeline

### Team Information

- **Team Name:** Bull Analytics
- **Dataset:** S&P 500 Company Information with Historical Stock Data
- **Data Sources:**
  - **sp500_companies.csv**: 502 companies with business descriptions and metadata
  - **sp500_stocks.csv**: 1.9M rows of historical price data (2010-2024)
  - **sp500_index.csv**: 2,517 days of S&P 500 index levels (2014-2024)

---

## Project Structure

```
bull_analytics
├── data_loader.py      # Loads company data and enriches with stock statistics
├── chunk_maker.py          # Splits text into chunks with stock performance data (Step 1)
├── embedder.py         # Generates embeddings (Step 2)
├── vector_storage.py     # Stores vectors in database (Step 3)
├── query_engine.py     # Handles query answering (Step 4)
└── README.md

```

## Text → Chunks

### Implementation Location

- **File:** `chunk_maker.py`
- **Class:** `TextChunker`
- **Key Methods:**
  - `create_enriched_text()` - Combines business description with stock performance
  - `chunk_companies()` - Creates chunks with metadata

### Description

The chunking module creates text chunks by combining:

**Company business descriptions and stock performance statistics both dervied from sp500_stocks.csv**

### How to Run

Open VSC Terminal:

```
python chunk_maker.py
```

#### Usage:

```python
from data_loader import load_companies, enrich_companies_with_stock_data
from chunker import TextChunker

# Load and enrich data
companies = load_companies('sp500_companies.csv')
enriched = enrich_companies_with_stock_data(companies, 'sp500_stocks.csv')

# Create chunker (with stock data)
chunker = TextChunker(chunk_size=500, overlap=50, include_stock_data=True)

print(f"Total chunks: {len(chunks)}")
print(f"Chunks with stock data: {sum(1 for c in chunks if c['has_stock_data'])}")
```

**Stock Performance Metrics Included:**

- Trading period (first date → last date)
- Price range (min, max, average)
- Total return percentage
- Volatility (standard deviation of daily returns)
- Average daily volume
- Performance categorization (exceptional/strong/positive/negative)

---

## Chunks → Vectors

### Implementation Location

- **File:** `embedder.py`
- **Class:** `Embedder`
- **Key Methods:**
  - `embed_text()` - Converts single text to vector
  - `embed_chunks()` - Converts all chunks to vectors
  - `save_embeddings()` / `load_embeddings()` - Persistence

### Description

The embedding module converts text chunks into numerical vector representations (embeddings). These vectors capture semantic meaning and enable similarity search.

### Supported Embedding Methods

**1. TF-IDF (Term Frequency-Inverse Document Frequency)**

- Weighs words by importance across documents
- 384-dimensional vectors
- Fast and interpretable
- Good for domain-specific text (financial/business descriptions)

**2. Simple Bag-of-Words (Hash-based)**

- Hash-based word mapping
- 384-dimensional vectors
- Very fast, no training needed
- Good for keyword matching

### How to Run

#### Generate embeddings for demo (first 20 companies):

Open VSC Terminal:

```
python embedder.py
```

#### Usage:

```python
from embedder import Embedder
from chunker import TextChunker
from data_loader import load_companies, enrich_companies_with_stock_data

# Prepare chunks
companies = load_companies('sp500_companies.csv')
enriched = enrich_companies_with_stock_data(companies, 'sp500_stocks.csv')
chunker = TextChunker(chunk_size=500, overlap=50, include_stock_data=True)
chunks = chunker.chunk_companies(enriched)

# Generate embeddings
embedder = Embedder(method="tfidf", dimension=384)
embedded_chunks = embedder.embed_chunks(chunks)

# Save for later use
embedder.save_embeddings(embedded_chunks, 'my_embeddings.json')

# Load embeddings
loaded_chunks = embedder.load_embeddings('my_embeddings.json')
```

### Output Files

- `embeddings_simple_bow.json` - Simple bag-of-words embeddings
- `embeddings_tfidf.json` - TF-IDF embeddings

---

## Vectors → VectorDB

### Implementation Location

- **File:** `vector_storage.py`
- **Class:** `VectorDatabase`
- **Key Methods:**
  - `add_vectors()` - Adds vectors to the database
  - `build_index()` - Builds search index
  - `search()` - Performs similarity search
  - `cosine_similarity()` - Cosine similarity search
  - `euclidean_distance()` - Euclidean distance search
  - `save()` / `load()` - Database persistence

### Description

The vector database stores embeddings in an efficient in-memory structure and provides fast similarity search using multiple algorithms.

### Supported Search Algorithms

**1. Cosine Similarity**

- Measures angle between vectors (0° = identical, 180° = opposite)
- Scale-independent (works with normalized vectors)
- Score range: -1 to 1 (higher = more similar)
- Best for semantic similarity

**2. Euclidean Distance** (L2 Distance)

- Measures straight-line distance in vector space
- Sensitive to vector magnitude
- Lower distance = more similar
- Good for exact matching

**3. Dot Product** (also implemented)

- Simple multiplication and sum
- Fast computation
- Works well with normalized vectors

### How to Run

#### Test the vector database:

```
python vector_storage.py
```

#### Usage:

```python
from vector_store import VectorDatabase
from embedder import Embedder

# Load embeddings
embedder = Embedder(method="tfidf")
chunks = embedder.load_embeddings('embeddings_tfidf.json')

# Create and populate database
vector_db = VectorDatabase(dimension=384)
vector_db.add_vectors(chunks)
vector_db.build_index()

# Search
query_text = "cloud computing software"
query_vector = embedder.embed_text(query_text)
results, search_time = vector_db.search(query_vector, k=5, method="cosine")

# Process results
for metadata, score in results:
    print(f"{metadata['company_symbol']}: {score:.4f}")

# Save database
vector_db.save('my_vector_db.json')

# Load database
new_db = VectorDatabase()
new_db.load('my_vector_db.json')
```

### Example Search Results

**Query**: "payment processing financial services"

| Rank | Company    | Sector             | Score (Cosine) |
| ---- | ---------- | ------------------ | -------------- |
| 1    | Mastercard | Financial Services | 0.3925         |
| 2    | Mastercard | Financial Services | 0.2838         |
| 3    | Mastercard | Financial Services | 0.2663         |
| 4    | Walmart    | Consumer Defensive | 0.2186         |
| 5    | Visa       | Financial Services | 0.1882         |

### Output Files

- `vector_database.json` - Complete vector database with metadata
  - Contains all 87 embedded chunks
  - Vectors stored as lists (JSON-serializable)
  - Includes company metadata (symbol, sector, industry, etc.)

### Advanced Features

**Metadata Filtering**: Results can be filtered by sector, industry, or other attributes:

```python
# Get results
results, _ = vector_db.search(query_vector, k=20)

# Filter by sector
tech_results = [
    (meta, score) for meta, score in results
    if meta['sector'] == 'Technology'
]
```

**Algorithm Comparison**: Test different algorithms on same query:

```python
for method in ["cosine", "euclidean", "dot_product"]:
    results, time = vector_db.search(query_vector, k=5, method=method)
    print(f"{method}: {time*1000:.2f}ms")
```

---

## Query Answering

### Implementation Location

- **File:** `query_engine.py`
- **Class:** `QueryEngine`
- **Key Methods:**
  - `query()` - Main query function that returns ranked results
  - `display_results()` - Formats and displays query results
  - `answer_with_citations()` - Generates answers with source attribution
  - `build_database()` - Builds complete database from scratch
  - `load_database()` - Loads pre-built database

### Description

The query engine is the final component that ties everything together into a complete RAG (Retrieval-Augmented Generation) system. It takes user questions, converts them to vectors, searches the database, and returns ranked, relevant results with citations.

### How It Works

```
User Question
     ↓
1. Text → Vector (Embedder)
     ↓
2. Vector → Similar Vectors (VectorDB Search)
     ↓
3. Retrieve Metadata & Text (Top-k Results)
     ↓
4. Format & Display Results
```

### Key Features

**1. Natural Language Queries**

- Accepts questions in plain English
- Handles diverse query types
- No special syntax required

**2. Ranked Results**

- Returns top-k most relevant chunks
- Similarity scores for each result
- Multiple search algorithms supported

**3. Rich Metadata**

- Company symbol and name
- Sector and industry
- Stock performance indicators
- Source text with context

**4. Citations & Attribution**

- Links results to source companies
- Shows relevant text excerpts
- Tracks unique companies retrieved

### How to Run

#### Basic query example:

```
python query_engine.py
```

### Usage

```python
from query_engine import QueryEngine

# Initialize with pre-built database
engine = QueryEngine(
    vector_db_path='vector_database.json',
    embedder_method='tfidf'
)

# Run a query
result = engine.query(
    question="What companies work in payment processing?",
    k=5,
    method='cosine'
)

# Display results
engine.display_results(result)

# Or get formatted answer with citations
answer = engine.answer_with_citations(
    question="What are some major tech companies?",
    k=3
)
print(answer)
```

### Query Types Supported

**1. Sector/Industry Queries**

- "What companies are in technology?"
- "Find financial services companies"
- "Cloud computing companies"
- "Healthcare sector"

**2. Product/Service Queries**

- "Which companies make [insert product]?"

**3. Performance Queries** (with stock data enrichment)

- "Companies with high stock returns"
- "Low volatility stocks"
- "Strong growth companies"

---

## Dependencies

- Python 3.8+
- Standard library only (csv, typing)

## Data Files Required

- `sp500_companies.csv`
- `sp500_index.csv`
- `sp500_stocks.csv`
