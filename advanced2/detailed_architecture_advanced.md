# PDF Intelligence System - Detailed Architecture & Data Flow

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow: Ingestion Pipeline](#data-flow-ingestion-pipeline)
5. [Data Flow: Query Pipeline](#data-flow-query-pipeline)
6. [Technology Stack](#technology-stack)
7. [Design Decisions & Rationale](#design-decisions--rationale)
8. [Performance Considerations](#performance-considerations)

---

## System Overview

The PDF Intelligence System is a **multi-stage document processing and query answering system** that combines:
- **Advanced PDF parsing** with intelligent junk removal
- **Vector-based semantic search** using ChromaDB
- **Structured data analytics** using SQLite
- **AI-powered natural language understanding** via OpenAI GPT
- **Interactive web interface** built with Streamlit

### Key Capabilities
✅ Semantic text search across documents  
✅ SQL-based table aggregation and analytics  
✅ Intelligent query routing (text vs analytical)  
✅ Automatic junk/boilerplate removal  
✅ Top-K retrieval queries  
✅ Document summarization  
✅ Conversational interface with context  

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)                   │
│                         Query Input / Results Display             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR (PDFIntelligenceSystem)           │
│  • Coordinates all components                                    │
│  • Manages workflow and state                                    │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                               │
    ┌───────────▼──────────┐       ┌───────────▼──────────────┐
    │  INGESTION PIPELINE  │       │   QUERY PIPELINE         │
    │  (Document Loading)  │       │   (Query Processing)     │
    └──────────────────────┘       └──────────────────────────┘
```

---

## Component Details

### 1. AdvancedPDFParser
**Purpose**: Extract and clean text and tables from PDF documents

**Responsibilities**:
- Multi-tool PDF text extraction (PyMuPDF + pdfplumber fallback)
- Advanced table detection (Camelot lattice/stream + pdfplumber)
- Intelligent junk removal using heuristics
- Text cleaning and normalization

**Key Methods**:
```python
extract_text_hybrid(pdf_path)      # Text extraction with fallback
extract_tables_advanced(pdf_path)  # Table extraction with quality filtering
clean_text(text)                   # Remove junk and boilerplate
is_likely_junk(text)               # Heuristic-based junk detection
```

**Junk Removal Heuristics**:
- Page numbers (e.g., "Page 1 of 10")
- Headers/footers (Confidential, Copyright)
- Separator lines (---, ===, ***)
- Excessive whitespace
- Very short lines (< 15 chars)
- High special character ratio (> 40%)
- Repeated character patterns

---

### 2. VectorStore (ChromaDB)
**Purpose**: Store and search document embeddings for semantic retrieval

**Responsibilities**:
- Create vector embeddings using sentence-transformers
- Store embeddings with metadata in ChromaDB
- Perform cosine similarity search
- Support filtered queries

**Key Methods**:
```python
create_or_get_collection(name)     # Initialize or load collection
add_documents(text_chunks)         # Add documents with embeddings
search(query, top_k)               # Semantic similarity search
get_stats()                        # Collection statistics
```

**Embedding Model**: `all-MiniLM-L6-v2`
- Dimension: 384
- Fast inference (runs locally)
- Good quality for general text

**Storage**: Persistent local storage in `./chroma_db/`

---

### 3. TableDatabase (SQLite)
**Purpose**: Store and query structured table data

**Responsibilities**:
- Convert DataFrames to SQL tables
- Store table metadata (source, page, columns)
- Execute SQL queries
- Handle aggregations and JOINs

**Key Methods**:
```python
add_tables(tables)                 # Store extracted tables
execute_query(sql)                 # Run SQL queries
get_schema_info()                  # Get table schemas
get_sample_data(table_name)        # Preview table contents
```

**Storage**: SQLite database file `tables.db`

**Metadata Columns**:
- `_source`: Source PDF filename
- `_page`: Page number in PDF
- `_table_id`: Table index within page
- `_accuracy`: Extraction quality score (Camelot)

---

### 4. IntelligentQueryRouter (OpenAI GPT)
**Purpose**: Classify queries and generate responses

**Responsibilities**:
- Classify query intent (SEARCH, ANALYTICAL, SUMMARY, HYBRID)
- Extract parameters (top_k, aggregation type)
- Generate SQL queries from natural language
- Synthesize final answers
- Maintain conversation history

**Key Methods**:
```python
classify_query(query, schema)      # Determine query type
generate_sql_query(question)       # NL to SQL conversion
generate_response(query, context)  # Create final answer
summarize_content(texts)           # Document summarization
```

**Query Classification Categories**:
- **SEARCH**: Text-based semantic search
- **ANALYTICAL**: SQL queries on tables
- **SUMMARY**: Document summarization
- **HYBRID**: Combination of text + tables

---

## Data Flow: Ingestion Pipeline

### Stage 1: PDF Loading
```
Input: Directory path containing PDF files
↓
Action: Scan directory for *.pdf files
↓
Output: List of PDF file paths
```

**Code Flow**:
```python
pdf_files = Path(directory).glob("*.pdf")
for pdf_file in pdf_files:
    process_pdf(pdf_file)
```

---

### Stage 2: Text Extraction
```
Input: PDF file path
↓
Primary Method: PyMuPDF (fitz)
  ├─ Extract text with layout preservation
  ├─ Try block-level extraction if needed
  └─ Fast, works for most PDFs
↓
Fallback Method: pdfplumber
  ├─ Used if PyMuPDF fails
  └─ Better for complex layouts
↓
Output: List of text chunks with metadata
```

**Data Structure**:
```python
{
    'text': "Cleaned text content...",
    'source': "document.pdf",
    'page': 5,
    'type': 'text',
    'char_count': 1234
}
```

---

### Stage 3: Junk Removal
```
Input: Raw extracted text
↓
Step 1: Split into lines
↓
Step 2: Apply heuristic filters
  ├─ Remove page numbers
  ├─ Remove headers/footers
  ├─ Remove separator lines
  ├─ Filter short lines
  └─ Check special char ratio
↓
Step 3: Join clean lines
↓
Step 4: Normalize whitespace
↓
Output: Clean text (only meaningful content)
```

**Example Transformation**:
```
BEFORE:
=====================================
                Page 5 of 20
           CONFIDENTIAL
=====================================

The quarterly revenue increased...

Copyright © 2024. All Rights Reserved.

AFTER:
The quarterly revenue increased...
```

---

### Stage 4: Table Extraction
```
Input: PDF file path
↓
Primary: Camelot (Lattice mode)
  ├─ Best for bordered tables
  ├─ Uses visual structure
  ├─ Quality filtering (accuracy > 50%)
  └─ Returns DataFrame
↓
Fallback 1: Camelot (Stream mode)
  ├─ For borderless tables
  └─ Uses whitespace detection
↓
Fallback 2: pdfplumber
  ├─ Final attempt
  └─ Different algorithm
↓
Post-processing:
  ├─ Clean headers (use first row if appropriate)
  ├─ Remove empty rows/columns
  ├─ Strip whitespace
  ├─ Convert numeric columns
  └─ Add metadata columns
↓
Output: List of cleaned DataFrames
```

**Quality Checks**:
- Minimum 2 rows (header + data)
- No completely empty tables
- Valid column names
- Accuracy score (if available)

---

### Stage 5: Vector Embedding Creation
```
Input: List of text chunks
↓
Step 1: Extract text content
texts = [chunk['text'] for chunk in chunks]
↓
Step 2: Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)
  ├─ Batch processing (32 per batch)
  ├─ 384-dimensional vectors
  └─ Progress bar shown
↓
Step 3: Store in ChromaDB
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadata_list,
    ids=unique_ids
)
↓
Output: Searchable vector database
```

**Vector Properties**:
- Dimension: 384
- Similarity metric: Cosine
- Index type: HNSW (fast ANN search)

---

### Stage 6: Table Storage
```
Input: List of DataFrames
↓
For each table:
  ├─ Generate unique table name (table_0, table_1, ...)
  ├─ Add metadata columns (_source, _page, _table_id)
  ├─ Create SQL table
  │   └─ df.to_sql(table_name, conn)
  └─ Store schema metadata
↓
Output: SQLite database with all tables
```

**Schema Metadata Stored**:
```python
{
    'name': 'table_0',
    'columns': ['product', 'revenue', 'date'],
    'row_count': 42,
    'source': 'report.pdf',
    'page': 7
}
```

---

## Data Flow: Query Pipeline

### Stage 1: Query Classification
```
Input: User's natural language query
↓
Send to OpenAI GPT with prompt:
  ├─ Query text
  ├─ Available table schemas
  └─ Classification instructions
↓
GPT analyzes and returns JSON:
{
    "primary_intent": "ANALYTICAL",
    "requires_text": false,
    "requires_tables": true,
    "top_k": 5,
    "aggregation_type": "SUM",
    "reasoning": "User wants top 5 by sum"
}
↓
Output: Classification dictionary
```

**Intent Examples**:
- "What are the key findings?" → **SEARCH**
- "Show top 5 products by revenue" → **ANALYTICAL** (top_k=5)
- "Calculate average sales" → **ANALYTICAL** (aggregation=AVG)
- "Summarize the document" → **SUMMARY**
- "Compare text with table data" → **HYBRID**

---

### Stage 2: Data Retrieval

#### Path A: Text Search (requires_text = True)
```
Input: User query + top_k
↓
Step 1: Create query embedding
query_embedding = model.encode([query])
↓
Step 2: Search ChromaDB
results = collection.query(
    query_embeddings=query_embedding,
    n_results=top_k
)
↓
Step 3: Format results
For each result:
  ├─ Extract text
  ├─ Get metadata (source, page)
  ├─ Calculate relevance score
  └─ Truncate for display
↓
Output: List of relevant text chunks with scores
```

**Ranking**: Cosine similarity (1.0 = perfect match, 0.0 = unrelated)

---

#### Path B: Table Query (requires_tables = True)
```
Input: User query + schema info
↓
Step 1: Generate SQL query
Prompt GPT with:
  ├─ User's question
  ├─ Database schema
  ├─ Query classification
  └─ SQL generation instructions
↓
GPT returns SQL query:
"SELECT product_name, SUM(revenue) as total
 FROM table_0
 GROUP BY product_name
 ORDER BY total DESC
 LIMIT 5"
↓
Step 2: Execute SQL on SQLite
result_df = pd.read_sql_query(sql, conn)
↓
Step 3: Format results as markdown table
↓
Output: DataFrame with query results
```

**Error Handling**: If SQL fails, return error message to user

---

#### Path C: Summary (primary_intent = SUMMARY)
```
Input: User query
↓
Step 1: Retrieve relevant documents (top_k=10-15)
search_results = vector_store.search(query, top_k=10)
↓
Step 2: Extract text from results
texts = [result['text'] for result in search_results]
↓
Step 3: Send to GPT for summarization
Prompt: "Summarize key concepts from these excerpts..."
↓
Output: Comprehensive summary
```

---

### Stage 3: Response Generation
```
Input: Query + Retrieved context + Classification
↓
Build context string:
  ├─ Text search results (if any)
  ├─ SQL query + results (if any)
  └─ Source citations
↓
Send to OpenAI GPT:
System prompt: "You are a helpful AI assistant..."
Conversation history: [last 3 exchanges]
User message: Query + Context
↓
GPT synthesizes answer:
  ├─ Direct answer to question
  ├─ Supporting details from context
  ├─ Source citations
  └─ Relevant insights
↓
Update conversation history
↓
Output: Natural language response
```

**Response Structure**:
```markdown
Based on the documents, [answer to question]

Key findings:
- Point 1 (Source: document.pdf, Page 5)
- Point 2 (Source: report.pdf, Page 12)

[Additional insights or context]
```

---

## Technology Stack

### Core Libraries

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Web Interface | Streamlit | 1.29+ | Interactive UI |
| PDF Text | PyMuPDF (fitz) | 1.23+ | Primary text extraction |
| PDF Fallback | pdfplumber | 0.10+ | Backup text extraction |
| Table Extraction | Camelot-py | 0.11+ | Table detection |
| Data Processing | Pandas | 2.1+ | DataFrame operations |
| Vector DB | ChromaDB | 0.4+ | Embedding storage |
| Embeddings | sentence-transformers | 2.2+ | Local embedding model |
| SQL Database | SQLite | 3.x | Built-in with Python |
| AI/LLM | OpenAI | 1.6+ | GPT API client |
| Image Processing | OpenCV | 4.8+ | Table detection support |

---

### System Architecture Pattern

**Pattern**: **Microservices-inspired Modular Monolith**

**Why This Pattern?**
- **Separation of Concerns**: Each class has single responsibility
- **Loose Coupling**: Components interact through well-defined interfaces
- **Testability**: Each component can be tested independently
- **Maintainability**: Easy to update individual components
- **Scalability**: Can be split into services later if needed

**Component Hierarchy**:
```
PDFIntelligenceSystem (Orchestrator)
├── AdvancedPDFParser (PDF Processing)
├── VectorStore (Text Search)
├── TableDatabase (Structured Data)
└── IntelligentQueryRouter (AI/LLM)
```

---

## Design Decisions & Rationale

### 1. Why ChromaDB for Text?
**Decision**: Use ChromaDB instead of Elasticsearch or traditional databases

**Rationale**:
- ✅ **Local-first**: No external server required
- ✅ **Persistent**: Data survives restarts
- ✅ **Fast**: HNSW index for approximate nearest neighbor search
- ✅ **Semantic**: Vector embeddings capture meaning, not just keywords
- ✅ **Simple**: Easy setup with Python
- ❌ **Not distributed**: But we don't need it for local use

**Alternative Considered**: FAISS (rejected - no persistence, requires manual management)

---

### 2. Why SQLite for Tables?
**Decision**: Use SQLite instead of PostgreSQL or CSV files

**Rationale**:
- ✅ **Powerful**: Full SQL support (JOINs, aggregations, subqueries)
- ✅ **Zero-config**: No server setup
- ✅ **Portable**: Single file database
- ✅ **Fast**: Optimized for read-heavy workloads
- ✅ **Reliable**: ACID compliant
- ❌ **Single-user**: But appropriate for local system

**Alternative Considered**: In-memory DataFrames (rejected - limited query capabilities)

---

### 3. Why Local Embeddings (sentence-transformers)?
**Decision**: Generate embeddings locally instead of OpenAI embeddings

**Rationale**:
- ✅ **No API cost**: Free after model download
- ✅ **Fast**: No network latency
- ✅ **Privacy**: Data stays local
- ✅ **No rate limits**: Process unlimited documents
- ✅ **Quality**: Good enough for most use cases
- ❌ **Slightly lower quality**: Than OpenAI's ada-002, but acceptable

**Cost Comparison**:
- OpenAI embeddings: $0.0001 per 1K tokens
- 10,000 documents ≈ $10-20 in API costs
- Local embeddings: $0 (one-time 90MB model download)

---

### 4. Why OpenAI GPT for Query Processing?
**Decision**: Use OpenAI API instead of local LLMs

**Rationale**:
- ✅ **Quality**: Superior understanding and generation
- ✅ **JSON mode**: Structured output for classification
- ✅ **Reasoning**: Better intent classification
- ✅ **SQL generation**: More reliable query generation
- ✅ **Easy integration**: Simple API
- ❌ **API cost**: ~$0.002 per query (affordable)
- ❌ **Network required**: But acceptable trade-off

**Alternative Considered**: Local LLMs like Llama-2 (rejected - quality not as good for complex reasoning)

---

### 5. Why Multi-Stage PDF Parsing?
**Decision**: Use multiple tools with fallbacks

**Rationale**:
- ✅ **Robustness**: If one tool fails, others try
- ✅ **Quality**: Different tools excel at different PDFs
- ✅ **Coverage**: Handles wider variety of PDF formats
- ✅ **Accuracy**: Camelot's lattice mode is excellent for bordered tables
- ❌ **Complexity**: More code, but worth it

**Extraction Strategy**:
1. **PyMuPDF**: Fast, works 80% of time
2. **pdfplumber**: Fallback for complex layouts
3. **Camelot lattice**: Best for bordered tables
4. **Camelot stream**: For borderless tables
5. **pdfplumber tables**: Final fallback

---

### 6. Why Intelligent Query Routing?
**Decision**: Classify queries before processing

**Rationale**:
- ✅ **Efficiency**: Don't search vectors if query needs SQL
- ✅ **Quality**: Use appropriate tool for each query type
- ✅ **User experience**: Return relevant results faster
- ✅ **Flexibility**: Easy to add new query types

**Without Routing**: Would need to search both text and tables for every query (slower, noisier results)

---

## Performance Considerations

### Ingestion Performance

**Bottlenecks**:
1. **PDF Parsing**: ~2-5 seconds per PDF
2. **Embedding Generation**: ~0.1 seconds per document
3. **Table Extraction**: ~1-3 seconds per page with tables

**Optimization Strategies**:
```python
# Batch embedding generation
embeddings = model.encode(texts, batch_size=32)  # Not one-by-one

# Parallel PDF processing (optional)
from multiprocessing import Pool
with Pool(4) as p:
    results = p.map(process_pdf, pdf_files)

# Progress tracking
for i, pdf in enumerate(pdf_files):
    progress = (i + 1) / len(pdf_files)
    update_progress_bar(progress)
```

**Expected Times** (on average laptop):
- 10 PDFs (100 pages): ~2-3 minutes
- 100 PDFs (1000 pages): ~20-30 minutes
- 1000 PDFs: ~3-5 hours

---

### Query Performance

**Target Latency**: < 3 seconds end-to-end

**Breakdown**:
- Vector search: ~50-100ms (local)
- SQL query: ~10-50ms (simple queries)
- OpenAI API call: ~1-2 seconds (network + generation)
- Response rendering: ~50-100ms

**Optimization Strategies**:
```python
# Limit context sent to GPT
context = context[:3000]  # Truncate to save tokens

# Cache embeddings (already done by ChromaDB)

# Use gpt-3.5-turbo instead of gpt-4
# 10x faster, 10x cheaper, usually sufficient

# Limit top_k for faster search
top_k = 5  # Not 50
```

---

### Scalability Limits

**Current System Capacity**:
- Documents: Up to 100,000 chunks (depends on RAM)
- Tables: Unlimited (SQLite handles GBs easily)
- Concurrent users: 1 (Streamlit limitation)

**To Scale Beyond**:
- **More documents**: Use persistent ChromaDB, add pagination
- **Multiple users**: Deploy with authentication, use cloud hosting
- **Distributed**: Migrate to Elasticsearch + PostgreSQL
- **Real-time**: Add document watching and incremental updates

---

### Memory Usage

**Typical Memory Footprint**:
- Base system: ~500 MB
- Embedding model: ~90 MB (loaded once)
- ChromaDB index: ~100-500 MB (10K-50K documents)
- Per query: ~50-100 MB (temporary)

**Total**: ~1-2 GB for normal usage

---

## Summary

This architecture provides:

✅ **Robust PDF processing** with multiple extraction strategies  
✅ **Semantic search** for text-based queries  
✅ **Structured analytics** for table data  
✅ **Intelligent routing** between query types  
✅ **Natural language interface** with conversation context  
✅ **Local-first** with optional cloud AI  
✅ **Production-ready** with error handling and progress tracking  

The system balances **quality, performance, cost, and simplicity** to create a practical document intelligence solution that runs locally while leveraging cloud AI where it adds most value.

---

**Document Version**: 1.0  
**Last Updated**: 2025  
**System**: PDF Intelligence System with Streamlit + ChromaDB + OpenAI
