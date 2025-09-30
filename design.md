PDF Query System - Complete Architecture & Design
🎯 System Overview
A local, chat-based system for querying PDF documents with text and tables using Python, running entirely on your machine with optional GPU acceleration.

🏗️ Architecture Diagram
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
│                    (Streamlit Chat Interface)                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QUERY ORCHESTRATOR                          │
│  • Classifies query type (search/summarize/aggregate/top-K)     │
│  • Routes to appropriate handler                                 │
│  • Manages conversation context                                  │
└────────┬────────────────┬───────────────┬───────────────────────┘
         │                │               │
         ▼                ▼               ▼
┌────────────────┐ ┌─────────────┐ ┌──────────────────┐
│ TEXT SEARCH    │ │ SUMMARIZER  │ │ TABLE AGGREGATOR │
│ • Vector search│ │ • LLM-based │ │ • SQL queries    │
│ • Top-K results│ │ • Context   │ │ • Column matching│
└───────┬────────┘ └──────┬──────┘ └────────┬─────────┘
        │                 │                  │
        └─────────────────┴──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐          │
│  │ Vector Store │  │  SQLite DB  │  │ Cache Layer  │          │
│  │ (ChromaDB)   │  │  (Tables)   │  │ (LRU Cache)  │          │
│  └──────────────┘  └─────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌─────────────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                             │
│                                                                   │
│  PDF Input → PDF Parser → Content Cleaner → Processor → Storage │
│                                                                   │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐        │
│  │ PyMuPDF    │  │ Text Extract │  │ Junk Removal    │        │
│  │ (fitz)     │  │ Table Extract│  │ • Headers       │        │
│  │            │  │ (Camelot)    │  │ • Footers       │        │
│  └────────────┘  └──────────────┘  │ • Page numbers  │        │
│                                     │ • Watermarks    │        │
│                                     └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                         ▲
                         │
┌─────────────────────────────────────────────────────────────────┐
│                    LLM LAYER (Local)                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Ollama (Local LLM Server)                                │  │
│  │ • Model: llama3.2 or mistral                             │  │
│  │ • Used for: Summarization, Q&A, Context understanding    │  │
│  │ • Why: Privacy, no API costs, full control              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Sentence Transformers (Embeddings)                       │  │
│  │ • Model: all-MiniLM-L6-v2                                │  │
│  │ • Used for: Text embeddings for semantic search          │  │
│  │ • Why: Fast, lightweight, runs on CPU                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
📊 Data Flow Diagram
INGESTION FLOW:
═══════════════

PDF File
   │
   ▼
┌─────────────────┐
│ PyMuPDF Parser  │
│ • Extract text  │
│ • Get metadata  │
│ • Page info     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Camelot Parser  │
│ • Detect tables │
│ • Extract data  │
│ • Structure     │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Content Cleaner     │
│ • Remove headers    │
│ • Remove footers    │
│ • Filter junk       │
│ • Normalize text    │
└────────┬────────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌──────────────┐    ┌───────────────┐
│ Text Chunker │    │ Table Parser  │
│ • Split text │    │ • Normalize   │
│ • Overlap    │    │ • Type detect │
└──────┬───────┘    └───────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐    ┌───────────────┐
│ Embeddings   │    │ SQLite DB     │
│ Generator    │    │ • Structured  │
└──────┬───────┘    │   table data  │
       │            └───────────────┘
       ▼
┌──────────────┐
│ ChromaDB     │
│ Vector Store │
└──────────────┘


QUERY FLOW:
═══════════

User Query
   │
   ▼
┌────────────────────┐
│ Query Classifier   │
│ • Search           │
│ • Summarize        │
│ • Aggregate        │
│ • Top-K            │
└─────────┬──────────┘
          │
          ├─────────────┬──────────────┬──────────────┐
          │             │              │              │
          ▼             ▼              ▼              ▼
    ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐
    │ Search  │  │Summarize │  │ Aggregate │  │  Top-K   │
    │ Handler │  │ Handler  │  │  Handler  │  │  Handler │
    └────┬────┘  └─────┬────┘  └─────┬─────┘  └────┬─────┘
         │             │              │             │
         ▼             ▼              ▼             ▼
    ┌─────────────────────────────────────────────────┐
    │           Response Generator                     │
    │  • Combine results                               │
    │  • Format output                                 │
    │  • Add citations                                 │
    └──────────────────┬──────────────────────────────┘
                       │
                       ▼
                  User Interface
🔧 Component Details
1. PDF Ingestion Pipeline
PyMuPDF (fitz)
Purpose: Primary PDF parsing
Why: Fast, reliable, handles complex PDFs
Extracts: Text, images, metadata, page structure
Camelot
Purpose: Table extraction
Why: Best open-source table detection
Extracts: Structured table data with headers
Content Cleaner
Removes:
Page headers/footers
Page numbers
Watermarks
Repeated boilerplate text
Non-content elements
Why: Improves search quality and reduces noise
2. Storage Layer
ChromaDB (Vector Store)
Purpose: Store text embeddings for semantic search
Why:
Runs locally
Fast similarity search
Persistent storage
Built-in filtering
SQLite (Relational DB)
Purpose: Store structured table data
Why:
Serverless, file-based
SQL queries for aggregation
Join across tables
Fast for structured data
3. LLM Layer
Ollama (Local LLM Server)
Models: llama3.2, mistral, phi-3
Used For:
Answering questions with context
Summarizing documents
Understanding query intent
Generating human-readable responses
Why Local:
✅ Privacy (no data leaves your machine)
✅ No API costs
✅ Unlimited usage
✅ Works offline
⚠️ Requires: 8GB+ RAM, better with GPU
Sentence Transformers
Model: all-MiniLM-L6-v2
Used For:
Converting text to embeddings (384-dim vectors)
Semantic similarity search
Why This Model:
Fast on CPU (50ms per encoding)
Small size (80MB)
Good quality for semantic search
No GPU required
4. Query Processing
Query Types & Handlers
1. Text Search (Top-K)

Query: "Find revenue information"
   │
   ▼
Generate embedding → Search ChromaDB → Return top-K chunks
   │
   ▼
LLM synthesizes answer with context
2. Summarization

Query: "Summarize financial performance"
   │
   ▼
Retrieve relevant chunks → Send to LLM → Generate summary
3. Table Aggregation

Query: "Total revenue across all PDFs"
   │
   ▼
Parse query → Match columns → SQL aggregation → Format results
4. Top-K Retrieval

Query: "Top 5 most relevant sections about X"
   │
   ▼
Semantic search → Rank by relevance → Return top-K with sources
🎨 UI Architecture (Streamlit)
┌─────────────────────────────────────────────────────┐
│                  SIDEBAR                             │
│  • Upload PDFs                                       │
│  • View ingested documents                           │
│  • Settings (Top-K, model selection)                 │
│  • Clear conversation                                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                 MAIN CHAT AREA                       │
│                                                       │
│  User: "What's the total revenue?"                   │
│                                                       │
│  Assistant: "Based on 3 documents..."               │
│  📊 [Table showing aggregated data]                  │
│  📄 Sources: doc1.pdf (p.5), doc2.pdf (p.12)        │
│                                                       │
│  ┌────────────────────────────────────────┐         │
│  │ Type your question...          [Send]  │         │
│  └────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────┘
🚀 Processing Pipeline Details
Text Processing
Extract: Get raw text from PDF pages
Clean: Remove headers, footers, page numbers
Chunk: Split into ~500 token chunks with 50 token overlap
Embed: Generate embeddings using Sentence Transformers
Store: Save to ChromaDB with metadata (page, doc_id, etc.)
Table Processing
Detect: Use Camelot to find tables in PDFs
Extract: Get structured data (headers + rows)
Clean: Remove empty rows, normalize headers
Normalize: Standardize column names for matching
Store: Insert into SQLite with foreign keys to documents
Junk Removal Strategy
python
Patterns to Remove:
- Page X of Y
- Document headers/footers (repeated text)
- Watermarks (low-opacity text)
- Navigation elements
- Copyright notices on every page
- URLs and email addresses (optional)
- Excessive whitespace

Detection Methods:
- Frequency analysis (appears on most pages)
- Position-based (always at top/bottom)
- Font size analysis (very small text)
- Regular expressions for patterns
💡 Why This Architecture?
RAG (Retrieval-Augmented Generation) Pattern
Retrieval: Find relevant chunks using vector search
Augmentation: Add retrieved context to LLM prompt
Generation: LLM generates accurate, grounded answers
Benefits:
✅ Accuracy: LLM uses actual document content ✅ Citations: Know which PDF and page ✅ Scalability: Add more PDFs without retraining ✅ Privacy: Everything runs locally ✅ Cost: No API fees

📦 Technology Stack
Component	Library	Purpose
PDF Parsing	PyMuPDF (fitz)	Text extraction
Table Extraction	Camelot-py	Table detection
Vector DB	ChromaDB	Semantic search
Relational DB	SQLite3	Structured data
Embeddings	Sentence Transformers	Text → Vectors
LLM	Ollama	Q&A, Summarization
UI	Streamlit	Chat interface
Chunking	LangChain	Text splitting
SQL	Pandas + SQLAlchemy	Data manipulation
🎯 Query Examples
python
# Text Search
"Find all mentions of revenue growth"
→ Vector search → Top-K chunks → LLM answer

# Summarization  
"Summarize the key findings from Q3 reports"
→ Retrieve Q3 chunks → LLM summarize

# Table Aggregation
"What's the total revenue across all PDFs?"
→ SQL: SELECT SUM(revenue) FROM all_tables

# Top-K Retrieval
"Show me the 5 most relevant sections about AI"
→ Vector search → Rank → Return top-5

# Hybrid Query
"Compare revenue trends and explain key drivers"
→ SQL aggregation + vector search + LLM synthesis
🔄 Conversation Flow
User inputs query
   ↓
System classifies intent
   ↓
Retrieves relevant data (vector/SQL/both)
   ↓
Passes context to LLM
   ↓
LLM generates response
   ↓
System formats with citations
   ↓
Displays in chat UI
   ↓
Maintains conversation context for follow-ups
📈 Scalability Considerations
ChromaDB: Handles millions of vectors efficiently
SQLite: Good for <100GB, can upgrade to PostgreSQL
Chunking: Keeps context windows manageable
Caching: LRU cache for frequent queries
Batch Processing: Process multiple PDFs in parallel
This architecture provides a robust, local, privacy-focused system for querying PDF documents with both unstructured text and structured tables.

