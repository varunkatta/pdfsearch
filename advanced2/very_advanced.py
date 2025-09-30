"""
Advanced PDF Intelligence System with Streamlit
Features:
- Advanced PDF parsing with junk removal
- ChromaDB for vector storage
- Intelligent query routing (text vs analytical)
- OpenAI GPT or local LLM support
- Top-K, aggregation, summary, and search queries
- Interactive Streamlit interface
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# PDF Processing
import fitz  # PyMuPDF
import camelot
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract

# Data Processing
import pandas as pd
import numpy as np

# Vector Storage & Embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# LLM Integration
from openai import OpenAI

# Web Interface
import streamlit as st


# ============================================================================
# ADVANCED PDF PARSER WITH JUNK REMOVAL
# ============================================================================

class AdvancedPDFParser:
    """Advanced PDF parsing with intelligent junk removal"""
    
    def __init__(self):
        self.junk_patterns = [
            r'Page\s+\d+\s+of\s+\d+',
            r'^\d+$',  # Standalone page numbers
            r'^[ivxIVX]+$',  # Roman numerals
            r'Confidential|CONFIDENTIAL',
            r'Copyright|©|\(c\)\s*\d{4}',
            r'All rights reserved',
            r'Draft|DRAFT',
            r'^-+$|^=+$|^\*+$',  # Separator lines
            r'^\s*$',  # Empty lines
        ]
        self.junk_regex = re.compile('|'.join(self.junk_patterns), re.IGNORECASE)
        
    def is_likely_junk(self, text: str) -> bool:
        """Determine if text is likely boilerplate/junk"""
        text = text.strip()
        
        # Too short
        if len(text) < 15:
            return True
        
        # Matches junk patterns
        if self.junk_regex.search(text):
            return True
        
        # Too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.4:
            return True
        
        # Repeated characters (like "...............")
        if re.search(r'(.)\1{5,}', text):
            return True
        
        return False
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        # Split into lines
        lines = text.split('\n')
        
        # Filter junk lines
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not self.is_likely_junk(line):
                clean_lines.append(line)
        
        # Join and clean
        text = ' '.join(clean_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (optional)
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text.strip()
    
    def extract_text_hybrid(self, pdf_path: str) -> List[Dict]:
        """Hybrid extraction using multiple tools for best results"""
        chunks = []
        
        try:
            # Method 1: PyMuPDF (fast, good for most PDFs)
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout preservation
                text = page.get_text("text")
                
                # Try to get better layout if needed
                if not text or len(text.strip()) < 50:
                    text = page.get_text("blocks")
                    if isinstance(text, list):
                        text = ' '.join([block[4] for block in text if len(block) > 4])
                
                cleaned_text = self.clean_text(text)
                
                if cleaned_text and len(cleaned_text) > 30:
                    chunks.append({
                        'text': cleaned_text,
                        'source': pdf_path,
                        'page': page_num + 1,
                        'type': 'text',
                        'char_count': len(cleaned_text)
                    })
            
            doc.close()
            
        except Exception as e:
            print(f"PyMuPDF failed for {pdf_path}: {e}")
            
            # Fallback: pdfplumber
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            cleaned_text = self.clean_text(text)
                            if cleaned_text and len(cleaned_text) > 30:
                                chunks.append({
                                    'text': cleaned_text,
                                    'source': pdf_path,
                                    'page': page_num + 1,
                                    'type': 'text',
                                    'char_count': len(cleaned_text)
                                })
            except Exception as e2:
                print(f"pdfplumber also failed: {e2}")
        
        return chunks
    
    def extract_tables_advanced(self, pdf_path: str) -> List[pd.DataFrame]:
        """Advanced table extraction with quality filtering"""
        tables = []
        
        # Try Camelot first (best for bordered tables)
        try:
            camelot_tables = camelot.read_pdf(
                pdf_path, 
                pages='all', 
                flavor='lattice',
                suppress_stdout=True
            )
            
            for idx, table in enumerate(camelot_tables):
                if table.accuracy > 50:  # Quality threshold
                    df = table.df
                    
                    # Clean and validate
                    df = self.clean_table(df)
                    
                    if not df.empty and len(df) > 1:
                        df['_source'] = os.path.basename(pdf_path)
                        df['_page'] = table.page
                        df['_table_id'] = idx
                        df['_accuracy'] = table.accuracy
                        tables.append(df)
        
        except Exception as e:
            print(f"Camelot lattice failed, trying stream mode: {e}")
            
            # Try stream mode for non-bordered tables
            try:
                camelot_tables = camelot.read_pdf(
                    pdf_path,
                    pages='all',
                    flavor='stream',
                    suppress_stdout=True
                )
                
                for idx, table in enumerate(camelot_tables):
                    df = table.df
                    df = self.clean_table(df)
                    
                    if not df.empty and len(df) > 1:
                        df['_source'] = os.path.basename(pdf_path)
                        df['_page'] = table.page
                        df['_table_id'] = idx
                        tables.append(df)
            
            except Exception as e2:
                print(f"Camelot stream also failed: {e2}")
        
        # Fallback: pdfplumber
        if not tables:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        page_tables = page.extract_tables()
                        
                        for idx, table in enumerate(page_tables):
                            if table and len(table) > 1:
                                # Create DataFrame
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df = self.clean_table(df)
                                
                                if not df.empty:
                                    df['_source'] = os.path.basename(pdf_path)
                                    df['_page'] = page_num + 1
                                    df['_table_id'] = idx
                                    tables.append(df)
            
            except Exception as e:
                print(f"pdfplumber table extraction failed: {e}")
        
        return tables
    
    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate table data"""
        if df.empty:
            return df
        
        # Use first row as header if it looks like headers
        if df.iloc[0].astype(str).str.len().mean() < 50:
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df[1:]
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)
        df = df.dropna(how='all', axis=1)
        
        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Remove rows that are all empty strings
        df = df[~(df.astype(str).apply(lambda x: x.str.strip() == '').all(axis=1))]
        
        # Try to convert numeric columns
        for col in df.columns:
            if col.startswith('_'):
                continue
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
        
        return df.reset_index(drop=True)
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict], List[pd.DataFrame]]:
        """Process single PDF with advanced parsing"""
        print(f"📄 Processing: {os.path.basename(pdf_path)}")
        
        text_chunks = self.extract_text_hybrid(pdf_path)
        tables = self.extract_tables_advanced(pdf_path)
        
        print(f"   ✓ Extracted {len(text_chunks)} text chunks, {len(tables)} tables")
        
        return text_chunks, tables


# ============================================================================
# VECTOR STORE WITH CHROMADB
# ============================================================================

class VectorStore:
    """Manages vector embeddings using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✓ Initialized vector store at: {persist_directory}")
        
    def create_or_get_collection(self, name: str = "pdf_documents"):
        """Create or retrieve collection"""
        try:
            self.collection = self.client.get_collection(name)
            print(f"✓ Loaded existing collection: {name} ({self.collection.count()} documents)")
        except:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"✓ Created new collection: {name}")
    
    def add_documents(self, text_chunks: List[Dict]):
        """Add documents to vector store"""
        if not text_chunks:
            print("⚠ No text chunks to add")
            return
        
        documents = [chunk['text'] for chunk in text_chunks]
        metadatas = [{
            'source': chunk['source'],
            'page': chunk['page'],
            'type': chunk['type'],
            'char_count': chunk.get('char_count', 0)
        } for chunk in text_chunks]
        ids = [f"doc_{i}_{hash(chunk['text'])}" for i, chunk in enumerate(text_chunks)]
        
        # Create embeddings in batches
        print(f"🔄 Creating embeddings for {len(documents)} documents...")
        embeddings = self.embedding_model.encode(
            documents, 
            show_progress_bar=True,
            batch_size=32
        ).tolist()
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"✓ Added {len(documents)} documents to vector store")
    
    def search(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant documents"""
        # Ensure top_k is always a valid integer
        if top_k is None or not isinstance(top_k, int):
            top_k = 5
        
        # Ensure top_k is at least 1
        top_k = max(1, top_k)
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=int(top_k),  # Explicitly cast to int
            where=filter_dict
        )

    def searchOld(self, query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=filter_dict
        )
        
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'relevance_score': 1 - results['distances'][0][i]
                })
        
        return formatted_results
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        if self.collection:
            return {
                'total_documents': self.collection.count(),
                'collection_name': self.collection.name
            }
        return {}


# ============================================================================
# TABLE DATABASE WITH SQLITE
# ============================================================================

class TableDatabase:
    """Manages structured table data using SQLite"""
    
    def __init__(self, db_path: str = "tables.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.tables_metadata = []
        print(f"✓ Initialized table database: {db_path}")
        
    def add_tables(self, tables: List[pd.DataFrame]):
        """Add all tables to database"""
        if not tables:
            print("⚠ No tables to add")
            return
        
        for i, df in enumerate(tables):
            if not df.empty:
                table_name = f"table_{i}"
                
                try:
                    df.to_sql(table_name, self.conn, if_exists='replace', index=False)
                    
                    # Store metadata
                    metadata = {
                        'name': table_name,
                        'columns': [col for col in df.columns if not col.startswith('_')],
                        'all_columns': list(df.columns),
                        'row_count': len(df),
                        'source': df['_source'].iloc[0] if '_source' in df.columns else 'unknown',
                        'page': int(df['_page'].iloc[0]) if '_page' in df.columns else 0
                    }
                    self.tables_metadata.append(metadata)
                    
                except Exception as e:
                    print(f"⚠ Error adding table {table_name}: {e}")
        
        print(f"✓ Added {len(self.tables_metadata)} tables to database")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query"""
        try:
            result = pd.read_sql_query(query, self.conn)
            return result
        except Exception as e:
            error_df = pd.DataFrame({
                'error': [str(e)],
                'query': [query]
            })
            return error_df
    
    def get_schema_info(self) -> str:
        """Get detailed schema information"""
        if not self.tables_metadata:
            return "No tables available in database."
        
        info = "📊 **Available Tables:**\n\n"
        
        for metadata in self.tables_metadata:
            info += f"**Table: {metadata['name']}**\n"
            info += f"- Source: {metadata['source']}, Page: {metadata['page']}\n"
            info += f"- Rows: {metadata['row_count']}\n"
            info += f"- Columns: {', '.join(metadata['columns'])}\n\n"
        
        return info
    
    def get_sample_data(self, table_name: str, limit: int = 3) -> pd.DataFrame:
        """Get sample rows from table"""
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_tables': len(self.tables_metadata),
            'total_rows': sum(m['row_count'] for m in self.tables_metadata)
        }
    
    def close(self):
        self.conn.close()


# ============================================================================
# INTELLIGENT QUERY ROUTER
# ============================================================================

class IntelligentQueryRouter:
    """Routes queries to appropriate handler using LLM"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.conversation_history = []
        
    def classify_query(self, query: str, schema_info: str) -> Dict[str, Any]:
        """Classify query intent and extract parameters"""
        
        classification_prompt = f"""Analyze this user query and classify it into one or more categories.
Also extract any relevant parameters like top_k values or aggregation needs.

Query: "{query}"

Available table schema:
{schema_info[:500]}

Provide a JSON response with:
{{
    "primary_intent": "one of: SEARCH, ANALYTICAL, SUMMARY, HYBRID",
    "secondary_intents": ["list of additional intents if any"],
    "requires_tables": true/false,
    "requires_text": true/false,
    "top_k": number or null,
    "aggregation_type": "one of: SUM, AVG, COUNT, MAX, MIN, GROUP_BY, null",
    "reasoning": "brief explanation"
}}

Intent descriptions:
- SEARCH: Find specific information in documents (semantic search)
- ANALYTICAL: Requires SQL queries, aggregations, calculations on table data
- SUMMARY: Needs summarization of concepts or documents
- HYBRID: Requires both text search and analytical queries

Examples:
"What are the key findings?" -> SEARCH or SUMMARY
"Show me top 5 products by revenue" -> ANALYTICAL (top_k=5)
"What is the average sales?" -> ANALYTICAL (aggregation_type=AVG)
"Compare methodology with results" -> HYBRID"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a query classification expert. Respond only with valid JSON."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            classification = json.loads(response.choices[0].message.content)
            return classification
            
        except Exception as e:
            print(f"Classification error: {e}")
            # Fallback classification
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['top', 'highest', 'largest', 'best']):
                return {
                    "primary_intent": "ANALYTICAL",
                    "requires_tables": True,
                    "requires_text": False,
                    "top_k": 5
                }
            elif any(word in query_lower for word in ['total', 'sum', 'average', 'count', 'calculate']):
                return {
                    "primary_intent": "ANALYTICAL",
                    "requires_tables": True,
                    "requires_text": False
                }
            elif any(word in query_lower for word in ['summarize', 'summary', 'overview', 'key points']):
                return {
                    "primary_intent": "SUMMARY",
                    "requires_text": True,
                    "requires_tables": False
                }
            else:
                return {
                    "primary_intent": "SEARCH",
                    "requires_text": True,
                    "requires_tables": False
                }
    
    def generate_sql_query(self, question: str, schema_info: str, classification: Dict) -> str:
        """Generate SQL query from natural language"""
        
        sql_prompt = f"""Generate a SQL query to answer this question.

Question: {question}

Database Schema:
{schema_info}

Classification: {classification.get('primary_intent')}
Top-K: {classification.get('top_k', 'N/A')}
Aggregation: {classification.get('aggregation_type', 'N/A')}

Requirements:
- Return ONLY valid SQL, no explanations
- Use proper SQL syntax for SQLite
- Include LIMIT clause if top_k is specified
- Handle NULL values appropriately
- Use appropriate JOINs if multiple tables needed

SQL Query:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Generate only valid SQLite queries."},
                    {"role": "user", "content": sql_prompt}
                ],
                temperature=0
            )
            
            sql_query = response.choices[0].message.content.strip()
            # Clean up
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            return sql_query
            
        except Exception as e:
            print(f"SQL generation error: {e}")
            return ""
    
    def generate_response(self, query: str, context: str, classification: Dict) -> str:
        """Generate final natural language response"""
        
        system_message = """You are a helpful AI assistant analyzing PDF documents.
You provide clear, accurate, and well-structured answers based on the data provided.
Always cite sources when possible and be honest about limitations."""

        user_message = f"""Question: {query}

Query Type: {classification.get('primary_intent', 'SEARCH')}

Retrieved Data/Context:
{context}

Please provide a clear, well-structured answer. Include:
1. Direct answer to the question
2. Supporting details from the data
3. Source references where applicable
4. Any relevant insights or patterns

Keep the response concise but comprehensive."""

        try:
            messages = [{"role": "system", "content": system_message}]
            
            # Add recent conversation history
            messages.extend(self.conversation_history[-4:])
            
            # Add current query
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_response = response.choices[0].message.content
            
            # Update history
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def summarize_content(self, texts: List[str]) -> str:
        """Generate summary from multiple text chunks"""
        
        combined_text = "\n\n".join(texts[:15])  # Limit for token budget
        
        summary_prompt = f"""Analyze and summarize the key concepts, findings, and insights from these document excerpts.

Document Content:
{combined_text}

Provide a well-structured summary with:
1. Main themes and concepts
2. Key findings or conclusions
3. Important data points or facts
4. Overall insights

Use clear headings and bullet points for readability."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing and summarizing documents."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"


# ============================================================================
# MAIN SYSTEM ORCHESTRATOR
# ============================================================================

class PDFIntelligenceSystem:
    """Main system integrating all components"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.parser = AdvancedPDFParser()
        self.vector_store = VectorStore()
        self.table_db = TableDatabase()
        self.query_router = IntelligentQueryRouter(openai_api_key, model)
        self.initialized = False
        self.stats = {}
        
    def ingest_documents(self, pdf_directory: str, progress_callback=None):
        """Ingest PDFs from directory with progress tracking"""
        
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        if not pdf_files:
            return {"status": "error", "message": f"No PDF files found in {pdf_directory}"}
        
        all_text_chunks = []
        all_tables = []
        
        total_files = len(pdf_files)
        
        for idx, pdf_file in enumerate(pdf_files):
            if progress_callback:
                progress_callback(idx + 1, total_files, str(pdf_file.name))
            
            try:
                text_chunks, tables = self.parser.process_pdf(str(pdf_file))
                all_text_chunks.extend(text_chunks)
                all_tables.extend(tables)
            except Exception as e:
                print(f"⚠ Error processing {pdf_file.name}: {e}")
        
        # Setup vector store
        self.vector_store.create_or_get_collection()
        self.vector_store.add_documents(all_text_chunks)
        
        # Setup table database
        self.table_db.add_tables(all_tables)
        
        self.initialized = True
        
        # Collect stats
        self.stats = {
            'total_pdfs': total_files,
            'text_chunks': len(all_text_chunks),
            'tables': len(all_tables),
            'vector_stats': self.vector_store.get_stats(),
            'table_stats': self.table_db.get_stats()
        }
        
        return {
            "status": "success",
            "message": f"Successfully processed {total_files} PDFs",
            "stats": self.stats
        }
    
    def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process user query with intelligent routing"""
        
        if not self.initialized:
            return {
                "status": "error",
                "message": "System not initialized. Please ingest documents first.",
                "answer": ""
            }
        
        try:
            # Get schema information
            schema_info = self.table_db.get_schema_info()
            
            # Classify query
            classification = self.query_router.classify_query(user_query, schema_info)
            
            primary_intent = classification.get('primary_intent', 'SEARCH')
            extracted_top_k = classification.get('top_k', top_k)
            
            # FIX: Ensure extracted_top_k is valid integer
            if extracted_top_k is None or not isinstance(extracted_top_k, (int, float)):
                extracted_top_k = top_k
            extracted_top_k = max(1, int(extracted_top_k))  # At least 1, cast to int
            
            context_parts = []
            sql_query = None
            
            # Handle based on intent
            if classification.get('requires_text', True):
                # Perform text search
                search_results = self.vector_store.search(user_query, top_k=extracted_top_k)
                
                if search_results:
                    context_parts.append("📄 **Relevant Text Content:**\n")
                    for idx, result in enumerate(search_results, 1):
                        context_parts.append(
                            f"\n[{idx}] Source: {result['metadata']['source']}, "
                            f"Page: {result['metadata']['page']}, "
                            f"Relevance: {result['relevance_score']:.2f}\n"
                            f"{result['text'][:300]}...\n"
                        )
            
            if classification.get('requires_tables', False):
                # Generate and execute SQL
                sql_query = self.query_router.generate_sql_query(
                    user_query, 
                    schema_info, 
                    classification
                )
                
                if sql_query:
                    result_df = self.table_db.execute_query(sql_query)
                    
                    context_parts.append("\n\n📊 **Query Results:**\n")
                    context_parts.append(f"SQL: `{sql_query}`\n\n")
                    
                    if 'error' in result_df.columns:
                        context_parts.append(f"⚠ Error: {result_df['error'].iloc[0]}\n")
                    else:
                        context_parts.append(result_df.to_markdown(index=False))
            
            # Handle summarization
            if primary_intent == "SUMMARY":
                search_results = self.vector_store.search(user_query, top_k=min(10, extracted_top_k * 2))
                texts = [r['text'] for r in search_results]
                summary = self.query_router.summarize_content(texts)
                
                return {
                    "status": "success",
                    "intent": classification,
                    "answer": summary,
                    "context": "\n".join(context_parts),
                    "sql_query": None
                }
            
            # Generate final response
            context = "\n".join(context_parts)
            
            answer = self.query_router.generate_response(
                user_query,
                context,
                classification
            )
            
            return {
                "status": "success",
                "intent": classification,
                "answer": answer,
                "context": context,
                "sql_query": sql_query
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in query processing: {error_trace}")
            
            return {
                "status": "error",
                "message": str(e),
                "answer": f"An error occurred: {str(e)}\n\nPlease try rephrasing your question."
            }

    def queryOld(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """Process user query with intelligent routing"""
        
        if not self.initialized:
            return {
                "status": "error",
                "message": "System not initialized. Please ingest documents first.",
                "answer": ""
            }
        
        try:
            # Get schema information
            schema_info = self.table_db.get_schema_info()
            
            # Classify query
            classification = self.query_router.classify_query(user_query, schema_info)
            
            primary_intent = classification.get('primary_intent', 'SEARCH')
            extracted_top_k = classification.get('top_k', top_k)
            
            context_parts = []
            
            # Handle based on intent
            if classification.get('requires_text', True):
                # Perform text search
                search_results = self.vector_store.search(user_query, top_k=extracted_top_k)
                
                if search_results:
                    context_parts.append("📄 **Relevant Text Content:**\n")
                    for idx, result in enumerate(search_results, 1):
                        context_parts.append(
                            f"\n[{idx}] Source: {result['metadata']['source']}, "
                            f"Page: {result['metadata']['page']}, "
                            f"Relevance: {result['relevance_score']:.2f}\n"
                            f"{result['text'][:300]}...\n"
                        )
            
            if classification.get('requires_tables', False):
                # Generate and execute SQL
                sql_query = self.query_router.generate_sql_query(
                    user_query, 
                    schema_info, 
                    classification
                )
                
                if sql_query:
                    result_df = self.table_db.execute_query(sql_query)
                    
                    context_parts.append("\n\n📊 **Query Results:**\n")
                    context_parts.append(f"SQL: `{sql_query}`\n\n")
                    
                    if 'error' in result_df.columns:
                        context_parts.append(f"⚠ Error: {result_df['error'].iloc[0]}\n")
                    else:
                        context_parts.append(result_df.to_markdown(index=False))
            
            # Handle summarization
            if primary_intent == "SUMMARY":
                search_results = self.vector_store.search(user_query, top_k=10)
                texts = [r['text'] for r in search_results]
                summary = self.query_router.summarize_content(texts)
                
                return {
                    "status": "success",
                    "intent": classification,
                    "answer": summary,
                    "context": "\n".join(context_parts),
                    "sql_query": None
                }
            
            # Generate final response
            context = "\n".join(context_parts)
            
            answer = self.query_router.generate_response(
                user_query,
                context,
                classification
            )
            
            return {
                "status": "success",
                "intent": classification,
                "answer": answer,
                "context": context,
                "sql_query": sql_query if classification.get('requires_tables') else None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "answer": f"An error occurred: {str(e)}"
            }
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return self.stats


# ============================================================================
# STREAMLIT INTERFACE
# ============================================================================

def create_streamlit_app():
    """Create Streamlit interface"""
    
    st.set_page_config(
        page_title="PDF Intelligence System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .stats-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📚 PDF Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown("**Advanced PDF analysis with AI-powered search, analytics, and summarization**")
    
    # Initialize session state
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key"
        )
        
        model_choice = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            help="Select the OpenAI model to use"
        )
        
        st.markdown("---")
        
        st.header("📁 Document Ingestion")
        
        pdf_directory = st.text_input(
            "PDF Directory Path",
            value="./pdf_documents",
            help="Path to directory containing PDF files"
        )
        
        if st.button("🚀 Ingest Documents", type="primary"):
            if not api_key:
                st.error("Please provide an OpenAI API key")
            elif not os.path.exists(pdf_directory):
                st.error(f"Directory not found: {pdf_directory}")
            else:
                with st.spinner("Processing PDFs..."):
                    # Initialize system
                    st.session_state.system = PDFIntelligenceSystem(api_key, model_choice)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(current, total, filename):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {current}/{total}: {filename}")
                    
                    # Ingest
                    result = st.session_state.system.ingest_documents(
                        pdf_directory,
                        progress_callback=update_progress
                    )
                    
                    if result['status'] == 'success':
                        st.session_state.initialized = True
                        st.success("✅ Documents ingested successfully!")
                        
                        stats = result['stats']
                        st.markdown("### 📊 Statistics")
                        st.json(stats)
                    else:
                        st.error(f"Error: {result['message']}")
        
        if st.session_state.initialized:
            st.markdown("---")
            st.success("✓ System Ready")
            
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
    
    # Main area
    if not st.session_state.initialized:
        st.info("👈 Please configure the system and ingest documents using the sidebar")
        
        # Show example queries
        st.markdown("### 📝 Example Queries")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Search Queries:**")
            st.markdown("""
            - What are the key findings about climate change?
            - Find all mentions of data privacy
            - What methodologies are described?
            """)
            
            st.markdown("**📊 Analytical Queries:**")
            st.markdown("""
            - What is the total revenue across all tables?
            - Show me the top 5 products by sales
            - Calculate average scores by category
            """)
        
        with col2:
            st.markdown("**📄 Summary Queries:**")
            st.markdown("""
            - Summarize the key concepts
            - What are the main conclusions?
            - Give me an overview of the findings
            """)
            
            st.markdown("**🔀 Hybrid Queries:**")
            st.markdown("""
            - Compare methodology with results
            - How do qualitative findings relate to data?
            - What patterns emerge from text and tables?
            """)
    
    else:
        # Chat interface
        st.markdown("### 💬 Chat with Your Documents")
        
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['role'] == 'user':
                st.markdown(f"**🧑 You:** {message['content']}")
            else:
                st.markdown(f"**🤖 Assistant:** {message['content']}")
                
                if 'metadata' in message:
                    with st.expander("📋 View Details"):
                        if message['metadata'].get('intent'):
                            st.markdown(f"**Intent:** {message['metadata']['intent'].get('primary_intent')}")
                        
                        if message['metadata'].get('sql_query'):
                            st.code(message['metadata']['sql_query'], language='sql')
                        
                        if message['metadata'].get('context'):
                            st.markdown("**Retrieved Context:**")
                            st.text(message['metadata']['context'][:500] + "...")
            
            st.markdown("---")
        
        # Query input
        with st.form(key='query_form', clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_query = st.text_input(
                    "Ask a question",
                    placeholder="e.g., What are the top 5 products by revenue?",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("🚀 Ask", type="primary")
        
        if submit_button and user_query:
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_query
            })
            
            with st.spinner("🤔 Thinking..."):
                # Process query
                result = st.session_state.system.query(user_query)
                
                # Add assistant message
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'metadata': {
                        'intent': result.get('intent'),
                        'sql_query': result.get('sql_query'),
                        'context': result.get('context')
                    }
                })
            
            st.rerun()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    create_streamlit_app()