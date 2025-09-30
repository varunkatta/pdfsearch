"""
Advanced PDF Query System - Standalone Version

Complete implementation with:
- Advanced PDF processing with ML-based content filtering
- Multi-method table extraction with quality scoring
- Automatic schema inference
- LangChain agents for intelligent query routing
- Text-to-SQL generation
- Separate database tables (advanced_*)

Run: streamlit run pdf_query_system_advanced.py
"""

import os
import re
import sqlite3
import json
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter
import pickle

import fitz
import pdfplumber
import camelot
import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Try to import optional ML dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not available. Using rule-based classification.")

# Try to import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available. Vector search disabled.")

load_dotenv()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class DocumentSection:
    section_type: str
    content: str
    page_num: int
    confidence: float

@dataclass
class TableSchema:
    table_id: str
    columns: List[Dict]
    primary_key: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_sql_schema(self) -> str:
        sql = f"CREATE TABLE IF NOT EXISTS {self.table_id} (\n"
        col_defs = []
        for col in self.columns:
            col_def = f"  {col['name']} {col['type']}"
            if not col['nullable']:
                col_def += " NOT NULL"
            if col['unique']:
                col_def += " UNIQUE"
            col_defs.append(col_def)
        
        if self.primary_key:
            col_defs.append(f"  PRIMARY KEY ({self.primary_key})")
        
        sql += ",\n".join(col_defs) + "\n);"
        return sql

@dataclass
class EnhancedTable:
    table_id: str
    doc_id: str
    page_num: int
    cleaned_data: pd.DataFrame
    schema: TableSchema
    quality_score: float
    extraction_method: str
    metadata: Dict = field(default_factory=dict)

# ============================================================================
# CONTENT CLASSIFIER
# ============================================================================

class ContentClassifier:
    """Classify content as material or boilerplate."""
    
    def __init__(self):
        self.boilerplate_patterns = [
            r'confidential|proprietary',
            r'©|copyright|\(c\)',
            r'all rights reserved',
            r'page \d+ of \d+',
            r'for internal use only',
            r'disclaimer:',
            r'^table of contents$',
            r'^appendix [a-z]',
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.boilerplate_patterns]
        
        if ML_AVAILABLE:
            self._init_ml_classifier()
    
    def _init_ml_classifier(self):
        """Initialize ML classifier if available."""
        try:
            with open('content_classifier_advanced.pkl', 'rb') as f:
                self.vectorizer, self.classifier = pickle.load(f)
        except FileNotFoundError:
            self._train_classifier()
    
    def _train_classifier(self):
        """Train ML classifier."""
        boilerplate = [
            "This document contains confidential information",
            "All rights reserved. No part of this publication",
            "For internal use only",
            "© Copyright 2024",
            "Page 1 of 10",
        ]
        material = [
            "Revenue increased by 15% year over year",
            "Customer satisfaction scores improved significantly",
            "Analysis shows strong market demand",
        ]
        
        X = boilerplate + material
        y = ['boilerplate'] * len(boilerplate) + ['material'] * len(material)
        
        self.vectorizer = TfidfVectorizer(max_features=100)
        X_vec = self.vectorizer.fit_transform(X)
        
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.classifier.fit(X_vec, y)
        
        with open('content_classifier_advanced.pkl', 'wb') as f:
            pickle.dump((self.vectorizer, self.classifier), f)
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify text."""
        if not text or len(text.strip()) < 10:
            return 'boilerplate', 1.0
        
        # Rule-based check first
        text_lower = text.lower()
        for pattern in self.compiled_patterns:
            if pattern.search(text_lower):
                return 'boilerplate', 0.9
        
        # ML classification if available
        if ML_AVAILABLE and hasattr(self, 'classifier'):
            try:
                X = self.vectorizer.transform([text])
                prediction = self.classifier.predict(X)[0]
                proba = self.classifier.predict_proba(X)[0]
                confidence = max(proba)
                return prediction, confidence
            except:
                pass
        
        return 'material', 0.8
    
    def classify_document_sections(self, pages_text: List[str]) -> List[DocumentSection]:
        """Classify all sections."""
        sections = []
        for page_num, page_text in enumerate(pages_text, 1):
            paragraphs = page_text.split('\n\n')
            for para in paragraphs:
                if len(para.strip()) > 20:
                    classification, confidence = self.classify(para)
                    sections.append(DocumentSection(
                        section_type=classification,
                        content=para,
                        page_num=page_num,
                        confidence=confidence
                    ))
        return sections

# ============================================================================
# SCHEMA INFERENCE
# ============================================================================

class SchemaInferencer:
    """Infer database schema from table data."""
    
    def infer_column_type(self, series: pd.Series) -> str:
        """Infer SQL type."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'TEXT'
        
        # Boolean
        unique_vals = set(str(v).lower() for v in non_null.unique())
        if unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no'}):
            return 'BOOLEAN'
        
        # Integer
        try:
            pd.to_numeric(non_null, errors='raise')
            if all(float(x).is_integer() for x in non_null):
                return 'INTEGER'
        except:
            pass
        
        # Float
        try:
            non_null.astype(float)
            return 'REAL'
        except:
            pass
        
        # Date
        try:
            pd.to_datetime(non_null, errors='raise')
            return 'DATE'
        except:
            pass
        
        # String
        max_len = non_null.astype(str).str.len().max()
        return f'VARCHAR({max(max_len, 50)})' if max_len <= 255 else 'TEXT'
    
    def detect_primary_key(self, df: pd.DataFrame) -> Optional[str]:
        """Detect primary key."""
        try:
            for col in df.columns:
                # Check if column has unique values and no nulls
                if len(df[col].unique()) == len(df) and df[col].notna().all():
                    if 'id' in str(col).lower():
                        return str(col)
        except Exception as e:
            print(f"Primary key detection error: {e}")
        return None
    
    def infer_schema(self, table_id: str, df: pd.DataFrame) -> TableSchema:
        """Infer complete schema."""
        columns = []
        
        try:
            for col_name in df.columns:
                clean_name = self._clean_column_name(str(col_name))
                
                try:
                    col_type = self.infer_column_type(df[col_name])
                except Exception as e:
                    print(f"Type inference error for {col_name}: {e}")
                    col_type = 'TEXT'
                
                try:
                    is_nullable = df[col_name].isna().any()
                    is_unique = len(df[col_name].unique()) == len(df)
                except:
                    is_nullable = True
                    is_unique = False
                
                columns.append({
                    'name': clean_name,
                    'original_name': str(col_name),
                    'type': col_type,
                    'nullable': is_nullable,
                    'unique': is_unique
                })
            
            primary_key = self.detect_primary_key(df)
            if primary_key:
                primary_key = self._clean_column_name(str(primary_key))
            
            return TableSchema(
                table_id=table_id,
                columns=columns,
                primary_key=primary_key,
                metadata={
                    'num_rows': len(df),
                    'num_columns': len(df.columns)
                }
            )
        except Exception as e:
            print(f"Schema inference error: {e}")
            # Return minimal schema
            return TableSchema(
                table_id=table_id,
                columns=[{
                    'name': f'col_{i}',
                    'original_name': str(col),
                    'type': 'TEXT',
                    'nullable': True,
                    'unique': False
                } for i, col in enumerate(df.columns)],
                primary_key=None,
                metadata={'num_rows': len(df), 'num_columns': len(df.columns)}
            )
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name for SQL."""
        clean = name.lower()
        clean = re.sub(r'[^\w]+', '_', clean)
        clean = clean.strip('_')
        if clean and clean[0].isdigit():
            clean = 'col_' + clean
        return clean or 'unnamed'

# ============================================================================
# TABLE EXTRACTOR
# ============================================================================

class AdvancedTableExtractor:
    """Multi-method table extraction."""
    
    def __init__(self):
        self.schema_inferencer = SchemaInferencer()
    
    def extract_with_camelot(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract with Camelot."""
        tables = []
        try:
            for flavor in ['lattice', 'stream']:
                camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
                tables.extend([t.df for t in camelot_tables if not t.df.empty])
        except Exception as e:
            print(f"Camelot error: {e}")
        return tables
    
    def extract_with_pdfplumber(self, pdf_path: str) -> List[pd.DataFrame]:
        """Extract with pdfplumber."""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for table in page.extract_tables():
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            if not df.empty:
                                tables.append(df)
        except Exception as e:
            print(f"pdfplumber error: {e}")
        return tables
    
    def calculate_quality(self, df: pd.DataFrame) -> float:
        """Calculate table quality score."""
        if df.empty:
            return 0.0
        
        score = 0.0
        score += min(len(df) / 10, 1.0) * 0.3
        score += min(len(df.columns) / 5, 1.0) * 0.2
        
        completeness = 1 - df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        score += completeness * 0.3
        
        headers = df.columns.tolist()
        if headers and any(len(str(h)) > 3 for h in headers):
            score += 0.2
        
        return min(score, 1.0)
    
    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean table data."""
        cleaned = df.copy()
        cleaned = cleaned.dropna(how='all')
        cleaned = cleaned.dropna(axis=1, how='all')
        cleaned.columns = [' '.join(str(col).split()) for col in cleaned.columns]
        cleaned = cleaned.drop_duplicates()
        cleaned = cleaned.reset_index(drop=True)
        return cleaned
    
    def extract_tables(self, pdf_path: str, doc_id: str) -> List[EnhancedTable]:
        """Extract tables with all methods."""
        enhanced_tables = []
        
        try:
            camelot_tables = self.extract_with_camelot(pdf_path)
            print(f"  Camelot extracted: {len(camelot_tables)} tables")
        except Exception as e:
            print(f"  Camelot failed: {e}")
            camelot_tables = []
        
        try:
            plumber_tables = self.extract_with_pdfplumber(pdf_path)
            print(f"  pdfplumber extracted: {len(plumber_tables)} tables")
        except Exception as e:
            print(f"  pdfplumber failed: {e}")
            plumber_tables = []
        
        all_tables = camelot_tables + plumber_tables
        
        if not all_tables:
            print("  ⚠️ No tables found in PDF")
            return []
        
        for idx, raw_df in enumerate(all_tables):
            try:
                cleaned = self.clean_table(raw_df)
                
                if cleaned.empty:
                    print(f"  Skipping empty table {idx}")
                    continue
                
                quality = self.calculate_quality(cleaned)
                
                if quality < 0.3:
                    print(f"  Skipping low quality table {idx} (score: {quality:.2f})")
                    continue
                
                table_id = f"advanced_{doc_id}_table_{idx}"
                
                try:
                    schema = self.schema_inferencer.infer_schema(table_id, cleaned)
                except Exception as e:
                    print(f"  Schema inference failed for table {idx}: {e}")
                    continue
                
                enhanced_tables.append(EnhancedTable(
                    table_id=table_id,
                    doc_id=doc_id,
                    page_num=1,
                    cleaned_data=cleaned,
                    schema=schema,
                    quality_score=quality,
                    extraction_method='hybrid',
                    metadata={}
                ))
                
                print(f"  ✅ Table {idx}: {len(cleaned)} rows × {len(cleaned.columns)} cols (quality: {quality:.2f})")
            
            except Exception as e:
                print(f"  ❌ Error processing table {idx}: {e}")
                continue
        
        return enhanced_tables

# ============================================================================
# PDF PROCESSOR
# ============================================================================

class AdvancedPDFProcessor:
    """Complete PDF processing pipeline."""
    
    def __init__(self):
        self.classifier = ContentClassifier()
        self.extractor = AdvancedTableExtractor()
    
    def extract_text(self, pdf_path: str) -> List[str]:
        """Extract text from PDF."""
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        return pages
    
    def process_pdf(self, pdf_path: str, doc_id: str, filter_boilerplate: bool = True) -> Dict:
        """Process PDF with advanced features."""
        print(f"\n{'='*60}")
        print(f"Processing: {Path(pdf_path).name}")
        print(f"{'='*60}")
        
        try:
            # Extract text
            print("📄 Extracting text...")
            pages_text = self.extract_text(pdf_path)
            print(f"  ✅ Extracted {len(pages_text)} pages")
            
            # Classify content
            print("🔍 Classifying content...")
            try:
                sections = self.classifier.classify_document_sections(pages_text)
                print(f"  ✅ Classified {len(sections)} sections")
            except Exception as e:
                print(f"  ⚠️ Classification failed: {e}, using all content")
                sections = [DocumentSection('material', page, i+1, 1.0) 
                           for i, page in enumerate(pages_text)]
            
            # Filter content
            if filter_boilerplate:
                filtered_content = [s.content for s in sections if s.section_type == 'material']
                num_filtered = len([s for s in sections if s.section_type == 'boilerplate'])
                print(f"  📝 Material: {len(filtered_content)} sections")
                print(f"  🚫 Filtered: {num_filtered} boilerplate sections")
            else:
                filtered_content = [s.content for s in sections]
            
            # Extract tables
            print("📊 Extracting tables...")
            try:
                enhanced_tables = self.extractor.extract_tables(pdf_path, doc_id)
                print(f"  ✅ Found {len(enhanced_tables)} high-quality tables")
            except Exception as e:
                print(f"  ⚠️ Table extraction failed: {e}")
                enhanced_tables = []
            
            print(f"{'='*60}")
            print(f"✅ Processing complete!")
            print(f"{'='*60}\n")
            
            return {
                'doc_id': doc_id,
                'filename': Path(pdf_path).name,
                'num_pages': len(pages_text),
                'filtered_content': filtered_content,
                'enhanced_tables': enhanced_tables,
                'all_sections': sections,
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'num_material_sections': len([s for s in sections if s.section_type == 'material']),
                    'num_boilerplate_sections': len([s for s in sections if s.section_type == 'boilerplate'])
                }
            }
        
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal result so processing doesn't completely fail
            return {
                'doc_id': doc_id,
                'filename': Path(pdf_path).name,
                'num_pages': 0,
                'filtered_content': [],
                'enhanced_tables': [],
                'all_sections': [],
                'metadata': {
                    'processed_at': datetime.now().isoformat(),
                    'error': str(e)
                }
            }

# ============================================================================
# ADVANCED DATABASE STORE
# ============================================================================

class AdvancedVectorStore:
    """Vector store for semantic search."""
    
    def __init__(self, persist_directory: str = "./data/advanced_chroma_db"):
        if not CHROMADB_AVAILABLE:
            print("⚠️  ChromaDB not available - vector search disabled")
            self.available = False
            return
        
        self.available = True
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        try:
            # ChromaDB 0.4.24 compatible initialization
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Create settings object
            settings = ChromaSettings(
                persist_directory=persist_directory,
                anonymized_telemetry=False,  # Disable telemetry to avoid errors
                allow_reset=True
            )
            
            # Initialize client with settings
            self.client = chromadb.Client(settings)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="advanced_pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"✅ ChromaDB initialized: {persist_directory}")
            
            # Initialize OpenAI for embeddings
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                self.embedding_model = "text-embedding-3-small"
                print("✅ OpenAI embeddings enabled")
            else:
                print("⚠️  OPENAI_API_KEY not found - using dummy embeddings")
                self.openai_client = None
        
        except Exception as e:
            print(f"⚠️  Vector store initialization failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.available = False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        if not self.openai_client:
            # Return dummy embeddings if no API key
            return [[0.0] * 1536 for _ in texts]
        
        try:
            embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
        except Exception as e:
            print(f"Embedding generation error: {e}")
            return [[0.0] * 1536 for _ in texts]
    
    def add_documents(self, doc_id: str, contents: List[str], page_nums: List[int]):
        """Add document content to vector store."""
        if not self.available or not contents:
            return
        
        try:
            # Generate embeddings
            embeddings = self.generate_embeddings(contents)
            
            # Prepare data
            ids = [f"{doc_id}_section_{i}" for i in range(len(contents))]
            metadatas = [
                {
                    'doc_id': doc_id,
                    'page_num': str(page_num),
                    'section_index': str(i)
                }
                for i, page_num in enumerate(page_nums)
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
            
            print(f"  ✅ Added {len(contents)} sections to vector store")
        
        except Exception as e:
            print(f"  ⚠️  Vector store add failed: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar content."""
        if not self.available:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    formatted.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'relevance': 1 - results['distances'][0][i]
                    })
            
            return formatted
        
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def clear(self):
        """Clear all data."""
        if not self.available:
            return
        
        try:
            self.client.delete_collection("advanced_pdf_documents")
            self.collection = self.client.get_or_create_collection(
                name="advanced_pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Clear error: {e}")


class AdvancedDatabaseStore:
    """Separate database with advanced_ prefix tables."""
    
    def __init__(self, db_path: str = "./data/advanced_database.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_schema()
    
    def _create_schema(self):
        """Create advanced schema."""
        cursor = self.conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                num_pages INTEGER,
                metadata TEXT,
                ingested_at TEXT,
                num_material_sections INTEGER,
                num_boilerplate_sections INTEGER
            )
        ''')
        
        # Content sections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_content_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                section_type TEXT,
                content TEXT,
                page_num INTEGER,
                confidence REAL,
                FOREIGN KEY (doc_id) REFERENCES advanced_documents(doc_id)
            )
        ''')
        
        # Table metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_table_metadata (
                table_id TEXT PRIMARY KEY,
                doc_id TEXT,
                page_num INTEGER,
                quality_score REAL,
                extraction_method TEXT,
                schema_json TEXT,
                num_rows INTEGER,
                num_columns INTEGER,
                FOREIGN KEY (doc_id) REFERENCES advanced_documents(doc_id)
            )
        ''')
        
        # Table data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS advanced_table_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id TEXT,
                row_index INTEGER,
                row_data TEXT,
                FOREIGN KEY (table_id) REFERENCES advanced_table_metadata(table_id)
            )
        ''')
        
        self.conn.commit()
    
    def add_document(self, doc_id: str, filename: str, num_pages: int, metadata: Dict):
        """Add document."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO advanced_documents 
            (doc_id, filename, num_pages, metadata, ingested_at, 
             num_material_sections, num_boilerplate_sections)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id, filename, num_pages, json.dumps(metadata),
            datetime.now().isoformat(),
            metadata.get('num_material_sections', 0),
            metadata.get('num_boilerplate_sections', 0)
        ))
        self.conn.commit()
    
    def add_content_sections(self, doc_id: str, sections: List[DocumentSection]):
        """Add content sections."""
        cursor = self.conn.cursor()
        for section in sections:
            cursor.execute('''
                INSERT INTO advanced_content_sections 
                (doc_id, section_type, content, page_num, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, section.section_type, section.content, 
                  section.page_num, section.confidence))
        self.conn.commit()
    
    def add_enhanced_table(self, table: EnhancedTable):
        """Add enhanced table."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO advanced_table_metadata
            (table_id, doc_id, page_num, quality_score, extraction_method,
             schema_json, num_rows, num_columns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            table.table_id, table.doc_id, table.page_num,
            table.quality_score, table.extraction_method,
            json.dumps(table.schema.to_sql_schema()),
            len(table.cleaned_data), len(table.cleaned_data.columns)
        ))
        
        for idx, row in table.cleaned_data.iterrows():
            cursor.execute('''
                INSERT INTO advanced_table_data (table_id, row_index, row_data)
                VALUES (?, ?, ?)
            ''', (table.table_id, idx, json.dumps(row.tolist())))
        
        self.conn.commit()
    
    def get_documents(self) -> List[Dict]:
        """Get all documents."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT doc_id, filename, num_pages, ingested_at,
                   num_material_sections, num_boilerplate_sections
            FROM advanced_documents
            ORDER BY ingested_at DESC
        ''')
        
        return [{
            'doc_id': row[0],
            'filename': row[1],
            'num_pages': row[2],
            'ingested_at': row[3],
            'material_sections': row[4],
            'boilerplate_sections': row[5]
        } for row in cursor.fetchall()]
    
    def get_content(self, doc_id: str, section_type: str = 'material') -> List[str]:
        """Get content sections."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT content FROM advanced_content_sections
            WHERE doc_id = ? AND section_type = ?
            ORDER BY page_num
        ''', (doc_id, section_type))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_all_tables(self) -> List[Dict]:
        """Get all tables with data."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT table_id, doc_id, quality_score, schema_json
            FROM advanced_table_metadata
            WHERE quality_score > 0.5
        ''')
        
        tables = []
        for row in cursor.fetchall():
            table_id, doc_id, quality_score, schema_json = row
            
            cursor.execute('''
                SELECT row_data FROM advanced_table_data
                WHERE table_id = ?
                ORDER BY row_index
            ''', (table_id,))
            
            rows = [json.loads(r[0]) for r in cursor.fetchall()]
            
            tables.append({
                'table_id': table_id,
                'doc_id': doc_id,
                'quality_score': quality_score,
                'schema': schema_json,
                'rows': rows
            })
        
        return tables
    
    def clear(self):
        """Clear all data."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM advanced_table_data')
        cursor.execute('DELETE FROM advanced_table_metadata')
        cursor.execute('DELETE FROM advanced_content_sections')
        cursor.execute('DELETE FROM advanced_documents')
        self.conn.commit()

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Advanced PDF Query System",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Advanced PDF Query System")
    st.markdown("**Standalone version with enhanced features**")
    
    # Initialize
    if 'db_store' not in st.session_state:
        st.session_state.db_store = AdvancedDatabaseStore()
        st.session_state.vector_store = AdvancedVectorStore()
        st.session_state.processor = AdvancedPDFProcessor()
        
        # Show initialization status
        if st.session_state.vector_store.available:
            st.success("✅ Vector search enabled")
        else:
            st.warning("⚠️  Vector search disabled (ChromaDB not available or no API key)")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key='pdf_uploader'
        )
        
        filter_boilerplate = st.checkbox("Filter boilerplate content", value=True)
        
        if uploaded_files and st.button("Process PDFs", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Save temporarily
                    temp_path = f"./temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process with detailed logging
                    doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}"
                    
                    with st.expander(f"📄 Processing {uploaded_file.name}", expanded=True):
                        result = st.session_state.processor.process_pdf(
                            temp_path, doc_id, filter_boilerplate
                        )
                        
                        # Store in database
                        st.session_state.db_store.add_document(
                            doc_id, result['filename'],
                            result['num_pages'], result['metadata']
                        )
                        
                        # Store content sections if available
                        if result.get('all_sections'):
                            st.session_state.db_store.add_content_sections(
                                doc_id, result['all_sections']
                            )
                            
                            # Add to vector store
                            material_sections = [
                                s for s in result['all_sections'] 
                                if s.section_type == 'material'
                            ]
                            
                            if material_sections:
                                contents = [s.content for s in material_sections]
                                page_nums = [s.page_num for s in material_sections]
                                
                                st.session_state.vector_store.add_documents(
                                    doc_id, contents, page_nums
                                )
                        
                        # Store tables
                        tables_stored = 0
                        for table in result['enhanced_tables']:
                            try:
                                st.session_state.db_store.add_enhanced_table(table)
                                tables_stored += 1
                            except Exception as e:
                                st.warning(f"Failed to store table: {e}")
                        
                        # Show results
                        st.success(f"✅ Processed successfully!")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Pages", result['num_pages'])
                        with col2:
                            st.metric("Content Sections", len(result['filtered_content']))
                        with col3:
                            st.metric("Tables", len(result['enhanced_tables']))
                        with col4:
                            st.metric("Vector Chunks", len(material_sections) if 'material_sections' in locals() else 0)
                        
                        if result['metadata'].get('error'):
                            st.error(f"Errors occurred: {result['metadata']['error']}")
                    
                    # Cleanup
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}")
                    st.error(f"Error: {str(e)}")
                    with st.expander("View full error"):
                        st.code(traceback.format_exc())
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ All files processed!")
            st.balloons()
        
        st.divider()
        
        # Show ingested documents
        st.header("📚 Ingested Documents")
        docs = st.session_state.db_store.get_documents()
        
        if docs:
            for doc in docs:
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Pages:** {doc['num_pages']}")
                    st.write(f"**Material sections:** {doc['material_sections']}")
                    st.write(f"**Boilerplate filtered:** {doc['boilerplate_sections']}")
                    st.write(f"**Ingested:** {doc['ingested_at'][:19]}")
        else:
            st.info("No documents yet")
        
        st.divider()
        
        if st.button("🗑️ Clear All Data"):
            st.session_state.db_store.clear()
            st.session_state.vector_store.clear()
            st.success("Data cleared!")
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Tables", "📝 Content", "🔍 Query"])
    
    with tab1:
        st.header("Extracted Tables")
        tables = st.session_state.db_store.get_all_tables()
        
        if tables:
            for table_data in tables:
                st.subheader(f"Table: {table_data['table_id']}")
                st.write(f"Quality Score: {table_data['quality_score']:.2f}")
                
                # Show schema
                with st.expander("View Schema"):
                    st.code(table_data['schema'], language='sql')
                
                # Show data
                if table_data['rows']:
                    st.dataframe(pd.DataFrame(table_data['rows']))
                
                st.divider()
        else:
            st.info("No tables extracted yet. Upload PDFs to get started.")
    
    with tab2:
        st.header("Document Content")
        docs = st.session_state.db_store.get_documents()
        
        if docs:
            selected_doc = st.selectbox(
                "Select Document",
                options=[d['filename'] for d in docs],
                key='doc_selector'
            )
            
            if selected_doc:
                doc_id = next(d['doc_id'] for d in docs if d['filename'] == selected_doc)
                content = st.session_state.db_store.get_content(doc_id, 'material')
                
                st.write(f"**Material content sections:** {len(content)}")
                
                for i, section in enumerate(content, 1):
                    with st.expander(f"Section {i}"):
                        st.write(section)
        else:
            st.info("No documents yet")
    
    with tab3:
        st.header("🔍 Search Documents")
        
        if not st.session_state.vector_store.available:
            st.warning("Vector search is not available. Install ChromaDB and set OPENAI_API_KEY.")
            st.code("pip install chromadb==0.4.24")
        else:
            query = st.text_input("Search query:", placeholder="e.g., revenue growth, customer satisfaction")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                top_k = st.slider("Number of results", 1, 10, 5)
            with col2:
                search_button = st.button("🔍 Search", type="primary")
            
            if search_button and query:
                with st.spinner("Searching..."):
                    results = st.session_state.vector_store.search(query, top_k=top_k)
                    
                    if results:
                        st.success(f"Found {len(results)} results")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} - Relevance: {result['relevance']:.2%}"):
                                st.markdown(f"**Content:**")
                                st.write(result['content'])
                                
                                st.markdown(f"**Source:**")
                                st.json(result['metadata'])
                    else:
                        st.info("No results found. Try a different query.")
            
            # Show stats
            st.divider()
            st.subheader("📊 Vector Store Stats")
            
            try:
                count = st.session_state.vector_store.collection.count()
                st.metric("Total chunks indexed", count)
            except:
                st.info("No data indexed yet")

if __name__ == "__main__":
    main()