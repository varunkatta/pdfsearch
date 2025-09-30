"""
PDF Query System - Complete Working Implementation
Compatible with Python 3.13.5 and OpenAI API

Usage:
    streamlit run pdf_query_system.py
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import json
from datetime import datetime
import traceback

import fitz
import camelot
import chromadb
from chromadb.config import Settings
import pandas as pd
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


@dataclass
class Document:
    doc_id: str
    filename: str
    num_pages: int
    metadata: Dict
    ingested_at: str = None
    
    def __post_init__(self):
        if self.ingested_at is None:
            self.ingested_at = datetime.now().isoformat()


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    content: str
    page_num: int
    metadata: Dict


@dataclass
class Table:
    table_id: str
    doc_id: str
    page_num: int
    headers: List[str]
    rows: List[List[str]]
    context: str = ""


class ContentCleaner:
    def __init__(self):
        self.patterns = [
            r'Page \d+ of \d+',
            r'^\d+$',
            r'©.*?\d{4}',
            r'www\.\S+',
        ]
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]
    
    def detect_repeated_text(self, pages_text: List[str], threshold: float = 0.7) -> List[str]:
        if len(pages_text) < 3:
            return []
        
        all_lines = [page.split('\n') for page in pages_text]
        line_counts = Counter()
        
        for lines in all_lines:
            for line in lines:
                line = line.strip()
                if len(line) > 5:
                    line_counts[line] += 1
        
        threshold_count = len(pages_text) * threshold
        repeated_lines = [line for line, count in line_counts.items() 
                         if count >= threshold_count]
        
        return repeated_lines
    
    def clean_text(self, text: str, repeated_text: List[str] = None) -> str:
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        
        if repeated_text:
            for repeated in repeated_text:
                text = text.replace(repeated, '')
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def clean_document(self, pages_text: List[str]) -> List[str]:
        repeated_text = self.detect_repeated_text(pages_text)
        
        cleaned_pages = []
        for page_text in pages_text:
            cleaned = self.clean_text(page_text, repeated_text)
            if cleaned:
                cleaned_pages.append(cleaned)
        
        return cleaned_pages


class PDFProcessor:
    def __init__(self):
        self.cleaner = ContentCleaner()
    
    def extract_text(self, pdf_path: str) -> Tuple[List[str], Dict]:
        doc = fitz.open(pdf_path)
        pages_text = []
        metadata = {
            'title': doc.metadata.get('title', ''),
            'author': doc.metadata.get('author', ''),
            'num_pages': len(doc)
        }
        
        for page in doc:
            text = page.get_text()
            pages_text.append(text)
        
        doc.close()
        cleaned_pages = self.cleaner.clean_document(pages_text)
        
        return cleaned_pages, metadata
    
    def extract_tables(self, pdf_path: str) -> List[Table]:
        tables = []
        
        try:
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            if len(camelot_tables) == 0:
                camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            
            for idx, table in enumerate(camelot_tables):
                df = table.df
                
                if df.empty or len(df) < 2:
                    continue
                
                headers = [str(h).strip() for h in df.iloc[0].tolist()]
                rows = df.iloc[1:].values.tolist()
                
                table_obj = Table(
                    table_id=f"table_{idx}",
                    doc_id="",
                    page_num=table.page,
                    headers=headers,
                    rows=rows,
                    context=""
                )
                tables.append(table_obj)
        
        except Exception as e:
            print(f"Table extraction warning: {e}")
        
        return tables
    
    def process_pdf(self, pdf_path: str, doc_id: str) -> Tuple[Document, List[str], List[Table]]:
        pages_text, metadata = self.extract_text(pdf_path)
        tables = self.extract_tables(pdf_path)
        
        for table in tables:
            table.doc_id = doc_id
        
        document = Document(
            doc_id=doc_id,
            filename=Path(pdf_path).name,
            num_pages=len(pages_text),
            metadata=metadata
        )
        
        return document, pages_text, tables


class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(self, pages_text: List[str], doc_id: str) -> List[TextChunk]:
        chunks = []
        
        for page_num, page_text in enumerate(pages_text, 1):
            page_chunks = self.splitter.split_text(page_text)
            
            for idx, chunk_text in enumerate(page_chunks):
                chunk = TextChunk(
                    chunk_id=f"{doc_id}_p{page_num}_c{idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    page_num=page_num,
                    metadata={'page': page_num, 'chunk_index': idx}
                )
                chunks.append(chunk)
        
        return chunks


class OpenAIService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def answer_question(self, question: str, context_chunks: List[str]) -> str:
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
        
        system_msg = "You are a helpful assistant that answers questions based on provided document excerpts. Always cite which excerpt number(s) support your answer."
        user_msg = f"Context from documents:\n{context}\n\nQuestion: {question}\n\nProvide a detailed answer based on the context. Cite the excerpt numbers."
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        return self.chat_completion(messages)
    
    def summarize(self, texts: List[str], focus: str = None) -> str:
        combined_text = "\n\n".join(texts)
        focus_instruction = f" Focus on: {focus}" if focus else ""
        
        system_msg = "You are a helpful assistant that creates concise summaries."
        user_msg = f"Summarize the following text.{focus_instruction}\n\n{combined_text}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        return self.chat_completion(messages, temperature=0.3)
    
    def analyze_tables(self, tables_data: str, question: str) -> str:
        system_msg = "You are a data analyst. Analyze table data and answer questions."
        user_msg = f"Table Data:\n{tables_data}\n\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        return self.chat_completion(messages)


class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(self, openai_service: OpenAIService, persist_directory: str = "./chroma_db"):
        self.openai_service = openai_service
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(persist_directory, exist_ok=True)
        
        # Fixed initialization for ChromaDB 0.5.3+
        try:
            # Try new API first
            self.client = chromadb.PersistentClient(path=persist_directory)
        except TypeError:
            # Fallback for older versions
            from chromadb.config import Settings
            settings = Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
            self.client = chromadb.Client(settings)
        
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[TextChunk]):
        """Add text chunks to vector store"""
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.openai_service.generate_embeddings(texts)
        
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                'doc_id': chunk.doc_id,
                'page_num': str(chunk.page_num),
                **{k: str(v) for k, v in chunk.metadata.items()}
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search for similar chunks"""
        query_embedding = self.openai_service.generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance': 1 - results['distances'][0][i]
            })
        
        return formatted_results
    
    def clear(self):
        """Clear all data"""
        try:
            self.client.delete_collection("pdf_documents")
        except:
            pass
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )

class VectorStoreOld:
    def __init__(self, openai_service: OpenAIService, persist_directory: str = "./chroma_db"):
        self.openai_service = openai_service
        self.client = chromadb.PersistentClient(path=persist_directory)

    # def __init__(self, openai_service: OpenAIService, persist_directory: str = "./chroma_db"):
    #     self.openai_service = openai_service
    #     # NEW - Fixed initialization
    #     settings = chromadb.config.Settings(
    #         anonymized_telemetry=False,
    #         allow_reset=True
    #     )
    #     self.client = chromadb.PersistentClient(
    #         path=persist_directory,
    #         settings=settings
    #     )
    
    def add_chunks(self, chunks: List[TextChunk]):
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.openai_service.generate_embeddings(texts)
        
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                'doc_id': chunk.doc_id,
                'page_num': str(chunk.page_num),
                **{k: str(v) for k, v in chunk.metadata.items()}
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        query_embedding = self.openai_service.generate_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'chunk_id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'relevance': 1 - results['distances'][0][i]
            })
        
        return formatted_results
    
    def clear(self):
        self.client.delete_collection("pdf_documents")
        self.collection = self.client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )


class TableStore:
    def __init__(self, db_path: str = "./tables.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT,
                num_pages INTEGER,
                metadata TEXT,
                ingested_at TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS table_metadata (
                table_id TEXT PRIMARY KEY,
                doc_id TEXT,
                page_num INTEGER,
                headers TEXT,
                num_rows INTEGER,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS table_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                table_id TEXT,
                row_index INTEGER,
                row_data TEXT,
                FOREIGN KEY (table_id) REFERENCES table_metadata(table_id)
            )
        ''')
        
        self.conn.commit()
    
    def add_document(self, document: Document):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO documents (doc_id, filename, num_pages, metadata, ingested_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (document.doc_id, document.filename, document.num_pages, 
              json.dumps(document.metadata), document.ingested_at))
        self.conn.commit()
    
    def add_table(self, table: Table):
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO table_metadata (table_id, doc_id, page_num, headers, num_rows)
            VALUES (?, ?, ?, ?, ?)
        ''', (table.table_id, table.doc_id, table.page_num, 
              json.dumps(table.headers), len(table.rows)))
        
        for idx, row in enumerate(table.rows):
            cursor.execute('''
                INSERT INTO table_data (table_id, row_index, row_data)
                VALUES (?, ?, ?)
            ''', (table.table_id, idx, json.dumps(row)))
        
        self.conn.commit()
    
    def get_all_tables(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT tm.table_id, tm.doc_id, tm.page_num, tm.headers, d.filename
            FROM table_metadata tm
            JOIN documents d ON tm.doc_id = d.doc_id
        ''')
        
        tables = []
        for row in cursor.fetchall():
            table_id, doc_id, page_num, headers_json, filename = row
            headers = json.loads(headers_json)
            
            cursor.execute('''
                SELECT row_data FROM table_data 
                WHERE table_id = ? 
                ORDER BY row_index
            ''', (table_id,))
            
            rows = [json.loads(r[0]) for r in cursor.fetchall()]
            
            tables.append({
                'table_id': table_id,
                'doc_id': doc_id,
                'filename': filename,
                'page_num': page_num,
                'headers': headers,
                'rows': rows
            })
        
        return tables
    
    def get_documents(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT doc_id, filename, num_pages, ingested_at FROM documents')
        
        docs = []
        for row in cursor.fetchall():
            docs.append({
                'doc_id': row[0],
                'filename': row[1],
                'num_pages': row[2],
                'ingested_at': row[3]
            })
        
        return docs
    
    def clear(self):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM table_data')
        cursor.execute('DELETE FROM table_metadata')
        cursor.execute('DELETE FROM documents')
        self.conn.commit()


class QueryEngine:
    def __init__(self, openai_api_key: str = None):
        self.openai_service = OpenAIService(openai_api_key)
        self.vector_store = VectorStore(self.openai_service)
        self.table_store = TableStore()
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker()
    
    def ingest_pdf(self, pdf_path: str) -> Document:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        doc_id = f"doc_{Path(pdf_path).stem}_{timestamp}"
        
        document, pages_text, tables = self.pdf_processor.process_pdf(pdf_path, doc_id)
        chunks = self.text_chunker.chunk_document(pages_text, doc_id)
        
        self.vector_store.add_chunks(chunks)
        self.table_store.add_document(document)
        
        for table in tables:
            self.table_store.add_table(table)
        
        return document
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        query_type = self._classify_query(question)
        
        if "table" in query_type or "aggregate" in query_type:
            return self._handle_table_query(question)
        elif "summarize" in query_type:
            return self._handle_summarize_query(question, top_k)
        else:
            return self._handle_search_query(question, top_k)
    
    def _classify_query(self, question: str) -> str:
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['table', 'aggregate', 'sum', 'total', 'average']):
            return 'table'
        elif any(word in q_lower for word in ['summarize', 'summary', 'overview']):
            return 'summarize'
        else:
            return 'search'
    
    def _handle_search_query(self, question: str, top_k: int) -> Dict:
        results = self.vector_store.search(question, top_k=top_k)
        
        if not results:
            return {
                'answer': "I could not find relevant information.",
                'sources': []
            }
        
        context_chunks = [r['content'] for r in results]
        answer = self.openai_service.answer_question(question, context_chunks)
        
        sources = []
        for r in results:
            sources.append({
                'doc_id': r['metadata'].get('doc_id', 'Unknown'),
                'page': r['metadata'].get('page_num', 'N/A'),
                'snippet': r['content'][:200] + '...',
                'relevance': f"{r['relevance']:.2%}"
            })
        
        return {'answer': answer, 'sources': sources}
    
    def _handle_summarize_query(self, question: str, top_k: int) -> Dict:
        results = self.vector_store.search(question, top_k=top_k * 2)
        
        if not results:
            return {
                'answer': "I could not find relevant information to summarize.",
                'sources': []
            }
        
        texts = [r['content'] for r in results]
        summary = self.openai_service.summarize(texts, focus=question)
        
        sources = []
        for r in results[:5]:
            sources.append({
                'doc_id': r['metadata'].get('doc_id', 'Unknown'),
                'page': r['metadata'].get('page_num', 'N/A')
            })
        
        return {'answer': summary, 'sources': sources}
    
    def _handle_table_query(self, question: str) -> Dict:
        all_tables = self.table_store.get_all_tables()
        
        if not all_tables:
            return {
                'answer': "No tables found in documents.",
                'sources': [],
                'tables': []
            }
        
        tables_text = ""
        dfs = []
        
        for table_data in all_tables:
            try:
                df = pd.DataFrame(table_data['rows'], columns=table_data['headers'])
                
                filename = table_data.get('filename', 'Unknown')
                page_num = table_data.get('page_num', 'N/A')
                source_info = f"{filename} (page {page_num})"
                
                df['_source'] = source_info
                dfs.append(df)
                
                tables_text += f"\n\nTable from {source_info}:\n"
                tables_text += df.to_string()
                
            except Exception as e:
                print(f"Warning: Error processing table: {e}")
                continue
        
        if not dfs:
            return {
                'answer': "Tables were found but could not be processed.",
                'sources': [],
                'tables': []
            }
        
        answer = self.openai_service.analyze_tables(tables_text, question)
        
        return {
            'answer': answer,
            'sources': [],
            'tables': dfs
        }


def main():
    st.set_page_config(
        page_title="PDF Query System",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 PDF Query System")
    st.markdown("Ask questions about your PDF documents")
    
    if 'engine' not in st.session_state:
        try:
            st.session_state.engine = QueryEngine()
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Error initializing: {e}")
            st.stop()
    
    with st.sidebar:
        st.header("📁 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process PDFs"):
                with st.spinner("Processing PDFs..."):
                    for uploaded_file in uploaded_files:
                        try:
                            temp_path = f"./temp_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            doc = st.session_state.engine.ingest_pdf(temp_path)
                            st.success(f"✅ {doc.filename}")
                            
                            os.remove(temp_path)
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}: {e}")
        
        st.divider()
        
        st.header("📊 Settings")
        top_k = st.slider("Top-K Results", 1, 10, 5)
        
        st.divider()
        
        docs = st.session_state.engine.table_store.get_documents()
        if docs:
            st.header("📚 Ingested Documents")
            for doc in docs:
                st.text(f"• {doc['filename']} ({doc['num_pages']} pages)")
        
        if st.button("Clear All Data"):
            st.session_state.engine.vector_store.clear()
            st.session_state.engine.table_store.clear()
            st.session_state.messages = []
            st.success("Data cleared!")
            st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "tables" in message and message["tables"]:
                for df in message["tables"]:
                    st.dataframe(df, use_container_width=True)
            
            if "sources" in message and message["sources"]:
                with st.expander("📎 Sources"):
                    for source in message["sources"]:
                        doc_id = source.get('doc_id', 'Unknown')
                        page = source.get('page', 'N/A')
                        st.text(f"• {doc_id} (page {page})")
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.engine.query(prompt, top_k=top_k)
                    
                    st.markdown(result['answer'])
                    
                    if 'tables' in result and result['tables']:
                        for df in result['tables']:
                            st.dataframe(df, use_container_width=True)
                    
                    if result.get('sources'):
                        with st.expander("📎 Sources"):
                            for source in result['sources']:
                                doc_id = source.get('doc_id', 'Unknown')
                                page = source.get('page', 'N/A')
                                st.text(f"• {doc_id} (page {page})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result.get('sources', []),
                        "tables": result.get('tables', [])
                    })
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
                    st.error(error_msg)


if __name__ == "__main__":
    main()