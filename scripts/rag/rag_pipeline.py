"""
Main RAG pipeline for medical document query and retrieval
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from document_processor import MedicalDocumentProcessor
from chunking import MedicalDocumentChunker
from vector_store import MedicalVectorStore
from config import RAGConfig


class MedicalRAGPipeline:
    """
    Complete RAG pipeline for processing and querying medical documents
    """


    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RAGConfig()

        # Initialize components
        self.document_processor = MedicalDocumentProcessor()
        self.chunker = MedicalDocumentChunker()
        self.vector_store = MedicalVectorStore()

        self.logger.info("RAG Pipeline initialized successfully")


    def ingest_document(self, pdf_path: str) -> bool:
        """
        Process and ingest a single PDF document into the vector store

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Success boolean
        """
        try:
            self.logger.info(f"Starting document ingestion for: {pdf_path}")

            # Step 1: Extract text and metadata
            processed_doc = self.document_processor.process_document(pdf_path)

            # Step 2: Chunk the document
            chunks = self.chunker.chunk_document(
                processed_doc["text"],
                processed_doc["metadata"]
            )

            self.logger.info(f"Document chunked into {len(chunks)} pieces")

            # Step 3: Add to vector store
            success = self.vector_store.add_documents(chunks)

            if success:
                self.logger.info(f"Successfully ingested document: {pdf_path}")
                return True
            else:
                self.logger.error(f"Failed to ingest document: {pdf_path}")
                return False

        except Exception as e:
            self.logger.error(f"Error ingesting document {pdf_path}: {str(e)}")
            return False


    def query_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        patient_id: Optional[str] = None,
        max_results: int = None
    ) -> Dict[str, Any]:
        """
        Query the medical document corpus

        Args:
            query: Natural language query
            document_types: Filter by document types
            patient_id: Filter by patient ID
            max_results: Maximum number of results to return

        Returns:
            Query results with context and metadata
        """
        try:
            self.logger.info(f"Processing query: {query}")

            # Step 1: Preprocess query
            processed_query = self._preprocess_query(query)

            # Step 2: Build metadata filters
            filter_metadata = {}
            if patient_id:
                filter_metadata["patient_id"] = patient_id

            # Step 3: Determine collections to search
            collection_names = None
            if document_types:
                collection_names = [
                    self.config.COLLECTIONS.get(doc_type, doc_type)
                    for doc_type in document_types
                ]

            # Step 4: Search vector store
            search_results = self.vector_store.search_documents(
                query=processed_query,
                collection_names=collection_names,
                n_results=max_results or self.config.MAX_RESULTS,
                filter_metadata=filter_metadata if filter_metadata else None
            )

            # Step 5: Rank and format results
            formatted_results = self._format_search_results(search_results)

            # Step 6: Generate response context
            context = self._build_response_context(formatted_results)

            return {
                "query": query,
                "processed_query": processed_query,
                "results": formatted_results,
                "context": context,
                "total_results": len(formatted_results)
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "results": [],
                "context": "",
                "total_results": 0
            }


    def _preprocess_query(self, query: str) -> str:
        """Preprocess and expand medical queries"""

        # Basic cleaning
        query = query.strip()

        # Expand common medical abbreviations
        abbreviations = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "hx": "history",
            "dx": "diagnosis",
            "tx": "treatment",
            "pt": "patient",
            "htn": "hypertension",
            "dm": "diabetes mellitus",
            "copd": "chronic obstructive pulmonary disease",
            "cad": "coronary artery disease",
            "chf": "congestive heart failure"
        }

        query_words = query.lower().split()
        expanded_words = []

        for word in query_words:
            # Remove punctuation for abbreviation matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in abbreviations:
                expanded_words.append(abbreviations[clean_word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)


    def _format_search_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format and enhance search results"""

        formatted = []

        for result in raw_results:
            formatted_result = {
                "text": result["text"],
                "relevance_score": 1.0 - result["distance"],  # Convert distance to similarity
                "source": {
                    "document_id": result["metadata"].get("document_id"),
                    "source_file": result["metadata"].get("source_file"),
                    "document_type": result["metadata"].get("document_type"),
                    "patient_id": result["metadata"].get("patient_id"),
                    "date": result["metadata"].get("date"),
                    "page_count": result["metadata"].get("page_count")
                },
                "chunk_info": {
                    "chunk_id": result["id"],
                    "chunk_index": result["metadata"].get("chunk_index"),
                    "start_char": result["metadata"].get("start_char"),
                    "end_char": result["metadata"].get("end_char")
                },
                "collection": result["collection"]
            }

            formatted.append(formatted_result)

        return formatted


    def _build_response_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context string for response generation"""

        if not results:
            return "No relevant documents found."

        context_parts = []

        for i, result in enumerate(results[:3]):  # Use top 3 results for context
            source_info = f"Source: {result['source']['source_file']}"
            if result['source']['patient_id']:
                source_info += f" (Patient: {result['source']['patient_id']})"

            context_parts.append(f"[Context {i+1}] {source_info}\n{result['text']}\n")

        return "\n".join(context_parts)


    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and collection information"""

        try:
            collection_info = self.vector_store.get_collection_info()

            total_documents = sum(
                info.get("count", 0) for info in collection_info.values()
                if isinstance(info.get("count"), int)
            )

            return {
                "status": "healthy",
                "collections": collection_info,
                "total_documents": total_documents,
                "chromadb_url": self.config.get_chromadb_url()
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "collections": {},
                "total_documents": 0
            }


    def reset_system(self) -> bool:
        """Reset the entire system (delete all data)"""

        try:
            self.logger.warning("Resetting RAG system - all data will be deleted")
            success = self.vector_store.reset_collections()

            if success:
                self.logger.info("RAG system reset successfully")
            else:
                self.logger.error("Failed to reset RAG system")

            return success

        except Exception as e:
            self.logger.error(f"Error resetting system: {str(e)}")
            return False
