"""
ChromaDB vector store interface for medical documents
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import RAGConfig
from chunking import DocumentChunk


class MedicalVectorStore:
    """
    ChromaDB interface for storing and retrieving medical document vectors
    """


    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = RAGConfig()
        self.client = None
        self.collections = {}
        self._connect()


    def _connect(self):
        """Connect to ChromaDB instance"""
        try:
            self.client = chromadb.HttpClient(
                host=self.config.CHROMADB_HOST,
                port=self.config.CHROMADB_PORT,
                settings=Settings(allow_reset=True)
            )

            # Test connection
            self.client.heartbeat()
            self.logger.info(f"Connected to ChromaDB at {self.config.get_chromadb_url()}")

            # Initialize collections
            self._initialize_collections()

        except Exception as e:
            self.logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise


    def _initialize_collections(self):
        """Initialize ChromaDB collections for different document types"""

        for collection_name in self.config.COLLECTIONS.values():
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": f"Medical {collection_name} collection"}
                )
                self.collections[collection_name] = collection
                self.logger.info(f"Initialized collection: {collection_name}")

            except Exception as e:
                self.logger.error(f"Failed to initialize collection {collection_name}: {str(e)}")
                raise


    def add_documents(self, chunks: List[DocumentChunk], collection_name: str = None) -> bool:
        """
        Add document chunks to the appropriate collection

        Args:
            chunks: List of DocumentChunk objects
            collection_name: Specific collection name (optional)

        Returns:
            Success boolean
        """
        try:
            # Group chunks by document type if no specific collection provided
            if collection_name:
                collections_to_update = {collection_name: chunks}
            else:
                collections_to_update = self._group_chunks_by_type(chunks)

            for coll_name, chunk_list in collections_to_update.items():
                if coll_name not in self.collections:
                    self.logger.error(f"Collection {coll_name} not found")
                    continue

                # Prepare data for ChromaDB
                texts = [chunk.text for chunk in chunk_list]
                ids = [chunk.chunk_id for chunk in chunk_list]
                metadatas = [chunk.metadata for chunk in chunk_list]

                # Add additional metadata, filtering out None values
                for i, metadata in enumerate(metadatas):
                    # Remove None values and convert to string
                    clean_metadata = {}
                    for key, value in metadata.items():
                        if value is not None:
                            clean_metadata[key] = str(value)

                    metadatas[i] = {
                        **clean_metadata,
                        "chunk_index": str(chunk_list[i].chunk_index),
                        "document_id": str(chunk_list[i].document_id),
                        "start_char": str(chunk_list[i].start_char),
                        "end_char": str(chunk_list[i].end_char)
                    }

                # Add to collection
                collection = self.collections[coll_name]
                collection.add(
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas
                )

                self.logger.info(f"Added {len(chunk_list)} chunks to {coll_name} collection")

            return True

        except Exception as e:
            self.logger.error(f"Error adding documents: {str(e)}")
            return False


    def _group_chunks_by_type(self, chunks: List[DocumentChunk]) -> Dict[str, List[DocumentChunk]]:
        """Group chunks by their document type for appropriate collection assignment"""

        grouped = {}

        for chunk in chunks:
            doc_type = chunk.metadata.get("document_type", "clinical_notes")

            # Map to collection name
            collection_name = self.config.COLLECTIONS.get(doc_type, "clinical_notes")

            if collection_name not in grouped:
                grouped[collection_name] = []

            grouped[collection_name].append(chunk)

        return grouped


    def search_documents(
        self,
        query: str,
        collection_names: Optional[List[str]] = None,
        n_results: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity

        Args:
            query: Search query
            collection_names: Specific collections to search (optional)
            n_results: Number of results to return
            filter_metadata: Metadata filters for search

        Returns:
            List of matching documents with metadata
        """

        if n_results is None:
            n_results = self.config.MAX_RESULTS

        if collection_names is None:
            collection_names = list(self.collections.keys())

        all_results = []

        try:
            for collection_name in collection_names:
                if collection_name not in self.collections:
                    self.logger.warning(f"Collection {collection_name} not found")
                    continue

                collection = self.collections[collection_name]

                # Perform search
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter_metadata
                )

                # Process results
                if results['documents'] and results['documents'][0]:
                    for i in range(len(results['documents'][0])):
                        result = {
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i],
                            'collection': collection_name,
                            'id': results['ids'][0][i]
                        }
                        all_results.append(result)

            # Sort by similarity (lower distance = higher similarity)
            all_results.sort(key=lambda x: x['distance'])

            # Apply similarity threshold (distance is 0=identical, 1=very different)
            # Convert threshold to distance: if threshold=0.7, we want distance <= 0.3
            max_distance = 1.0 - self.config.SIMILARITY_THRESHOLD
            filtered_results = [
                result for result in all_results
                if result['distance'] <= max_distance
            ]

            return filtered_results[:n_results]

        except Exception as e:
            self.logger.error(f"Error searching documents: {str(e)}")
            return []


    def get_collection_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all collections"""

        info = {}

        for name, collection in self.collections.items():
            try:
                count = collection.count()
                info[name] = {
                    "count": count,
                    "name": name
                }
            except Exception as e:
                self.logger.error(f"Error getting info for collection {name}: {str(e)}")
                info[name] = {"error": str(e)}

        return info


    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its documents"""

        try:
            if collection_name in self.collections:
                self.client.delete_collection(name=collection_name)
                del self.collections[collection_name]
                self.logger.info(f"Deleted collection: {collection_name}")
                return True
            else:
                self.logger.warning(f"Collection {collection_name} not found")
                return False

        except Exception as e:
            self.logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False


    def reset_collections(self) -> bool:
        """Reset all collections (delete all data)"""

        try:
            for collection_name in list(self.collections.keys()):
                self.delete_collection(collection_name)

            # Reinitialize collections
            self._initialize_collections()
            return True

        except Exception as e:
            self.logger.error(f"Error resetting collections: {str(e)}")
            return False
