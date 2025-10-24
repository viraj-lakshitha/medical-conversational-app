"""
Document chunking strategies optimized for medical content
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from config import RAGConfig

@dataclass


class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    text: str
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class MedicalDocumentChunker:
    """
    Intelligent chunking for medical documents that preserves medical context
    """


    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or RAGConfig.CHUNK_SIZE
        self.overlap = overlap or RAGConfig.CHUNK_OVERLAP

        # Medical section headers that should start new chunks
        self.section_headers = [
            r"chief complaint",
            r"history of present illness",
            r"past medical history",
            r"medications",
            r"physical examination",
            r"assessment and plan",
            r"impression",
            r"diagnosis",
            r"treatment plan",
            r"laboratory results",
            r"vital signs",
            r"allergies",
            r"social history",
            r"family history",
            r"review of systems"
        ]


    def chunk_document(self, text: str, document_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Chunk a medical document using intelligent strategies

        Args:
            text: Document text
            document_metadata: Metadata about the document

        Returns:
            List of DocumentChunk objects
        """

        # Try section-based chunking first
        chunks = self._section_based_chunking(text, document_metadata)

        # If section-based chunking doesn't work well, fall back to semantic chunking
        if len(chunks) == 1 and len(chunks[0].text) > self.chunk_size * 2:
            chunks = self._semantic_chunking(text, document_metadata)

        # If still too large, use sliding window
        final_chunks = []
        for chunk in chunks:
            if len(chunk.text) > self.chunk_size:
                sub_chunks = self._sliding_window_chunking(
                    chunk.text,
                    document_metadata,
                    start_index=len(final_chunks)
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks


    def _section_based_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split document by medical sections"""

        chunks = []
        document_id = f"{metadata.get('source_file', 'unknown')}_{metadata.get('extracted_at', '')}"

        # Find section boundaries
        section_positions = []
        for header_pattern in self.section_headers:
            matches = re.finditer(rf"^{header_pattern}:?\s*$", text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                section_positions.append((match.start(), header_pattern))

        # Sort by position
        section_positions.sort(key=lambda x: x[0])

        if not section_positions:
            # No sections found, return whole document as single chunk
            return [DocumentChunk(
                text=text,
                chunk_id=f"{document_id}_chunk_0",
                document_id=document_id,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
                metadata=metadata
            )]

        # Create chunks based on sections
        for i, (start_pos, section_name) in enumerate(section_positions):
            end_pos = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(text)

            section_text = text[start_pos:end_pos].strip()

            if section_text:  # Skip empty sections
                chunk = DocumentChunk(
                    text=section_text,
                    chunk_id=f"{document_id}_section_{i}",
                    document_id=document_id,
                    chunk_index=i,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={**metadata, "section": section_name}
                )
                chunks.append(chunk)

        return chunks


    def _semantic_chunking(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Chunk based on semantic boundaries like paragraphs and sentences"""

        chunks = []
        document_id = f"{metadata.get('source_file', 'unknown')}_{metadata.get('extracted_at', '')}"

        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""
        chunk_index = 0
        start_char = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk = DocumentChunk(
                    text=current_chunk.strip(),
                    chunk_id=f"{document_id}_semantic_{chunk_index}",
                    document_id=document_id,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    metadata=metadata
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                    start_char += len(current_chunk) - len(overlap_text) - len(paragraph) - 2
                else:
                    current_chunk = paragraph
                    start_char += len(current_chunk)

                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                chunk_id=f"{document_id}_semantic_{chunk_index}",
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                metadata=metadata
            )
            chunks.append(chunk)

        return chunks


    def _sliding_window_chunking(self, text: str, metadata: Dict[str, Any], start_index: int = 0) -> List[DocumentChunk]:
        """Fallback sliding window chunking for large texts"""

        chunks = []
        document_id = f"{metadata.get('source_file', 'unknown')}_{metadata.get('extracted_at', '')}"

        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk_text = text[i:i + self.chunk_size]

            if not chunk_text.strip():
                continue

            chunk = DocumentChunk(
                text=chunk_text,
                chunk_id=f"{document_id}_window_{start_index + len(chunks)}",
                document_id=document_id,
                chunk_index=start_index + len(chunks),
                start_char=i,
                end_char=min(i + self.chunk_size, len(text)),
                metadata=metadata
            )
            chunks.append(chunk)

            # Break if we've reached the end
            if i + self.chunk_size >= len(text):
                break

        return chunks


    def chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage"""
        return {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata
        }
