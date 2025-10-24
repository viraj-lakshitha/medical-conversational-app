"""
PDF document processing pipeline for medical documents
"""
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import fitz  # PyMuPDF
from dataclasses import dataclass

@dataclass


class DocumentMetadata:
    """Metadata extracted from medical documents"""
    document_type: str
    patient_id: Optional[str] = None
    date: Optional[datetime] = None
    source_file: str = ""
    page_count: int = 0
    extracted_at: datetime = None


    def __post_init__(self):
        if self.extracted_at is None:
            self.extracted_at = datetime.now()


class MedicalDocumentProcessor:
    """
    Processes medical PDF documents and extracts text with metadata
    """


    def __init__(self):
        self.logger = logging.getLogger(__name__)


    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, DocumentMetadata]:
        """
        Extract text from PDF and identify document metadata

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            doc = fitz.open(pdf_path)
            text_content = ""

            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content += page.get_text()
                text_content += "\n\n"  # Add page breaks

            # Extract metadata
            page_count = doc.page_count
            doc.close()

            metadata = self._extract_metadata(text_content, pdf_path, page_count)

            # Clean and normalize text
            cleaned_text = self._clean_text(text_content)

            return cleaned_text, metadata

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise


    def _extract_metadata(self, text: str, source_file: str, page_count: int) -> DocumentMetadata:
        """Extract metadata from document text"""

        # Determine document type based on content patterns
        doc_type = self._classify_document_type(text)

        # Extract patient ID (common patterns in medical documents)
        patient_id = self._extract_patient_id(text)

        # Extract dates
        doc_date = self._extract_document_date(text)

        return DocumentMetadata(
            document_type=doc_type,
            patient_id=patient_id,
            date=doc_date,
            source_file=os.path.basename(source_file),
            page_count=page_count
        )


    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content patterns"""
        text_lower = text.lower()

        if any(term in text_lower for term in ["clinical note", "progress note", "consultation"]):
            return "clinical_notes"
        elif any(term in text_lower for term in ["discharge summary", "hospital discharge"]):
            return "clinical_notes"
        elif any(term in text_lower for term in ["pathology report", "lab report", "laboratory"]):
            return "patient_records"
        elif any(term in text_lower for term in ["disease information", "medical condition", "diagnosis guide"]):
            return "disease_info"
        else:
            return "clinical_notes"  # Default


    def _extract_patient_id(self, text: str) -> Optional[str]:
        """Extract patient ID using common medical document patterns"""
        patterns = [
            r"patient\s+id\s*:?\s*(\w+)",
            r"mrn\s*:?\s*(\w+)",
            r"medical\s+record\s+number\s*:?\s*(\w+)",
            r"patient\s+number\s*:?\s*(\w+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None


    def _extract_document_date(self, text: str) -> Optional[datetime]:
        """Extract document date using common patterns"""
        date_patterns = [
            r"date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"date\s+of\s+service\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    date_str = matches[0]
                    # Try different date formats
                    for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"]:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
                except:
                    continue

        return None


    def _clean_text(self, text: str) -> str:
        """Clean and normalize medical text"""

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)

        # Preserve medical formatting but clean up
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Remove excessive line breaks

        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.,;:()/-]', '', text)

        return text.strip()


    def process_document(self, pdf_path: str) -> Dict:
        """
        Process a single document and return structured data

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with text content and metadata
        """
        text, metadata = self.extract_text_from_pdf(pdf_path)

        return {
            "text": text,
            "metadata": {
                "document_type": metadata.document_type,
                "patient_id": metadata.patient_id,
                "date": metadata.date.isoformat() if metadata.date else None,
                "source_file": metadata.source_file,
                "page_count": metadata.page_count,
                "extracted_at": metadata.extracted_at.isoformat()
            }
        }
