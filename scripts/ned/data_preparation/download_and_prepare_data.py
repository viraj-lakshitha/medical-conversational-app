import pandas as pd
import numpy as np
from datasets import load_dataset
import json
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalDatasetDownloader:


    def __init__(self, output_dir="./data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.dataset = None
        self.combined_data = []


    def download_dataset(self, dataset_name="FreedomIntelligence/medical-o1-reasoning-SFT", config="en"):
        """Download the medical reasoning dataset from HuggingFace"""
        try:
            logger.info(f"Downloading dataset: {dataset_name} (config: {config})")
            self.dataset = load_dataset(dataset_name, config)
            logger.info(f"Successfully downloaded dataset with {len(self.dataset)} splits")

            for split in self.dataset.keys():
                logger.info(f"Split '{split}': {len(self.dataset[split])} examples")

            return True

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return False


    def combine_dataset_columns(self):
        """Combine all columns into a single text for processing"""
        if not self.dataset:
            logger.error("No dataset loaded. Call download_dataset() first.")
            return False

        combined_texts = []

        for split_name, split_data in self.dataset.items():
            logger.info(f"Processing split: {split_name}")

            # First, let's examine the structure of a few examples
            if len(split_data) > 0:
                sample_example = split_data[0]
                logger.info(f"Sample example fields: {list(sample_example.keys())}")
                for key, value in sample_example.items():
                    if isinstance(value, str):
                        logger.info(f"Field '{key}': {value[:100]}..." if len(value) > 100 else f"Field '{key}': {value}")
                    else:
                        logger.info(f"Field '{key}': {type(value)} - {value}")

            for idx, example in enumerate(split_data):
                # Extract all text content from the example
                text_parts = []

                # Check all available fields in the example
                for field, value in example.items():
                    if value:  # Skip empty/None values
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, list):
                            # Handle list of messages/conversations
                            for item in value:
                                if isinstance(item, dict):
                                    # Extract text from message dict
                                    for key in ['content', 'text', 'message', 'value']:
                                        if key in item and isinstance(item[key], str):
                                            text_parts.append(item[key])
                                elif isinstance(item, str):
                                    text_parts.append(item)
                        elif isinstance(value, dict):
                            # Handle nested dictionaries
                            for key in ['content', 'text', 'message', 'value']:
                                if key in value and isinstance(value[key], str):
                                    text_parts.append(value[key])

                # Combine all text parts
                combined_text = " ".join(text_parts).strip()

                if combined_text:
                    combined_texts.append({
                        'id': f"{split_name}_{idx}",
                        'split': split_name,
                        'original_example': example,
                        'combined_text': combined_text,
                        'text_length': len(combined_text)
                    })

        self.combined_data = combined_texts
        logger.info(f"Combined {len(combined_texts)} examples with text content")

        # Save combined data
        combined_df = pd.DataFrame([{
            'id': item['id'],
            'split': item['split'],
            'combined_text': item['combined_text'],
            'text_length': item['text_length']
        } for item in combined_texts])

        output_path = self.output_dir / "combined_medical_texts.csv"
        combined_df.to_csv(output_path, index=False)
        logger.info(f"Saved combined data to {output_path}")

        return True


    def prepare_annotation_corpus(self, min_length=50, max_length=5000):
        """Prepare text corpus for disease-symptom annotation"""
        if not self.combined_data:
            logger.error("No combined data available. Call combine_dataset_columns() first.")
            return False

        # Filter texts by length
        filtered_texts = [
            item for item in self.combined_data
            if min_length <= item['text_length'] <= max_length
        ]

        logger.info(f"Filtered to {len(filtered_texts)} texts (length: {min_length}-{max_length} chars)")

        # Create annotation corpus
        annotation_corpus = []

        for item in filtered_texts:
            # Clean text for better processing
            cleaned_text = self._clean_text(item['combined_text'])

            if cleaned_text and len(cleaned_text) >= min_length:
                annotation_corpus.append({
                    'id': item['id'],
                    'split': item['split'],
                    'text': cleaned_text,
                    'original_length': item['text_length'],
                    'cleaned_length': len(cleaned_text),
                    'source': 'FreedomIntelligence/medical-o1-reasoning-SFT'
                })

        # Save annotation corpus
        corpus_df = pd.DataFrame(annotation_corpus)
        output_path = self.output_dir / "annotation_corpus.csv"
        corpus_df.to_csv(output_path, index=False)
        logger.info(f"Saved annotation corpus to {output_path} ({len(annotation_corpus)} texts)")

        # Also save as JSONL for easier processing
        jsonl_path = self.output_dir / "annotation_corpus.jsonl"
        with open(jsonl_path, 'w') as f:
            for item in annotation_corpus:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved annotation corpus as JSONL to {jsonl_path}")

        # Save statistics
        self._save_corpus_statistics(annotation_corpus)

        return annotation_corpus


    def _clean_text(self, text):
        """Clean and normalize text for annotation"""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{2,}', '--', text)

        # Basic sentence boundary cleanup
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)

        return text.strip()


    def _save_corpus_statistics(self, annotation_corpus):
        """Save statistics about the annotation corpus"""
        stats = {
            'total_texts': len(annotation_corpus),
            'avg_length': np.mean([item['cleaned_length'] for item in annotation_corpus]),
            'median_length': np.median([item['cleaned_length'] for item in annotation_corpus]),
            'min_length': min([item['cleaned_length'] for item in annotation_corpus]),
            'max_length': max([item['cleaned_length'] for item in annotation_corpus]),
            'splits': {}
        }

        # Count by split
        for item in annotation_corpus:
            split = item['split']
            if split not in stats['splits']:
                stats['splits'][split] = 0
            stats['splits'][split] += 1

        stats_path = self.output_dir / "corpus_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Corpus statistics saved to {stats_path}")
        logger.info(f"Total texts: {stats['total_texts']}")
        logger.info(f"Average length: {stats['avg_length']:.1f} characters")
        logger.info(f"Length range: {stats['min_length']}-{stats['max_length']} characters")

        for split, count in stats['splits'].items():
            logger.info(f"Split '{split}': {count} texts")


def main():
    """Main function to download and prepare data"""
    downloader = MedicalDatasetDownloader()

    # Download dataset
    if not downloader.download_dataset():
        logger.error("Failed to download dataset")
        return False

    # Combine columns
    if not downloader.combine_dataset_columns():
        logger.error("Failed to combine dataset columns")
        return False

    # Prepare annotation corpus
    annotation_corpus = downloader.prepare_annotation_corpus()
    if not annotation_corpus:
        logger.error("Failed to prepare annotation corpus")
        return False

    logger.info("Data preparation completed successfully!")
    logger.info("Ready for annotation with auto_annotator.py")

    return True

if __name__ == "__main__":
    main()
