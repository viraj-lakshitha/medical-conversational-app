#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from auto_annotator import MedicalEntityAnnotator


def main():
    parser = argparse.ArgumentParser(description="Run full corpus annotation")
    parser.add_argument("--input", default="data/annotation_corpus.jsonl", help="Input corpus file")
    parser.add_argument("--output", default="data/full_annotated_corpus.jsonl", help="Output file")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size (default: 25)")
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j_user", default="neo4j", help="Neo4j user")
    parser.add_argument("--neo4j_password", default="password", help="Neo4j password")

    args = parser.parse_args()

    print("=== Medical NED Data Preparation Pipeline ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Check input file
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} not found!")
        print("Run download_and_prepare_data.py first to create the corpus.")
        return 1

    # Initialize annotator
    print("Initializing medical entity annotator...")
    annotator = MedicalEntityAnnotator(
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password
    )

    if not annotator.kg_builder.driver:
        print("Error: Could not connect to Neo4j database!")
        print("Please ensure Neo4j is running and knowledge base is populated.")
        return 1

    print(f"Connected to Neo4j - {len(annotator.disease_entities)} diseases, {len(annotator.symptom_entities)} symptoms loaded")
    print()

    # Run annotation
    print("Starting annotation process...")
    success = annotator.annotate_corpus(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size
    )

    if success:
        print()
        print("=== Annotation Completed Successfully! ===")
        print(f"Results saved to: {args.output}")
        print()
        print("Next steps:")
        print("1. Review annotation results")
        print("2. Extract training data: python auto_annotator.py --extract_training")
        print("3. Use annotated data for NER model training")
    else:
        print("Annotation failed!")
        return 1

    annotator.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
