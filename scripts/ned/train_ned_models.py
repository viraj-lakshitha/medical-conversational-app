"""
Training script for the complete Named Entity Disambiguation pipeline.
Trains both the BioBERT NER model and the neural ranking model.
"""

import argparse
import logging
import os
import json
from pathlib import Path
from typing import Tuple

from biobert_ner_model import OptimizedMedicalNERTrainer
from neural_ranking_model import FastEntityDisambiguator
from knowledge_base_interface import KnowledgeBaseInterface
from ned_pipeline import OptimizedNEDPipeline
from evaluation_metrics import NEDEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Named Entity Disambiguation Models")

    # Data arguments
    parser.add_argument("--annotated_corpus",
                       default="data_preparation/data/full_annotated_corpus.jsonl",
                       help="Path to annotated corpus")
    parser.add_argument("--test_split", type=float, default=0.2,
                       help="Proportion of data to use for testing")

    # Model arguments
    parser.add_argument("--medical_model",
                       default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                       help="Medical BERT model name")
    parser.add_argument("--ner_output_dir", default="./optimized_medical_ner_model",
                       help="Output directory for NER model")
    parser.add_argument("--ranking_output_path", default="./fast_ranking_model.pth",
                       help="Output path for ranking model")

    # Training arguments - memory-efficient with LoRA
    parser.add_argument("--ner_epochs", type=int, default=2,
                       help="Number of epochs for NER training")
    parser.add_argument("--ner_batch_size", type=int, default=16,
                       help="Batch size for NER training (conservative for LoRA)")
    parser.add_argument("--ranking_epochs", type=int, default=2,
                       help="Number of epochs for ranking model training")
    parser.add_argument("--ranking_batch_size", type=int, default=64,
                       help="Batch size for ranking model training (LoRA allows larger)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for memory-efficient training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate for ranking model")

    # Neo4j arguments
    parser.add_argument("--neo4j_uri", default="bolt://localhost:7687",
                       help="Neo4j URI")
    parser.add_argument("--neo4j_user", default="neo4j",
                       help="Neo4j username")
    parser.add_argument("--neo4j_password", default="password",
                       help="Neo4j password")

    # Evaluation arguments
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after training")
    parser.add_argument("--eval_output_dir", default="./evaluation_results",
                       help="Directory for evaluation outputs")

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.ner_output_dir, exist_ok=True)
    if args.evaluate:
        os.makedirs(args.eval_output_dir, exist_ok=True)

    print(" Starting Memory-Efficient NED Model Training Pipeline")
    print(f" Model: {args.medical_model}")
    print(f" LoRA Enabled: {getattr(args, 'use_lora', True)}")
    print(f" Training: NER({args.ner_epochs}e, bs{args.ner_batch_size}) + Ranking({args.ranking_epochs}e, bs{args.ranking_batch_size})")
    print("=" * 80)

    # Check if annotated corpus exists
    if not os.path.exists(args.annotated_corpus):
        logger.error(f"Annotated corpus not found: {args.annotated_corpus}")
        logger.error("Please run data preparation first:")
        logger.error("cd data_preparation && python run_full_annotation.py")
        return

    # Step 1: Prepare data splits
    logger.info("Preparing data splits...")
    train_data, test_data = prepare_data_splits(args.annotated_corpus, args.test_split)

    # Step 2: Train Memory-Efficient Medical NER Model
    logger.info(" Training Memory-Efficient Medical NER Model with LoRA...")
    ner_trainer = OptimizedMedicalNERTrainer(args.medical_model, use_lora=True)

    try:
        ner_trainer.train(
            annotated_corpus_path=train_data,
            output_dir=args.ner_output_dir,
            epochs=args.ner_epochs,
            batch_size=args.ner_batch_size
        )
        logger.info("NER model training completed")
    except Exception as e:
        logger.error(f"NER model training failed: {e}")
        return

    # Step 3: Train Memory-Efficient Ranking Model
    logger.info(" Training Memory-Efficient Ranking Model with LoRA...")
    disambiguator = FastEntityDisambiguator(use_lora=True)

    try:
        training_data = disambiguator.prepare_training_data(train_data)
        disambiguator.train(
            training_data=training_data,
            epochs=args.ranking_epochs,
            batch_size=args.ranking_batch_size,
            learning_rate=args.learning_rate
        )

        # Save ranking model
        disambiguator.save_model(args.ranking_output_path)
        logger.info("Neural ranking model training completed")
    except Exception as e:
        logger.error(f"Neural ranking model training failed: {e}")
        return

    # Step 4: Evaluation (if requested)
    if args.evaluate:
        logger.info("Running Evaluation...")

        try:
            # Initialize optimized pipeline with trained models
            pipeline = OptimizedNEDPipeline(
                ner_model_path=args.ner_output_dir,
                ranking_model_path=args.ranking_output_path
            )

            # Run evaluation on test set
            evaluator = NEDEvaluator()

            # Process test data through pipeline
            test_predictions_file = os.path.join(args.eval_output_dir, "test_predictions.jsonl")
            create_predictions_file(pipeline, test_data, test_predictions_file)

            # Evaluate
            result = evaluator.evaluate_corpus(test_predictions_file, test_data)

            # Generate report
            report_file = os.path.join(args.eval_output_dir, "evaluation_report.txt")
            evaluator.generate_evaluation_report(result, report_file)

            # Generate plots
            plot_file = os.path.join(args.eval_output_dir, "evaluation_plots.png")
            evaluator.plot_evaluation_metrics(result, plot_file)

            logger.info("Evaluation completed")
            logger.info(f"Results saved in {args.eval_output_dir}")

            # Print summary
            print("\n" + "="*50)
            print("EVALUATION SUMMARY")
            print("="*50)
            print(f"Precision: {result.precision:.4f}")
            print(f"Recall: {result.recall:.4f}")
            print(f"F1-Score: {result.f1_score:.4f}")
            print(f"Accuracy: {result.accuracy:.4f}")
            print(f"Total Entities: {result.total_ground_truth}")
            print(f"Correct Predictions: {result.correct_predictions}")

            pipeline.close()

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return

    logger.info("Training pipeline completed successfully!")
    logger.info(f"NER model saved to: {args.ner_output_dir}")
    logger.info(f"Ranking model saved to: {args.ranking_output_path}")

    if args.evaluate:
        logger.info(f"Evaluation results saved to: {args.eval_output_dir}")


def prepare_data_splits(corpus_path: str, test_split: float) -> Tuple[str, str]:
    """Split annotated corpus into train and test sets"""

    # Read all data
    all_data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_data.append(json.loads(line))

    # Shuffle data
    import random
    random.seed(42)
    random.shuffle(all_data)

    # Split data
    split_idx = int(len(all_data) * (1 - test_split))
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # Save splits
    train_file = corpus_path.replace('.jsonl', '_train.jsonl')
    test_file = corpus_path.replace('.jsonl', '_test.jsonl')

    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

    logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test")

    return train_file, test_file


def create_predictions_file(pipeline, test_data_file: str, output_file: str):
    """Create predictions file for evaluation"""

    predictions = []

    with open(test_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data['text']

            # Process through pipeline
            result = pipeline.process_text(text)

            # Store prediction
            predictions.append({
                'id': data.get('id', ''),
                'text': text,
                'entities': result.entities
            })

    # Save predictions
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')

    logger.info(f"Created predictions file with {len(predictions)} examples")

if __name__ == "__main__":
    main()
