"""
Evaluation metrics and testing framework for Named Entity Disambiguation.
Provides comprehensive evaluation of NED pipeline performance.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass


class EntityMatch:
    predicted_entity: Dict
    ground_truth_entity: Dict
    match_type: str  # 'exact', 'partial', 'type_mismatch', 'no_match'
    overlap_ratio: float = 0.0

@dataclass


class EvaluationResult:
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    total_predictions: int
    total_ground_truth: int
    correct_predictions: int

    # Detailed metrics
    entity_level_metrics: Dict
    type_level_metrics: Dict

    # Error analysis
    false_positives: List
    false_negatives: List
    type_errors: List


class NEDEvaluator:


    def __init__(self):
        self.evaluation_results = []


    def evaluate_predictions(self, predictions: List[Dict], ground_truth: List[Dict],
                           text: str) -> Dict:
        """
        Evaluate predictions against ground truth for a single text

        Args:
            predictions: List of predicted entities
            ground_truth: List of ground truth entities
            text: Original text

        Returns:
            Dictionary containing evaluation metrics
        """

        # Convert to consistent format
        pred_entities = self._normalize_entities(predictions, 'prediction')
        gt_entities = self._normalize_entities(ground_truth, 'ground_truth')

        # Find matches
        matches = self._find_entity_matches(pred_entities, gt_entities, text)

        # Calculate metrics
        metrics = self._calculate_metrics(matches, pred_entities, gt_entities)

        return metrics


    def _normalize_entities(self, entities: List[Dict], source: str) -> List[Dict]:
        """Normalize entity format"""
        normalized = []

        for entity in entities:
            if source == 'prediction':
                normalized.append({
                    'text': entity.get('mention', entity.get('text', '')),
                    'type': entity.get('concept_type', entity.get('type', '')),
                    'start': entity.get('start_pos', entity.get('position', 0)),
                    'end': entity.get('end_pos', entity.get('position', 0) + len(entity.get('text', ''))),
                    'concept_id': entity.get('concept_id', ''),
                    'confidence': entity.get('confidence_score', 0.0)
                })
            else:  # ground_truth
                text = entity.get('entity', entity.get('text', ''))
                start = entity.get('position', 0)
                normalized.append({
                    'text': text,
                    'type': entity.get('type', 'UNKNOWN'),
                    'start': start,
                    'end': start + len(text),
                    'concept_id': entity.get('id', ''),
                    'confidence': 1.0
                })

        return normalized


    def _find_entity_matches(self, predictions: List[Dict], ground_truth: List[Dict],
                           text: str) -> List[EntityMatch]:
        """Find matches between predictions and ground truth"""
        matches = []
        used_gt_indices = set()

        for pred in predictions:
            best_match = None
            best_overlap = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(ground_truth):
                if gt_idx in used_gt_indices:
                    continue

                # Calculate overlap
                overlap_start = max(pred['start'], gt['start'])
                overlap_end = min(pred['end'], gt['end'])

                if overlap_start < overlap_end:
                    overlap_length = overlap_end - overlap_start
                    pred_length = pred['end'] - pred['start']
                    gt_length = gt['end'] - gt['start']

                    # Calculate overlap ratio
                    overlap_ratio = overlap_length / max(pred_length, gt_length)

                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_gt_idx = gt_idx

                        # Determine match type
                        if overlap_ratio >= 0.8:  # High overlap threshold
                            if pred['type'] == gt['type']:
                                match_type = 'exact'
                            else:
                                match_type = 'type_mismatch'
                        else:
                            match_type = 'partial'

                        best_match = EntityMatch(
                            predicted_entity=pred,
                            ground_truth_entity=gt,
                            match_type=match_type,
                            overlap_ratio=overlap_ratio
                        )

            if best_match:
                matches.append(best_match)
                used_gt_indices.add(best_gt_idx)
            else:
                # No match found - false positive
                matches.append(EntityMatch(
                    predicted_entity=pred,
                    ground_truth_entity=None,
                    match_type='no_match',
                    overlap_ratio=0.0
                ))

        # Add unmatched ground truth entities as false negatives
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx not in used_gt_indices:
                matches.append(EntityMatch(
                    predicted_entity=None,
                    ground_truth_entity=gt,
                    match_type='missing',
                    overlap_ratio=0.0
                ))

        return matches


    def _calculate_metrics(self, matches: List[EntityMatch],
                          predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate evaluation metrics from matches"""

        # Count different types of matches
        exact_matches = len([m for m in matches if m.match_type == 'exact'])
        partial_matches = len([m for m in matches if m.match_type == 'partial'])
        type_mismatches = len([m for m in matches if m.match_type == 'type_mismatch'])
        false_positives = len([m for m in matches if m.match_type == 'no_match'])
        false_negatives = len([m for m in matches if m.match_type == 'missing'])

        # Calculate basic metrics
        correct_predictions = exact_matches
        total_predictions = len(predictions)
        total_ground_truth = len(ground_truth)

        precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = correct_predictions / max(total_predictions, total_ground_truth) if max(total_predictions, total_ground_truth) > 0 else 0.0

        # Entity-level metrics
        entity_metrics = {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'type_mismatches': type_mismatches,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

        # Type-level metrics
        type_metrics = self._calculate_type_metrics(matches)

        # Error analysis
        errors = {
            'false_positives': [m.predicted_entity for m in matches if m.match_type == 'no_match'],
            'false_negatives': [m.ground_truth_entity for m in matches if m.match_type == 'missing'],
            'type_errors': [m for m in matches if m.match_type == 'type_mismatch']
        }

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'correct_predictions': correct_predictions,
            'entity_metrics': entity_metrics,
            'type_metrics': type_metrics,
            'errors': errors
        }


    def _calculate_type_metrics(self, matches: List[EntityMatch]) -> Dict:
        """Calculate per-type evaluation metrics"""
        type_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for match in matches:
            if match.match_type == 'exact':
                entity_type = match.predicted_entity['type']
                type_stats[entity_type]['tp'] += 1
            elif match.match_type == 'no_match':
                entity_type = match.predicted_entity['type']
                type_stats[entity_type]['fp'] += 1
            elif match.match_type == 'missing':
                entity_type = match.ground_truth_entity['type']
                type_stats[entity_type]['fn'] += 1
            elif match.match_type == 'type_mismatch':
                pred_type = match.predicted_entity['type']
                gt_type = match.ground_truth_entity['type']
                type_stats[pred_type]['fp'] += 1
                type_stats[gt_type]['fn'] += 1

        # Calculate metrics for each type
        type_metrics = {}
        for entity_type, stats in type_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            type_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn
            }

        return type_metrics


    def evaluate_corpus(self, predictions_file: str, ground_truth_file: str) -> EvaluationResult:
        """Evaluate on entire corpus"""
        logger.info("Evaluating corpus...")

        # Load predictions
        predictions_data = []
        with open(predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                predictions_data.append(json.loads(line))

        # Load ground truth
        ground_truth_data = []
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                ground_truth_data.append(json.loads(line))

        # Evaluate each text
        all_metrics = []
        for pred_data, gt_data in zip(predictions_data, ground_truth_data):
            # Extract entities from ground truth
            gt_entities = []
            for disease in gt_data.get('diseases', []):
                gt_entities.append({
                    'entity': disease['entity'],
                    'type': 'DISEASE',
                    'position': disease['position']
                })
            for symptom in gt_data.get('symptoms', []):
                gt_entities.append({
                    'entity': symptom['entity'],
                    'type': 'SYMPTOM',
                    'position': symptom['position']
                })

            # Get predictions
            pred_entities = pred_data.get('entities', [])

            # Evaluate
            metrics = self.evaluate_predictions(pred_entities, gt_entities, gt_data['text'])
            all_metrics.append(metrics)

        # Aggregate metrics
        return self._aggregate_metrics(all_metrics)


    def _aggregate_metrics(self, metrics_list: List[Dict]) -> EvaluationResult:
        """Aggregate metrics across all texts"""

        # Calculate weighted averages
        total_predictions = sum(m['total_predictions'] for m in metrics_list)
        total_ground_truth = sum(m['total_ground_truth'] for m in metrics_list)
        total_correct = sum(m['correct_predictions'] for m in metrics_list)

        # Overall metrics
        overall_precision = total_correct / total_predictions if total_predictions > 0 else 0.0
        overall_recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        overall_accuracy = total_correct / max(total_predictions, total_ground_truth) if max(total_predictions, total_ground_truth) > 0 else 0.0

        # Aggregate entity-level metrics
        entity_metrics = {
            'exact_matches': sum(m['entity_metrics']['exact_matches'] for m in metrics_list),
            'partial_matches': sum(m['entity_metrics']['partial_matches'] for m in metrics_list),
            'type_mismatches': sum(m['entity_metrics']['type_mismatches'] for m in metrics_list),
            'false_positives': sum(m['entity_metrics']['false_positives'] for m in metrics_list),
            'false_negatives': sum(m['entity_metrics']['false_negatives'] for m in metrics_list)
        }

        # Aggregate type-level metrics
        all_type_metrics = defaultdict(list)
        for m in metrics_list:
            for entity_type, type_metrics in m['type_metrics'].items():
                all_type_metrics[entity_type].append(type_metrics)

        aggregated_type_metrics = {}
        for entity_type, type_metrics_list in all_type_metrics.items():
            # Weighted average by support
            total_support = sum(tm['support'] for tm in type_metrics_list)
            if total_support > 0:
                avg_precision = sum(tm['precision'] * tm['support'] for tm in type_metrics_list) / total_support
                avg_recall = sum(tm['recall'] * tm['support'] for tm in type_metrics_list) / total_support
                avg_f1 = sum(tm['f1_score'] * tm['support'] for tm in type_metrics_list) / total_support
            else:
                avg_precision = avg_recall = avg_f1 = 0.0

            aggregated_type_metrics[entity_type] = {
                'precision': avg_precision,
                'recall': avg_recall,
                'f1_score': avg_f1,
                'support': total_support
            }

        # Aggregate errors
        all_false_positives = []
        all_false_negatives = []
        all_type_errors = []

        for m in metrics_list:
            all_false_positives.extend(m['errors']['false_positives'])
            all_false_negatives.extend(m['errors']['false_negatives'])
            all_type_errors.extend(m['errors']['type_errors'])

        return EvaluationResult(
            precision=overall_precision,
            recall=overall_recall,
            f1_score=overall_f1,
            accuracy=overall_accuracy,
            total_predictions=total_predictions,
            total_ground_truth=total_ground_truth,
            correct_predictions=total_correct,
            entity_level_metrics=entity_metrics,
            type_level_metrics=aggregated_type_metrics,
            false_positives=all_false_positives,
            false_negatives=all_false_negatives,
            type_errors=all_type_errors
        )


    def generate_evaluation_report(self, result: EvaluationResult, output_file: str = None):
        """Generate comprehensive evaluation report"""

        report = []
        report.append("=" * 60)
        report.append("NAMED ENTITY DISAMBIGUATION EVALUATION REPORT")
        report.append("=" * 60)

        # Overall metrics
        report.append("\nOVERALL METRICS")
        report.append("-" * 30)
        report.append(f"Precision: {result.precision:.4f}")
        report.append(f"Recall: {result.recall:.4f}")
        report.append(f"F1-Score: {result.f1_score:.4f}")
        report.append(f"Accuracy: {result.accuracy:.4f}")
        report.append(f"Total Predictions: {result.total_predictions}")
        report.append(f"Total Ground Truth: {result.total_ground_truth}")
        report.append(f"Correct Predictions: {result.correct_predictions}")

        # Entity-level metrics
        report.append("\nENTITY-LEVEL METRICS")
        report.append("-" * 30)
        for metric, value in result.entity_level_metrics.items():
            report.append(f"{metric.replace('_', ' ').title()}: {value}")

        # Type-level metrics
        report.append("\nTYPE-LEVEL METRICS")
        report.append("-" * 30)
        for entity_type, metrics in result.type_level_metrics.items():
            report.append(f"\n{entity_type}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"  Support: {metrics['support']}")

        # Error analysis
        report.append("\nERROR ANALYSIS")
        report.append("-" * 30)
        report.append(f"False Positives: {len(result.false_positives)}")
        report.append(f"False Negatives: {len(result.false_negatives)}")
        report.append(f"Type Errors: {len(result.type_errors)}")

        # Most common errors
        if result.false_positives:
            fp_texts = [fp['text'] for fp in result.false_positives]
            common_fps = Counter(fp_texts).most_common(5)
            report.append("\nMost common false positives:")
            for text, count in common_fps:
                report.append(f"  - '{text}': {count} times")

        if result.false_negatives:
            fn_texts = [fn['text'] for fn in result.false_negatives]
            common_fns = Counter(fn_texts).most_common(5)
            report.append("\nMost common false negatives:")
            for text, count in common_fns:
                report.append(f"  - '{text}': {count} times")

        report_text = "\n".join(report)

        # Print report
        print(report_text)

        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_file}")

        return report_text


    def plot_evaluation_metrics(self, result: EvaluationResult, save_path: str = None):
        """Create visualization plots for evaluation results"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Named Entity Disambiguation Evaluation Results', fontsize=16)

        # 1. Overall metrics bar chart
        ax1 = axes[0, 0]
        metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        values = [result.precision, result.recall, result.f1_score, result.accuracy]
        bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Score')
        ax1.set_title('Overall Performance Metrics')
        ax1.set_ylim(0, 1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # 2. Entity-level metrics pie chart
        ax2 = axes[0, 1]
        entity_counts = [
            result.entity_level_metrics['exact_matches'],
            result.entity_level_metrics['partial_matches'],
            result.entity_level_metrics['type_mismatches'],
            result.entity_level_metrics['false_positives'],
            result.entity_level_metrics['false_negatives']
        ]
        entity_labels = ['Exact Match', 'Partial Match', 'Type Mismatch', 'False Positive', 'False Negative']
        colors = ['#2ca02c', '#ffbb33', '#ff7f0e', '#d62728', '#9467bd']

        ax2.pie(entity_counts, labels=entity_labels, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Entity-Level Results Distribution')

        # 3. Type-level F1 scores
        ax3 = axes[1, 0]
        type_names = list(result.type_level_metrics.keys())
        f1_scores = [result.type_level_metrics[t]['f1_score'] for t in type_names]

        bars = ax3.bar(type_names, f1_scores, color='#2ca02c')
        ax3.set_ylabel('F1-Score')
        ax3.set_title('F1-Score by Entity Type')
        ax3.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')

        # 4. Support by entity type
        ax4 = axes[1, 1]
        supports = [result.type_level_metrics[t]['support'] for t in type_names]

        bars = ax4.bar(type_names, supports, color='#1f77b4')
        ax4.set_ylabel('Support (Number of Entities)')
        ax4.set_title('Support by Entity Type')
        ax4.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar, support in zip(bars, supports):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{support}', ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")

        plt.show()


def main():
    """Example usage"""
    evaluator = NEDEvaluator()

    # Example evaluation (you would replace these with actual file paths)
    # result = evaluator.evaluate_corpus('predictions.jsonl', 'ground_truth.jsonl')
    # evaluator.generate_evaluation_report(result, 'evaluation_report.txt')
    # evaluator.plot_evaluation_metrics(result, 'evaluation_plots.png')

    # Demo with sample data
    sample_predictions = [
        {'mention': 'diabetes', 'concept_type': 'DISEASE', 'start_pos': 10, 'end_pos': 18, 'confidence_score': 0.95},
        {'mention': 'chest pain', 'concept_type': 'SYMPTOM', 'start_pos': 30, 'end_pos': 40, 'confidence_score': 0.87}
    ]

    sample_ground_truth = [
        {'entity': 'diabetes', 'type': 'DISEASE', 'position': 10},
        {'entity': 'chest pain', 'type': 'SYMPTOM', 'position': 30}
    ]

    sample_text = "Patient has diabetes and reports chest pain."

    metrics = evaluator.evaluate_predictions(sample_predictions, sample_ground_truth, sample_text)

    print("Sample evaluation results:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")

if __name__ == "__main__":
    main()
