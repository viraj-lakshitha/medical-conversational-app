"""
End-to-end Named Entity Disambiguation Pipeline.
Combines BioBERT NER, knowledge base candidate generation, and neural ranking
for accurate medical entity disambiguation.
"""

import logging
import json
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import torch

from biobert_ner_model import OptimizedMedicalNERTrainer, EntitySpan
from knowledge_base_interface import KnowledgeBaseInterface, CandidateConcept
from neural_ranking_model import FastEntityDisambiguator, DisambiguationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass


class NEDResult:
    text: str
    entities: List[Dict]
    processing_time: float
    confidence_threshold: float = 0.5


    def to_dict(self):
        return asdict(self)

@dataclass


class EntityResult:
    mention: str
    start_pos: int
    end_pos: int
    concept_id: str
    concept_name: str
    display_name: str
    concept_type: str
    confidence_score: float
    candidates_considered: int
    disambiguation_method: str = "neural"


class OptimizedNEDPipeline:


    def __init__(self,
                 ner_model_path: str = "./optimized_medical_ner_model",
                 ranking_model_path: str = "./fast_ranking_model.pth",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 confidence_threshold: float = 0.6):
        """
        Initialize the complete NED pipeline

        Args:
            ner_model_path: Path to trained BioBERT NER model
            ranking_model_path: Path to trained neural ranking model
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            confidence_threshold: Minimum confidence for entity predictions
        """
        self.confidence_threshold = confidence_threshold

        # Initialize components
        logger.info("Initializing NED Pipeline components...")

        # 1. Optimized NER Model
        logger.info("Loading optimized medical NER model...")
        self.ner_trainer = OptimizedMedicalNERTrainer()
        try:
            self.ner_trainer.load_model(ner_model_path)
            logger.info("NER model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load NER model from {ner_model_path}: {e}")
            logger.info("NER model will need to be trained first")
            self.ner_trainer = None

        # 2. Knowledge Base Interface
        logger.info("Connecting to knowledge base...")
        try:
            self.kb_interface = KnowledgeBaseInterface(neo4j_uri, neo4j_user, neo4j_password)
            logger.info("Knowledge base connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to knowledge base: {e}")
            raise

        # 3. Fast Neural Ranking Model
        logger.info("Loading fast neural ranking model...")
        self.disambiguator = FastEntityDisambiguator()
        try:
            self.disambiguator.load_model(ranking_model_path)
            logger.info("Neural ranking model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ranking model from {ranking_model_path}: {e}")
            logger.info("Ranking model will use default weights")

        logger.info("NED Pipeline initialized successfully!")


    def process_text(self, text: str, max_candidates: int = 10) -> NEDResult:
        """
        Process text through the complete NED pipeline

        Args:
            text: Input text to process
            max_candidates: Maximum candidates to consider per entity

        Returns:
            NEDResult containing all identified and disambiguated entities
        """
        start_time = time.time()

        logger.info(f"Processing text: '{text[:100]}...'")

        # Step 1: Named Entity Recognition
        logger.info("Step 1: Performing Named Entity Recognition...")
        if self.ner_trainer is None:
            # Fallback to simple pattern matching if NER model not available
            entity_spans = self._fallback_entity_extraction(text)
        else:
            entity_spans = self.ner_trainer.predict(text)

        logger.info(f"Found {len(entity_spans)} potential entities")

        # Step 2: Process each entity span
        results = []
        for span in entity_spans:
            try:
                entity_result = self._process_entity_span(span, text, max_candidates)
                if entity_result and entity_result.confidence_score >= self.confidence_threshold:
                    results.append(entity_result)
            except Exception as e:
                logger.error(f"Error processing entity span '{span.text}': {e}")
                continue

        processing_time = time.time() - start_time

        logger.info(f"Completed processing in {processing_time:.2f}s. Found {len(results)} entities.")

        # Convert results to dict format
        entity_dicts = []
        for result in results:
            entity_dicts.append({
                'mention': result.mention,
                'start_pos': result.start_pos,
                'end_pos': result.end_pos,
                'concept_id': result.concept_id,
                'concept_name': result.concept_name,
                'display_name': result.display_name,
                'concept_type': result.concept_type,
                'confidence_score': result.confidence_score,
                'candidates_considered': result.candidates_considered,
                'disambiguation_method': result.disambiguation_method
            })

        return NEDResult(
            text=text,
            entities=entity_dicts,
            processing_time=processing_time,
            confidence_threshold=self.confidence_threshold
        )


    def _fallback_entity_extraction(self, text: str) -> List[EntitySpan]:
        """Fallback entity extraction using knowledge base matching"""
        entities = []
        text_lower = text.lower()

        # Check for disease entities
        for disease_name in list(self.kb_interface.disease_entities.keys())[:1000]:  # Limit for performance
            if len(disease_name) >= 3 and disease_name in text_lower:
                start_pos = text_lower.find(disease_name)
                end_pos = start_pos + len(disease_name)

                entities.append(EntitySpan(
                    start=start_pos,
                    end=end_pos,
                    label="DISEASE",
                    text=text[start_pos:end_pos],
                    confidence=0.8
                ))

        # Check for symptom entities
        for symptom_name in list(self.kb_interface.symptom_entities.keys())[:1000]:  # Limit for performance
            if len(symptom_name) >= 3 and symptom_name in text_lower:
                start_pos = text_lower.find(symptom_name)
                end_pos = start_pos + len(symptom_name)

                entities.append(EntitySpan(
                    start=start_pos,
                    end=end_pos,
                    label="SYMPTOM",
                    text=text[start_pos:end_pos],
                    confidence=0.8
                ))

        # Remove duplicates and overlaps
        entities = self._remove_overlapping_spans(entities)

        return entities


    def _remove_overlapping_spans(self, spans: List[EntitySpan]) -> List[EntitySpan]:
        """Remove overlapping entity spans, keeping the longer ones"""
        spans.sort(key=lambda x: (x.start, -x.end))

        non_overlapping = []
        for span in spans:
            # Check for overlap with existing spans
            overlaps = False
            for existing in non_overlapping:
                if (span.start < existing.end and span.end > existing.start):
                    overlaps = True
                    break

            if not overlaps:
                non_overlapping.append(span)

        return non_overlapping


    def _process_entity_span(self, span: EntitySpan, full_text: str,
                           max_candidates: int) -> Optional[EntityResult]:
        """Process a single entity span through disambiguation"""

        mention = span.text
        logger.debug(f"Processing entity: '{mention}' ({span.label})")

        # Step 2: Generate Candidates
        candidates = self.kb_interface.generate_candidates(
            mention=mention,
            context=full_text,
            max_candidates=max_candidates
        )

        if not candidates:
            logger.debug(f"No candidates found for '{mention}'")
            return None

        logger.debug(f"Generated {len(candidates)} candidates for '{mention}'")

        # Step 3: Rank Candidates using Neural Model
        disambiguation_result = self.disambiguator.disambiguate(
            mention=mention,
            candidates=candidates,
            context=full_text,
            start_pos=span.start,
            end_pos=span.end
        )

        # Get the predicted candidate details
        predicted_candidate = next(
            (c for c in candidates if c.concept_id == disambiguation_result.predicted_concept_id),
            None
        )

        if predicted_candidate is None:
            logger.warning(f"Could not find predicted candidate for '{mention}'")
            return None

        return EntityResult(
            mention=mention,
            start_pos=span.start,
            end_pos=span.end,
            concept_id=predicted_candidate.concept_id,
            concept_name=predicted_candidate.name,
            display_name=predicted_candidate.display_name,
            concept_type=predicted_candidate.concept_type,
            confidence_score=disambiguation_result.confidence_score,
            candidates_considered=len(candidates),
            disambiguation_method="neural"
        )


    def train_models(self, annotated_corpus_path: str):
        """Train both NER and ranking models"""

        logger.info("Starting model training...")

        # Train NER model
        logger.info("Training BioBERT NER model...")
        if self.ner_trainer is None:
            self.ner_trainer = MedicalNERTrainer()

        self.ner_trainer.train(annotated_corpus_path)

        # Train ranking model
        logger.info("Training neural ranking model...")
        training_data = self.disambiguator.prepare_training_data(annotated_corpus_path)
        self.disambiguator.train(training_data, epochs=3, batch_size=16)

        # Save models
        self.disambiguator.save_model("./neural_ranking_model.pth")

        logger.info("Model training completed!")


    def batch_process(self, texts: List[str], batch_size: int = 10) -> List[NEDResult]:
        """Process multiple texts in batches"""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            for text in batch:
                result = self.process_text(text)
                results.append(result)

        return results


    def evaluate_on_test_set(self, test_corpus_path: str) -> Dict:
        """Evaluate the pipeline on a test corpus"""
        logger.info("Evaluating pipeline on test set...")

        results = []
        ground_truth = []

        with open(test_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']

                # Get ground truth
                gt_entities = []
                for disease in data.get('diseases', []):
                    gt_entities.append({
                        'text': disease['entity'],
                        'type': 'DISEASE',
                        'position': disease['position']
                    })

                for symptom in data.get('symptoms', []):
                    gt_entities.append({
                        'text': symptom['entity'],
                        'type': 'SYMPTOM',
                        'position': symptom['position']
                    })

                # Get predictions
                ned_result = self.process_text(text)

                results.append(ned_result.entities)
                ground_truth.append(gt_entities)

        # Calculate metrics (simplified)
        total_predicted = sum(len(r) for r in results)
        total_ground_truth = sum(len(gt) for gt in ground_truth)

        return {
            'total_predicted': total_predicted,
            'total_ground_truth': total_ground_truth,
            'texts_processed': len(results)
        }


    def close(self):
        """Clean up resources"""
        if self.kb_interface:
            self.kb_interface.close()
        logger.info("Pipeline closed")


def main():
    """Example usage and demonstration"""

    # Initialize pipeline
    pipeline = OptimizedNEDPipeline(
        confidence_threshold=0.3  # Lower threshold for demo
    )

    # Test texts
    test_texts = [
        "Patient presents with chest pain, shortness of breath, and has diabetes mellitus.",
        "The patient reports severe headache, nausea, and dizziness after the accident.",
        "Diagnosis includes hypertension, coronary artery disease, and chronic kidney disease.",
        "Symptoms include fever, cough, and fatigue lasting for several days.",
        "Patient has a history of stroke and currently experiences memory loss and confusion."
    ]

    print("Medical Named Entity Disambiguation Pipeline Demo")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"Text: {text}")
        print("-" * 40)

        # Process text
        result = pipeline.process_text(text)

        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Entities found: {len(result.entities)}")

        for j, entity in enumerate(result.entities, 1):
            print(f"  {j}. {entity['display_name']} ({entity['concept_type']})")
            print(f"     Mention: '{entity['mention']}'")
            print(f"     Confidence: {entity['confidence_score']:.3f}")
            print(f"     Candidates considered: {entity['candidates_considered']}")

    # Clean up
    pipeline.close()

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
