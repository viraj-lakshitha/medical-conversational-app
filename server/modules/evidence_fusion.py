"""
Multi-Modal Evidence Fusion System

Advanced fusion of evidence from multiple medical AI components:
- Neural network predictions with uncertainty
- Symbolic knowledge graph evidence
- Document retrieval confidence scores
- Conversation context relevance
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy.special import softmax
from sklearn.calibration import calibration_curve
import math

logger = logging.getLogger(__name__)

@dataclass
class EvidenceSource:
    """Container for evidence from a single source"""
    source_type: str  # 'neural', 'symbolic', 'rag', 'context'
    predictions: Dict[str, float]  # entity/disease -> confidence
    confidence: float  # overall source confidence
    uncertainty: float  # uncertainty measure
    metadata: Dict[str, Any]  # additional source-specific data

@dataclass
class FusedEvidence:
    """Result of evidence fusion"""
    predictions: Dict[str, Dict[str, float]]  # entity -> {'confidence', 'uncertainty', 'sources'}
    overall_confidence: float
    overall_uncertainty: float
    source_weights: Dict[str, float]
    reasoning_chain: List[str]
    calibration_quality: float

class EvidenceFusionEngine:
    """
    Advanced evidence fusion engine that combines predictions from multiple
    medical AI sources with uncertainty quantification and calibration.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the evidence fusion engine
        
        Args:
            config: Configuration dictionary with fusion parameters
        """
        self.config = config or self._get_default_config()
        self.source_reliability = self._initialize_source_reliability()
        self.calibration_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for evidence fusion"""
        return {
            'fusion_method': 'weighted_average',  # 'weighted_average', 'bayesian', 'dempster_shafer'
            'uncertainty_threshold': 0.3,
            'confidence_threshold': 0.7,
            'min_source_weight': 0.1,
            'calibration_window': 100,  # Number of predictions to track for calibration
            'source_weights': {
                'neural': 0.4,
                'symbolic': 0.3,
                'rag': 0.2,
                'context': 0.1
            },
            'dynamic_weighting': True,  # Adjust weights based on performance
            'uncertainty_propagation': 'monte_carlo'
        }
    
    def _initialize_source_reliability(self) -> Dict[str, Dict[str, float]]:
        """Initialize source reliability tracking"""
        return {
            'neural': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'calibration': 0.75},
            'symbolic': {'accuracy': 0.78, 'precision': 0.89, 'recall': 0.72, 'calibration': 0.90},
            'rag': {'accuracy': 0.70, 'precision': 0.75, 'recall': 0.68, 'calibration': 0.65},
            'context': {'accuracy': 0.60, 'precision': 0.65, 'recall': 0.58, 'calibration': 0.60}
        }
    
    def fuse_evidence(self, evidence_sources: List[EvidenceSource]) -> FusedEvidence:
        """
        Fuse evidence from multiple sources into unified predictions
        
        Args:
            evidence_sources: List of evidence from different sources
            
        Returns:
            FusedEvidence with combined predictions and metadata
        """
        if not evidence_sources:
            return self._create_empty_fusion_result()
        
        # Calculate dynamic source weights
        source_weights = self._calculate_dynamic_weights(evidence_sources)
        
        # Perform evidence fusion
        if self.config['fusion_method'] == 'weighted_average':
            fused_predictions = self._weighted_average_fusion(evidence_sources, source_weights)
        elif self.config['fusion_method'] == 'bayesian':
            fused_predictions = self._bayesian_fusion(evidence_sources, source_weights)
        else:
            fused_predictions = self._weighted_average_fusion(evidence_sources, source_weights)
        
        # Calculate overall confidence and uncertainty
        overall_confidence = self._calculate_overall_confidence(fused_predictions, evidence_sources)
        overall_uncertainty = self._calculate_overall_uncertainty(fused_predictions, evidence_sources)
        
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(evidence_sources, source_weights)
        
        # Assess calibration quality
        calibration_quality = self._assess_calibration_quality(evidence_sources)
        
        return FusedEvidence(
            predictions=fused_predictions,
            overall_confidence=overall_confidence,
            overall_uncertainty=overall_uncertainty,
            source_weights=source_weights,
            reasoning_chain=reasoning_chain,
            calibration_quality=calibration_quality
        )
    
    def _calculate_dynamic_weights(self, evidence_sources: List[EvidenceSource]) -> Dict[str, float]:
        """Calculate dynamic weights based on source performance and confidence"""
        if not self.config['dynamic_weighting']:
            return self.config['source_weights']
        
        weights = {}
        total_weight = 0
        
        for source in evidence_sources:
            source_type = source.source_type
            
            # Base weight from configuration
            base_weight = self.config['source_weights'].get(source_type, 0.25)
            
            # Adjust based on source confidence
            confidence_factor = source.confidence
            
            # Adjust based on uncertainty (lower uncertainty = higher weight)
            uncertainty_factor = max(0.1, 1.0 - source.uncertainty)
            
            # Adjust based on historical reliability
            reliability = self.source_reliability.get(source_type, {})
            reliability_factor = (
                reliability.get('accuracy', 0.7) * 0.4 +
                reliability.get('calibration', 0.7) * 0.6
            )
            
            # Calculate final weight
            final_weight = base_weight * confidence_factor * uncertainty_factor * reliability_factor
            final_weight = max(self.config['min_source_weight'], final_weight)
            
            weights[source_type] = final_weight
            total_weight += final_weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _weighted_average_fusion(self, evidence_sources: List[EvidenceSource], 
                                source_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Perform weighted average fusion of evidence"""
        all_entities = set()
        for source in evidence_sources:
            all_entities.update(source.predictions.keys())
        
        fused_predictions = {}
        
        for entity in all_entities:
            weighted_confidence = 0
            weighted_uncertainty = 0
            total_weight = 0
            contributing_sources = []
            
            for source in evidence_sources:
                if entity in source.predictions:
                    weight = source_weights.get(source.source_type, 0)
                    confidence = source.predictions[entity]
                    uncertainty = source.uncertainty
                    
                    weighted_confidence += weight * confidence
                    weighted_uncertainty += weight * uncertainty
                    total_weight += weight
                    contributing_sources.append(source.source_type)
            
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
                final_uncertainty = weighted_uncertainty / total_weight
                
                # Apply calibration if available
                final_confidence = self._apply_calibration(final_confidence, contributing_sources)
                
                fused_predictions[entity] = {
                    'confidence': final_confidence,
                    'uncertainty': final_uncertainty,
                    'sources': contributing_sources,
                    'source_count': len(contributing_sources)
                }
        
        return fused_predictions
    
    def _bayesian_fusion(self, evidence_sources: List[EvidenceSource], 
                        source_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Perform Bayesian fusion of evidence"""
        # Simplified Bayesian fusion - can be enhanced with proper priors
        all_entities = set()
        for source in evidence_sources:
            all_entities.update(source.predictions.keys())
        
        fused_predictions = {}
        
        for entity in all_entities:
            # Start with uniform prior
            prior = 0.5
            
            # Update with evidence from each source
            posterior = prior
            uncertainty_accumulator = []
            contributing_sources = []
            
            for source in evidence_sources:
                if entity in source.predictions:
                    likelihood = source.predictions[entity]
                    weight = source_weights.get(source.source_type, 0)
                    
                    # Bayesian update (simplified)
                    posterior = (posterior * likelihood * weight) / \
                               (posterior * likelihood * weight + (1 - posterior) * (1 - likelihood) * weight)
                    
                    uncertainty_accumulator.append(source.uncertainty * weight)
                    contributing_sources.append(source.source_type)
            
            if contributing_sources:
                final_uncertainty = np.mean(uncertainty_accumulator) if uncertainty_accumulator else 0.5
                
                fused_predictions[entity] = {
                    'confidence': posterior,
                    'uncertainty': final_uncertainty,
                    'sources': contributing_sources,
                    'source_count': len(contributing_sources)
                }
        
        return fused_predictions
    
    def _apply_calibration(self, confidence: float, sources: List[str]) -> float:
        """Apply calibration correction based on source performance"""
        # Simplified calibration - can be enhanced with isotonic regression
        calibration_factors = []
        
        for source in sources:
            reliability = self.source_reliability.get(source, {})
            calibration_factor = reliability.get('calibration', 1.0)
            calibration_factors.append(calibration_factor)
        
        if calibration_factors:
            avg_calibration = np.mean(calibration_factors)
            # Apply sigmoid-like calibration correction
            calibrated_confidence = confidence * avg_calibration + (1 - avg_calibration) * 0.5
            return max(0.0, min(1.0, calibrated_confidence))
        
        return confidence
    
    def _calculate_overall_confidence(self, fused_predictions: Dict, 
                                    evidence_sources: List[EvidenceSource]) -> float:
        """Calculate overall confidence in the fused evidence"""
        if not fused_predictions:
            return 0.0
        
        confidences = [pred['confidence'] for pred in fused_predictions.values()]
        source_confidences = [source.confidence for source in evidence_sources]
        
        # Weighted combination of prediction confidences and source confidences
        pred_confidence = np.mean(confidences) if confidences else 0.0
        source_confidence = np.mean(source_confidences) if source_confidences else 0.0
        
        overall_confidence = 0.7 * pred_confidence + 0.3 * source_confidence
        
        # Adjust based on number of sources (more sources = higher confidence)
        source_diversity_bonus = min(0.1, len(evidence_sources) * 0.02)
        overall_confidence += source_diversity_bonus
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_overall_uncertainty(self, fused_predictions: Dict, 
                                     evidence_sources: List[EvidenceSource]) -> float:
        """Calculate overall uncertainty in the fused evidence"""
        if not fused_predictions:
            return 1.0
        
        uncertainties = [pred['uncertainty'] for pred in fused_predictions.values()]
        source_uncertainties = [source.uncertainty for source in evidence_sources]
        
        # Combine uncertainties (lower when more sources agree)
        pred_uncertainty = np.mean(uncertainties) if uncertainties else 1.0
        source_uncertainty = np.mean(source_uncertainties) if source_uncertainties else 1.0
        
        overall_uncertainty = 0.6 * pred_uncertainty + 0.4 * source_uncertainty
        
        # Reduce uncertainty when multiple sources agree
        if len(evidence_sources) > 1:
            agreement_factor = 1.0 / math.sqrt(len(evidence_sources))
            overall_uncertainty *= agreement_factor
        
        return max(0.0, min(1.0, overall_uncertainty))
    
    def _generate_reasoning_chain(self, evidence_sources: List[EvidenceSource], 
                                source_weights: Dict[str, float]) -> List[str]:
        """Generate explanation of the fusion process"""
        reasoning = []
        
        reasoning.append(f"Fused evidence from {len(evidence_sources)} sources using {self.config['fusion_method']} method")
        
        for source in evidence_sources:
            weight = source_weights.get(source.source_type, 0)
            reasoning.append(
                f"{source.source_type.title()} source: {len(source.predictions)} predictions, "
                f"confidence={source.confidence:.3f}, weight={weight:.3f}"
            )
        
        if self.config['dynamic_weighting']:
            reasoning.append("Applied dynamic weighting based on source reliability and confidence")
        
        reasoning.append("Applied calibration corrections based on historical performance")
        
        return reasoning
    
    def _assess_calibration_quality(self, evidence_sources: List[EvidenceSource]) -> float:
        """Assess the calibration quality of the evidence sources"""
        calibration_scores = []
        
        for source in evidence_sources:
            reliability = self.source_reliability.get(source.source_type, {})
            calibration_score = reliability.get('calibration', 0.5)
            calibration_scores.append(calibration_score * source.confidence)
        
        if calibration_scores:
            return np.mean(calibration_scores)
        return 0.5
    
    def _create_empty_fusion_result(self) -> FusedEvidence:
        """Create empty fusion result when no evidence is available"""
        return FusedEvidence(
            predictions={},
            overall_confidence=0.0,
            overall_uncertainty=1.0,
            source_weights={},
            reasoning_chain=["No evidence sources available"],
            calibration_quality=0.0
        )
    
    def update_source_reliability(self, source_type: str, ground_truth: Dict[str, bool], 
                                 predictions: Dict[str, float]):
        """Update source reliability based on ground truth feedback"""
        if source_type not in self.source_reliability:
            self.source_reliability[source_type] = {
                'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'calibration': 0.5
            }
        
        # Calculate metrics
        tp = fp = tn = fn = 0
        for entity, pred_conf in predictions.items():
            if entity in ground_truth:
                predicted = pred_conf > 0.5
                actual = ground_truth[entity]
                
                if predicted and actual:
                    tp += 1
                elif predicted and not actual:
                    fp += 1
                elif not predicted and actual:
                    fn += 1
                else:
                    tn += 1
        
        # Update reliability metrics with exponential moving average
        alpha = 0.1  # Learning rate
        
        if tp + fp + tn + fn > 0:
            accuracy = (tp + tn) / (tp + fp + tn + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            reliability = self.source_reliability[source_type]
            reliability['accuracy'] = (1 - alpha) * reliability['accuracy'] + alpha * accuracy
            reliability['precision'] = (1 - alpha) * reliability['precision'] + alpha * precision
            reliability['recall'] = (1 - alpha) * reliability['recall'] + alpha * recall
            
            logger.info(f"Updated {source_type} reliability: acc={reliability['accuracy']:.3f}")

class ConversationContextManager:
    """
    Manages conversation context and tracks entity evolution over time.
    Maintains coherent medical context across conversation turns.
    """
    
    def __init__(self, max_context_length: int = 10):
        """
        Initialize conversation context manager
        
        Args:
            max_context_length: Maximum number of turns to maintain in context
        """
        self.max_context_length = max_context_length
        self.context_history = []
        self.entity_timeline = {}
        self.topic_tracking = {}
        
    def add_turn(self, user_query: str, entities: List[Dict], 
                 predictions: Dict, response: str):
        """Add a new conversation turn to the context"""
        turn = {
            'timestamp': time.time(),
            'user_query': user_query,
            'entities': entities,
            'predictions': predictions,
            'response': response,
            'turn_id': len(self.context_history)
        }
        
        self.context_history.append(turn)
        
        # Maintain context length limit
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)
        
        # Update entity timeline
        self._update_entity_timeline(entities, turn['turn_id'])
        
        # Update topic tracking
        self._update_topic_tracking(user_query, entities, turn['turn_id'])
    
    def _update_entity_timeline(self, entities: List[Dict], turn_id: int):
        """Update the timeline of entity mentions"""
        for entity in entities:
            entity_name = entity.get('concept_name', entity.get('text', ''))
            if entity_name not in self.entity_timeline:
                self.entity_timeline[entity_name] = []
            
            self.entity_timeline[entity_name].append({
                'turn_id': turn_id,
                'confidence': entity.get('confidence_score', 0.5),
                'type': entity.get('concept_type', 'unknown')
            })
    
    def _update_topic_tracking(self, query: str, entities: List[Dict], turn_id: int):
        """Track topic evolution in the conversation"""
        # Simple topic tracking based on entity types and keywords
        topics = set()
        
        for entity in entities:
            entity_type = entity.get('concept_type', 'unknown')
            topics.add(entity_type.lower())
        
        # Add keyword-based topics
        medical_keywords = {
            'pain': 'symptom_pain',
            'fever': 'symptom_fever',
            'treatment': 'treatment_discussion',
            'medication': 'medication_discussion',
            'diagnosis': 'diagnosis_discussion'
        }
        
        query_lower = query.lower()
        for keyword, topic in medical_keywords.items():
            if keyword in query_lower:
                topics.add(topic)
        
        for topic in topics:
            if topic not in self.topic_tracking:
                self.topic_tracking[topic] = []
            self.topic_tracking[topic].append(turn_id)
    
    def get_relevant_context(self, current_entities: List[Dict], 
                           num_turns: int = 3) -> Dict[str, Any]:
        """Get relevant context for current query based on entity overlap"""
        if not self.context_history:
            return {'relevant_turns': [], 'entity_context': {}, 'topic_context': {}}
        
        current_entity_names = set(
            entity.get('concept_name', entity.get('text', ''))
            for entity in current_entities
        )
        
        # Find turns with overlapping entities
        relevant_turns = []
        for turn in self.context_history[-num_turns:]:
            turn_entities = set(
                entity.get('concept_name', entity.get('text', ''))
                for entity in turn['entities']
            )
            
            overlap = current_entity_names.intersection(turn_entities)
            if overlap:
                relevance_score = len(overlap) / max(len(current_entity_names), 1)
                relevant_turns.append({
                    'turn': turn,
                    'entity_overlap': list(overlap),
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance
        relevant_turns.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Build entity context
        entity_context = {}
        for entity_name in current_entity_names:
            if entity_name in self.entity_timeline:
                timeline = self.entity_timeline[entity_name]
                entity_context[entity_name] = {
                    'mentions': len(timeline),
                    'first_mention': min(t['turn_id'] for t in timeline),
                    'last_mention': max(t['turn_id'] for t in timeline),
                    'confidence_trend': [t['confidence'] for t in timeline[-3:]]
                }
        
        return {
            'relevant_turns': relevant_turns[:3],  # Top 3 most relevant
            'entity_context': entity_context,
            'topic_context': self.topic_tracking,
            'conversation_length': len(self.context_history)
        }
    
    def get_context_summary(self) -> str:
        """Generate a summary of the conversation context"""
        if not self.context_history:
            return "No conversation history available."
        
        recent_entities = set()
        for turn in self.context_history[-3:]:
            for entity in turn['entities']:
                entity_name = entity.get('concept_name', entity.get('text', ''))
                recent_entities.add(entity_name)
        
        summary_parts = [
            f"Conversation has {len(self.context_history)} turns.",
            f"Recent medical entities discussed: {', '.join(list(recent_entities)[:5])}",
            f"Active topics: {', '.join(self.topic_tracking.keys())}"
        ]
        
        return " ".join(summary_parts)