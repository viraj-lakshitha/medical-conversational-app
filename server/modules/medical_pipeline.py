"""
Medical Conversational Pipeline

End-to-end medical query processing pipeline that integrates:
- Named Entity Disambiguation (NED)
- Retrieval-Augmented Generation (RAG)
- Neuro-Symbolic Disease Prediction
- Medical Knowledge Graph Reasoning
"""

import sys
import os
import logging
import time
import importlib.util
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add scripts directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_PATH = os.path.join(PROJECT_ROOT, "scripts")
sys.path.append(SCRIPTS_PATH)

from config.medical_config import MedicalSystemConfig
from modules.evidence_fusion import EvidenceFusionEngine, EvidenceSource, ConversationContextManager
from modules.medical_safety import MedicalSafetyValidator
from modules.caching_system import MedicalCacheManager, ComponentCacheWrapper, CacheConfig
from modules.monitoring_system import MetricsCollector, PerformanceMonitor, AlertingSystem
from modules.llm_integration import MedicalLLMInterface, LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalQueryResult:
    """Result container for medical query processing"""
    response: str
    entities: List[Dict[str, Any]]
    predictions: Dict[str, Any]
    context_documents: List[Dict[str, Any]]
    confidence: float
    uncertainty: float
    reasoning_chain: List[str]
    processing_time: float
    warnings: List[str]

class MedicalConversationalPipeline:
    """
    Main orchestration class for the medical conversational AI system.
    
    Integrates all medical AI components into a unified pipeline for
    processing user medical queries and generating evidence-based responses.
    """
    
    def __init__(self):
        """Initialize the medical pipeline with all components"""
        self.config = MedicalSystemConfig()
        self.components_loaded = {}
        self.initialization_errors = []
        
        # Initialize production optimization components
        self.cache_manager = MedicalCacheManager()
        self.metrics_collector = MetricsCollector()
        self.alerting_system = AlertingSystem(self.metrics_collector)
        
        # Initialize advanced components
        self.evidence_fusion_engine = EvidenceFusionEngine()
        self.conversation_manager = ConversationContextManager()
        self.safety_validator = MedicalSafetyValidator()
        
        # Initialize LLM interface
        llm_config = LLMConfig(
            model_path=self.config.LLM_CONFIG["model_path"],
            temperature=self.config.LLM_CONFIG["temperature"],
            top_p=self.config.LLM_CONFIG["top_p"],
            use_medical_reasoning=self.config.LLM_CONFIG["use_medical_reasoning"]
        )
        self.llm_interface = MedicalLLMInterface(llm_config)
        
        # Initialize component cache wrappers
        self.cache_wrappers = {
            'ned': ComponentCacheWrapper('ned', self.cache_manager),
            'rag': ComponentCacheWrapper('rag', self.cache_manager),
            'neuro_symbolic': ComponentCacheWrapper('neuro_symbolic', self.cache_manager),
            'knowledge_base': ComponentCacheWrapper('knowledge_base', self.cache_manager)
        }
        
        # Initialize components
        self._initialize_components()
        
        # Start monitoring
        self._start_monitoring()
        
        logger.info(f"Medical Pipeline initialized. Components loaded: {list(self.components_loaded.keys())}")
        if self.initialization_errors:
            logger.warning(f"Component initialization errors: {self.initialization_errors}")
    
    def _initialize_components(self):
        """Initialize all medical AI components with error handling"""
        
        # Initialize NED Pipeline
        self._initialize_ned_pipeline()
        
        # Initialize RAG Pipeline
        self._initialize_rag_pipeline()
        
        # Initialize Neuro-Symbolic Model
        self._initialize_neuro_symbolic_model()
        
        # Initialize Knowledge Base
        self._initialize_knowledge_base()
        
        # Initialize LLM (when available)
        self._initialize_llm()
    
    def _initialize_ned_pipeline(self):
        """Initialize Named Entity Disambiguation pipeline"""
        try:
            if self.config.PIPELINE_CONFIG["enable_ned"]:
                from ned.ned_pipeline import NamedEntityDisambiguationPipeline
                
                self.ned_pipeline = NamedEntityDisambiguationPipeline(
                    ner_model_path=self.config.NED_CONFIG["ner_model_path"],
                    ranking_model_path=self.config.NED_CONFIG["ranking_model_path"],
                    neo4j_uri=self.config.NEO4J_CONFIG["uri"],
                    neo4j_user=self.config.NEO4J_CONFIG["user"],
                    neo4j_password=self.config.NEO4J_CONFIG["password"],
                    confidence_threshold=self.config.NED_CONFIG["confidence_threshold"]
                )
                self.components_loaded["ned"] = True
                logger.info("NED Pipeline initialized successfully")
            else:
                logger.info("NED Pipeline disabled in configuration")
        except Exception as e:
            error_msg = f"Failed to initialize NED Pipeline: {str(e)}"
            self.initialization_errors.append(error_msg)
            logger.error(error_msg)
            self.components_loaded["ned"] = False
    
    def _initialize_rag_pipeline(self):
        """Initialize RAG (Retrieval-Augmented Generation) pipeline"""
        try:
            if self.config.PIPELINE_CONFIG["enable_rag"]:
                from rag.rag_pipeline import MedicalRAGPipeline
                
                self.rag_pipeline = MedicalRAGPipeline()
                self.components_loaded["rag"] = True
                logger.info("RAG Pipeline initialized successfully")
            else:
                logger.info("RAG Pipeline disabled in configuration")
        except Exception as e:
            error_msg = f"Failed to initialize RAG Pipeline: {str(e)}"
            self.initialization_errors.append(error_msg)
            logger.error(error_msg)
            self.components_loaded["rag"] = False
    
    def _initialize_neuro_symbolic_model(self):
        """Initialize Neuro-Symbolic disease prediction model"""
        try:
            if self.config.PIPELINE_CONFIG["enable_neuro_symbolic"]:
                # Import with importlib to handle hyphenated directory name
                import importlib.util
                import sys
                
                # Load the disease model module
                neuro_symbolic_path = os.path.join(SCRIPTS_PATH, "neuro-symbolic", "disease_model.py")
                spec = importlib.util.spec_from_file_location("disease_model", neuro_symbolic_path)
                disease_model = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(disease_model)
                
                OptimizedDiseasePredictor = disease_model.OptimizedDiseasePredictor
                
                self.neuro_symbolic_model = OptimizedDiseasePredictor()
                
                # Load trained model if available
                model_path = self.config.NEURO_SYMBOLIC_CONFIG["model_path"]
                if os.path.exists(model_path):
                    self.neuro_symbolic_model.load_model(model_path)
                    logger.info("Neuro-Symbolic model loaded successfully")
                else:
                    logger.warning(f"Neuro-Symbolic model not found at {model_path}")
                
                self.components_loaded["neuro_symbolic"] = True
            else:
                logger.info("Neuro-Symbolic model disabled in configuration")
        except Exception as e:
            error_msg = f"Failed to initialize Neuro-Symbolic model: {str(e)}"
            self.initialization_errors.append(error_msg)
            logger.error(error_msg)
            self.components_loaded["neuro_symbolic"] = False
    
    def _initialize_knowledge_base(self):
        """Initialize Medical Knowledge Graph interface"""
        try:
            from knowledge_base.medical_knowledge_interface import MedicalKnowledgeBase
            
            self.knowledge_base = MedicalKnowledgeBase(
                uri=self.config.NEO4J_CONFIG["uri"],
                user=self.config.NEO4J_CONFIG["user"],
                password=self.config.NEO4J_CONFIG["password"]
            )
            self.components_loaded["knowledge_base"] = True
            logger.info("Knowledge Base initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize Knowledge Base: {str(e)}"
            self.initialization_errors.append(error_msg)
            logger.error(error_msg)
            self.components_loaded["knowledge_base"] = False
    
    def _initialize_llm(self):
        """Initialize fine-tuned LLM (when available)"""
        try:
            if self.config.PIPELINE_CONFIG["enable_llm_reasoning"]:
                # TODO: Implement when Qwen2.5 fine-tuning is complete
                logger.info("LLM reasoning not yet implemented")
                self.components_loaded["llm"] = False
            else:
                logger.info("LLM reasoning disabled in configuration")
                self.components_loaded["llm"] = False
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {str(e)}"
            self.initialization_errors.append(error_msg)
            logger.error(error_msg)
            self.components_loaded["llm"] = False
    
    def _start_monitoring(self):
        """Start monitoring and alerting systems"""
        try:
            # Setup default alert handlers
            def log_alert_handler(alert):
                logger.critical(f"MEDICAL AI ALERT: {alert['name']} - {alert['message']}")
            
            self.alerting_system.add_alert_handler(log_alert_handler)
            
            # Start periodic alert checking in background
            import threading
            def alert_checker():
                while True:
                    try:
                        self.alerting_system.check_alerts()
                        time.sleep(300)  # Check every 5 minutes
                    except Exception as e:
                        logger.error(f"Alert checking error: {str(e)}")
                        time.sleep(60)
            
            alert_thread = threading.Thread(target=alert_checker, daemon=True)
            alert_thread.start()
            
            logger.info("Monitoring and alerting systems started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {str(e)}")
    
    def process_medical_query(self, query: str, conversation_context: List[Dict] = None) -> MedicalQueryResult:
        """
        Process a medical query through the complete pipeline with caching and monitoring
        
        Args:
            query: User's medical question or description
            conversation_context: Previous conversation messages for context
            
        Returns:
            MedicalQueryResult with comprehensive analysis and response
        """
        # Monitor overall performance
        with PerformanceMonitor(self.metrics_collector, 'medical_pipeline', 'process_query') as monitor:
            monitor.add_metadata('query_length', len(query))
            monitor.add_metadata('has_context', bool(conversation_context))
            
            try:
                # Get conversation context for better understanding
                context_info = self.conversation_manager.get_relevant_context(
                    current_entities=[], num_turns=3
                ) if conversation_context else {}
                
                # Stage 1: Named Entity Recognition & Disambiguation
                entities = self._extract_entities(query)
                
                # Stage 2: Retrieve Relevant Medical Documents
                context_docs = self._retrieve_context(query, entities)
                
                # Stage 3: Knowledge Graph Query
                kg_evidence = self._query_knowledge_graph(entities)
                
                # Stage 4: Neuro-Symbolic Disease Prediction
                predictions = self._predict_diseases(entities, kg_evidence)
                
                # Stage 5: Advanced Evidence Fusion
                fused_evidence = self._fuse_evidence(entities, context_docs, kg_evidence, predictions)
                
                # Stage 6: Generate Medical Response (with LLM if available)
                if self.config.PIPELINE_CONFIG["enable_llm_reasoning"] and self.llm_interface.is_loaded:
                    llm_result = self.llm_interface.generate_medical_response(
                        query, entities, fused_evidence, context_docs, conversation_context
                    )
                    response = llm_result["response"]
                    monitor.add_metadata('llm_method', llm_result.get('method', 'unknown'))
                    monitor.add_metadata('llm_confidence', llm_result.get('confidence', 0))
                else:
                    response = self._generate_response(query, entities, context_docs, fused_evidence)
                
                # Stage 7: Safety Validation
                safety_assessment = self.safety_validator.validate_response(
                    response, entities, fused_evidence.predictions if fused_evidence else predictions,
                    fused_evidence.overall_confidence if fused_evidence else 0.5,
                    fused_evidence.overall_uncertainty if fused_evidence else 0.5
                )
                
                # Stage 8: Apply Safety Measures
                if safety_assessment.confidence_override is not None:
                    final_confidence = safety_assessment.confidence_override
                else:
                    final_confidence = fused_evidence.overall_confidence if fused_evidence else 0.5
                
                final_uncertainty = fused_evidence.overall_uncertainty if fused_evidence else 0.5
                
                # Sanitize response based on safety assessment
                final_response = self.safety_validator.sanitize_response(response, safety_assessment)
                
                # Generate enhanced reasoning chain
                reasoning_chain = self._generate_enhanced_reasoning_chain(
                    entities, fused_evidence, kg_evidence, safety_assessment, context_info
                )
                
                # Compile warnings from safety assessment
                warnings = safety_assessment.warnings + safety_assessment.disclaimers
                
                processing_time = time.time() - monitor.start_time
                
                # Record medical accuracy metrics
                self.metrics_collector.record_medical_accuracy(
                    query_type='general',
                    entities_found=len(entities),
                    predictions_count=len(fused_evidence.predictions) if fused_evidence else 0,
                    confidence=final_confidence,
                    uncertainty=final_uncertainty,
                    safety_flags=[flag.value for flag in safety_assessment.safety_flags]
                )
                
                # Add metadata for monitoring
                monitor.add_metadata('entities_found', len(entities))
                monitor.add_metadata('final_confidence', final_confidence)
                monitor.add_metadata('safety_flags', len(safety_assessment.safety_flags))
                
                # Save to conversation context
                self.conversation_manager.add_turn(
                    query, entities, fused_evidence.predictions if fused_evidence else predictions, final_response
                )
                
                return MedicalQueryResult(
                    response=final_response,
                    entities=entities,
                    predictions=fused_evidence.predictions if fused_evidence else predictions,
                    context_documents=context_docs,
                    confidence=final_confidence,
                    uncertainty=final_uncertainty,
                    reasoning_chain=reasoning_chain,
                    processing_time=processing_time,
                    warnings=warnings
                )
                
            except Exception as e:
                logger.error(f"Error processing medical query: {str(e)}")
                return self._generate_error_response(str(e), time.time() - monitor.start_time)
    
    def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract and disambiguate medical entities from query with caching"""
        if not self.components_loaded.get("ned", False):
            logger.warning("NED Pipeline not available, skipping entity extraction")
            return []
        
        # Use cached wrapper for entity extraction
        try:
            return self.cache_wrappers['ned'].cached_call(
                self._extract_entities_uncached, query
            )
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            self.metrics_collector.record_error('ned', 'extraction_error', str(e))
            return []
    
    def _extract_entities_uncached(self, query: str) -> List[Dict[str, Any]]:
        """Actual entity extraction without caching"""
        ned_result = self.ned_pipeline.process_text(query)
        return ned_result.entities if ned_result else []
    
    def _retrieve_context(self, query: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Retrieve relevant medical documents for context with caching"""
        if not self.components_loaded.get("rag", False):
            logger.warning("RAG Pipeline not available, skipping context retrieval")
            return []
        
        try:
            return self.cache_wrappers['rag'].cached_call(
                self._retrieve_context_uncached, query
            )
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            self.metrics_collector.record_error('rag', 'retrieval_error', str(e))
            return []
    
    def _retrieve_context_uncached(self, query: str) -> List[Dict[str, Any]]:
        """Actual context retrieval without caching"""
        rag_results = self.rag_pipeline.query_documents(query, max_results=self.config.RAG_CONFIG["max_results"])
        return rag_results.get("documents", []) if rag_results else []
    
    def _query_knowledge_graph(self, entities: List[Dict]) -> Dict[str, Any]:
        """Query medical knowledge graph for entity relationships"""
        if not self.components_loaded.get("knowledge_base", False):
            logger.warning("Knowledge Base not available, skipping KG query")
            return {}
        
        try:
            # Extract entity names for KG queries
            entity_names = [e.get("concept_name", e.get("text", "")) for e in entities]
            symptoms = [name for e, name in zip(entities, entity_names) if e.get("concept_type") == "SYMPTOM"]
            
            if symptoms:
                kg_results = self.knowledge_base.get_related_diseases(symptoms)
                return kg_results
            return {}
        except Exception as e:
            logger.error(f"Knowledge graph query failed: {str(e)}")
            return {}
    
    def _predict_diseases(self, entities: List[Dict], kg_evidence: Dict) -> Dict[str, Any]:
        """Generate disease predictions using neuro-symbolic model"""
        if not self.components_loaded.get("neuro_symbolic", False):
            logger.warning("Neuro-Symbolic model not available, skipping disease prediction")
            return {}
        
        try:
            # Extract symptoms for disease prediction
            symptoms = [e.get("concept_name", e.get("text", "")) for e in entities if e.get("concept_type") == "SYMPTOM"]
            
            if symptoms:
                predictions = self.neuro_symbolic_model.predict_with_uncertainty(
                    symptoms,
                    return_explanations=True
                )
                return predictions
            return {}
        except Exception as e:
            logger.error(f"Disease prediction failed: {str(e)}")
            return {}
    
    def _fuse_evidence(self, entities: List[Dict], context_docs: List[Dict], 
                      kg_evidence: Dict, predictions: Dict):
        """Fuse evidence from multiple sources using advanced fusion engine"""
        try:
            evidence_sources = []
            
            # Neural/NED evidence
            if entities:
                entity_confidences = {
                    e.get('concept_name', e.get('text', '')): e.get('confidence_score', 0.5)
                    for e in entities
                }
                evidence_sources.append(EvidenceSource(
                    source_type='neural',
                    predictions=entity_confidences,
                    confidence=np.mean([e.get('confidence_score', 0.5) for e in entities]),
                    uncertainty=0.3,  # Default uncertainty for NED
                    metadata={'entity_count': len(entities)}
                ))
            
            # Symbolic/Knowledge Graph evidence
            if kg_evidence and kg_evidence.get('diseases'):
                kg_confidences = {}
                for disease in kg_evidence['diseases']:
                    disease_name = disease.get('disease_name', '')
                    score = disease.get('total_score', 0)
                    # Normalize score to [0,1] range
                    normalized_score = min(1.0, score / 5.0)  # Assuming max score ~5
                    kg_confidences[disease_name] = normalized_score
                
                if kg_confidences:
                    evidence_sources.append(EvidenceSource(
                        source_type='symbolic',
                        predictions=kg_confidences,
                        confidence=np.mean(list(kg_confidences.values())),
                        uncertainty=0.2,  # Knowledge graphs typically have lower uncertainty
                        metadata={'disease_count': len(kg_confidences)}
                    ))
            
            # RAG evidence
            if context_docs:
                # Simple RAG confidence based on document relevance
                rag_confidence = min(1.0, len(context_docs) * 0.2)  # More docs = higher confidence
                evidence_sources.append(EvidenceSource(
                    source_type='rag',
                    predictions={'document_relevance': rag_confidence},
                    confidence=rag_confidence,
                    uncertainty=0.4,  # Document retrieval has higher uncertainty
                    metadata={'document_count': len(context_docs)}
                ))
            
            # Neuro-symbolic predictions
            if predictions and predictions.get('predictions'):
                ns_confidences = {}
                for disease, info in predictions['predictions'].items():
                    confidence = info.get('fused_probability', info.get('probability', 0))
                    ns_confidences[disease] = confidence
                
                if ns_confidences:
                    avg_uncertainty = np.mean([
                        info.get('uncertainty', 0.5) 
                        for info in predictions['predictions'].values()
                    ])
                    
                    evidence_sources.append(EvidenceSource(
                        source_type='neuro_symbolic',
                        predictions=ns_confidences,
                        confidence=predictions.get('confidence', 0.5),
                        uncertainty=avg_uncertainty,
                        metadata={'prediction_count': len(ns_confidences)}
                    ))
            
            # Perform evidence fusion
            if evidence_sources:
                return self.evidence_fusion_engine.fuse_evidence(evidence_sources)
            
            return None
            
        except Exception as e:
            logger.error(f"Evidence fusion failed: {str(e)}")
            return None
    
    def _generate_response(self, query: str, entities: List[Dict], context_docs: List[Dict], fused_evidence) -> str:
        """Generate medical response based on fused evidence and available information"""
        
        response_parts = []
        
        # Summarize entities found
        if entities:
            diseases = [e for e in entities if e.get("concept_type") == "DISEASE"]
            symptoms = [e for e in entities if e.get("concept_type") == "SYMPTOM"]
            
            if diseases:
                disease_names = [e.get("display_name", e.get("text", "")) for e in diseases]
                response_parts.append(f"**Medical conditions mentioned**: {', '.join(disease_names)}")
            
            if symptoms:
                symptom_names = [e.get("display_name", e.get("text", "")) for e in symptoms]
                response_parts.append(f"**Symptoms identified**: {', '.join(symptom_names)}")
        
        # Add fused evidence predictions
        if fused_evidence and fused_evidence.predictions:
            response_parts.append("**Analysis Results**:")
            
            # Sort predictions by confidence
            sorted_predictions = sorted(
                fused_evidence.predictions.items(),
                key=lambda x: x[1].get('confidence', 0),
                reverse=True
            )
            
            for entity, info in sorted_predictions[:3]:  # Top 3
                confidence = info.get('confidence', 0)
                uncertainty = info.get('uncertainty', 0)
                sources = info.get('sources', [])
                
                response_parts.append(
                    f"- {entity}: {confidence:.1%} confidence "
                    f"(uncertainty: {uncertainty:.1%}, sources: {len(sources)})"
                )
            
            # Add fusion quality information
            response_parts.append(
                f"**Overall Assessment**: Confidence {fused_evidence.overall_confidence:.1%}, "
                f"Uncertainty {fused_evidence.overall_uncertainty:.1%}"
            )
        
        # Add context from documents if available
        if context_docs:
            response_parts.append(f"**Reference Information**: Based on {len(context_docs)} relevant medical documents")
        
        # Add conversation context if available
        context_summary = self.conversation_manager.get_context_summary()
        if context_summary and "No conversation history" not in context_summary:
            response_parts.append(f"**Context**: {context_summary}")
        
        # Default response if no components are working
        if not response_parts:
            response_parts.append("I understand you have a medical question. Currently initializing medical analysis components. Please ensure all medical services are running.")
        
        return "\n\n".join(response_parts)
    
    def _calculate_overall_confidence(self, entities: List[Dict], predictions: Dict) -> float:
        """Calculate overall confidence in the medical response"""
        confidences = []
        
        # Entity extraction confidence
        if entities:
            entity_confidences = [e.get("confidence_score", 0.5) for e in entities]
            confidences.extend(entity_confidences)
        
        # Disease prediction confidence
        if predictions and predictions.get("confidence"):
            confidences.append(predictions["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _calculate_uncertainty(self, predictions: Dict) -> float:
        """Calculate uncertainty in predictions"""
        if predictions and predictions.get("predictions"):
            uncertainties = []
            for disease_info in predictions["predictions"].values():
                uncertainty = disease_info.get("uncertainty", 0.5)
                uncertainties.append(uncertainty)
            return sum(uncertainties) / len(uncertainties) if uncertainties else 0.5
        return 0.5
    
    def _generate_enhanced_reasoning_chain(self, entities: List[Dict], fused_evidence, 
                                          kg_evidence: Dict, safety_assessment, context_info: Dict) -> List[str]:
        """Generate enhanced step-by-step reasoning chain with fusion and safety details"""
        reasoning = []
        
        # Entity extraction step
        if entities:
            diseases = len([e for e in entities if e.get("concept_type") == "DISEASE"])
            symptoms = len([e for e in entities if e.get("concept_type") == "SYMPTOM"])
            reasoning.append(f"Entity Recognition: Identified {diseases} diseases and {symptoms} symptoms")
        
        # Evidence fusion step
        if fused_evidence:
            source_weights = fused_evidence.source_weights
            active_sources = [source for source, weight in source_weights.items() if weight > 0]
            reasoning.append(f"Evidence Fusion: Combined evidence from {len(active_sources)} sources")
            reasoning.extend(fused_evidence.reasoning_chain)
        
        # Knowledge graph step
        if kg_evidence and kg_evidence.get('diseases'):
            reasoning.append(f"Knowledge Graph: Found {len(kg_evidence['diseases'])} related disease patterns")
        
        # Safety validation step
        risk_level = safety_assessment.risk_level.value if safety_assessment else "unknown"
        safety_flags = len(safety_assessment.safety_flags) if safety_assessment else 0
        reasoning.append(f"Safety Validation: Risk level {risk_level}, {safety_flags} safety flags raised")
        
        # Conversation context step
        if context_info and context_info.get('conversation_length', 0) > 0:
            reasoning.append(f"Context Analysis: Considered {context_info['conversation_length']} previous conversation turns")
        
        # Calibration and confidence step
        if fused_evidence:
            reasoning.append(f"Confidence Calibration: Final confidence {fused_evidence.overall_confidence:.1%}, uncertainty {fused_evidence.overall_uncertainty:.1%}")
        
        return reasoning
    
    def _generate_reasoning_chain(self, entities: List[Dict], predictions: Dict, kg_evidence: Dict) -> List[str]:
        """Generate step-by-step reasoning chain (legacy method for backward compatibility)"""
        reasoning = []
        
        if entities:
            reasoning.append(f"Identified {len(entities)} medical entities in the query")
        
        if predictions:
            reasoning.append("Applied neuro-symbolic disease prediction model")
        
        if kg_evidence:
            reasoning.append("Consulted medical knowledge graph for evidence-based relationships")
        
        reasoning.append("Generated response with appropriate medical disclaimers")
        
        return reasoning
    
    def _generate_safety_warnings(self, confidence: float, uncertainty: float, predictions: Dict) -> List[str]:
        """Generate safety warnings based on confidence and uncertainty"""
        warnings = []
        
        if confidence < self.config.SAFETY_CONFIG["minimum_confidence_for_diagnosis"]:
            warnings.append("Low confidence prediction - seek professional medical advice")
        
        if uncertainty > self.config.SAFETY_CONFIG["require_human_review_threshold"]:
            warnings.append("High uncertainty in analysis - human review recommended")
        
        if self.config.SAFETY_CONFIG["enable_uncertainty_warnings"] and uncertainty > 0.7:
            warnings.append("Significant uncertainty detected in medical analysis")
        
        return warnings
    
    def _generate_error_response(self, error_message: str, processing_time: float) -> MedicalQueryResult:
        """Generate error response when pipeline fails"""
        return MedicalQueryResult(
            response="I apologize, but I encountered an error while processing your medical query. Please try again or consult with a healthcare professional.",
            entities=[],
            predictions={},
            context_documents=[],
            confidence=0.0,
            uncertainty=1.0,
            reasoning_chain=["Error occurred during processing"],
            processing_time=processing_time,
            warnings=[f"Processing error: {error_message}"]
        )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get health status of all medical pipeline components"""
        return {
            "components_loaded": self.components_loaded,
            "initialization_errors": self.initialization_errors,
            "advanced_features": {
                "evidence_fusion": True,
                "conversation_context": True,
                "safety_validation": True,
                "confidence_calibration": True,
                "llm_integration": self.llm_interface.is_loaded if hasattr(self, 'llm_interface') else False,
                "caching_system": self.cache_manager.redis_client is not None if hasattr(self, 'cache_manager') else False,
                "monitoring_system": True
            },
            "configuration": {
                "ned_enabled": self.config.PIPELINE_CONFIG["enable_ned"],
                "rag_enabled": self.config.PIPELINE_CONFIG["enable_rag"],
                "neuro_symbolic_enabled": self.config.PIPELINE_CONFIG["enable_neuro_symbolic"],
                "llm_enabled": self.config.PIPELINE_CONFIG["enable_llm_reasoning"]
            },
            "safety_metrics": self.safety_validator.get_safety_metrics(),
            "conversation_stats": {
                "active_conversations": 1 if hasattr(self, 'conversation_manager') else 0,
                "context_length": len(self.conversation_manager.context_history) if hasattr(self, 'conversation_manager') else 0
            },
            "fusion_config": self.evidence_fusion_engine.config if hasattr(self, 'evidence_fusion_engine') else {},
            "connection_urls": self.config.get_connection_urls()
        }