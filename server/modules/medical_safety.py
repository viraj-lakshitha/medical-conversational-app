"""
Medical Safety and Validation Layer

Comprehensive safety framework for medical AI responses including:
- Medical disclaimer generation
- Confidence thresholding and safety checks
- Regulatory compliance validation
- Risk assessment and warning generation
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk levels for medical predictions and responses"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class SafetyFlag(Enum):
    """Types of safety flags that can be raised"""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_UNCERTAINTY = "high_uncertainty"
    EMERGENCY_SYMPTOMS = "emergency_symptoms"
    DRUG_INTERACTION = "drug_interaction"
    DIAGNOSIS_CLAIM = "diagnosis_claim"
    TREATMENT_ADVICE = "treatment_advice"
    DOSAGE_INFORMATION = "dosage_information"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"

@dataclass
class SafetyAssessment:
    """Result of safety assessment"""
    risk_level: RiskLevel
    safety_flags: List[SafetyFlag]
    warnings: List[str]
    disclaimers: List[str]
    blocked_content: List[str]
    recommendations: List[str]
    confidence_override: Optional[float] = None
    human_review_required: bool = False

class MedicalSafetyValidator:
    """
    Comprehensive medical safety validation system that ensures
    AI responses meet medical safety standards and regulatory requirements.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the medical safety validator
        
        Args:
            config: Configuration for safety parameters
        """
        self.config = config or self._get_default_config()
        self._initialize_safety_patterns()
        self._initialize_emergency_symptoms()
        self._initialize_restricted_terms()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default safety configuration"""
        return {
            'confidence_thresholds': {
                'diagnosis_claim': 0.95,  # Very high threshold for diagnosis claims
                'treatment_advice': 0.90,
                'medication_info': 0.85,
                'general_info': 0.70
            },
            'uncertainty_thresholds': {
                'acceptable': 0.3,
                'concerning': 0.5,
                'unacceptable': 0.7
            },
            'enable_emergency_detection': True,
            'enable_drug_interaction_check': True,
            'require_disclaimers': True,
            'block_diagnosis_claims': True,
            'block_treatment_prescriptions': True,
            'enable_human_review_flags': True,
            'regulatory_compliance': 'FDA_GUIDANCE',  # FDA, EMA, etc.
            'risk_tolerance': 'conservative'  # conservative, moderate, permissive
        }
    
    def _initialize_safety_patterns(self):
        """Initialize patterns for detecting unsafe content"""
        self.diagnosis_patterns = [
            r'\byou have\b.*\b(disease|condition|disorder|syndrome)\b',
            r'\bdiagnosed with\b',
            r'\byou are suffering from\b',
            r'\byour condition is\b',
            r'\byou definitely have\b',
            r'\bconfirm(ed)? diagnosis of\b'
        ]
        
        self.treatment_patterns = [
            r'\btake\b.*\b(mg|medication|drug|pill|tablet)\b',
            r'\bdosage\b.*\bmg\b',
            r'\bprescrib(e|ed|ing)\b',
            r'\btreatment plan\b',
            r'\byou should take\b',
            r'\brecommended dose\b'
        ]
        
        self.emergency_patterns = [
            r'\bemergency\b',
            r'\bimmediate(ly)?\b.*\b(help|attention|care)\b',
            r'\bcall\b.*\b(911|ambulance|emergency)\b',
            r'\blife.threatening\b',
            r'\bseek immediate\b'
        ]
    
    def _initialize_emergency_symptoms(self):
        """Initialize list of emergency symptoms that require immediate attention"""
        self.emergency_symptoms = {
            'chest_pain', 'severe_chest_pain', 'crushing_chest_pain',
            'difficulty_breathing', 'shortness_of_breath', 'severe_dyspnea',
            'loss_of_consciousness', 'fainting', 'syncope',
            'severe_headache', 'sudden_severe_headache',
            'stroke_symptoms', 'facial_drooping', 'speech_difficulty',
            'severe_abdominal_pain', 'acute_abdominal_pain',
            'heavy_bleeding', 'severe_bleeding', 'hemorrhage',
            'seizure', 'convulsions',
            'severe_allergic_reaction', 'anaphylaxis',
            'poisoning', 'overdose',
            'severe_burns', 'major_trauma'
        }
    
    def _initialize_restricted_terms(self):
        """Initialize terms that should trigger safety warnings"""
        self.restricted_terms = {
            'diagnosis': ['diagnose', 'diagnosed', 'diagnosis'],
            'prescription': ['prescribe', 'prescribed', 'prescription', 'dosage', 'mg'],
            'emergency': ['emergency', 'urgent', 'critical', 'life-threatening'],
            'certainty': ['definitely', 'certainly', 'absolutely', 'confirm', 'guaranteed']
        }
    
    def validate_response(self, response: str, entities: List[Dict], 
                         predictions: Dict, confidence: float, 
                         uncertainty: float) -> SafetyAssessment:
        """
        Comprehensive safety validation of medical response
        
        Args:
            response: Generated medical response text
            entities: Extracted medical entities
            predictions: Disease/condition predictions
            confidence: Overall confidence score
            uncertainty: Overall uncertainty score
            
        Returns:
            SafetyAssessment with validation results
        """
        safety_flags = []
        warnings = []
        disclaimers = []
        blocked_content = []
        recommendations = []
        
        # Check confidence thresholds
        confidence_flags = self._check_confidence_thresholds(response, confidence)
        safety_flags.extend(confidence_flags)
        
        # Check uncertainty levels
        uncertainty_flags = self._check_uncertainty_levels(uncertainty)
        safety_flags.extend(uncertainty_flags)
        
        # Check for emergency symptoms
        emergency_flags = self._check_emergency_symptoms(entities)
        safety_flags.extend(emergency_flags)
        
        # Check for unsafe language patterns
        pattern_flags, blocked = self._check_unsafe_patterns(response)
        safety_flags.extend(pattern_flags)
        blocked_content.extend(blocked)
        
        # Check for contradictory evidence
        contradiction_flags = self._check_contradictory_evidence(predictions)
        safety_flags.extend(contradiction_flags)
        
        # Generate warnings based on flags
        warnings = self._generate_warnings(safety_flags, confidence, uncertainty)
        
        # Generate disclaimers
        disclaimers = self._generate_disclaimers(safety_flags, entities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(safety_flags, predictions)
        
        # Determine risk level
        risk_level = self._assess_risk_level(safety_flags, confidence, uncertainty)
        
        # Determine if human review is required
        human_review_required = self._requires_human_review(safety_flags, risk_level)
        
        # Apply confidence override if necessary
        confidence_override = self._calculate_confidence_override(safety_flags, confidence)
        
        return SafetyAssessment(
            risk_level=risk_level,
            safety_flags=safety_flags,
            warnings=warnings,
            disclaimers=disclaimers,
            blocked_content=blocked_content,
            recommendations=recommendations,
            confidence_override=confidence_override,
            human_review_required=human_review_required
        )
    
    def _check_confidence_thresholds(self, response: str, confidence: float) -> List[SafetyFlag]:
        """Check if confidence meets safety thresholds for different content types"""
        flags = []
        
        # Check for diagnosis-related content
        if any(re.search(pattern, response.lower()) for pattern in self.diagnosis_patterns):
            if confidence < self.config['confidence_thresholds']['diagnosis_claim']:
                flags.append(SafetyFlag.DIAGNOSIS_CLAIM)
        
        # Check for treatment-related content
        if any(re.search(pattern, response.lower()) for pattern in self.treatment_patterns):
            if confidence < self.config['confidence_thresholds']['treatment_advice']:
                flags.append(SafetyFlag.TREATMENT_ADVICE)
        
        # General low confidence check
        if confidence < self.config['confidence_thresholds']['general_info']:
            flags.append(SafetyFlag.LOW_CONFIDENCE)
        
        return flags
    
    def _check_uncertainty_levels(self, uncertainty: float) -> List[SafetyFlag]:
        """Check if uncertainty levels are within acceptable ranges"""
        flags = []
        
        if uncertainty > self.config['uncertainty_thresholds']['unacceptable']:
            flags.append(SafetyFlag.HIGH_UNCERTAINTY)
        elif uncertainty > self.config['uncertainty_thresholds']['concerning']:
            flags.append(SafetyFlag.HIGH_UNCERTAINTY)
        
        return flags
    
    def _check_emergency_symptoms(self, entities: List[Dict]) -> List[SafetyFlag]:
        """Check for emergency symptoms that require immediate attention"""
        flags = []
        
        if not self.config['enable_emergency_detection']:
            return flags
        
        for entity in entities:
            entity_name = entity.get('concept_name', entity.get('text', '')).lower()
            if entity_name in self.emergency_symptoms:
                flags.append(SafetyFlag.EMERGENCY_SYMPTOMS)
                break
        
        return flags
    
    def _check_unsafe_patterns(self, response: str) -> Tuple[List[SafetyFlag], List[str]]:
        """Check for unsafe language patterns in the response"""
        flags = []
        blocked_content = []
        
        response_lower = response.lower()
        
        # Check for diagnosis claims
        if self.config['block_diagnosis_claims']:
            for pattern in self.diagnosis_patterns:
                matches = re.findall(pattern, response_lower)
                if matches:
                    flags.append(SafetyFlag.DIAGNOSIS_CLAIM)
                    blocked_content.extend(matches)
        
        # Check for treatment prescriptions
        if self.config['block_treatment_prescriptions']:
            for pattern in self.treatment_patterns:
                matches = re.findall(pattern, response_lower)
                if matches:
                    flags.append(SafetyFlag.TREATMENT_ADVICE)
                    blocked_content.extend(matches)
        
        return flags, blocked_content
    
    def _check_contradictory_evidence(self, predictions: Dict) -> List[SafetyFlag]:
        """Check for contradictory evidence in predictions"""
        flags = []
        
        if not predictions or 'predictions' not in predictions:
            return flags
        
        prediction_confidences = []
        for disease_info in predictions.get('predictions', {}).values():
            confidence = disease_info.get('confidence', disease_info.get('fused_probability', 0))
            prediction_confidences.append(confidence)
        
        # Check for high variance in predictions (indicating contradiction)
        if len(prediction_confidences) > 1:
            import numpy as np
            variance = np.var(prediction_confidences)
            if variance > 0.2:  # High variance threshold
                flags.append(SafetyFlag.CONTRADICTORY_EVIDENCE)
        
        return flags
    
    def _generate_warnings(self, safety_flags: List[SafetyFlag], 
                          confidence: float, uncertainty: float) -> List[str]:
        """Generate appropriate warnings based on safety flags"""
        warnings = []
        
        if SafetyFlag.EMERGENCY_SYMPTOMS in safety_flags:
            warnings.append(
                "IMPORTANT: Some symptoms mentioned may require immediate medical attention. "
                "If you are experiencing a medical emergency, call emergency services immediately."
            )
        
        if SafetyFlag.LOW_CONFIDENCE in safety_flags:
            warnings.append(
                f"Low confidence in analysis (confidence: {confidence:.1%}). "
                "Please consult with a healthcare professional for accurate assessment."
            )
        
        if SafetyFlag.HIGH_UNCERTAINTY in safety_flags:
            warnings.append(
                f"High uncertainty in predictions (uncertainty: {uncertainty:.1%}). "
                "Multiple interpretations are possible."
            )
        
        if SafetyFlag.DIAGNOSIS_CLAIM in safety_flags:
            warnings.append(
                "This response contains diagnostic-like statements. "
                "Only qualified healthcare professionals can provide medical diagnoses."
            )
        
        if SafetyFlag.TREATMENT_ADVICE in safety_flags:
            warnings.append(
                "This response contains treatment-related information. "
                "Do not use this as a substitute for professional medical advice."
            )
        
        if SafetyFlag.CONTRADICTORY_EVIDENCE in safety_flags:
            warnings.append(
                "Contradictory evidence detected in analysis. "
                "Further evaluation by a medical professional is recommended."
            )
        
        return warnings
    
    def _generate_disclaimers(self, safety_flags: List[SafetyFlag], 
                            entities: List[Dict]) -> List[str]:
        """Generate appropriate medical disclaimers"""
        disclaimers = []
        
        if not self.config['require_disclaimers']:
            return disclaimers
        
        # General medical disclaimer
        disclaimers.append(
            "This AI-generated response is for informational purposes only and "
            "should not be considered as professional medical advice, diagnosis, or treatment. "
            "Always consult with a qualified healthcare provider for medical concerns."
        )
        
        # Entity-specific disclaimers
        disease_entities = [e for e in entities if e.get('concept_type') == 'DISEASE']
        if disease_entities:
            disclaimers.append(
                "Information about medical conditions is general in nature and "
                "may not apply to your specific situation."
            )
        
        # Emergency disclaimer
        if SafetyFlag.EMERGENCY_SYMPTOMS in safety_flags:
            disclaimers.append(
                "For medical emergencies, contact your local emergency services immediately. "
                "This AI system cannot replace emergency medical care."
            )
        
        # Uncertainty disclaimer
        if SafetyFlag.HIGH_UNCERTAINTY in safety_flags:
            disclaimers.append(
                "This analysis contains significant uncertainty. "
                "Professional medical evaluation is strongly recommended."
            )
        
        return disclaimers
    
    def _generate_recommendations(self, safety_flags: List[SafetyFlag], 
                                predictions: Dict) -> List[str]:
        """Generate actionable recommendations based on safety assessment"""
        recommendations = []
        
        if SafetyFlag.EMERGENCY_SYMPTOMS in safety_flags:
            recommendations.append(
                "Seek immediate medical attention from a healthcare professional or emergency services"
            )
        
        if SafetyFlag.LOW_CONFIDENCE in safety_flags or SafetyFlag.HIGH_UNCERTAINTY in safety_flags:
            recommendations.append(
                "Schedule an appointment with your healthcare provider for proper evaluation"
            )
        
        if SafetyFlag.CONTRADICTORY_EVIDENCE in safety_flags:
            recommendations.append(
                "Consider getting a second opinion from another qualified healthcare professional"
            )
        
        if predictions and predictions.get('predictions'):
            recommendations.append(
                "Discuss these potential conditions with your healthcare provider, "
                "providing them with your complete medical history and current symptoms"
            )
        
        return recommendations
    
    def _assess_risk_level(self, safety_flags: List[SafetyFlag], 
                          confidence: float, uncertainty: float) -> RiskLevel:
        """Assess overall risk level based on safety flags and metrics"""
        
        # Critical risk indicators
        if SafetyFlag.EMERGENCY_SYMPTOMS in safety_flags:
            return RiskLevel.CRITICAL
        
        # High risk indicators
        high_risk_flags = [SafetyFlag.DIAGNOSIS_CLAIM, SafetyFlag.TREATMENT_ADVICE]
        if any(flag in safety_flags for flag in high_risk_flags):
            return RiskLevel.HIGH
        
        # Moderate risk based on confidence/uncertainty
        if (confidence < 0.5 or uncertainty > 0.7 or 
            SafetyFlag.CONTRADICTORY_EVIDENCE in safety_flags):
            return RiskLevel.MODERATE
        
        # Low confidence or uncertainty
        if SafetyFlag.LOW_CONFIDENCE in safety_flags or SafetyFlag.HIGH_UNCERTAINTY in safety_flags:
            return RiskLevel.MODERATE
        
        return RiskLevel.LOW
    
    def _requires_human_review(self, safety_flags: List[SafetyFlag], 
                              risk_level: RiskLevel) -> bool:
        """Determine if human review is required"""
        
        if not self.config['enable_human_review_flags']:
            return False
        
        # Always require review for critical and high risk
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            return True
        
        # Require review for specific flags
        review_flags = [
            SafetyFlag.EMERGENCY_SYMPTOMS,
            SafetyFlag.DIAGNOSIS_CLAIM,
            SafetyFlag.TREATMENT_ADVICE,
            SafetyFlag.CONTRADICTORY_EVIDENCE
        ]
        
        return any(flag in safety_flags for flag in review_flags)
    
    def _calculate_confidence_override(self, safety_flags: List[SafetyFlag], 
                                     original_confidence: float) -> Optional[float]:
        """Calculate confidence override based on safety concerns"""
        
        # Reduce confidence for safety concerns
        confidence_reduction = 0.0
        
        if SafetyFlag.HIGH_UNCERTAINTY in safety_flags:
            confidence_reduction += 0.2
        
        if SafetyFlag.CONTRADICTORY_EVIDENCE in safety_flags:
            confidence_reduction += 0.3
        
        if SafetyFlag.LOW_CONFIDENCE in safety_flags:
            confidence_reduction += 0.1
        
        if confidence_reduction > 0:
            new_confidence = max(0.0, original_confidence - confidence_reduction)
            return new_confidence
        
        return None
    
    def sanitize_response(self, response: str, safety_assessment: SafetyAssessment) -> str:
        """Sanitize response by removing or modifying unsafe content"""
        
        sanitized_response = response
        
        # Remove blocked content
        for blocked in safety_assessment.blocked_content:
            sanitized_response = sanitized_response.replace(blocked, "[content removed for safety]")
        
        # Add safety warnings at the beginning
        if safety_assessment.warnings:
            warning_text = "\n".join([f"WARNING: {warning}" for warning in safety_assessment.warnings])
            sanitized_response = f"{warning_text}\n\n{sanitized_response}"
        
        # Add disclaimers at the end
        if safety_assessment.disclaimers:
            disclaimer_text = "\n".join([f"DISCLAIMER: {disclaimer}" for disclaimer in safety_assessment.disclaimers])
            sanitized_response = f"{sanitized_response}\n\n{disclaimer_text}"
        
        # Add recommendations
        if safety_assessment.recommendations:
            recommendation_text = "\n".join([f"RECOMMENDATION: {rec}" for rec in safety_assessment.recommendations])
            sanitized_response = f"{sanitized_response}\n\n{recommendation_text}"
        
        return sanitized_response
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety system metrics for monitoring"""
        return {
            'config': self.config,
            'emergency_symptoms_count': len(self.emergency_symptoms),
            'safety_patterns_count': len(self.diagnosis_patterns) + len(self.treatment_patterns),
            'risk_tolerance': self.config['risk_tolerance'],
            'regulatory_compliance': self.config['regulatory_compliance']
        }