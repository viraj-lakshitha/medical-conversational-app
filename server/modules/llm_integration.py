"""
LLM Integration Module for Medical Conversational AI

Integrates fine-tuned medical reasoning models (Qwen2.5-7B) with the pipeline:
- Medical reasoning chain generation
- Context-aware response synthesis
- Step-by-step clinical thinking
- Integration with evidence fusion and safety systems
"""

import logging
import os
import time
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM integration"""
    model_path: str = ""
    model_type: str = "qwen2.5"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_medical_reasoning: bool = True
    enable_chain_of_thought: bool = True
    safety_filtering: bool = True
    batch_size: int = 1

class MedicalLLMInterface:
    """
    Interface for medical fine-tuned language models.
    Designed to work with Qwen2.5-7B fine-tuned for medical reasoning.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the medical LLM interface
        
        Args:
            config: LLM configuration parameters
        """
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.load_error = None
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned medical model"""
        try:
            # Check if model path exists
            if not self.config.model_path or not os.path.exists(self.config.model_path):
                logger.warning(f"Medical LLM model not found at {self.config.model_path}")
                self.load_error = "Model path not found"
                return
            
            # Try to import transformers
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                try:
                    import torch
                except ImportError:
                    logger.warning("PyTorch not available, LLM will use CPU inference")
                    torch = None
            except ImportError as e:
                logger.warning(f"Transformers not available for LLM: {str(e)}")
                self.load_error = "Transformers library not available"
                return
            
            # Load tokenizer and model
            logger.info(f"Loading medical LLM from {self.config.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Configure model loading parameters
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "low_cpu_mem_usage": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            self.is_loaded = True
            logger.info("Medical LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load medical LLM: {str(e)}")
            self.load_error = str(e)
            self.is_loaded = False
    
    def generate_medical_response(self, query: str, entities: List[Dict], 
                                 fused_evidence: Dict, context_docs: List[Dict],
                                 conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate medical response using fine-tuned LLM
        
        Args:
            query: User's medical question
            entities: Extracted medical entities
            fused_evidence: Fused evidence from multiple sources
            context_docs: Retrieved context documents
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary containing generated response and metadata
        """
        if not self.is_loaded:
            return {
                "response": self._generate_fallback_response(query, entities, fused_evidence),
                "reasoning_chain": ["LLM not available, using fallback response"],
                "confidence": 0.3,
                "method": "fallback"
            }
        
        try:
            # Construct medical reasoning prompt
            prompt = self._construct_medical_prompt(
                query, entities, fused_evidence, context_docs, conversation_history
            )
            
            # Generate response
            response_data = self._generate_with_model(prompt)
            
            # Parse and validate response
            parsed_response = self._parse_medical_response(response_data)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return {
                "response": self._generate_fallback_response(query, entities, fused_evidence),
                "reasoning_chain": [f"LLM generation error: {str(e)}"],
                "confidence": 0.2,
                "method": "fallback_error"
            }
    
    def _construct_medical_prompt(self, query: str, entities: List[Dict], 
                                 fused_evidence: Dict, context_docs: List[Dict],
                                 conversation_history: List[Dict] = None) -> str:
        """Construct comprehensive medical reasoning prompt"""
        
        prompt_parts = []
        
        # System instruction for medical reasoning
        system_instruction = """You are a medical AI assistant designed to provide informative responses about health and medical topics. Follow these guidelines:

1. Provide evidence-based information when possible
2. Use step-by-step reasoning to analyze medical queries
3. Always include appropriate medical disclaimers
4. Clearly state confidence levels and uncertainties
5. Recommend consulting healthcare professionals for specific medical advice
6. Do not provide definitive diagnoses or treatment prescriptions

Format your response with:
- Clear analysis of the medical query
- Step-by-step reasoning process
- Evidence-based conclusions
- Appropriate disclaimers and recommendations"""
        
        prompt_parts.append(f"<system>\n{system_instruction}\n</system>")
        
        # Add conversation history if available
        if conversation_history:
            prompt_parts.append("\n<conversation_history>")
            for turn in conversation_history[-3:]:  # Last 3 turns
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            prompt_parts.append("</conversation_history>")
        
        # Add extracted entities
        if entities:
            prompt_parts.append("\n<extracted_entities>")
            diseases = [e for e in entities if e.get('concept_type') == 'DISEASE']
            symptoms = [e for e in entities if e.get('concept_type') == 'SYMPTOM']
            
            if diseases:
                disease_names = [e.get('display_name', e.get('text', '')) for e in diseases]
                prompt_parts.append(f"Medical conditions mentioned: {', '.join(disease_names)}")
            
            if symptoms:
                symptom_names = [e.get('display_name', e.get('text', '')) for e in symptoms]
                prompt_parts.append(f"Symptoms mentioned: {', '.join(symptom_names)}")
            
            prompt_parts.append("</extracted_entities>")
        
        # Add evidence from knowledge fusion
        if fused_evidence and fused_evidence.get('predictions'):
            prompt_parts.append("\n<medical_evidence>")
            
            # Sort predictions by confidence
            sorted_predictions = sorted(
                fused_evidence['predictions'].items(),
                key=lambda x: x[1].get('confidence', 0),
                reverse=True
            )
            
            prompt_parts.append("Evidence-based analysis:")
            for entity, info in sorted_predictions[:3]:  # Top 3
                confidence = info.get('confidence', 0)
                uncertainty = info.get('uncertainty', 0)
                sources = info.get('sources', [])
                
                prompt_parts.append(
                    f"- {entity}: {confidence:.1%} confidence, "
                    f"{uncertainty:.1%} uncertainty, from {len(sources)} sources"
                )
            
            overall_confidence = fused_evidence.get('overall_confidence', 0)
            overall_uncertainty = fused_evidence.get('overall_uncertainty', 0)
            
            prompt_parts.append(
                f"\nOverall assessment: {overall_confidence:.1%} confidence, "
                f"{overall_uncertainty:.1%} uncertainty"
            )
            
            prompt_parts.append("</medical_evidence>")
        
        # Add context documents
        if context_docs:
            prompt_parts.append(f"\n<reference_documents>")
            prompt_parts.append(f"Based on {len(context_docs)} relevant medical documents in knowledge base")
            prompt_parts.append("</reference_documents>")
        
        # Add the user query
        prompt_parts.append(f"\n<user_query>\n{query}\n</user_query>")
        
        # Add response format instruction
        response_format = """
Please provide a comprehensive medical response that includes:

1. **Analysis**: Break down the medical query and identify key components
2. **Evidence Review**: Analyze the available evidence and entity relationships  
3. **Clinical Reasoning**: Step-by-step reasoning process for medical assessment
4. **Conclusions**: Evidence-based conclusions with confidence levels
5. **Recommendations**: Appropriate next steps and medical advice
6. **Disclaimers**: Include necessary medical disclaimers and safety warnings

Begin your response with "**Medical Analysis:**" and structure your reasoning clearly."""
        
        prompt_parts.append(f"\n<response_format>{response_format}</response_format>")
        
        return "\n".join(prompt_parts)
    
    def _generate_with_model(self, prompt: str) -> str:
        """Generate response using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length - 512  # Leave room for response
            )
            
            # Move to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Model generation error: {str(e)}")
            raise
    
    def _parse_medical_response(self, response: str) -> Dict[str, Any]:
        """Parse and structure the model's response"""
        
        # Extract reasoning chain if present
        reasoning_chain = []
        if "**Clinical Reasoning:**" in response:
            reasoning_section = response.split("**Clinical Reasoning:**")[1]
            if "**Conclusions:**" in reasoning_section:
                reasoning_text = reasoning_section.split("**Conclusions:**")[0]
                reasoning_chain = [line.strip() for line in reasoning_text.split('\n') if line.strip()]
        
        # Estimate confidence based on language used
        confidence = self._estimate_response_confidence(response)
        
        # Clean up response
        cleaned_response = self._clean_response(response)
        
        return {
            "response": cleaned_response,
            "reasoning_chain": reasoning_chain,
            "confidence": confidence,
            "method": "llm_generated",
            "word_count": len(cleaned_response.split()),
            "has_disclaimers": "disclaimer" in response.lower() or "consult" in response.lower()
        }
    
    def _estimate_response_confidence(self, response: str) -> float:
        """Estimate confidence based on language patterns in response"""
        
        # Count confidence indicators
        high_confidence_terms = ["clearly", "definitely", "certainly", "confirmed", "established"]
        low_confidence_terms = ["possibly", "might", "could", "uncertain", "unclear", "may"]
        uncertainty_terms = ["however", "although", "but", "requires further", "needs evaluation"]
        
        response_lower = response.lower()
        
        high_count = sum(1 for term in high_confidence_terms if term in response_lower)
        low_count = sum(1 for term in low_confidence_terms if term in response_lower)
        uncertainty_count = sum(1 for term in uncertainty_terms if term in response_lower)
        
        # Base confidence
        base_confidence = 0.6
        
        # Adjust based on language patterns
        confidence_adjustment = (high_count * 0.1) - (low_count * 0.1) - (uncertainty_count * 0.05)
        
        # Check for structured reasoning
        if "**Analysis:**" in response and "**Conclusions:**" in response:
            confidence_adjustment += 0.1
        
        # Check for medical disclaimers (good practice, slight confidence boost)
        if "consult" in response_lower or "healthcare professional" in response_lower:
            confidence_adjustment += 0.05
        
        final_confidence = max(0.1, min(0.9, base_confidence + confidence_adjustment))
        return final_confidence
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the response"""
        # Remove any prompt artifacts
        cleaned = response.replace("<system>", "").replace("</system>", "")
        cleaned = cleaned.replace("<user_query>", "").replace("</user_query>", "")
        
        # Ensure proper formatting
        cleaned = cleaned.strip()
        
        # Add medical disclaimer if not present
        if "consult" not in cleaned.lower() and "healthcare professional" not in cleaned.lower():
            disclaimer = "\n\n**Important**: This information is for educational purposes only. Please consult with a qualified healthcare professional for personalized medical advice."
            cleaned += disclaimer
        
        return cleaned
    
    def _generate_fallback_response(self, query: str, entities: List[Dict], 
                                   fused_evidence: Dict) -> str:
        """Generate fallback response when LLM is not available"""
        
        response_parts = []
        
        response_parts.append("**Medical Analysis** (Generated using rule-based system)")
        
        # Analyze entities
        if entities:
            diseases = [e for e in entities if e.get('concept_type') == 'DISEASE']
            symptoms = [e for e in entities if e.get('concept_type') == 'SYMPTOM']
            
            if diseases:
                disease_names = [e.get('display_name', e.get('text', '')) for e in diseases]
                response_parts.append(f"\n**Medical Conditions Identified**: {', '.join(disease_names)}")
            
            if symptoms:
                symptom_names = [e.get('display_name', e.get('text', '')) for e in symptoms]
                response_parts.append(f"\n**Symptoms Identified**: {', '.join(symptom_names)}")
        
        # Add evidence if available
        if fused_evidence and fused_evidence.get('predictions'):
            response_parts.append("\n**Evidence-Based Analysis**:")
            
            sorted_predictions = sorted(
                fused_evidence['predictions'].items(),
                key=lambda x: x[1].get('confidence', 0),
                reverse=True
            )
            
            for entity, info in sorted_predictions[:3]:
                confidence = info.get('confidence', 0)
                response_parts.append(f"- {entity}: {confidence:.1%} confidence based on available evidence")
        
        # Add recommendations
        response_parts.append("\n**Recommendations**:")
        response_parts.append("- Consult with a qualified healthcare professional for proper evaluation")
        response_parts.append("- Provide complete medical history and current symptoms to your doctor")
        response_parts.append("- Seek immediate medical attention if experiencing severe symptoms")
        
        # Add disclaimer
        response_parts.append("\n**Medical Disclaimer**: This analysis is generated by an AI system for informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions.")
        
        return "\n".join(response_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "is_loaded": self.is_loaded,
            "load_error": self.load_error,
            "model_path": self.config.model_path,
            "model_type": self.config.model_type,
            "config": {
                "max_length": self.config.max_length,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "use_medical_reasoning": self.config.use_medical_reasoning
            }
        }
    
    def warm_up(self) -> bool:
        """Warm up the model with a test query"""
        if not self.is_loaded:
            return False
        
        try:
            test_query = "What are the common symptoms of hypertension?"
            test_entities = [{"concept_type": "DISEASE", "text": "hypertension"}]
            test_evidence = {"predictions": {}}
            
            result = self.generate_medical_response(
                test_query, test_entities, test_evidence, []
            )
            
            return result.get("method") == "llm_generated"
            
        except Exception as e:
            logger.error(f"Model warm-up failed: {str(e)}")
            return False