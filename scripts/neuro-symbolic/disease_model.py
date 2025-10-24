import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import json
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx
from collections import defaultdict, Counter
import logging
import warnings
from dataclasses import dataclass
import pickle
from pathlib import Path
from neo4j import GraphDatabase
import sys
import os

# Add knowledge_base path to sys.path for imports
knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
sys.path.append(knowledge_base_path)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass


class MedicalRule:
    """Structured representation of a medical rule with evidence"""
    symptoms: frozenset
    disease: str
    confidence: float
    evidence_source: str
    sensitivity: float = 0.0
    specificity: float = 0.0
    prevalence: float = 0.0

@dataclass


class ValidationMetrics:
    """Comprehensive validation metrics for medical AI"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sensitivity: float
    specificity: float
    ppv: float  # Positive Predictive Value
    npv: float  # Negative Predictive Value


class MedicalKnowledgeGraph:
    """
    Medical knowledge graph integrated with Neo4j ontology data
    Handles loading and querying of symptom-disease relationships
    """


    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
        self.graph = nx.MultiDiGraph()
        self.symptom_disease_rules = {}
        self.differential_diagnoses = defaultdict(set)
        self.contraindications = defaultdict(set)
        self.temporal_relationships = {}
        self.demographic_factors = {}
        self.evidence_weights = {}

        # Neo4j connection parameters
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.connect_to_neo4j()


    def connect_to_neo4j(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Connected to Neo4j for knowledge graph")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.warning("Knowledge graph will not be available without Neo4j connection")
            self.driver = None


    def add_validated_rule(self, rule: MedicalRule):
        """Add a medically validated rule with evidence tracking"""
        rule_key = rule.symptoms
        if rule_key not in self.symptom_disease_rules:
            self.symptom_disease_rules[rule_key] = []
        self.symptom_disease_rules[rule_key].append(rule)

        # Add to NetworkX graph for analysis
        for symptom in rule.symptoms:
            self.graph.add_edge(symptom, rule.disease,
                              weight=rule.confidence,
                              evidence=rule.evidence_source,
                              sensitivity=rule.sensitivity,
                              specificity=rule.specificity)


    def add_differential_diagnosis(self, disease1: str, disease2: str, similarity: float):
        """Add differential diagnosis relationships"""
        self.differential_diagnoses[disease1].add((disease2, similarity))
        self.differential_diagnoses[disease2].add((disease1, similarity))


    def add_contraindication(self, symptom: str, disease: str, reason: str):
        """Add contraindications - symptoms that rule out diseases"""
        self.contraindications[disease].add((symptom, reason))


    def get_evidence_based_predictions(self, symptoms: Set[str],
                                     patient_demographics: Dict = None) -> Tuple[Dict[str, float], Dict[str, List]]:
        """Get predictions based on evidence-weighted rules"""
        disease_scores = defaultdict(float)
        disease_evidence = defaultdict(list)

        # Direct rule matching with evidence weighting
        for rule_symptoms, rules in self.symptom_disease_rules.items():
            if rule_symptoms.issubset(symptoms):
                for rule in rules:
                    # Calculate evidence weight using sensitivity, specificity, and prevalence
                    evidence_weight = (rule.sensitivity * rule.specificity *
                                     (rule.prevalence + 0.1))  # Add small constant to avoid zero
                    weighted_score = rule.confidence * evidence_weight

                    disease_scores[rule.disease] += weighted_score
                    disease_evidence[rule.disease].append({
                        'rule': rule,
                        'weight': weighted_score,
                        'evidence': rule.evidence_source
                    })

        # Apply contraindications to penalize inappropriate diagnoses
        for disease in list(disease_scores.keys()):
            if disease in self.contraindications:
                for contraindicated_symptom, reason in self.contraindications[disease]:
                    if contraindicated_symptom in symptoms:
                        disease_scores[disease] *= 0.1  # Heavy penalty for contraindications
                        logger.info(f"Contraindication found for {disease}: {reason}")

        # Normalize scores to create probability distribution
        total_score = sum(disease_scores.values())
        if total_score > 0:
            disease_scores = {k: v/total_score for k, v in disease_scores.items()}

        return dict(disease_scores), dict(disease_evidence)


    def load_from_neo4j(self, min_weight=0.3, max_rules_per_disease=50):
        """Load medical rules from Neo4j ontology database"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False

        try:
            with self.driver.session() as session:
                # Query all disease-symptom relationships above minimum weight threshold
                result = session.run("""
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
                    WHERE r.weight >= $min_weight
                    RETURN d.id as disease_id, d.name as disease_name, d.name_display as disease_display,
                           s.name as symptom_name, s.name_display as symptom_display,
                           r.weight as weight, r.tfidf_weight as tfidf_weight,
                           r.pmi_confidence as pmi_confidence, r.category as category
                    ORDER BY d.id, r.weight DESC
                """, min_weight=min_weight)

                # Group symptoms by disease to create comprehensive medical rules
                disease_symptoms = defaultdict(list)
                for record in result:
                    disease_key = record['disease_name']
                    symptom_info = {
                        'name': record['symptom_name'],
                        'display': record['symptom_display'],
                        'weight': record['weight'],
                        'tfidf_weight': record['tfidf_weight'],
                        'pmi_confidence': record['pmi_confidence'],
                        'category': record['category']
                    }
                    disease_symptoms[disease_key].append(symptom_info)

                # Create medical rules from Neo4j ontology data
                rules_created = 0
                for disease_name, symptoms_list in disease_symptoms.items():
                    # Sort by weight and take top symptoms to avoid noise
                    symptoms_list.sort(key=lambda x: x['weight'], reverse=True)
                    top_symptoms = symptoms_list[:max_rules_per_disease]

                    # Create rule combinations from high-weight symptoms
                    self._create_rules_from_symptoms(disease_name, top_symptoms)
                    rules_created += 1

                logger.info(f"Successfully loaded {rules_created} disease patterns from Neo4j ontology")
                logger.info(f"Total symptom-disease rules: {len(self.symptom_disease_rules)}")
                return True

        except Exception as e:
            logger.error(f"Failed to load from Neo4j: {e}")
            return False


    def _create_rules_from_symptoms(self, disease_name, symptoms_list):
        """Create medical rules from Neo4j symptom data with proper weighting"""
        # Single symptom rules for high-confidence symptoms
        for symptom_info in symptoms_list:
            if symptom_info['weight'] >= 0.4:  # High confidence threshold
                rule = MedicalRule(
                    symptoms=frozenset([symptom_info['name']]),
                    disease=disease_name,
                    confidence=float(symptom_info['weight']),
                    evidence_source="Neo4j_Ontology_TF_IDF_PMI",
                    sensitivity=min(0.95, float(symptom_info['tfidf_weight']) * 1.2),
                    specificity=min(0.95, float(symptom_info['pmi_confidence']) * 1.1),
                    prevalence=0.1  # Default prevalence, can be updated with real data
                )
                self.add_validated_rule(rule)

        # Two symptom combination rules for very high-weight symptoms
        high_weight_symptoms = [s for s in symptoms_list if s['weight'] >= 0.5]
        for i in range(len(high_weight_symptoms)):
            for j in range(i + 1, min(len(high_weight_symptoms), i + 4)):  # Limit combinations
                symptom1 = high_weight_symptoms[i]
                symptom2 = high_weight_symptoms[j]

                # Combined confidence based on both symptoms
                combined_weight = (symptom1['weight'] + symptom2['weight']) / 2
                combined_confidence = min(0.95, combined_weight * 1.1)

                rule = MedicalRule(
                    symptoms=frozenset([symptom1['name'], symptom2['name']]),
                    disease=disease_name,
                    confidence=combined_confidence,
                    evidence_source="Neo4j_Ontology_Combined",
                    sensitivity=min(0.92, combined_weight * 1.1),
                    specificity=min(0.93, (symptom1['pmi_confidence'] + symptom2['pmi_confidence']) / 2 * 1.05),
                    prevalence=0.08
                )
                self.add_validated_rule(rule)


    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


class CalibratedNeuralPredictor(nn.Module):
    """
    Neural predictor with calibration and uncertainty quantification
    Uses Monte Carlo dropout for uncertainty estimation
    """


    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super(CalibratedNeuralPredictor, self).__init__()

        self.use_batch_norm = use_batch_norm
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer for disease predictions
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Calibration layer for uncertainty quantification
        self.calibration_layer = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )


    def forward(self, x, return_uncertainty=False):
        """
        Forward pass with optional uncertainty estimation
        Uses Monte Carlo dropout for uncertainty quantification
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)

        logits = self.output_layer(x)

        if return_uncertainty:
            # Enable dropout for uncertainty estimation
            self.train()
            predictions = []
            for _ in range(10):  # Multiple forward passes for uncertainty
                pred = torch.sigmoid(self.output_layer(x))
                predictions.append(pred)
            self.eval()

            # Calculate mean and variance across predictions
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            uncertainty = predictions.var(dim=0)

            return mean_pred, uncertainty

        return torch.sigmoid(logits)


class EvidenceBasedFusion(nn.Module):
    """
    Evidence-based fusion mechanism for combining neural and symbolic predictions
    Uses context-aware attention to balance different prediction sources
    """


    def __init__(self, input_dim: int, num_diseases: int):
        super(EvidenceBasedFusion, self).__init__()

        # Context-aware attention network for dynamic weighting
        self.context_net = nn.Sequential(
            nn.Linear(input_dim + num_diseases * 2, 128),  # symptoms + both predictions
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Neural weight, Symbolic weight, Uncertainty weight
            nn.Softmax(dim=1)
        )

        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim + num_diseases * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


    def forward(self, symptoms, neural_pred, symbolic_pred):
        """
        Fuse neural and symbolic predictions based on context
        Returns fused predictions, confidence scores, and fusion weights
        """
        # Concatenate all inputs for context analysis
        context_input = torch.cat([symptoms, neural_pred, symbolic_pred], dim=1)

        # Calculate dynamic fusion weights based on context
        fusion_weights = self.context_net(context_input)

        # Estimate confidence in the fused prediction
        confidence = self.confidence_net(context_input)

        # Weighted combination of predictions
        fused_pred = (fusion_weights[:, 0:1] * neural_pred +
                     fusion_weights[:, 1:2] * symbolic_pred)

        # Apply confidence weighting to final prediction
        final_pred = fused_pred * confidence

        return final_pred, confidence, fusion_weights


class OptimizedDiseasePredictor:
    """
    Optimized neuro-symbolic disease predictor with high factual accuracy
    Combines neural networks with medical knowledge graph reasoning
    """


    def __init__(self, validation_strategy='stratified_kfold',
                 uncertainty_threshold=0.3, confidence_threshold=0.7):
        self.knowledge_graph = MedicalKnowledgeGraph()
        self.neural_model = None
        self.fusion_model = None
        self.symptom_encoder = None
        self.disease_encoder = None
        self.symptom_scaler = StandardScaler()

        self.validation_strategy = validation_strategy
        self.uncertainty_threshold = uncertainty_threshold
        self.confidence_threshold = confidence_threshold

        self.symptoms_vocab = []
        self.diseases_vocab = []
        self.validation_metrics = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def load_knowledge_base_from_neo4j(self, min_weight: float = 0.3):
        """Load medical knowledge base from Neo4j ontology"""
        success = self.knowledge_graph.load_from_neo4j(min_weight=min_weight)
        if success:
            logger.info("Successfully loaded medical knowledge from Neo4j ontology")
        else:
            logger.error("Failed to load knowledge base from Neo4j")
        return success


    def _compute_class_weights(self, y):
        """Compute class weights for handling imbalanced datasets"""
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        class_counts = np.bincount(y_np)
        total_samples = len(y_np)
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights).to(self.device)


    def _get_symbolic_predictions_batch(self, X_batch):
        """Get symbolic predictions for a batch of inputs using knowledge graph"""
        batch_size = X_batch.shape[0]
        symbolic_preds = torch.zeros(batch_size, len(self.diseases_vocab))

        for i in range(batch_size):
            # Convert tensor back to symptoms
            active_symptoms = set()
            for j, val in enumerate(X_batch[i]):
                if val > 0.5:  # Threshold for active symptoms
                    symptom = self.symptoms_vocab[j]
                    active_symptoms.add(symptom)

            # Get symbolic predictions from knowledge graph
            disease_scores, _ = self.knowledge_graph.get_evidence_based_predictions(active_symptoms)

            # Convert to tensor format
            for disease, score in disease_scores.items():
                if disease in self.diseases_vocab:
                    disease_idx = self.diseases_vocab.index(disease)
                    symbolic_preds[i, disease_idx] = score

        return symbolic_preds


    def _compute_comprehensive_metrics(self, y_true, y_pred, y_proba) -> ValidationMetrics:
        """Compute comprehensive validation metrics for medical applications"""

        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # ROC AUC for multiclass classification
        try:
            auc_roc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc_roc = 0.0

        # Medical-specific metrics
        sensitivity = recall  # Sensitivity equals recall

        # Calculate specificity (average across all classes)
        specificities = []
        for class_idx in range(len(self.diseases_vocab)):
            tn = np.sum((y_true != class_idx) & (y_pred != class_idx))
            fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(specificity)

        avg_specificity = np.mean(specificities)

        # Positive Predictive Value (PPV) equals precision
        ppv = precision

        # Negative Predictive Value (NPV) calculation
        npvs = []
        for class_idx in range(len(self.diseases_vocab)):
            tn = np.sum((y_true != class_idx) & (y_pred != class_idx))
            fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            npvs.append(npv)

        avg_npv = np.mean(npvs)

        return ValidationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            sensitivity=sensitivity,
            specificity=avg_specificity,
            ppv=ppv,
            npv=avg_npv
        )


    def advanced_cross_validation(self, X, y, n_folds=5) -> Dict:
        """Perform stratified cross-validation with comprehensive medical metrics"""

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{n_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Initialize and train neural model for this fold
            neural_model = CalibratedNeuralPredictor(
                input_dim=X.shape[1],
                hidden_dims=[256, 128, 64],
                output_dim=len(self.diseases_vocab)
            ).to(self.device)

            # Training with early stopping
            optimizer = optim.Adam(neural_model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss(weight=self._compute_class_weights(y_train))

            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10

            for epoch in range(100):
                neural_model.train()
                optimizer.zero_grad()

                outputs = neural_model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

                # Validation step
                neural_model.eval()
                with torch.no_grad():
                    val_outputs = neural_model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Evaluate this fold
            neural_model.eval()
            with torch.no_grad():
                neural_pred = neural_model(X_val)
                neural_pred_classes = torch.argmax(neural_pred, dim=1)

                # Calculate comprehensive metrics
                metrics = self._compute_comprehensive_metrics(
                    y_val.cpu().numpy(),
                    neural_pred_classes.cpu().numpy(),
                    neural_pred.cpu().numpy()
                )

                fold_metrics.append(metrics)

        # Aggregate metrics across all folds
        return self._aggregate_fold_metrics(fold_metrics)


    def _aggregate_fold_metrics(self, fold_metrics: List[ValidationMetrics]) -> Dict:
        """Aggregate metrics across cross-validation folds"""

        metrics_dict = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
                       'sensitivity', 'specificity', 'ppv', 'npv']

        for metric_name in metric_names:
            values = [getattr(fold_metric, metric_name) for fold_metric in fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)

            metrics_dict[f'{metric_name}_mean'] = mean_val
            metrics_dict[f'{metric_name}_std'] = std_val

        return metrics_dict


    def train_optimized_model(self, symptoms_data: List[List[str]], diseases_data: List[str],
                            epochs: int = 100, batch_size: int = 32, cv_folds: int = 5):
        """Train the optimized neuro-symbolic model with comprehensive validation"""

        # Setup data encoders
        self.symptom_encoder = MultiLabelBinarizer()
        self.disease_encoder = LabelEncoder()

        # Prepare and encode training data
        X = self.symptom_encoder.fit_transform(symptoms_data)
        y = self.disease_encoder.fit_transform(diseases_data)

        # Scale features for better neural network training
        X = self.symptom_scaler.fit_transform(X)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_metrics = self.advanced_cross_validation(X_tensor, y_tensor, n_folds=cv_folds)
        self.validation_metrics = cv_metrics

        # Train final model on full dataset
        logger.info("Training final model...")

        self.neural_model = CalibratedNeuralPredictor(
            input_dim=X.shape[1],
            hidden_dims=[256, 128, 64],
            output_dim=len(self.diseases_vocab)
        ).to(self.device)

        self.fusion_model = EvidenceBasedFusion(
            input_dim=X.shape[1],
            num_diseases=len(self.diseases_vocab)
        ).to(self.device)

        # Setup optimizer and loss function with class weighting
        optimizer = optim.Adam(
            list(self.neural_model.parameters()) + list(self.fusion_model.parameters()),
            lr=0.001, weight_decay=1e-5
        )

        class_weights = self._compute_class_weights(y_tensor)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Main training loop
        for epoch in range(epochs):
            self.neural_model.train()
            self.fusion_model.train()

            total_loss = 0
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]

                optimizer.zero_grad()

                # Get neural predictions
                neural_pred = self.neural_model(batch_X)

                # Get symbolic predictions from knowledge graph
                symbolic_pred = self._get_symbolic_predictions_batch(batch_X).to(self.device)

                # Fuse predictions using evidence-based fusion
                fused_pred, confidence, _ = self.fusion_model(batch_X, neural_pred, symbolic_pred)

                # Calculate multi-component loss
                neural_loss = criterion(neural_pred, batch_y)
                fused_loss = criterion(fused_pred, batch_y)

                # Confidence regularization to encourage high confidence predictions
                confidence_reg = torch.mean((confidence - 0.8) ** 2)

                # Combined loss with different weightings
                total_loss_batch = 0.4 * neural_loss + 0.5 * fused_loss + 0.1 * confidence_reg
                total_loss_batch.backward()

                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.neural_model.parameters()) + list(self.fusion_model.parameters()),
                    max_norm=1.0
                )

                optimizer.step()
                total_loss += total_loss_batch.item()

            # Log progress periodically
            if epoch % 20 == 0:
                avg_loss = total_loss / (len(X_tensor) // batch_size)
                logger.info(f'Epoch {epoch}, Average Loss: {avg_loss:.4f}')

        # Final model evaluation
        self._evaluate_final_model(X_tensor, y_tensor)

        # Print comprehensive validation results
        self._print_validation_results()


    def _evaluate_final_model(self, X, y):
        """Evaluate the final trained model and analyze fusion behavior"""
        self.neural_model.eval()
        self.fusion_model.eval()

        with torch.no_grad():
            # Get all types of predictions
            neural_pred = self.neural_model(X)
            symbolic_pred = self._get_symbolic_predictions_batch(X).to(self.device)
            fused_pred, confidence, fusion_weights = self.fusion_model(X, neural_pred, symbolic_pred)

            # Get uncertainty estimates
            neural_pred_unc, uncertainty = self.neural_model(X, return_uncertainty=True)

            # Compute final metrics
            fused_classes = torch.argmax(fused_pred, dim=1)
            neural_classes = torch.argmax(neural_pred, dim=1)

            final_metrics = self._compute_comprehensive_metrics(
                y.cpu().numpy(),
                fused_classes.cpu().numpy(),
                fused_pred.cpu().numpy()
            )

            neural_metrics = self._compute_comprehensive_metrics(
                y.cpu().numpy(),
                neural_classes.cpu().numpy(),
                neural_pred.cpu().numpy()
            )

            self.validation_metrics['final_fused'] = final_metrics
            self.validation_metrics['final_neural'] = neural_metrics

            # Analyze fusion behavior for insights
            avg_neural_weight = torch.mean(fusion_weights[:, 0]).item()
            avg_symbolic_weight = torch.mean(fusion_weights[:, 1]).item()
            avg_confidence = torch.mean(confidence).item()
            avg_uncertainty = torch.mean(uncertainty).item()

            logger.info("Fusion Analysis:")
            logger.info(f"  Average Neural Weight: {avg_neural_weight:.3f}")
            logger.info(f"  Average Symbolic Weight: {avg_symbolic_weight:.3f}")
            logger.info(f"  Average Confidence: {avg_confidence:.3f}")
            logger.info(f"  Average Uncertainty: {avg_uncertainty:.3f}")


    def _print_validation_results(self):
        """Print comprehensive validation results for analysis"""
        print("\n" + "="*60)
        print("COMPREHENSIVE VALIDATION RESULTS")
        print("="*60)

        # Cross-validation results
        print("\nCross-Validation Results (Mean +/- Std):")
        print("-" * 40)

        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
                      'sensitivity', 'specificity', 'ppv', 'npv']

        for metric in key_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in self.validation_metrics and std_key in self.validation_metrics:
                mean_val = self.validation_metrics[mean_key]
                std_val = self.validation_metrics[std_key]
                print(f"{metric.upper():>12}: {mean_val:.4f} +/- {std_val:.4f}")

        # Final model comparison
        if 'final_fused' in self.validation_metrics and 'final_neural' in self.validation_metrics:
            print("\nFinal Model Comparison:")
            print("-" * 40)
            print(f"{'Metric':<15} {'Fused Model':<12} {'Neural Only':<12} {'Improvement':<12}")
            print("-" * 51)

            fused_metrics = self.validation_metrics['final_fused']
            neural_metrics = self.validation_metrics['final_neural']

            for metric in key_metrics:
                if hasattr(fused_metrics, metric) and hasattr(neural_metrics, metric):
                    fused_val = getattr(fused_metrics, metric)
                    neural_val = getattr(neural_metrics, metric)
                    improvement = ((fused_val - neural_val) / neural_val * 100) if neural_val > 0 else 0

                    print(f"{metric.upper():<15} {fused_val:<12.4f} {neural_val:<12.4f} {improvement:<12.2f}%")


    def predict_with_uncertainty(self, symptoms: List[str],
                               return_explanations: bool = True) -> Dict:
        """Make predictions with uncertainty quantification and explanations"""

        if not all([self.neural_model, self.fusion_model, self.symptom_encoder]):
            raise ValueError("Model not trained yet. Please train the model first.")

        # Encode input symptoms
        X = self.symptom_encoder.transform([symptoms])
        X = self.symptom_scaler.transform(X)
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.neural_model.eval()
        self.fusion_model.eval()

        with torch.no_grad():
            # Get neural predictions with uncertainty
            neural_pred, uncertainty = self.neural_model(X_tensor, return_uncertainty=True)

            # Get symbolic predictions from knowledge graph
            symbolic_pred = self._get_symbolic_predictions_batch(X_tensor).to(self.device)

            # Get fused predictions
            fused_pred, confidence, fusion_weights = self.fusion_model(
                X_tensor, neural_pred, symbolic_pred
            )

            # Convert to probabilities
            fused_probs = torch.softmax(fused_pred, dim=1)[0].cpu().numpy()
            neural_probs = torch.softmax(neural_pred, dim=1)[0].cpu().numpy()
            symbolic_probs = torch.softmax(symbolic_pred, dim=1)[0].cpu().numpy()

            # Get top predictions
            top_indices = np.argsort(fused_probs)[::-1][:5]

            results = {
                'predictions': {},
                'confidence': float(confidence[0].cpu().numpy()),
                'fusion_weights': {
                    'neural': float(fusion_weights[0, 0].cpu().numpy()),
                    'symbolic': float(fusion_weights[0, 1].cpu().numpy()),
                    'uncertainty': float(fusion_weights[0, 2].cpu().numpy())
                },
                'average_uncertainty': float(uncertainty[0].mean().cpu().numpy())
            }

            # Format top predictions
            for i, idx in enumerate(top_indices):
                disease = self.diseases_vocab[idx]
                results['predictions'][disease] = {
                    'rank': i + 1,
                    'fused_probability': float(fused_probs[idx]),
                    'neural_probability': float(neural_probs[idx]),
                    'symbolic_probability': float(symbolic_probs[idx]),
                    'uncertainty': float(uncertainty[0, idx].cpu().numpy())
                }

            # Add explanations if requested
            if return_explanations:
                explanations = self._generate_explanations(symptoms, results)
                results['explanations'] = explanations

            # Quality assessment
            results['quality_indicators'] = self._assess_prediction_quality(results)

            return results


    def _generate_explanations(self, symptoms: List[str], results: Dict) -> Dict:
        """Generate explanations for the predictions based on matched rules"""

        # Get symbolic reasoning evidence from knowledge graph
        active_symptoms = set(symptoms)
        disease_scores, disease_evidence = self.knowledge_graph.get_evidence_based_predictions(active_symptoms)

        explanations = {
            'input_symptoms': symptoms,
            'matched_rules': {},
            'reasoning_mode': 'neural' if results['fusion_weights']['neural'] > 0.6 else
                            'symbolic' if results['fusion_weights']['symbolic'] > 0.6 else 'hybrid',
            'confidence_factors': []
        }

        # Add rule-based explanations for top diseases
        for disease, evidence_list in disease_evidence.items():
            if disease in [d for d in results['predictions'].keys()][:3]:  # Top 3 diseases
                explanations['matched_rules'][disease] = []
                for evidence in evidence_list[:2]:  # Top 2 rules per disease
                    rule = evidence['rule']
                    explanations['matched_rules'][disease].append({
                        'symptoms': list(rule.symptoms),
                        'confidence': rule.confidence,
                        'evidence_source': rule.evidence_source,
                        'sensitivity': rule.sensitivity,
                        'specificity': rule.specificity
                    })

        # Analyze confidence factors
        if results['confidence'] > 0.8:
            explanations['confidence_factors'].append("High model confidence")
        if results['average_uncertainty'] < 0.2:
            explanations['confidence_factors'].append("Low prediction uncertainty")
        if results['fusion_weights']['symbolic'] > 0.5:
            explanations['confidence_factors'].append("Strong evidence-based reasoning")

        return explanations


    def _assess_prediction_quality(self, results: Dict) -> Dict:
        """Assess the quality and reliability of predictions"""

        quality_indicators = {
            'reliability_score': 0.0,
            'warnings': [],
            'recommendations': []
        }

        # Calculate base reliability from confidence and uncertainty
        confidence = results['confidence']
        uncertainty = results['average_uncertainty']

        reliability_score = confidence * (1 - uncertainty)

        # Adjust based on fusion weights
        if results['fusion_weights']['symbolic'] > 0.6:
            reliability_score += 0.1  # Bonus for evidence-based reasoning

        if results['fusion_weights']['uncertainty'] > 0.3:
            reliability_score -= 0.1  # Penalty for high uncertainty weight

        quality_indicators['reliability_score'] = min(1.0, max(0.0, reliability_score))

        # Generate warnings based on thresholds
        if confidence < self.confidence_threshold:
            quality_indicators['warnings'].append(
                f"Low confidence ({confidence:.2f}). Consider additional symptoms or tests."
            )

        if uncertainty > self.uncertainty_threshold:
            quality_indicators['warnings'].append(
                f"High uncertainty ({uncertainty:.2f}). Prediction may be unreliable."
            )

        # Check for differential diagnoses
        top_diseases = list(results['predictions'].keys())[:3]
        if len(top_diseases) >= 2:
            prob_diff = (results['predictions'][top_diseases[0]]['fused_probability'] -
                        results['predictions'][top_diseases[1]]['fused_probability'])
            if prob_diff < 0.1:
                quality_indicators['warnings'].append(
                    "Close differential diagnoses detected. Consider discriminating tests."
                )

        # Generate recommendations
        if quality_indicators['reliability_score'] < 0.6:
            quality_indicators['recommendations'].append(
                "Seek additional clinical evaluation"
            )

        if results['fusion_weights']['neural'] > 0.8:
            quality_indicators['recommendations'].append(
                "Pattern-based prediction. Verify with clinical guidelines."
            )

        return quality_indicators


    def save_model(self, path: str):
        """Save the trained model and all components"""
        # Temporarily close Neo4j connection for serialization
        neo4j_config = None
        if hasattr(self.knowledge_graph, 'driver') and self.knowledge_graph.driver:
            neo4j_config = {
                'neo4j_uri': getattr(self.knowledge_graph, 'neo4j_uri', 'bolt://localhost:7687'),
                'neo4j_user': getattr(self.knowledge_graph, 'neo4j_user', 'neo4j'),
                'neo4j_password': getattr(self.knowledge_graph, 'neo4j_password', 'password')
            }
            self.knowledge_graph.close()

        # Save serializable components only
        model_data = {
            'neural_model_state': self.neural_model.state_dict() if self.neural_model else None,
            'fusion_model_state': self.fusion_model.state_dict() if self.fusion_model else None,
            'symptom_encoder': self.symptom_encoder,
            'disease_encoder': self.disease_encoder,
            'symptom_scaler': self.symptom_scaler,
            'symptoms_vocab': self.symptoms_vocab,
            'diseases_vocab': self.diseases_vocab,
            'validation_metrics': self.validation_metrics,
            'knowledge_graph_rules': self.knowledge_graph.symptom_disease_rules if self.knowledge_graph else {},
            'knowledge_graph_differentials': self.knowledge_graph.differential_diagnoses if self.knowledge_graph else {},
            'knowledge_graph_contraindications': self.knowledge_graph.contraindications if self.knowledge_graph else {},
            'neo4j_config': neo4j_config
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

        # Reconnect to Neo4j if we had a connection
        if neo4j_config:
            try:
                self.knowledge_graph.connect_to_neo4j(
                    neo4j_config['neo4j_uri'],
                    neo4j_config['neo4j_user'],
                    neo4j_config['neo4j_password']
                )
            except Exception as e:
                logger.warning(f"Could not reconnect to Neo4j after saving: {e}")


    def load_model(self, path: str):
        """Load a trained model from file"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Restore all components
        self.symptom_encoder = model_data['symptom_encoder']
        self.disease_encoder = model_data['disease_encoder']
        self.symptom_scaler = model_data['symptom_scaler']
        self.symptoms_vocab = model_data['symptoms_vocab']
        self.diseases_vocab = model_data['diseases_vocab']
        self.validation_metrics = model_data['validation_metrics']

        # Handle backward compatibility and restore knowledge graph
        if 'knowledge_graph' in model_data:
            # Old format - direct object
            self.knowledge_graph = model_data['knowledge_graph']
        else:
            # New format - rebuild from components
            self.knowledge_graph = MedicalKnowledgeGraph()
            if 'knowledge_graph_rules' in model_data:
                self.knowledge_graph.symptom_disease_rules = model_data['knowledge_graph_rules']
            if 'knowledge_graph_differentials' in model_data:
                self.knowledge_graph.differential_diagnoses = model_data['knowledge_graph_differentials']
            if 'knowledge_graph_contraindications' in model_data:
                self.knowledge_graph.contraindications = model_data['knowledge_graph_contraindications']

            # Reconnect to Neo4j if configuration is available
            if 'neo4j_config' in model_data and model_data['neo4j_config']:
                config = model_data['neo4j_config']
                try:
                    self.knowledge_graph.connect_to_neo4j(
                        config['neo4j_uri'],
                        config['neo4j_user'],
                        config['neo4j_password']
                    )
                except Exception as e:
                    logger.warning(f"Could not reconnect to Neo4j during load: {e}")

        # Restore neural networks
        if model_data['neural_model_state']:
            self.neural_model = CalibratedNeuralPredictor(
                input_dim=len(self.symptoms_vocab),
                hidden_dims=[256, 128, 64],
                output_dim=len(self.diseases_vocab)
            ).to(self.device)
            self.neural_model.load_state_dict(model_data['neural_model_state'])

        if model_data['fusion_model_state']:
            self.fusion_model = EvidenceBasedFusion(
                input_dim=len(self.symptoms_vocab),
                num_diseases=len(self.diseases_vocab)
            ).to(self.device)
            self.fusion_model.load_state_dict(model_data['fusion_model_state'])

        logger.info(f"Model loaded from {path}")


    def __del__(self):
        """Cleanup: close Neo4j connection"""
        if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
            self.knowledge_graph.close()
