import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import sys
import os
from collections import defaultdict, Counter

# Add knowledge_base path to sys.path for imports
knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', 'knowledge_base')
sys.path.append(knowledge_base_path)

from disease_model import OptimizedDiseasePredictor
from neo4j import GraphDatabase

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Neo4jDataManager:
    """
    Manages data extraction and preparation from Neo4j ontology database
    Handles connection, querying, and data generation for model training
    """


    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="password"):
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
            logger.info("Connected to Neo4j database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            return False


    def get_database_statistics(self):
        """Get statistics about the Neo4j database contents"""
        if not self.driver:
            return None

        try:
            with self.driver.session() as session:
                disease_count = session.run("MATCH (d:Disease) RETURN count(d) as count").single()['count']
                symptom_count = session.run("MATCH (s:Symptom) RETURN count(s) as count").single()['count']
                rel_count = session.run("MATCH ()-[r:HAS_SYMPTOM]->() RETURN count(r) as count").single()['count']

                return {
                    'diseases': disease_count,
                    'symptoms': symptom_count,
                    'relationships': rel_count
                }
        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return None


    def extract_training_data(self, min_weight=0.3, n_samples_per_disease=10):
        """Extract training data from Neo4j ontology"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return [], []

        try:
            with self.driver.session() as session:
                # Query disease-symptom relationships above weight threshold
                result = session.run("""
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
                    WHERE r.weight >= $min_weight
                    RETURN d.name as disease, s.name as symptom, r.weight as weight
                    ORDER BY d.name, r.weight DESC
                """, min_weight=min_weight)

                # Group symptoms by disease
                disease_symptom_map = defaultdict(list)
                for record in result:
                    disease_symptom_map[record['disease']].append({
                        'symptom': record['symptom'],
                        'weight': record['weight']
                    })

                if not disease_symptom_map:
                    logger.warning("No data found in Neo4j with the given weight threshold")
                    return [], []

                # Generate training samples based on symptom weights
                symptoms_data = []
                diseases_data = []

                np.random.seed(42)  # For reproducibility

                for disease, symptom_weights in disease_symptom_map.items():
                    # Generate multiple samples per disease
                    for _ in range(n_samples_per_disease):
                        case_symptoms = []

                        # Select symptoms based on their weights (higher weight = higher probability)
                        for symptom_info in symptom_weights:
                            # Use weight as probability, but cap at 0.9 to allow variation
                            inclusion_prob = min(0.9, symptom_info['weight'] * 1.5)
                            if np.random.random() < inclusion_prob:
                                case_symptoms.append(symptom_info['symptom'])

                        # Ensure each case has at least 2 symptoms
                        if len(case_symptoms) < 2 and symptom_weights:
                            # Add top symptoms to reach minimum
                            sorted_symptoms = sorted(symptom_weights, key=lambda x: x['weight'], reverse=True)
                            for symptom_info in sorted_symptoms:
                                if symptom_info['symptom'] not in case_symptoms:
                                    case_symptoms.append(symptom_info['symptom'])
                                    if len(case_symptoms) >= 2:
                                        break

                        # Add occasional noise symptoms for robustness (10% chance)
                        if np.random.random() < 0.1 and len(case_symptoms) > 0:
                            noise_symptoms = ['fatigue', 'general_malaise', 'mild_discomfort', 'anxiety']
                            noise_symptom = np.random.choice(noise_symptoms)
                            if noise_symptom not in case_symptoms:
                                case_symptoms.append(noise_symptom)

                        if case_symptoms:  # Only add valid cases
                            symptoms_data.append(case_symptoms)
                            diseases_data.append(disease)

                logger.info(f"Generated {len(symptoms_data)} training samples from Neo4j ontology")
                logger.info(f"Covering {len(set(diseases_data))} unique diseases")

                return symptoms_data, diseases_data

        except Exception as e:
            logger.error(f"Failed to extract training data from Neo4j: {e}")
            return [], []


    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


class ModelTrainer:
    """
    Complete training pipeline for the neuro-symbolic disease prediction model
    Handles data preparation, model training, validation, and evaluation
    """


    def __init__(self):
        self.predictor = None
        self.data_manager = None
        self.training_history = []
        self.validation_results = {}
        self.neo4j_config = {}


    def setup_environment(self):
        """Set up the training environment and check system requirements"""
        print("=" * 60)
        print("ENVIRONMENT SETUP")
        print("=" * 60)

        # Check system memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"System Memory:")
            print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
            print(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
            print(f"  Used RAM: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")

            # Memory warning
            if memory.available < 2 * 1024**3:  # Less than 2GB available
                print("     WARNING: Low available memory (<2GB). Consider:")
                print("     - Closing other applications")
                print("     - Using smaller batch sizes")
                print("     - Reducing samples_per_disease")
        except ImportError:
            print("psutil not available - unable to check memory")

        # Check PyTorch installation and hardware
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("Using CPU for training")

        # Create necessary directories
        directories = ["models", "data", "results"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            print(f"Directory created/verified: {directory}/")

        print("Environment setup complete")
        print()


    def initialize_neo4j_connection(self, config: Dict = None):
        """Initialize Neo4j connection with configuration"""
        print("=" * 60)
        print("NEO4J CONNECTION SETUP")
        print("=" * 60)

        # Default Neo4j configuration (memory-optimized)
        default_config = {
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'password',
            'min_weight_threshold': 0.5,  # Higher threshold = less data
            'samples_per_disease': 5      # Reduced samples for memory efficiency
        }

        # Auto-adjust based on available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < 3.0:
                print("  Auto-adjusting for low memory:")
                default_config['min_weight_threshold'] = 0.6
                default_config['samples_per_disease'] = 3
                print(f"     - Increased weight threshold to {default_config['min_weight_threshold']}")
                print(f"     - Reduced samples per disease to {default_config['samples_per_disease']}")
        except ImportError:
            pass

        if config:
            default_config.update(config)

        self.neo4j_config = default_config

        print("Neo4j Configuration:")
        for key, value in default_config.items():
            if 'password' in key:
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")

        # Initialize data manager with Neo4j connection
        self.data_manager = Neo4jDataManager(
            neo4j_uri=default_config['neo4j_uri'],
            neo4j_user=default_config['neo4j_user'],
            neo4j_password=default_config['neo4j_password']
        )

        if self.data_manager.driver:
            # Get database statistics
            stats = self.data_manager.get_database_statistics()
            if stats:
                print(f"\nDatabase Statistics:")
                print(f"  Diseases: {stats['diseases']:,}")
                print(f"  Symptoms: {stats['symptoms']:,}")
                print(f"  Relationships: {stats['relationships']:,}")

            print("Neo4j connection established successfully")
        else:
            print("Failed to establish Neo4j connection")
            print("Training will not be possible without Neo4j data")

        print()


    def initialize_model(self, model_config: Dict = None):
        """Initialize the disease prediction model"""
        print("=" * 60)
        print("MODEL INITIALIZATION")
        print("=" * 60)

        # Default model configuration
        default_config = {
            'uncertainty_threshold': 0.25,
            'confidence_threshold': 0.75,
            'validation_strategy': 'stratified_kfold'
        }

        if model_config:
            default_config.update(model_config)

        print("Model Configuration:")
        for key, value in default_config.items():
            print(f"  {key}: {value}")

        # Initialize the predictor model
        self.predictor = OptimizedDiseasePredictor(
            uncertainty_threshold=default_config['uncertainty_threshold'],
            confidence_threshold=default_config['confidence_threshold'],
            validation_strategy=default_config['validation_strategy']
        )

        print("Model initialized successfully")
        print()


    def prepare_training_data(self):
        """Prepare training data from Neo4j ontology"""
        print("=" * 60)
        print("DATA PREPARATION")
        print("=" * 60)

        if not self.data_manager or not self.data_manager.driver:
            raise RuntimeError("Neo4j connection not available. Cannot prepare training data.")

        print("Extracting training data from Neo4j ontology...")

        # Extract training data using configured parameters
        symptoms_data, diseases_data = self.data_manager.extract_training_data(
            min_weight=self.neo4j_config['min_weight_threshold'],
            n_samples_per_disease=self.neo4j_config['samples_per_disease']
        )

        if not symptoms_data or not diseases_data:
            raise RuntimeError("No training data extracted from Neo4j. Check database and weight threshold.")

        # Build vocabularies
        all_symptoms = set()
        for symptom_list in symptoms_data:
            all_symptoms.update(symptom_list)

        self.predictor.symptoms_vocab = sorted(list(all_symptoms))
        self.predictor.diseases_vocab = sorted(list(set(diseases_data)))

        # Filter data for minimum class representation (adjusted for low memory)
        disease_counts = Counter(diseases_data)
        min_samples = max(2, min(5, self.neo4j_config['samples_per_disease']))  # Adaptive minimum

        filtered_data = [(s, d) for s, d in zip(symptoms_data, diseases_data)
                        if disease_counts[d] >= min_samples]

        if len(filtered_data) < len(symptoms_data):
            removed_count = len(symptoms_data) - len(filtered_data)
            logger.warning(f"Filtered out {removed_count} samples due to insufficient class representation")

        if filtered_data:
            symptoms_data, diseases_data = zip(*filtered_data)
            symptoms_data, diseases_data = list(symptoms_data), list(diseases_data)
        else:
            raise RuntimeError("No valid training data after filtering")

        # Display data statistics
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(symptoms_data):,}")
        print(f"  Unique symptoms: {len(self.predictor.symptoms_vocab):,}")
        print(f"  Disease classes: {len(self.predictor.diseases_vocab):,}")
        print(f"  Average symptoms per case: {np.mean([len(s) for s in symptoms_data]):.1f}")

        # Show top disease classes by sample count
        print(f"\nTop 10 Disease Classes by Sample Count:")
        for disease, count in disease_counts.most_common(10):
            print(f"  {disease}: {count} samples ({count/len(diseases_data)*100:.1f}%)")

        print("Data preparation complete")
        print()

        return symptoms_data, diseases_data


    def load_knowledge_base(self):
        """Load medical knowledge base from Neo4j"""
        print("=" * 60)
        print("KNOWLEDGE BASE LOADING")
        print("=" * 60)

        print("Loading medical knowledge base from Neo4j ontology...")

        # Load knowledge base from Neo4j
        success = self.predictor.load_knowledge_base_from_neo4j(
            min_weight=self.neo4j_config['min_weight_threshold']
        )

        if not success:
            raise RuntimeError("Failed to load knowledge base from Neo4j")

        # Display knowledge base statistics
        kb = self.predictor.knowledge_graph
        print(f"\nKnowledge Base Statistics:")
        print(f"  Medical rules: {len(kb.symptom_disease_rules):,}")
        print(f"  Differential diagnoses: {len(kb.differential_diagnoses):,}")
        print(f"  Contraindications: {len(kb.contraindications):,}")

        # Show sample rules for verification
        print(f"\nSample Medical Rules:")
        rule_count = 0
        for symptoms, rules in kb.symptom_disease_rules.items():
            if rule_count >= 3:  # Show only first 3 rules
                break
            for rule in rules:
                print(f"  Rule {rule_count + 1}:")
                print(f"    Symptoms: {list(rule.symptoms)}")
                print(f"    Disease: {rule.disease}")
                print(f"    Confidence: {rule.confidence:.3f}")
                print(f"    Evidence: {rule.evidence_source}")
                print(f"    Sensitivity: {rule.sensitivity:.3f}, Specificity: {rule.specificity:.3f}")
                rule_count += 1
                break

        print("Knowledge base loaded successfully")
        print()


    def configure_training(self, training_config: Dict = None):
        """Configure training parameters"""
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)

        # Default training configuration (memory-optimized)
        default_config = {
            'epochs': 50,        # Reduced epochs
            'batch_size': 16,    # Smaller batch size for memory
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'early_stopping_patience': 10,  # Earlier stopping
            'cross_validation_folds': 3,    # Fewer folds
            'validation_split': 0.15        # Less validation data
        }

        # Further optimization for very low memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)

            if available_gb < 2.5:
                print("  Further memory optimization:")
                default_config['batch_size'] = 8
                default_config['epochs'] = 30
                default_config['cross_validation_folds'] = 2
                print(f"     - Reduced batch size to {default_config['batch_size']}")
                print(f"     - Reduced epochs to {default_config['epochs']}")
                print(f"     - Reduced CV folds to {default_config['cross_validation_folds']}")
        except ImportError:
            pass

        if training_config:
            default_config.update(training_config)

        self.training_config = default_config

        print("Training Configuration:")
        for key, value in self.training_config.items():
            print(f"  {key}: {value}")

        print("Training configuration set")
        print()


    def train_model(self, symptoms_data: List[List[str]], diseases_data: List[str]):
        """Train the neuro-symbolic model"""
        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        print("Starting model training...")
        print("This process may take several minutes depending on dataset size and hardware.")

        # Monitor memory before training
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Memory before training: {memory.available / (1024**3):.1f} GB available")
        except ImportError:
            pass
        print()

        try:
            # Train the model with configured parameters
            self.predictor.train_optimized_model(
                symptoms_data=symptoms_data,
                diseases_data=diseases_data,
                epochs=self.training_config['epochs'],
                batch_size=self.training_config['batch_size'],
                cv_folds=self.training_config['cross_validation_folds']
            )

            # Store validation results for analysis
            self.validation_results = self.predictor.validation_metrics

            print("Model training completed successfully")
            print()

        except Exception as e:
            print(f"Training failed with error: {e}")
            logger.error(f"Training failed: {e}")

            # Check if memory was the issue
            try:
                import psutil
                memory = psutil.virtual_memory()
                if memory.available < 1 * 1024**3:  # Less than 1GB
                    print("\nMEMORY OPTIMIZATION SUGGESTIONS:")
                    print("   - Close other applications to free memory")
                    print("   - Use smaller batch_size (try 4 or 8)")
                    print("   - Reduce samples_per_disease (try 2 or 3)")
                    print("   - Increase min_weight_threshold (try 0.7 or 0.8)")
            except ImportError:
                pass
            raise


    def evaluate_model(self):
        """Evaluate the trained model on test cases"""
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        # Test cases for model evaluation
        test_cases = [
            {
                'name': 'Cardiovascular Case',
                'symptoms': ['chest_pain', 'shortness_of_breath', 'diaphoresis', 'nausea']
            },
            {
                'name': 'Respiratory Case',
                'symptoms': ['cough', 'fever', 'shortness_of_breath', 'fatigue']
            },
            {
                'name': 'Neurological Case',
                'symptoms': ['headache', 'nausea', 'photophobia', 'dizziness']
            },
            {
                'name': 'Gastrointestinal Case',
                'symptoms': ['abdominal_pain', 'fever', 'nausea', 'vomiting']
            },
            {
                'name': 'Multi-symptom Case',
                'symptoms': ['fever', 'fatigue', 'cough', 'headache', 'muscle_aches']
            }
        ]

        print("Testing model predictions on sample cases:")
        print()

        for i, test_case in enumerate(test_cases, 1):
            print(f"Test Case {i}: {test_case['name']}")
            print(f"Input symptoms: {test_case['symptoms']}")

            try:
                results = self.predictor.predict_with_uncertainty(
                    test_case['symptoms'],
                    return_explanations=True
                )

                # Display top 3 predictions
                print("Top predictions:")
                for disease, pred_info in list(results['predictions'].items())[:3]:
                    print(f"  {pred_info['rank']}. {disease.replace('_', ' ').title()}")
                    print(f"     Probability: {pred_info['fused_probability']:.3f}")
                    print(f"     Uncertainty: {pred_info['uncertainty']:.3f}")
                    print(f"     Neural: {pred_info['neural_probability']:.3f}, Symbolic: {pred_info['symbolic_probability']:.3f}")

                # Display quality assessment
                quality = results['quality_indicators']
                print(f"Quality assessment:")
                print(f"  Reliability score: {quality['reliability_score']:.3f}")
                print(f"  Overall confidence: {results['confidence']:.3f}")
                print(f"  Reasoning mode: {results['explanations']['reasoning_mode']}")

                # Show warnings if any
                if quality['warnings']:
                    print(f"  Warnings: {'; '.join(quality['warnings'])}")

                print("-" * 50)

            except Exception as e:
                print(f"Error in prediction: {e}")
                print("-" * 50)

        print("Model evaluation completed")
        print()


    def save_model(self, model_path: str = "models/neo4j_disease_predictor.pkl"):
        """Save the trained model and training metadata"""
        print("=" * 60)
        print("MODEL SAVING")
        print("=" * 60)

        try:
            # Ensure models directory exists
            Path(model_path).parent.mkdir(exist_ok=True, parents=True)

            # Save the trained model
            self.predictor.save_model(model_path)

            # Calculate actual data statistics if available
            total_samples = 0
            if hasattr(self, 'symptoms_data') and hasattr(self, 'diseases_data'):
                total_samples = len(self.diseases_data)

            # Save training configuration and results
            training_metadata = {
                'neo4j_config': self.neo4j_config,
                'training_config': self.training_config,
                'validation_results': self.validation_results,
                'model_path': model_path,
                'data_statistics': {
                    'total_samples': total_samples,
                    'unique_symptoms': len(self.predictor.symptoms_vocab) if hasattr(self.predictor, 'symptoms_vocab') else 0,
                    'unique_diseases': len(self.predictor.diseases_vocab) if hasattr(self.predictor, 'diseases_vocab') else 0
                },
                'training_timestamp': pd.Timestamp.now().isoformat(),
                'pytorch_version': torch.__version__
            }

            # Save metadata as JSON
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(training_metadata, f, indent=2, default=str)

            print(f"Model saved successfully:")
            print(f"  Model file: {model_path}")
            print(f"  Metadata file: {metadata_path}")
            print(f"  Model size: {Path(model_path).stat().st_size / (1024*1024):.2f} MB")
            print()

        except Exception as e:
            print(f"Failed to save model: {e}")
            logger.error(f"Failed to save model: {e}")
            raise


    def generate_training_report(self):
        """Generate comprehensive training report"""
        print("=" * 60)
        print("TRAINING REPORT")
        print("=" * 60)

        print("NEURO-SYMBOLIC DISEASE PREDICTION MODEL")
        print("Training Summary Report")
        print("=" * 40)

        # Neo4j and data configuration
        print(f"\n1. DATA SOURCE CONFIGURATION:")
        print(f"   Neo4j URI: {self.neo4j_config.get('neo4j_uri', 'N/A')}")
        print(f"   Weight threshold: {self.neo4j_config.get('min_weight_threshold', 'N/A')}")
        print(f"   Samples per disease: {self.neo4j_config.get('samples_per_disease', 'N/A')}")

        # Training configuration
        print(f"\n2. TRAINING CONFIGURATION:")
        for key, value in self.training_config.items():
            print(f"   {key}: {value}")

        # Model performance metrics
        if self.validation_results:
            print(f"\n3. MODEL PERFORMANCE:")

            # Cross-validation results
            if any('_mean' in key for key in self.validation_results.keys()):
                print(f"\n   Cross-Validation Results (Mean +/- Std):")
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']

                for metric in metrics:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in self.validation_results:
                        mean_val = self.validation_results[mean_key]
                        std_val = self.validation_results.get(std_key, 0)
                        print(f"   {metric.upper()}: {mean_val:.4f} +/- {std_val:.4f}")

            # Final model performance
            if 'final_fused' in self.validation_results:
                print(f"\n   Final Model Performance:")
                final_metrics = self.validation_results['final_fused']
                print(f"   Accuracy: {final_metrics.accuracy:.4f}")
                print(f"   Precision: {final_metrics.precision:.4f}")
                print(f"   Recall (Sensitivity): {final_metrics.recall:.4f}")
                print(f"   F1-Score: {final_metrics.f1_score:.4f}")
                print(f"   Specificity: {final_metrics.specificity:.4f}")
                print(f"   PPV: {final_metrics.ppv:.4f}")
                print(f"   NPV: {final_metrics.npv:.4f}")

        # Next steps and recommendations
        print(f"\n4. NEXT STEPS:")
        print(f"   - Test the model on new cases using predict_with_uncertainty()")
        print(f"   - Validate model performance on real clinical data")
        print(f"   - Monitor model performance in production environment")
        print(f"   - Update Neo4j ontology with new medical evidence as available")
        print(f"   - Consider retraining with updated ontology or additional data")

        print(f"\n5. MODEL USAGE:")
        print(f"   - Load model: predictor = OptimizedDiseasePredictor()")
        print(f"   - Load trained model: predictor.load_model('path_to_model.pkl')")
        print(f"   - Make predictions: results = predictor.predict_with_uncertainty(symptoms)")

        print(f"\nTraining process completed successfully!")
        print(f"Neuro-symbolic disease prediction model is ready for deployment.")


    def cleanup(self):
        """Cleanup resources and connections"""
        if self.data_manager:
            self.data_manager.close()
        if self.predictor and hasattr(self.predictor, 'knowledge_graph'):
            self.predictor.knowledge_graph.close()


def train_with_neo4j_ontology(neo4j_config: Dict = None, training_config: Dict = None):
    """
    Complete training pipeline using Neo4j ontology data

    Args:
        neo4j_config: Neo4j connection and data extraction configuration
        training_config: Model training configuration
    """
    print("NEURO-SYMBOLIC DISEASE PREDICTION MODEL TRAINER")
    print("Training with Neo4j Ontology Data")
    print("=" * 60)

    trainer = ModelTrainer()

    try:
        # Step 1: Setup environment
        trainer.setup_environment()

        # Step 2: Initialize Neo4j connection
        trainer.initialize_neo4j_connection(neo4j_config)

        # Step 3: Initialize model
        trainer.initialize_model()

        # Step 4: Prepare training data from Neo4j
        symptoms_data, diseases_data = trainer.prepare_training_data()

        # Store data for later use in metadata
        trainer.symptoms_data = symptoms_data
        trainer.diseases_data = diseases_data

        # Step 5: Load knowledge base from Neo4j
        trainer.load_knowledge_base()

        # Step 6: Configure training parameters
        trainer.configure_training(training_config)

        # Step 7: Train the model
        trainer.train_model(symptoms_data, diseases_data)

        # Step 8: Evaluate model performance
        trainer.evaluate_model()

        # Step 9: Save trained model
        trainer.save_model()

        # Step 10: Generate comprehensive report
        trainer.generate_training_report()

    except Exception as e:
        print(f"Training failed: {e}")
        logger.error(f"Training pipeline failed: {e}")
        raise
    finally:
        # Always cleanup resources
        trainer.cleanup()


def load_and_test_model(model_path: str):
    """
    Load a trained model and test it on sample cases

    Args:
        model_path: Path to the saved model file
    """
    print(f"LOADING AND TESTING MODEL: {model_path}")
    print("=" * 60)

    if not Path(model_path).exists():
        print(f"Model file not found: {model_path}")
        return

    # Initialize predictor and load model
    predictor = OptimizedDiseasePredictor()
    predictor.load_model(model_path)

    # Test cases for evaluation
    test_symptoms = [
        ['chest_pain', 'shortness_of_breath', 'nausea', 'diaphoresis'],
        ['cough', 'fever', 'fatigue', 'shortness_of_breath'],
        ['headache', 'photophobia', 'nausea', 'neck_stiffness'],
        ['abdominal_pain', 'fever', 'vomiting', 'loss_of_appetite']
    ]

    print("Testing model on sample cases:")
    print()

    for i, symptoms in enumerate(test_symptoms, 1):
        print(f"Test Case {i}: {symptoms}")
        try:
            results = predictor.predict_with_uncertainty(symptoms, return_explanations=True)

            # Show top prediction
            top_disease = list(results['predictions'].keys())[0]
            top_info = results['predictions'][top_disease]

            print(f"Top Prediction: {top_disease.replace('_', ' ').title()}")
            print(f"  Probability: {top_info['fused_probability']:.3f}")
            print(f"  Confidence: {results['confidence']:.3f}")
            print(f"  Reliability: {results['quality_indicators']['reliability_score']:.3f}")

            print("-" * 40)

        except Exception as e:
            print(f"Error in prediction: {e}")
            print("-" * 40)

    # Cleanup
    predictor.knowledge_graph.close()


def main():
    """Main function to run training based on user choice"""
    print("NEURO-SYMBOLIC DISEASE PREDICTION MODEL TRAINER")
    print("=" * 60)
    print("Training Options:")
    print("1. Train with Neo4j ontology data (Default)")
    print("2. Train with custom Neo4j configuration")
    print("3. Load and test existing model")
    print()

    choice = input("Enter your choice (1-3, default=1): ").strip() or "1"

    if choice == "1":
        # Default training with Neo4j ontology
        train_with_neo4j_ontology()

    elif choice == "2":
        # Custom Neo4j configuration
        print("Enter custom Neo4j configuration:")
        neo4j_uri = input("Neo4j URI (default: bolt://localhost:7687): ").strip() or "bolt://localhost:7687"
        neo4j_user = input("Neo4j username (default: neo4j): ").strip() or "neo4j"
        neo4j_password = input("Neo4j password (default: password): ").strip() or "password"
        min_weight = float(input("Minimum weight threshold (default: 0.3): ").strip() or "0.3")
        samples_per_disease = int(input("Samples per disease (default: 10): ").strip() or "10")

        neo4j_config = {
            'neo4j_uri': neo4j_uri,
            'neo4j_user': neo4j_user,
            'neo4j_password': neo4j_password,
            'min_weight_threshold': min_weight,
            'samples_per_disease': samples_per_disease
        }

        train_with_neo4j_ontology(neo4j_config=neo4j_config)

    elif choice == "3":
        # Load and test existing model
        model_path = input("Enter path to model file: ").strip()
        if model_path:
            load_and_test_model(model_path)
        else:
            print("No model path provided")

    else:
        print("Invalid choice. Running default training...")
        train_with_neo4j_ontology()

if __name__ == "__main__":
    main()
