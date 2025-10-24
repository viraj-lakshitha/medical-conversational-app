import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import re
from neo4j import GraphDatabase
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .utils import make_symptom_id, normalise_symptom_text
except ImportError:
    from utils import make_symptom_id, normalise_symptom_text


class DiseaseSymptomKnowledgeGraphBuilder:


    def __init__(self, df, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        self.df = df.copy()
        self.disease_symptom_pairs = []
        self.weights = {}
        self.final_weights_df = None

        # Neo4j connection configuration with safe fallbacks
        self.neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        self.neo4j_user = neo4j_user or "neo4j"
        self.neo4j_password = neo4j_password or "password"
        self.driver = None

        # Attempt connection immediately; failures fall back to offline mode
        self.connect_to_neo4j()


    def connect_to_neo4j(self):
        """Establish connection to Neo4j database"""
        try:
            # More robust connection with additional configuration
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=60  # 60 seconds
            )

            # Test connection with more detailed error handling
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info(f"Successfully connected to Neo4j at {self.neo4j_uri}")
                    return True

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.error(f"URI: {self.neo4j_uri}")
            logger.error(f"User: {self.neo4j_user}")
            logger.error("Please check your Neo4j credentials and database status")
            self.driver = None
            return False


    def preprocess_data(self):
        """Extract and clean disease-symptom pairs from the dataset"""
        pairs = []

        for _, row in self.df.iterrows():
            disease_id = row['disease_id']
            disease = normalise_symptom_text(row['disease'])
            symptoms_text = row['common_symptom']

            # Clean and split symptoms
            if pd.notna(symptoms_text):
                # Split by comma and clean each symptom
                symptoms = [s.strip() for s in symptoms_text.split(',')]
                symptoms = [s for s in symptoms if s]  # Remove empty strings

                for symptom in symptoms:
                    symptom_clean = self._normalize_symptom(symptom)

                    if symptom_clean and len(symptom_clean) > 1:  # Ensure meaningful symptoms
                        pairs.append({
                            'disease_id': disease_id,
                            'disease': disease,
                            'symptom': symptom_clean
                        })

        # Remove duplicate disease-symptom pairs
        self.disease_symptom_pairs = pd.DataFrame(pairs).drop_duplicates()

        logger.info(f"Extracted {len(self.disease_symptom_pairs)} unique disease-symptom pairs")
        logger.info(f"Unique diseases: {self.disease_symptom_pairs['disease'].nunique()}")
        logger.info(f"Unique symptoms: {self.disease_symptom_pairs['symptom'].nunique()}")

        return self.disease_symptom_pairs


    def _normalize_symptom(self, symptom):
        """Normalize symptom text for consistency"""
        symptom = normalise_symptom_text(symptom)

        # Remove common medical articles and conjunctions
        stop_words = ['and', 'or', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with']
        words = symptom.split()
        words = [word for word in words if word not in stop_words or len(words) <= 2]
        symptom = ' '.join(words)

        # Standardize common medical terms
        replacements = {
            'high temperature': 'fever',
            'elevated temperature': 'fever',
            'high fever': 'fever',
            'low-grade fever': 'low grade fever',
            'mild fever': 'low grade fever',
            'severe headache': 'headache',
            'mild headache': 'headache',
            'stomach pain': 'abdominal pain',
            'belly pain': 'abdominal pain',
            'shortness of breath': 'dyspnea',
            'difficulty breathing': 'dyspnea',
            'trouble breathing': 'dyspnea',
            'weight loss': 'weight loss',
            'loss of weight': 'weight loss',
            'decreased appetite': 'loss of appetite',
            'poor appetite': 'loss of appetite',
            'reduced appetite': 'loss of appetite'
        }

        for old_term, new_term in replacements.items():
            if old_term in symptom:
                symptom = symptom.replace(old_term, new_term)

        # Remove extra whitespace
        symptom = re.sub(r'\s+', ' ', symptom).strip()

        return symptom


    def calculate_tfidf_weights(self):
        """Calculate TF-IDF inspired weights"""
        weights = {}

        # Calculate term frequency (symptom frequency in disease)
        disease_symptom_counts = self.disease_symptom_pairs.groupby(['disease_id', 'disease', 'symptom']).size().reset_index(name='tf')
        disease_totals = self.disease_symptom_pairs.groupby(['disease_id', 'disease']).size().reset_index(name='total_symptoms')

        # Calculate inverse document frequency (how rare is the symptom across diseases)
        symptom_disease_counts = self.disease_symptom_pairs.groupby('symptom')['disease_id'].nunique().reset_index(name='diseases_with_symptom')
        total_diseases = self.disease_symptom_pairs['disease_id'].nunique()
        symptom_disease_counts['idf'] = np.log(total_diseases / symptom_disease_counts['diseases_with_symptom'])

        # Merge and calculate TF-IDF
        merged = disease_symptom_counts.merge(disease_totals, on=['disease_id', 'disease'])
        merged['tf_normalized'] = merged['tf'] / merged['total_symptoms']
        merged = merged.merge(symptom_disease_counts, on='symptom')
        merged['tfidf_weight'] = merged['tf_normalized'] * merged['idf']

        for _, row in merged.iterrows():
            key = (row['disease_id'], row['symptom'])
            weights[key] = {
                'tfidf_weight': row['tfidf_weight'],
                'tf': row['tf_normalized'],
                'idf': row['idf'],
                'disease': row['disease']
            }

        return weights


    def calculate_pmi_weights(self):
        """Calculate Pointwise Mutual Information weights"""
        weights = {}

        total_pairs = len(self.disease_symptom_pairs)

        # Calculate joint probability P(disease, symptom)
        joint_counts = self.disease_symptom_pairs.groupby(['disease_id', 'symptom']).size().reset_index(name='joint_count')
        joint_counts['p_joint'] = joint_counts['joint_count'] / total_pairs

        # Calculate marginal probabilities
        disease_counts = self.disease_symptom_pairs.groupby('disease_id').size().reset_index(name='disease_count')
        disease_counts['p_disease'] = disease_counts['disease_count'] / total_pairs

        symptom_counts = self.disease_symptom_pairs.groupby('symptom').size().reset_index(name='symptom_count')
        symptom_counts['p_symptom'] = symptom_counts['symptom_count'] / total_pairs

        # Merge and calculate PMI
        merged = joint_counts.merge(disease_counts, on='disease_id')
        merged = merged.merge(symptom_counts, on='symptom')
        merged['pmi'] = np.log(merged['p_joint'] / (merged['p_disease'] * merged['p_symptom']))
        merged['pmi_positive'] = np.maximum(0, merged['pmi'])  # Positive PMI

        # Add disease names
        disease_names = self.df.set_index('disease_id')['disease'].to_dict()
        merged['disease'] = merged['disease_id'].map(disease_names)

        for _, row in merged.iterrows():
            key = (row['disease_id'], row['symptom'])
            weights[key] = {
                'pmi_weight': row['pmi_positive'],
                'pmi_raw': row['pmi'],
                'disease': row['disease']
            }

        return weights


    def calculate_optimal_weights(self):
        """Calculate TF-IDF + PMI confidence combination (recommended approach)"""
        print("Preprocessing data...")
        self.preprocess_data()

        print("Calculating TF-IDF weights...")
        tfidf_weights = self.calculate_tfidf_weights()

        print("Calculating PMI weights...")
        pmi_weights = self.calculate_pmi_weights()

        # Combine weights optimally
        combined_weights = {}
        all_keys = set(tfidf_weights.keys()) | set(pmi_weights.keys())

        # Calculate normalization factors
        all_tfidf = [tfidf_weights.get(key, {}).get('tfidf_weight', 0) for key in all_keys]
        tfidf_max = max(all_tfidf) if all_tfidf else 1

        for key in all_keys:
            tfidf_val = tfidf_weights.get(key, {}).get('tfidf_weight', 0)
            pmi_val = pmi_weights.get(key, {}).get('pmi_weight', 0)

            # Normalize TF-IDF to 0-1 range
            tfidf_normalized = tfidf_val / tfidf_max

            # Convert PMI to confidence score (0-1 range)
            pmi_confidence = 1 - np.exp(-pmi_val) if pmi_val > 0 else 0

            # Optimal combination: 70% TF-IDF + 30% PMI confidence
            final_weight = 0.7 * tfidf_normalized + 0.3 * pmi_confidence

            combined_weights[key] = {
                'disease_id': key[0],
                'symptom': key[1],
                'disease': tfidf_weights.get(key, {}).get('disease', '') or pmi_weights.get(key, {}).get('disease', ''),
                'tfidf_weight': tfidf_val,
                'tfidf_normalized': tfidf_normalized,
                'pmi_weight': pmi_val,
                'pmi_confidence': pmi_confidence,
                'final_weight': final_weight,
                'weight_category': self._categorize_weight(final_weight)
            }

        return combined_weights


    def _categorize_weight(self, weight):
        """Categorize weight strength for easier interpretation"""
        if weight < 0.3:
            return "Weak"
        elif weight < 0.6:
            return "Moderate"
        elif weight < 0.8:
            return "Strong"
        else:
            return "Very Strong"


    def export_weights_to_dataframe(self, weights):
        """Convert weights dictionary to DataFrame for easy analysis"""
        rows = []
        for key, values in weights.items():
            rows.append(values)
        return pd.DataFrame(rows)


    def create_knowledge_graph_schema(self):
        """Create Neo4j schema for the knowledge graph"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False

        schema_queries = [
            # Create constraints
            "CREATE CONSTRAINT disease_id_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT symptom_id_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT symptom_name_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",

            # Create indexes for performance
            "CREATE INDEX disease_name_index IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            "CREATE INDEX symptom_name_index IF NOT EXISTS FOR (s:Symptom) ON (s.name)",
        ]

        try:
            with self.driver.session() as session:
                for query in schema_queries:
                    session.run(query)
            logger.info("Knowledge graph schema created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create schema: {e}")
            return False


    def populate_knowledge_graph(self, min_weight_threshold=0.2):
        """Populate Neo4j with disease-symptom relationships"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return False

        if self.final_weights_df is None:
            logger.error("No weights calculated. Run calculate_optimal_weights first.")
            return False

        # Filter weights above threshold
        filtered_df = self.final_weights_df[self.final_weights_df['final_weight'] >= min_weight_threshold].copy()
        if filtered_df.empty:
            logger.warning("Filtered dataset is empty. No relationships will be created.")
            return True

        filtered_df['symptom_id'] = filtered_df['symptom'].apply(make_symptom_id)
        filtered_df['symptom_display'] = filtered_df['symptom'].str.title()
        filtered_df['disease_display'] = filtered_df['disease'].str.title()

        logger.info(
            "Populating knowledge graph with %s relationships (threshold=%.2f)",
            len(filtered_df),
            min_weight_threshold,
        )

        rows = filtered_df[
            [
                'disease_id',
                'disease',
                'disease_display',
                'symptom',
                'symptom_id',
                'symptom_display',
                'final_weight',
                'tfidf_weight',
                'pmi_confidence',
                'weight_category',
            ]
        ].to_dict('records')

        try:
            with self.driver.session() as session:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (d:Disease {id: row.disease_id})
                      ON CREATE SET d.name = row.disease,
                                    d.name_display = row.disease_display
                    MERGE (s:Symptom {id: row.symptom_id})
                      ON CREATE SET s.name = row.symptom,
                                    s.name_display = row.symptom_display
                    MERGE (d)-[r:HAS_SYMPTOM]->(s)
                      SET r.weight = row.final_weight,
                          r.tfidf_weight = row.tfidf_weight,
                          r.pmi_confidence = row.pmi_confidence,
                          r.category = row.weight_category,
                          r.updated_at = datetime()
                    MERGE (s)-[r2:INDICATES]->(d)
                      SET r2.weight = row.final_weight,
                          r2.tfidf_weight = row.tfidf_weight,
                          r2.pmi_confidence = row.pmi_confidence,
                          r2.category = row.weight_category,
                          r2.updated_at = datetime()
                    """,
                    rows=rows,
                )

            logger.info("Knowledge graph populated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to populate knowledge graph: {e}")
            return False


    def query_diseases_by_symptoms(self, symptoms, min_weight=0.3, top_n=10):
        """Query probable diseases given a list of symptoms"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return []

        # Normalize input symptoms to lowercase
        normalized_symptoms = [normalise_symptom_text(symptom) for symptom in symptoms]
        symptom_ids = [make_symptom_id(symptom) for symptom in normalized_symptoms]

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s:Symptom)-[r:INDICATES]->(d:Disease)
                    WHERE (s.name IN $symptoms OR s.id IN $symptom_ids) AND r.weight >= $min_weight
                    WITH d, collect(r.weight) as weights, collect(s.name) as matched_symptoms
                    RETURN d.id as disease_id, d.name_display as disease_name,
                           matched_symptoms,
                           reduce(total = 0, weight IN weights | total + weight) as total_score,
                           size(matched_symptoms) as symptom_count
                    ORDER BY total_score DESC, symptom_count DESC
                    LIMIT $top_n
                """, symptoms=normalized_symptoms, symptom_ids=symptom_ids, min_weight=min_weight, top_n=top_n)

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to query diseases: {e}")
            return []


    def query_symptoms_by_disease(self, disease_id, min_weight=0.3):
        """Query symptoms for a specific disease"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Disease {id: $disease_id})-[r:HAS_SYMPTOM]->(s:Symptom)
                    WHERE r.weight >= $min_weight
                    RETURN s.name_display as symptom, r.weight as weight, r.category as category
                    ORDER BY r.weight DESC
                """, disease_id=disease_id, min_weight=min_weight)

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to query symptoms: {e}")
            return []


    def search_diseases_by_name(self, disease_name_pattern):
        """Search diseases by name pattern (case-insensitive)"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Disease)
                    WHERE d.name CONTAINS $pattern OR d.name_display CONTAINS $pattern
                    RETURN d.id as disease_id, d.name_display as disease_name
                    ORDER BY d.name_display
                """, pattern=disease_name_pattern.lower())

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to search diseases: {e}")
            return []


    def search_symptoms_by_name(self, symptom_name_pattern):
        """Search symptoms by name pattern (case-insensitive)"""
        if not self.driver:
            logger.error("No Neo4j connection available")
            return []

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s:Symptom)
                    WHERE s.name CONTAINS $pattern OR s.name_display CONTAINS $pattern
                    RETURN DISTINCT s.name_display as symptom
                    ORDER BY s.name_display
                """, pattern=symptom_name_pattern.lower())

                return [record["symptom"] for record in result]

        except Exception as e:
            logger.error(f"Failed to search symptoms: {e}")
            return []


    def build_complete_knowledge_graph(self, min_weight_threshold=0.2):
        """Complete pipeline: calculate weights and build knowledge graph"""
        logger.info("Starting complete knowledge graph build process...")

        # Calculate optimal weights
        weights = self.calculate_optimal_weights()
        self.final_weights_df = self.export_weights_to_dataframe(weights)

        # Save weights for analysis
        self.final_weights_df.to_csv('optimized_disease_symptom_weights.csv', index=False)
        logger.info("Weights saved to 'optimized_disease_symptom_weights.csv'")

        # Create knowledge graph
        if self.driver:
            # Clear existing nodes to avoid constraint conflicts with legacy data
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")

            self.create_knowledge_graph_schema()
            self.populate_knowledge_graph(min_weight_threshold)

            # Print statistics
            self._print_graph_statistics()

        return self.final_weights_df


    def _print_graph_statistics(self):
        """Print knowledge graph statistics"""
        if not self.driver:
            return

        try:
            with self.driver.session() as session:
                # Count nodes and relationships
                disease_count = session.run("MATCH (d:Disease) RETURN count(d) as count").single()['count']
                symptom_count = session.run("MATCH (s:Symptom) RETURN count(s) as count").single()['count']
                relationship_count = session.run("MATCH ()-[r:HAS_SYMPTOM]->() RETURN count(r) as count").single()['count']

                logger.info(f"Knowledge Graph Statistics:")
                logger.info(f"  Diseases: {disease_count}")
                logger.info(f"  Symptoms: {symptom_count}")
                logger.info(f"  Relationships: {relationship_count}")

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")


    def test_neo4j_connection(self):
        """Test and diagnose Neo4j connection issues"""
        print("=== Neo4j Connection Diagnostics ===")
        print(f"URI: {self.neo4j_uri}")
        print(f"Username: {self.neo4j_user}")
        print(f"Password: {'*' * len(self.neo4j_password) if self.neo4j_password else 'None'}")

        # Test different connection scenarios
        test_configs = [
            {"uri": self.neo4j_uri, "auth": (self.neo4j_user, self.neo4j_password)},
            {"uri": "bolt://localhost:7687", "auth": ("neo4j", "password")},  # common defaults
            {"uri": "neo4j://localhost:7687", "auth": (self.neo4j_user, self.neo4j_password)},  # alternative protocol
        ]

        for i, config in enumerate(test_configs):
            print(f"\nTesting configuration {i+1}:")
            print(f"  URI: {config['uri']}")
            print(f"  User: {config['auth'][0]}")

            try:
                test_driver = GraphDatabase.driver(config["uri"], auth=config["auth"])
                with test_driver.session() as session:
                    result = session.run("RETURN 'Connection successful' as message")
                    message = result.single()["message"]
                    print(f"  SUCCESS: {message}")
                    test_driver.close()
                    return config
            except Exception as e:
                print(f"  FAILED: {e}")

        print("\n=== Troubleshooting Tips ===")
        print("1. Check if Neo4j is running: Open Neo4j Desktop and start your database")
        print("2. Verify credentials: Use the same username/password from Neo4j Desktop")
        print("3. Check port: Default is 7687 for bolt, 7474 for HTTP")
        print("4. Try connecting via Neo4j Browser first to verify credentials")
        return None


    def close(self):
        """Close the Neo4j driver if open."""
        if self.driver:
            self.driver.close()
            self.driver = None

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('disease_database.csv')

    # Option 1: Manual connection (replace with your credentials)
    kg_builder = DiseaseSymptomKnowledgeGraphBuilder(
        df,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )

    # Option 2: Interactive setup (uncomment to use)
    # uri, user, password = setup_neo4j_connection()
    # kg_builder = DiseaseSymptomKnowledgeGraphBuilder(df, uri, user, password)

    # Test connection before proceeding
    if not kg_builder.driver:
        print("Testing connection configurations...")
        working_config = kg_builder.test_neo4j_connection()

        if working_config:
            print(f"\nFound working configuration! Reconnecting...")
            kg_builder.neo4j_uri = working_config["uri"]
            kg_builder.neo4j_user = working_config["auth"][0]
            kg_builder.neo4j_password = working_config["auth"][1]
            kg_builder.connect_to_neo4j()
        else:
            print("\nCould not establish Neo4j connection.")
            print("The system will still calculate weights and save them to CSV.")
            print("You can import the CSV to Neo4j later using:")
            print("  LOAD CSV WITH HEADERS FROM 'file:///optimized_disease_symptom_weights.csv' AS row")

    # Build the complete knowledge graph (works with or without Neo4j)
    weights_df = kg_builder.build_complete_knowledge_graph(min_weight_threshold=0.25)

    # Example queries (only if Neo4j is connected)
    if kg_builder.driver:
        print("\n=== Testing Knowledge Graph Queries ===")

        # Test with lowercase symptoms (as they should be stored)
        test_symptoms = ["fever", "headache", "fatigue"]
        print(f"Testing with normalized symptoms: {test_symptoms}")

        # Query diseases by symptoms
        probable_diseases = kg_builder.query_diseases_by_symptoms(
            symptoms=test_symptoms,
            min_weight=0.3,
            top_n=5
        )
        print("Probable diseases for symptoms [fever, headache, fatigue]:")
        for disease in probable_diseases:
            print(f"  {disease['disease_name']}: Score={disease['total_score']:.2f}")
            print(f"    Matched symptoms: {disease['matched_symptoms']}")

        # Query symptoms for a specific disease
        disease_symptoms = kg_builder.query_symptoms_by_disease('1656164150939770881', min_weight=0.3)
        print("\nTop symptoms for disease ID 1656164150939770881:")
        for symptom in disease_symptoms:
            print(f"  {symptom['symptom']}: Weight={symptom['weight']:.3f} ({symptom['category']})")

        # Test search functionality
        print("\n=== Testing Search Functions ===")

        # Search for diseases containing "syndrome"
        syndrome_diseases = kg_builder.search_diseases_by_name("syndrome")
        print("Diseases containing 'syndrome':")
        for disease in syndrome_diseases[:5]:  # Show first 5
            print(f"  {disease['disease_name']}")

        # Search for symptoms containing "pain"
        pain_symptoms = kg_builder.search_symptoms_by_name("pain")
        print("Symptoms containing 'pain':")
        for symptom in pain_symptoms[:5]:  # Show first 5
            print(f"  {symptom}")

        # Show sample of processed data
        print("\n=== Sample Processed Data ===")
        sample_data = kg_builder.final_weights_df.head(10)
        print("First 10 disease-symptom pairs:")
        for _, row in sample_data.iterrows():
            print(f"  {row['disease']} -> {row['symptom']} (weight: {row['final_weight']:.3f})")
    else:
        print("\n=== Neo4j not connected ===")
        print("Weights have been calculated and saved to 'optimized_disease_symptom_weights.csv'")
        print("Sample of processed data:")
        if kg_builder.final_weights_df is not None:
            sample_data = kg_builder.final_weights_df.head(10)
            for _, row in sample_data.iterrows():
                print(f"  {row['disease']} -> {row['symptom']} (weight: {row['final_weight']:.3f})")
        print("\nYou can import this data to Neo4j later or use it with other graph databases.")

    # Close connection
    kg_builder.close()

    print("\nKnowledge graph construction completed!")
