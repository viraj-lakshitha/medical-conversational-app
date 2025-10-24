"""
Medical Knowledge Base Interface

Provides a unified interface for querying the Neo4j medical knowledge graph
for disease-symptom relationships and medical reasoning.
"""

import logging
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

try:
    from .utils import make_symptom_id, normalise_symptom_text
except ImportError:
    from utils import make_symptom_id, normalise_symptom_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalKnowledgeBase:
    """
    Interface for querying the medical knowledge graph stored in Neo4j.

    Provides methods for:
    - Disease-symptom relationship queries
    - Symptom-based disease prediction
    - Medical entity validation
    - Knowledge graph statistics
    """


    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        """
        Initialize connection to Neo4j medical knowledge base

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()


    def _connect(self):
        """Establish connection to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=60  # 60 seconds
            )

            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info(f"Successfully connected to Neo4j at {self.uri}")
                    return True

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            return False


    def get_related_diseases(self, symptoms: List[str], min_weight: float = 0.3, top_n: int = 10) -> Dict[str, Any]:
        """
        Get diseases related to given symptoms with evidence weights

        Args:
            symptoms: List of symptom names
            min_weight: Minimum relationship weight threshold
            top_n: Maximum number of diseases to return

        Returns:
            Dictionary containing disease predictions and evidence
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return {}

        normalized_symptoms = [normalise_symptom_text(sym) for sym in symptoms]
        symptom_ids = [make_symptom_id(symptom) for symptom in normalized_symptoms]

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s:Symptom)-[r:INDICATES]->(d:Disease)
                    WHERE (s.name IN $symptoms OR s.id IN $symptom_ids) AND r.weight >= $min_weight
                    WITH d, collect(s.name) as matched_symptoms, collect(r.weight) as matched_weights
                    OPTIONAL MATCH (d)-[r2:HAS_SYMPTOM]->(s2:Symptom)
                    WHERE NOT s2.name IN $symptoms AND r2.weight >= $min_weight
                    WITH
                      d.id as disease_id,
                      d.name_display as disease_name,
                      matched_symptoms,
                      reduce(score = 0, w IN matched_weights | score + w) as total_score,
                      size(matched_symptoms) as matched_symptom_count,
                      collect({symptom: s2.name_display, weight: r2.weight}) as additional_symptoms
                    RETURN
                      disease_id,
                      disease_name,
                      matched_symptoms,
                      total_score,
                      matched_symptom_count,
                      additional_symptoms
                    ORDER BY total_score DESC, matched_symptom_count DESC
                    LIMIT $top_n
                """, symptoms=normalized_symptoms, symptom_ids=symptom_ids, min_weight=min_weight, top_n=top_n)

                diseases = []
                for record in result:
                    disease_data = dict(record)
                    # Round scores for consistency
                    disease_data['total_score'] = round(disease_data['total_score'], 4)
                    for symptom in disease_data['additional_symptoms']:
                        symptom['weight'] = round(symptom['weight'], 4)
                    diseases.append(disease_data)

                return {
                    "diseases": diseases,
                    "input_symptoms": symptoms,
                    "matched_symptoms": normalized_symptoms,
                    "query_parameters": {
                        "min_weight": min_weight,
                        "top_n": top_n
                    }
                }

        except Exception as e:
            logger.error(f"Failed to query related diseases: {e}")
            return {}


    def validate_medical_entity(self, entity_name: str, entity_type: str = None) -> Dict[str, Any]:
        """
        Validate if an entity exists in the medical knowledge base

        Args:
            entity_name: Name of the entity to validate
            entity_type: Type of entity ('Disease' or 'Symptom')

        Returns:
            Dictionary with validation results
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return {"valid": False, "error": "No database connection"}

        try:
            with self.driver.session() as session:
                # Search in both Disease and Symptom nodes if type not specified
                if entity_type:
                    query = f"""
                        MATCH (n:{entity_type})
                        WHERE toLower(n.name) = toLower($entity_name)
                           OR toLower(n.name_display) = toLower($entity_name)
                        RETURN n.id as id, n.name as name, n.name_display as display_name, labels(n) as type
                        LIMIT 1
                    """
                else:
                    query = """
                        MATCH (n)
                        WHERE (n:Disease OR n:Symptom)
                          AND (toLower(n.name) = toLower($entity_name)
                               OR toLower(n.name_display) = toLower($entity_name))
                        RETURN n.id as id, n.name as name, n.name_display as display_name, labels(n) as type
                        LIMIT 1
                    """

                result = session.run(query, entity_name=entity_name)
                record = result.single()

                if record:
                    return {
                        "valid": True,
                        "entity": {
                            "id": record["id"],
                            "name": record["name"],
                            "display_name": record["display_name"],
                            "type": record["type"][0]  # First label
                        }
                    }
                else:
                    return {"valid": False, "message": "Entity not found in knowledge base"}

        except Exception as e:
            logger.error(f"Failed to validate entity: {e}")
            return {"valid": False, "error": str(e)}


    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the medical knowledge base

        Returns:
            Dictionary with database statistics
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return {}

        try:
            with self.driver.session() as session:
                # Count nodes and relationships
                stats_query = """
                    MATCH (d:Disease) WITH count(d) as disease_count
                    MATCH (s:Symptom) WITH disease_count, count(s) as symptom_count
                    MATCH ()-[r:INDICATES]->() WITH disease_count, symptom_count, count(r) as indicates_count
                    MATCH ()-[r2:HAS_SYMPTOM]->() WITH disease_count, symptom_count, indicates_count, count(r2) as has_symptom_count
                    RETURN disease_count, symptom_count, indicates_count, has_symptom_count
                """

                result = session.run(stats_query)
                record = result.single()

                if record:
                    return {
                        "total_diseases": record["disease_count"],
                        "total_symptoms": record["symptom_count"],
                        "indicates_relationships": record["indicates_count"],
                        "has_symptom_relationships": record["has_symptom_count"],
                        "connection_status": "connected",
                        "database_uri": self.uri
                    }
                else:
                    return {"error": "Failed to retrieve statistics"}

        except Exception as e:
            logger.error(f"Failed to get database statistics: {e}")
            return {"error": str(e), "connection_status": "error"}


    def search_symptoms_by_disease(self, disease_name: str, min_weight: float = 0.3) -> Dict[str, Any]:
        """
        Get all symptoms associated with a specific disease

        Args:
            disease_name: Name of the disease
            min_weight: Minimum relationship weight threshold

        Returns:
            Dictionary with disease symptoms and metadata
        """
        if not self.driver:
            logger.error("No Neo4j connection available")
            return {}

        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
                    WHERE toLower(d.name_display) = toLower($disease_name)
                      AND r.weight >= $min_weight
                    RETURN
                      d.id as disease_id,
                      d.name_display as disease_name,
                      collect({
                        symptom: s.name_display,
                        weight: r.weight,
                        category: r.category
                      }) as symptoms
                    ORDER BY r.weight DESC
                """, disease_name=disease_name, min_weight=min_weight)

                record = result.single()
                if record:
                    symptoms = record["symptoms"]
                    # Round weights
                    for symptom in symptoms:
                        symptom['weight'] = round(symptom['weight'], 4)

                    return {
                        "disease_id": record["disease_id"],
                        "disease_name": record["disease_name"],
                        "symptoms": symptoms,
                        "symptom_count": len(symptoms),
                        "min_weight_threshold": min_weight
                    }
                else:
                    return {"error": f"Disease '{disease_name}' not found"}

        except Exception as e:
            logger.error(f"Failed to search symptoms for disease: {e}")
            return {"error": str(e)}


    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


    def __enter__(self):
        """Context manager entry"""
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
