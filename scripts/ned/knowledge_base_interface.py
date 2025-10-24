"""
Knowledge Base Interface for candidate concept generation.
This component generates candidate concepts from the Neo4j medical knowledge base.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set
from neo4j import GraphDatabase
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass


class CandidateConcept:
    concept_id: str
    name: str
    display_name: str
    concept_type: str  # 'DISEASE' or 'SYMPTOM'
    similarity_score: float
    aliases: List[str] = None
    definition: str = ""


    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class KnowledgeBaseInterface:


    def __init__(self, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        """Initialize Neo4j connection and load medical entities"""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None

        # Entity caches for fast lookup
        self.disease_entities = {}
        self.symptom_entities = {}
        self.disease_aliases = {}
        self.symptom_aliases = {}

        self._connect_to_neo4j()
        self._load_entities()


    def _connect_to_neo4j(self):
        """Establish Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("MATCH (n) RETURN count(n) LIMIT 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise


    def _load_entities(self):
        """Load all diseases and symptoms from Neo4j into memory for fast lookup"""
        logger.info("Loading medical entities from Neo4j...")

        with self.driver.session() as session:
            # Load diseases
            disease_query = """
            MATCH (d:Disease)
            RETURN d.id as id, d.name as name, d.display_name as display_name
            """

            diseases = session.run(disease_query)
            for record in diseases:
                disease_id = str(record["id"])
                name = record["name"].lower()
                display_name = record["display_name"]

                self.disease_entities[name] = {
                    'id': disease_id,
                    'name': name,
                    'display_name': display_name
                }

                # Create aliases for partial matching
                self._create_aliases(name, disease_id, self.disease_aliases)

            # Load symptoms
            symptom_query = """
            MATCH (s:Symptom)
            RETURN s.name as name, s.display_name as display_name
            """

            symptoms = session.run(symptom_query)
            for record in symptoms:
                name = record["name"].lower()
                display_name = record["display_name"]

                self.symptom_entities[name] = {
                    'id': name,  # Symptoms use name as ID
                    'name': name,
                    'display_name': display_name
                }

                # Create aliases for partial matching
                self._create_aliases(name, name, self.symptom_aliases)

        logger.info(f"Loaded {len(self.disease_entities)} diseases and {len(self.symptom_entities)} symptoms")


    def _create_aliases(self, entity_name: str, entity_id: str, alias_dict: Dict):
        """Create aliases for better matching"""
        # Original name
        alias_dict[entity_name] = entity_id

        # Remove common medical terms
        cleaned_name = re.sub(r'\b(syndrome|disease|disorder|condition|symptom)\b', '', entity_name)
        cleaned_name = cleaned_name.strip()
        if cleaned_name and cleaned_name != entity_name:
            alias_dict[cleaned_name] = entity_id

        # Split compound terms
        if ' ' in entity_name:
            words = entity_name.split()
            if len(words) >= 2:
                # First word
                alias_dict[words[0]] = entity_id
                # Last word
                alias_dict[words[-1]] = entity_id

        # Remove parentheses content
        paren_removed = re.sub(r'\([^)]*\)', '', entity_name).strip()
        if paren_removed and paren_removed != entity_name:
            alias_dict[paren_removed] = entity_id


    def generate_candidates(self, mention: str, context: str = "",
                          max_candidates: int = 10) -> List[CandidateConcept]:
        """Generate candidate concepts for a given mention"""
        mention_lower = mention.lower().strip()
        candidates = []

        # Direct matching
        disease_candidates = self._find_disease_candidates(mention_lower)
        symptom_candidates = self._find_symptom_candidates(mention_lower)

        # Combine and rank candidates
        all_candidates = disease_candidates + symptom_candidates

        # Sort by similarity score
        all_candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        return all_candidates[:max_candidates]


    def _find_disease_candidates(self, mention: str) -> List[CandidateConcept]:
        """Find disease candidates for mention"""
        candidates = []

        # Exact match
        if mention in self.disease_entities:
            entity = self.disease_entities[mention]
            candidates.append(CandidateConcept(
                concept_id=entity['id'],
                name=entity['name'],
                display_name=entity['display_name'],
                concept_type='DISEASE',
                similarity_score=1.0
            ))

        # Alias matching
        if mention in self.disease_aliases and mention not in self.disease_entities:
            entity_id = self.disease_aliases[mention]
            entity = next((e for e in self.disease_entities.values() if e['id'] == entity_id), None)
            if entity:
                candidates.append(CandidateConcept(
                    concept_id=entity['id'],
                    name=entity['name'],
                    display_name=entity['display_name'],
                    concept_type='DISEASE',
                    similarity_score=0.9
                ))

        # Fuzzy matching
        for entity_name, entity_data in self.disease_entities.items():
            similarity = SequenceMatcher(None, mention, entity_name).ratio()
            if similarity >= 0.7 and similarity < 1.0:
                candidates.append(CandidateConcept(
                    concept_id=entity_data['id'],
                    name=entity_data['name'],
                    display_name=entity_data['display_name'],
                    concept_type='DISEASE',
                    similarity_score=similarity
                ))

        # Partial matching
        if len(candidates) < 5:
            for entity_name, entity_data in self.disease_entities.items():
                if mention in entity_name or entity_name in mention:
                    if len(mention) >= 3 and len(entity_name) >= 3:
                        similarity = min(len(mention), len(entity_name)) / max(len(mention), len(entity_name))
                        if similarity >= 0.5:
                            candidates.append(CandidateConcept(
                                concept_id=entity_data['id'],
                                name=entity_data['name'],
                                display_name=entity_data['display_name'],
                                concept_type='DISEASE',
                                similarity_score=similarity * 0.8  # Lower score for partial match
                            ))

        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.concept_id not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate.concept_id)

        return unique_candidates


    def _find_symptom_candidates(self, mention: str) -> List[CandidateConcept]:
        """Find symptom candidates for mention"""
        candidates = []

        # Exact match
        if mention in self.symptom_entities:
            entity = self.symptom_entities[mention]
            candidates.append(CandidateConcept(
                concept_id=entity['id'],
                name=entity['name'],
                display_name=entity['display_name'],
                concept_type='SYMPTOM',
                similarity_score=1.0
            ))

        # Alias matching
        if mention in self.symptom_aliases and mention not in self.symptom_entities:
            entity_id = self.symptom_aliases[mention]
            entity = next((e for e in self.symptom_entities.values() if e['id'] == entity_id), None)
            if entity:
                candidates.append(CandidateConcept(
                    concept_id=entity['id'],
                    name=entity['name'],
                    display_name=entity['display_name'],
                    concept_type='SYMPTOM',
                    similarity_score=0.9
                ))

        # Fuzzy matching
        for entity_name, entity_data in self.symptom_entities.items():
            similarity = SequenceMatcher(None, mention, entity_name).ratio()
            if similarity >= 0.7 and similarity < 1.0:
                candidates.append(CandidateConcept(
                    concept_id=entity_data['id'],
                    name=entity_data['name'],
                    display_name=entity_data['display_name'],
                    concept_type='SYMPTOM',
                    similarity_score=similarity
                ))

        # Partial matching
        if len(candidates) < 5:
            for entity_name, entity_data in self.symptom_entities.items():
                if mention in entity_name or entity_name in mention:
                    if len(mention) >= 3 and len(entity_name) >= 3:
                        similarity = min(len(mention), len(entity_name)) / max(len(mention), len(entity_name))
                        if similarity >= 0.5:
                            candidates.append(CandidateConcept(
                                concept_id=entity_data['id'],
                                name=entity_data['name'],
                                display_name=entity_data['display_name'],
                                concept_type='SYMPTOM',
                                similarity_score=similarity * 0.8
                            ))

        # Remove duplicates
        seen_ids = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate.concept_id not in seen_ids:
                unique_candidates.append(candidate)
                seen_ids.add(candidate.concept_id)

        return unique_candidates


    def get_concept_relationships(self, concept_id: str, concept_type: str) -> Dict:
        """Get relationships for a concept from knowledge base"""
        relationships = {'related_diseases': [], 'related_symptoms': [], 'weights': {}}

        with self.driver.session() as session:
            if concept_type == 'DISEASE':
                # Get symptoms related to disease
                query = """
                MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
                WHERE d.id = $concept_id
                RETURN s.name as symptom, r.weight as weight
                """
            else:
                # Get diseases related to symptom
                query = """
                MATCH (d:Disease)-[r:HAS_SYMPTOM]->(s:Symptom)
                WHERE s.name = $concept_id
                RETURN d.name as disease, r.weight as weight
                """

            try:
                results = session.run(query, concept_id=concept_id)
                for record in results:
                    if concept_type == 'DISEASE':
                        relationships['related_symptoms'].append(record['symptom'])
                        relationships['weights'][record['symptom']] = record['weight']
                    else:
                        relationships['related_diseases'].append(record['disease'])
                        relationships['weights'][record['disease']] = record['weight']
            except Exception as e:
                logger.warning(f"Error getting relationships for {concept_id}: {e}")

        return relationships


    def get_concept_definition(self, concept_id: str, concept_type: str) -> str:
        """Get definition/description for a concept"""
        # This could be enhanced with actual definitions stored in Neo4j
        return f"Medical {concept_type.lower()}: {concept_id}"


    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


def main():
    """Example usage"""
    kb = KnowledgeBaseInterface()

    # Test candidate generation
    test_mentions = [
        "diabetes",
        "chest pain",
        "shortness of breath",
        "hypertension",
        "fever"
    ]

    for mention in test_mentions:
        print(f"\nCandidates for '{mention}':")
        candidates = kb.generate_candidates(mention)
        for i, candidate in enumerate(candidates[:5]):
            print(f"  {i+1}. {candidate.display_name} ({candidate.concept_type}) - {candidate.similarity_score:.3f}")

    kb.close()

if __name__ == "__main__":
    main()
