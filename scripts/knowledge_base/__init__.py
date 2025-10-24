"""
Knowledge base tooling for the medical conversational application.

This package contains utilities to build and query the Neo4j-backed medical
ontology.  The most commonly used exports are:

``KnowledgeGraphBuilder`` – constructs the graph from the prepared dataset.
``MedicalKnowledgeBase`` – high level query interface used at runtime.
``Neo4jConnectionValidator`` – quick connectivity smoke tests.
``get_disease_suggestions`` – convenience wrapper for CLI usage.
"""

from .disease_ontology_neo4j import DiseaseSymptomKnowledgeGraphBuilder as KnowledgeGraphBuilder
from .medical_knowledge_interface import MedicalKnowledgeBase
from .neo4j_connection_test import Neo4jConnectionValidator
from .disease_symptom_query import get_disease_suggestions

__all__ = [
    "KnowledgeGraphBuilder",
    "MedicalKnowledgeBase",
    "Neo4jConnectionValidator",
    "get_disease_suggestions",
]
