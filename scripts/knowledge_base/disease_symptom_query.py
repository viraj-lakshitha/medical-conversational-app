"""
Convenience helpers for querying the medical knowledge graph from the CLI.

This module deliberately reuses :class:`MedicalKnowledgeBase` so there is a
single source of truth for Cypher queries and response shaping.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

try:
    from .medical_knowledge_interface import MedicalKnowledgeBase
    from .utils import normalise_symptom_text
except ImportError:  # pragma: no cover - fallback for direct execution
    from medical_knowledge_interface import MedicalKnowledgeBase  # type: ignore
    from utils import normalise_symptom_text  # type: ignore


logger = logging.getLogger(__name__)


def query_diseases_with_additional_symptoms(
    symptoms: List[str],
    *,
    min_weight: float = 0.3,
    top_n: int = 10,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str = "password",
) -> List[Dict[str, Any]]:
    """
    Return diseases that match the provided symptoms together with additional
    supporting evidence from the knowledge graph.
    """
    normalised = [normalise_symptom_text(symptom) for symptom in symptoms]
    with MedicalKnowledgeBase(uri=uri, user=user, password=password) as kb:
        payload = kb.get_related_diseases(normalised, min_weight=min_weight, top_n=top_n)
    return payload.get("diseases", [])


def get_disease_suggestions(
    symptoms: List[str],
    min_weight: float = 0.3,
    top_n: int = 5,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
) -> str:
    """
    Wrapper that returns a JSON serialisation suitable for CLI usage.
    """
    diseases = query_diseases_with_additional_symptoms(
        symptoms,
        min_weight=min_weight,
        top_n=top_n,
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )

    for entry in diseases:
        entry["total_score"] = round(entry.get("total_score", 0), 4)
        for symptom in entry.get("additional_symptoms", []):
            symptom["weight"] = round(symptom.get("weight", 0), 4)
    return json.dumps(diseases, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the medical knowledge graph for diseases related to symptoms.")
    parser.add_argument("symptoms", nargs="+", help="List of symptoms (e.g. fever cough fatigue)")
    parser.add_argument("--min-weight", type=float, default=0.3, help="Minimum relationship weight to consider")
    parser.add_argument("--top-n", type=int, default=5, help="Maximum number of diseases to return")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="password", help="Neo4j password")
    args = parser.parse_args()

    results = query_diseases_with_additional_symptoms(
        args.symptoms,
        min_weight=args.min_weight,
        top_n=args.top_n,
        uri=args.uri,
        user=args.user,
        password=args.password,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
