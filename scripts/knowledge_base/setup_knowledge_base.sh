#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/data_preparation/download_data.py"
python3 "${SCRIPT_DIR}/data_preparation/disease_symptom_weight_calculation.py"
python3 "${SCRIPT_DIR}/disease_ontology_neo4j.py"
python3 "${SCRIPT_DIR}/neo4j_connection_test.py"
