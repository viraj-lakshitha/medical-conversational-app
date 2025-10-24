import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import re


class DiseaseSymptomWeightCalculator:


    def __init__(self, df):
        self.df = df.copy()
        self.disease_symptom_pairs = []
        self.symptom_disease_matrix = None
        self.weights = {}


    def preprocess_data(self):
        """Extract and clean disease-symptom pairs from the dataset"""
        pairs = []

        for _, row in self.df.iterrows():
            disease_id = row['disease_id']
            disease = row['disease'].strip()
            symptoms_text = row['common_symptom']

            # Clean and split symptoms
            if pd.notna(symptoms_text):
                # Split by comma and clean each symptom
                symptoms = [s.strip() for s in symptoms_text.split(',')]
                symptoms = [s for s in symptoms if s]  # Remove empty strings

                for symptom in symptoms:
                    # Remove extra whitespace and standardize
                    symptom_clean = re.sub(r'\s+', ' ', symptom.strip())
                    if symptom_clean:
                        pairs.append({
                            'disease_id': disease_id,
                            'disease': disease,
                            'symptom': symptom_clean
                        })

        self.disease_symptom_pairs = pd.DataFrame(pairs)
        return self.disease_symptom_pairs


    def calculate_frequency_weights(self):
        """Calculate weights based on symptom frequency within each disease"""
        weights = {}

        # Count symptoms per disease
        disease_symptom_counts = self.disease_symptom_pairs.groupby(['disease_id', 'disease', 'symptom']).size().reset_index(name='count')
        disease_totals = self.disease_symptom_pairs.groupby(['disease_id', 'disease']).size().reset_index(name='total_symptoms')

        # Merge to calculate weights
        merged = disease_symptom_counts.merge(disease_totals, on=['disease_id', 'disease'])
        merged['frequency_weight'] = merged['count'] / merged['total_symptoms']

        for _, row in merged.iterrows():
            key = (row['disease_id'], row['symptom'])
            weights[key] = {
                'frequency_weight': row['frequency_weight'],
                'disease': row['disease'],
                'symptom_count': row['count']
            }

        return weights


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


    def calculate_all_weights(self):
        """Calculate all weight types and combine them"""
        print("Preprocessing data...")
        self.preprocess_data()
        print(f"Extracted {len(self.disease_symptom_pairs)} disease-symptom pairs")

        print("Calculating frequency weights...")
        freq_weights = self.calculate_frequency_weights()

        print("Calculating TF-IDF weights...")
        tfidf_weights = self.calculate_tfidf_weights()

        print("Calculating PMI weights...")
        pmi_weights = self.calculate_pmi_weights()

        # Combine all weights
        combined_weights = {}
        all_keys = set(freq_weights.keys()) | set(tfidf_weights.keys()) | set(pmi_weights.keys())

        for key in all_keys:
            combined_weights[key] = {
                'disease_id': key[0],
                'symptom': key[1],
                'disease': freq_weights.get(key, {}).get('disease', ''),
                'frequency_weight': freq_weights.get(key, {}).get('frequency_weight', 0),
                'tfidf_weight': tfidf_weights.get(key, {}).get('tfidf_weight', 0),
                'pmi_weight': pmi_weights.get(key, {}).get('pmi_weight', 0)
            }

        return combined_weights


    def export_weights_to_dataframe(self, weights):
        """Convert weights dictionary to DataFrame for easy analysis"""
        rows = []
        for key, values in weights.items():
            row = {
                'disease_id': key[0],
                'disease': values['disease'],
                'symptom': key[1],
                'frequency_weight': values['frequency_weight'],
                'tfidf_weight': values['tfidf_weight'],
                'pmi_weight': values['pmi_weight']
            }
            rows.append(row)

        return pd.DataFrame(rows)


    def get_top_symptoms_for_disease(self, disease_id, weight_type='tfidf_weight', top_n=10):
        """Get top symptoms for a specific disease based on weight type"""
        if not hasattr(self, 'weights_df'):
            weights = self.calculate_all_weights()
            self.weights_df = self.export_weights_to_dataframe(weights)

        disease_data = self.weights_df[self.weights_df['disease_id'] == disease_id]
        top_symptoms = disease_data.nlargest(top_n, weight_type)[['symptom', weight_type, 'disease']]
        return top_symptoms

# Load your data
df = pd.read_csv('disease_database.csv')

# Initialize calculator
calculator = DiseaseSymptomWeightCalculator(df)

# Calculate all weights
weights = calculator.calculate_all_weights()

# Export to DataFrame for analysis
weights_df = calculator.export_weights_to_dataframe(weights)

# Get top symptoms for a specific disease
top_symptoms = calculator.get_top_symptoms_for_disease('1656164150939770881', 'tfidf_weight', 5)
print(top_symptoms)

# Save weights for knowledge graph construction
weights_df.to_csv('disease_symptom_weights.csv', index=False)

print("Disease-Symptom Weight Calculator ready!")
print("Use the calculator to extract weighted relationships for your knowledge graph.")
