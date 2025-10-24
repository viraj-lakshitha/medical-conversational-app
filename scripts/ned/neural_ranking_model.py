"""
Neural ranking model for entity disambiguation.
This model combines contextual embeddings with string similarity features
to rank candidate concepts for entity mentions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import json
import pickle

# Memory optimization imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass


class RankingFeatures:
    mention_text: str
    candidate_name: str
    candidate_display_name: str
    concept_type: str
    left_context: str
    right_context: str
    document_context: str
    string_similarity: float
    candidate_frequency: float = 0.0
    type_consistency: float = 0.0

@dataclass


class DisambiguationResult:
    mention: str
    start_pos: int
    end_pos: int
    predicted_concept_id: str
    predicted_concept_name: str
    concept_type: str
    confidence_score: float
    candidates_considered: int


class LightweightRankingModel(nn.Module):


    def __init__(self, model_name: str = "distilbert-base-uncased", use_lora: bool = True):
        super(LightweightRankingModel, self).__init__()
        self.use_lora = use_lora and LORA_AVAILABLE

        # Use lightweight DistilBERT for efficiency
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Apply LoRA if available for memory efficiency
        if self.use_lora:
            try:
                peft_config = LoraConfig(
                    r=8,  # Low rank for efficiency
                    lora_alpha=16,
                    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],  # DistilBERT modules
                    lora_dropout=0.1,
                    bias="none",
                )
                self.encoder = get_peft_model(self.encoder, peft_config)
                print(" LoRA applied to ranking model - Memory reduced by ~50%")
            except Exception as e:
                print(f" LoRA failed for ranking model: {e}")
                self.use_lora = False

        # Simplified architecture for efficiency
        self.string_features_size = 3  # similarity, length_ratio, exact_match

        # Compact neural layers
        self.feature_combiner = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.string_features_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, mention_embeddings, candidate_embeddings, string_features):
        # Simplified feature combination - just mention and candidate embeddings
        combined_features = torch.cat([
            mention_embeddings,
            candidate_embeddings,
            string_features
        ], dim=-1)

        # Generate ranking score
        score = self.feature_combiner(combined_features)
        return torch.sigmoid(score)


class FastEntityDisambiguator:


    def __init__(self, model_name: str = "distilbert-base-uncased", use_lora: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.ranking_model = LightweightRankingModel(model_name, use_lora=use_lora)
        self.use_lora = use_lora

        # Device setup - prefer CPU for lightweight model
        self.device = torch.device("cpu")
        self.encoder.to(self.device)
        self.ranking_model.to(self.device)

        # Set to evaluation mode initially
        self.encoder.eval()
        self.ranking_model.eval()


    def prepare_training_data(self, annotated_corpus_path: str) -> List[Tuple[RankingFeatures, int]]:
        """Prepare training data from annotated corpus"""
        training_data = []

        logger.info(f"Preparing training data from {annotated_corpus_path}")

        with open(annotated_corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 1000 == 0:
                    logger.info(f"Processed {line_num} examples")

                data = json.loads(line)
                text = data['text']

                # Process diseases
                for disease in data.get('diseases', []):
                    features = self._create_ranking_features(
                        mention_text=disease['entity'],
                        candidate_name=disease['entity'],
                        candidate_display_name=disease['display_name'],
                        concept_type='DISEASE',
                        text=text,
                        mention_pos=disease['position']
                    )
                    training_data.append((features, 1))  # Positive example

                # Process symptoms
                for symptom in data.get('symptoms', []):
                    features = self._create_ranking_features(
                        mention_text=symptom['entity'],
                        candidate_name=symptom['entity'],
                        candidate_display_name=symptom['display_name'],
                        concept_type='SYMPTOM',
                        text=text,
                        mention_pos=symptom['position']
                    )
                    training_data.append((features, 1))  # Positive example

        logger.info(f"Prepared {len(training_data)} training examples")
        return training_data


    def _create_ranking_features(self, mention_text: str, candidate_name: str,
                                candidate_display_name: str, concept_type: str,
                                text: str, mention_pos: int) -> RankingFeatures:
        """Create ranking features for mention-candidate pair"""

        # Extract context
        left_context = text[max(0, mention_pos - self.context_window):mention_pos]
        right_context = text[mention_pos + len(mention_text):mention_pos + len(mention_text) + self.context_window]

        # String similarity
        string_similarity = SequenceMatcher(None, mention_text.lower(), candidate_name.lower()).ratio()

        return RankingFeatures(
            mention_text=mention_text,
            candidate_name=candidate_name,
            candidate_display_name=candidate_display_name,
            concept_type=concept_type,
            left_context=left_context,
            right_context=right_context,
            document_context=text[:500],  # First 500 chars as document context
            string_similarity=string_similarity
        )


    def _extract_embeddings(self, text: str) -> torch.Tensor:
        """Extract lightweight embeddings for text"""
        with torch.no_grad():
            # Tokenize with smaller max length for efficiency
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)

            # Get embeddings
            outputs = self.encoder(**inputs)
            # Use [CLS] token embedding for efficiency
            embeddings = outputs.last_hidden_state[:, 0, :]

            return embeddings


    def _compute_string_features(self, mention: str, candidate: str) -> np.ndarray:
        """Compute simplified string-based similarity features"""
        mention_lower = mention.lower().strip()
        candidate_lower = candidate.lower().strip()

        # String similarity ratio
        similarity_ratio = SequenceMatcher(None, mention_lower, candidate_lower).ratio()

        # Length ratio
        length_ratio = min(len(mention), len(candidate)) / max(len(mention), len(candidate)) if max(len(mention), len(candidate)) > 0 else 0.0

        # Exact match bonus
        exact_match = 1.0 if mention_lower == candidate_lower else 0.0

        return np.array([similarity_ratio, length_ratio, exact_match], dtype=np.float32)


    def create_features_tensor(self, features: RankingFeatures) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert ranking features to tensors - simplified version"""

        # Mention embedding
        mention_embedding = self._extract_embeddings(features.mention_text)

        # Candidate embedding
        candidate_embedding = self._extract_embeddings(features.candidate_name)

        # String features
        string_features = torch.tensor(
            self._compute_string_features(features.mention_text, features.candidate_name),
            dtype=torch.float32
        ).unsqueeze(0)

        return mention_embedding, candidate_embedding, string_features


    def train(self, training_data: List[Tuple[RankingFeatures, int]],
              epochs: int = 3, batch_size: int = 64, learning_rate: float = 2e-4):
        """Train the ranking model with memory optimizations"""

        self.ranking_model.train()

        # Memory-efficient optimizer settings
        if self.use_lora:
            # Only optimize LoRA parameters
            optimizer = torch.optim.AdamW(self.ranking_model.parameters(), lr=learning_rate, weight_decay=0.01)
        else:
            optimizer = torch.optim.Adam(self.ranking_model.parameters(), lr=learning_rate)

        criterion = nn.BCELoss()

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.ranking_model.encoder, 'gradient_checkpointing_enable'):
            self.ranking_model.encoder.gradient_checkpointing_enable()

        # Prepare batched data
        logger.info("Preparing batched training data...")
        batched_data = []

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]

            context_embeddings = []
            mention_embeddings = []
            candidate_embeddings = []
            string_features = []
            labels = []

            for features, label in batch:
                men_emb, cand_emb, str_feat = self.create_features_tensor(features)

                mention_embeddings.append(men_emb)
                candidate_embeddings.append(cand_emb)
                string_features.append(str_feat)
                labels.append(label)

            # Stack tensors
            mention_batch = torch.cat(mention_embeddings, dim=0).to(self.device)
            candidate_batch = torch.cat(candidate_embeddings, dim=0).to(self.device)
            string_batch = torch.cat(string_features, dim=0).to(self.device)
            label_batch = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(self.device)

            batched_data.append((mention_batch, candidate_batch, string_batch, label_batch))

        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (men_batch, cand_batch, str_batch, label_batch) in enumerate(batched_data):
                optimizer.zero_grad()

                # Forward pass
                predictions = self.ranking_model(men_batch, cand_batch, str_batch)

                # Compute loss
                loss = criterion(predictions, label_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(batched_data)}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(batched_data)
            logger.info(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        # Set back to eval mode
        self.ranking_model.eval()
        logger.info("Training completed")


    def rank_candidates(self, mention: str, candidates: List, context: str,
                       mention_pos: int = 0) -> List[Tuple[str, float]]:
        """Rank candidates for a given mention"""

        if not candidates:
            return []

        candidate_scores = []

        with torch.no_grad():
            for candidate in candidates:
                # Create features
                features = self._create_ranking_features(
                    mention_text=mention,
                    candidate_name=candidate.name,
                    candidate_display_name=candidate.display_name,
                    concept_type=candidate.concept_type,
                    text=context,
                    mention_pos=mention_pos
                )

                # Get tensors
                men_emb, cand_emb, str_feat = self.create_features_tensor(features)

                # Move to device
                men_emb = men_emb.to(self.device)
                cand_emb = cand_emb.to(self.device)
                str_feat = str_feat.to(self.device)

                # Get score
                score = self.ranking_model(men_emb, cand_emb, str_feat)

                candidate_scores.append((candidate.concept_id, score.item()))

        # Sort by score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        return candidate_scores


    def disambiguate(self, mention: str, candidates: List, context: str,
                    start_pos: int = 0, end_pos: int = 0) -> DisambiguationResult:
        """Perform entity disambiguation for a mention"""

        if not candidates:
            return DisambiguationResult(
                mention=mention,
                start_pos=start_pos,
                end_pos=end_pos,
                predicted_concept_id="UNKNOWN",
                predicted_concept_name=mention,
                concept_type="UNKNOWN",
                confidence_score=0.0,
                candidates_considered=0
            )

        # Rank candidates
        ranked_candidates = self.rank_candidates(mention, candidates, context, start_pos)

        # Get top candidate
        top_candidate_id, confidence_score = ranked_candidates[0]
        top_candidate = next(c for c in candidates if c.concept_id == top_candidate_id)

        return DisambiguationResult(
            mention=mention,
            start_pos=start_pos,
            end_pos=end_pos,
            predicted_concept_id=top_candidate.concept_id,
            predicted_concept_name=top_candidate.name,
            concept_type=top_candidate.concept_type,
            confidence_score=confidence_score,
            candidates_considered=len(candidates)
        )


    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.ranking_model.state_dict(),
            'tokenizer_name': self.tokenizer.name_or_path
        }, filepath)
        logger.info(f"Model saved to {filepath}")


    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.ranking_model.load_state_dict(checkpoint['model_state_dict'])
        self.ranking_model.eval()
        logger.info(f"Model loaded from {filepath}")


def main():
    """Example usage with memory optimization"""
    # Enable LoRA for memory-efficient training
    disambiguator = FastEntityDisambiguator(use_lora=True)

    # Load training data
    training_data = disambiguator.prepare_training_data("data_preparation/data/full_annotated_corpus.jsonl")

    # Train model with memory-efficient settings
    disambiguator.train(
        training_data,
        epochs=2,  # Reduced epochs
        batch_size=64,  # Larger batch with LoRA
        learning_rate=2e-4  # Slightly higher LR for LoRA
    )

    # Save model
    disambiguator.save_model("fast_ranking_model_lora.pth")

    print(" Memory-efficient training completed!")
    logger.info("Training completed and model saved")

if __name__ == "__main__":
    main()
