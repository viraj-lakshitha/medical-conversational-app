"""
BioBERT-based Named Entity Recognition model for medical entity span identification.
This is the first stage of the Named Entity Disambiguation pipeline.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification
)
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.metrics import classification_report, f1_score
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

# Unsloth and LoRA imports for memory-efficient training
try:
    from unsloth import FastLanguageModel
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth/PEFT not available. Using standard training.")
    UNSLOTH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass


class EntitySpan:
    start: int
    end: int
    label: str
    text: str
    confidence: float = 0.0


class BioBERTNERDataset(Dataset):


    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Label mapping
        self.label2id = {'O': 0, 'B-DISEASE': 1, 'I-DISEASE': 2, 'B-SYMPTOM': 3, 'I-SYMPTOM': 4}
        self.id2label = {v: k for k, v in self.label2id.items()}


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        # Tokenize and align labels
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Align labels with tokenized text
        aligned_labels = self._align_labels(text, labels, encoding)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


    def _align_labels(self, text: str, labels: List[str], encoding) -> List[int]:
        """Align BIO labels with tokenized text"""
        tokens = self.tokenizer.tokenize(text)
        aligned_labels = [self.label2id['O']] * self.max_length

        # Map original labels to tokenized positions
        # This is a simplified alignment - in practice, you'd need more sophisticated alignment
        for i, label in enumerate(labels[:len(tokens)]):
            if i < self.max_length - 2:  # Account for [CLS] and [SEP]
                aligned_labels[i + 1] = self.label2id.get(label, 0)

        return aligned_labels


class OptimizedMedicalNERModel(nn.Module):


    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels: int = 5, use_lora: bool = True):
        super(OptimizedMedicalNERModel, self).__init__()
        self.use_lora = use_lora and UNSLOTH_AVAILABLE

        if self.use_lora:
            # Use Unsloth for memory-efficient training
            try:
                self.medical_bert, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=512,
                    dtype=None,  # Auto-detect
                    load_in_4bit=True,  # 4-bit quantization
                )

                # Add LoRA adapters
                self.medical_bert = FastLanguageModel.get_peft_model(
                    self.medical_bert,
                    r=16,  # LoRA rank
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=3407,
                    use_rslora=False,
                    loftq_config=None,
                )
            except Exception as e:
                print(f"Failed to use Unsloth: {e}. Falling back to standard model.")
                self.use_lora = False
                self.medical_bert = AutoModel.from_pretrained(model_name)
        else:
            # Standard model loading
            self.medical_bert = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(0.1)
        hidden_size = getattr(self.medical_bert.config, 'hidden_size', 768)
        self.classifier = nn.Linear(hidden_size, num_labels)


    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.medical_bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return {'loss': loss, 'logits': logits}


class OptimizedMedicalNERTrainer:


    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", use_lora: bool = True):
        self.model_name = model_name
        self.use_lora = use_lora and UNSLOTH_AVAILABLE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        # Simplified label set - focus on medical entities
        self.label2id = {'O': 0, 'B-DISEASE': 1, 'I-DISEASE': 2, 'B-SYMPTOM': 3, 'I-SYMPTOM': 4}
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Memory optimization settings
        self.memory_efficient = True


    def _create_unsloth_model(self):
        """Create Unsloth-optimized model with LoRA for memory efficiency"""
        try:
            from unsloth import FastLanguageModel

            # Load model with Unsloth optimizations
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=512,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # 4-bit quantization for memory
            )

            # Replace tokenizer with the optimized one
            self.tokenizer = tokenizer

            # Convert to classification model
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )

            if UNSLOTH_AVAILABLE:
                # Apply LoRA for memory-efficient training
                peft_config = LoraConfig(
                    task_type=TaskType.TOKEN_CLS,
                    inference_mode=False,
                    r=8,  # LoRA rank (smaller for NER)
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["query", "key", "value", "dense"]
                )
                model = get_peft_model(model, peft_config)
                print(" LoRA enabled - Memory usage reduced by ~60%")

            return model

        except Exception as e:
            print(f" Unsloth initialization failed: {e}")
            print(" Falling back to standard model")
            return AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.label2id),
                id2label=self.id2label,
                label2id=self.label2id
            )


    def prepare_training_data(self, annotated_corpus_path: str) -> Tuple[List[str], List[List[str]]]:
        """Convert annotated corpus to NER training format"""
        texts = []
        labels = []

        logger.info(f"Loading annotated corpus from {annotated_corpus_path}")

        with open(annotated_corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']

                # Initialize BIO labels
                tokens = self.tokenizer.tokenize(text)
                token_labels = ['O'] * len(tokens)

                # Process diseases
                for disease in data.get('diseases', []):
                    start_pos = disease['position']
                    entity_text = disease['entity']
                    self._assign_bio_labels(text, tokens, token_labels, start_pos, entity_text, 'DISEASE')

                # Process symptoms
                for symptom in data.get('symptoms', []):
                    start_pos = symptom['position']
                    entity_text = symptom['entity']
                    self._assign_bio_labels(text, tokens, token_labels, start_pos, entity_text, 'SYMPTOM')

                texts.append(text)
                labels.append(token_labels)

        logger.info(f"Prepared {len(texts)} training examples")
        return texts, labels


    def _assign_bio_labels(self, text: str, tokens: List[str], token_labels: List[str],
                          start_pos: int, entity_text: str, entity_type: str):
        """Assign BIO labels for an entity span"""
        try:
            # Find the entity in the text
            entity_start = text.find(entity_text, start_pos)
            if entity_start == -1:
                entity_start = start_pos

            entity_end = entity_start + len(entity_text)

            # Convert character positions to token positions (approximation)
            char_to_token = self._create_char_to_token_mapping(text, tokens)

            token_start = char_to_token.get(entity_start, 0)
            token_end = char_to_token.get(entity_end - 1, len(tokens) - 1) + 1

            # Assign BIO labels
            for i in range(token_start, min(token_end, len(token_labels))):
                if i == token_start:
                    token_labels[i] = f'B-{entity_type}'
                else:
                    token_labels[i] = f'I-{entity_type}'

        except Exception as e:
            logger.warning(f"Error assigning BIO labels for '{entity_text}': {e}")


    def _create_char_to_token_mapping(self, text: str, tokens: List[str]) -> Dict[int, int]:
        """Create mapping from character positions to token indices"""
        char_to_token = {}
        current_pos = 0

        for token_idx, token in enumerate(tokens):
            # Skip special tokens
            if token.startswith('##'):
                token = token[2:]

            # Find token in text
            token_start = text.find(token, current_pos)
            if token_start != -1:
                for char_pos in range(token_start, token_start + len(token)):
                    char_to_token[char_pos] = token_idx
                current_pos = token_start + len(token)

        return char_to_token


    def train(self, annotated_corpus_path: str, output_dir: str = "./biobert_ner_model",
              epochs: int = 3, batch_size: int = 16):
        """Train the BioBERT NER model"""

        # Prepare data
        texts, labels = self.prepare_training_data(annotated_corpus_path)

        # Split data
        split_idx = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split_idx], texts[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]

        # Create datasets
        train_dataset = BioBERTNERDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = BioBERTNERDataset(val_texts, val_labels, self.tokenizer)

        # Initialize optimized model with LoRA if available
        if self.use_lora and UNSLOTH_AVAILABLE:
            # Use Unsloth's memory-efficient approach
            self.model = self._create_unsloth_model()
        else:
            # Standard model
            self.model = OptimizedMedicalNERModel(self.model_name, len(self.label2id), use_lora=False)

        # Memory-efficient training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size if not self.use_lora else batch_size * 2,  # LoRA allows larger batch
            per_device_eval_batch_size=batch_size if not self.use_lora else batch_size * 2,
            gradient_accumulation_steps=2 if self.use_lora else 4,  # Reduce with LoRA
            warmup_steps=100,  # Reduced warmup
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,  # More frequent logging
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            # Memory optimizations
            fp16=True,  # Half precision
            dataloader_pin_memory=False,
            save_total_limit=2,  # Keep only 2 checkpoints
            report_to="none"  # Disable wandb/tensorboard
        )

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()

        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        logger.info(f"Model saved to {output_dir}")


    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (padding tokens)
        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_predictions.append(self.id2label[pred])
                    true_labels.append(self.id2label[lab])

        # Compute F1 score
        f1 = f1_score(true_labels, true_predictions, average='weighted')

        return {"f1": f1}


    def load_model(self, model_path: str):
        """Load trained model"""
        self.model = OptimizedMedicalNERModel(self.model_name, len(self.label2id))
        self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu'))
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")


    def predict(self, text: str) -> List[EntitySpan]:
        """Predict entities in text"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.nn.functional.softmax(outputs['logits'], dim=-1)
            predicted_labels = torch.argmax(predictions, dim=-1)

        # Convert predictions to entity spans
        entities = self._extract_entities(text, predicted_labels[0], predictions[0])

        return entities


    def _extract_entities(self, text: str, predicted_labels: torch.Tensor,
                         confidences: torch.Tensor) -> List[EntitySpan]:
        """Extract entity spans from BIO predictions"""
        entities = []
        current_entity = None
        tokens = self.tokenizer.tokenize(text)

        for i, (label_id, confidence_scores) in enumerate(zip(predicted_labels, confidences)):
            if i >= len(tokens) + 1:  # Account for [CLS]
                break

            label = self.id2label[label_id.item()]
            confidence = confidence_scores[label_id].item()

            if label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]  # Remove 'B-'
                current_entity = EntitySpan(
                    start=max(0, i-1),  # Approximate character position
                    end=max(0, i-1),
                    label=entity_type,
                    text="",
                    confidence=confidence
                )

            elif label.startswith('I-') and current_entity:
                # Continue current entity
                entity_type = label[2:]  # Remove 'I-'
                if current_entity.label == entity_type:
                    current_entity.end = max(0, i-1)
                    current_entity.confidence = max(current_entity.confidence, confidence)

            elif label == 'O' and current_entity:
                # End current entity
                entities.append(current_entity)
                current_entity = None

        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)

        # Update entity text based on positions
        for entity in entities:
            # This is simplified - in practice you'd need better char-token mapping
            try:
                if entity.start < len(text) and entity.end < len(text):
                    entity.text = text[entity.start:entity.end+1]
            except:
                entity.text = "unknown"

        return entities


def main():
    """Example usage"""
    trainer = OptimizedMedicalNERTrainer()

    # Train model
    annotated_corpus_path = "data_preparation/data/full_annotated_corpus.jsonl"
    trainer.train(annotated_corpus_path, output_dir="./optimized_medical_ner_model")

    # Test prediction
    trainer.load_model("./optimized_medical_ner_model")
    test_text = "Patient presents with chest pain, shortness of breath, and has diabetes."
    entities = trainer.predict(test_text)

    print(f"Entities found in: '{test_text}'")
    for entity in entities:
        print(f"  - {entity.label}: '{entity.text}' (confidence: {entity.confidence:.3f})")

if __name__ == "__main__":
    main()
