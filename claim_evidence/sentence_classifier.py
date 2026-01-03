"""
EVINCE: Sentence Type Classifier

Classifies sentences into types for claim-evidence linking:
- CLAIM: Commitments, promises, goals ("cam kết", "sẽ", "hướng tới")
- EVIDENCE: Data, metrics, completed actions ("đã", numbers, %)
- CONTEXT: Background information, explanations
- NON_ESG: Not ESG-related content

Reference:
- A3CG (Ong et al. 2025): Implemented/Planning/Indeterminate classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SentenceType(Enum):
    """Sentence types for claim-evidence linking."""
    CLAIM = "CLAIM"           # Commitments, promises
    EVIDENCE = "EVIDENCE"     # Data, metrics, actions done
    CONTEXT = "CONTEXT"       # Background info
    NON_ESG = "NON_ESG"       # Not ESG-related


SENTENCE_TYPES = [t.value for t in SentenceType]
SENTENCE_TYPE_TO_ID = {t.value: i for i, t in enumerate(SentenceType)}
SENTENCE_ID_TO_TYPE = {i: t.value for i, t in enumerate(SentenceType)}

SENTENCE_TYPE_VN = {
    "CLAIM": "Cam kết/Tuyên bố",
    "EVIDENCE": "Bằng chứng/Số liệu",
    "CONTEXT": "Ngữ cảnh",
    "NON_ESG": "Không liên quan ESG"
}


@dataclass
class SentenceClassificationResult:
    """Result from sentence type classification."""
    sentence_type: str
    sentence_type_vn: str
    probabilities: Dict[str, float]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            "sentence_type": self.sentence_type,
            "sentence_type_vn": self.sentence_type_vn,
            "probabilities": self.probabilities,
            "confidence": self.confidence
        }


class SentenceClassifier(nn.Module):
    """
    Classifier for sentence types in claim-evidence linking.
    
    Architecture:
    - Encoder: PhoBERT-base (768-dim)
    - Classifier: MLP head with 4 classes
    
    This is a simpler model than ESGTopicClassifier since we only need
    4 broad categories, not fine-grained topic classification.
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        freeze_bert_layers: int = 8  # Freeze more layers for this simpler task
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _freeze_bert_layers(self, num_layers: int):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        return F.softmax(logits, dim=-1)


class SentenceClassifierInference:
    """Inference wrapper for SentenceClassifier."""
    
    def __init__(
        self,
        model: Optional[SentenceClassifier] = None,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = SentenceClassifier().to(self.device)
        
        self.model.eval()
    
    def _load_model(self, path: str) -> SentenceClassifier:
        model = SentenceClassifier()
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    @torch.no_grad()
    def predict(self, text: str, max_length: int = 256) -> SentenceClassificationResult:
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        probs = self.model.predict_proba(input_ids, attention_mask)
        probs = probs.cpu().numpy()[0]
        
        predicted_id = int(probs.argmax())
        predicted_type = SENTENCE_ID_TO_TYPE[predicted_id]
        
        return SentenceClassificationResult(
            sentence_type=predicted_type,
            sentence_type_vn=SENTENCE_TYPE_VN[predicted_type],
            probabilities={t: float(probs[i]) for i, t in enumerate(SENTENCE_TYPES)},
            confidence=float(probs[predicted_id])
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = 256,
        batch_size: int = 32
    ) -> List[SentenceClassificationResult]:
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            probs = self.model.predict_proba(input_ids, attention_mask)
            probs = probs.cpu().numpy()
            
            for prob in probs:
                predicted_id = int(prob.argmax())
                predicted_type = SENTENCE_ID_TO_TYPE[predicted_id]
                
                results.append(SentenceClassificationResult(
                    sentence_type=predicted_type,
                    sentence_type_vn=SENTENCE_TYPE_VN[predicted_type],
                    probabilities={t: float(prob[k]) for k, t in enumerate(SENTENCE_TYPES)},
                    confidence=float(prob[predicted_id])
                ))
        
        return results
