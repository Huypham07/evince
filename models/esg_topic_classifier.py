"""
EVINCE: ESG Topic Classifier

PhoBERT-based multi-class classifier for Vietnamese ESG topic detection.
Optimized for paragraph-level input with max_length=512 and full fine-tuning.

Labels:
- Environmental_Performance: Môi trường
- Social_Performance: Xã hội  
- Governance_Performance: Quản trị
- ESG_Financing: Tín dụng xanh
- Strategy_and_Policy: Chiến lược và Chính sách
- Not_ESG_Related: Không liên quan ESG

Reference:
- Schimanski et al. (2024): ESGBERT
- VinAI PhoBERT: https://github.com/VinAIResearch/PhoBERT
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed.")


# HuggingFace model path
HUGGINGFACE_ESG_MODEL = "huypham71/esgify_vn_class_weights"

# Maximum sequence length for paragraph-level input
MAX_SEQ_LENGTH = 512

# ESG Topic Labels (matching HuggingFace model)
ESG_LABELS = [
    "Environmental_Performance",
    "Social_Performance", 
    "Governance_Performance",
    "ESG_Financing",
    "Strategy_and_Policy",
    "Not_ESG_Related"
]
LABEL_TO_ID = {label: i for i, label in enumerate(ESG_LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(ESG_LABELS)}

# Vietnamese names for display
ESG_LABELS_VN = {
    "Environmental_Performance": "Môi trường",
    "Social_Performance": "Xã hội", 
    "Governance_Performance": "Quản trị",
    "ESG_Financing": "Tín dụng xanh",
    "Strategy_and_Policy": "Chiến lược và Chính sách",
    "Not_ESG_Related": "Không liên quan ESG"
}


@dataclass
class ESGClassificationResult:
    """Result from ESG topic classification."""
    predicted_label: str
    predicted_label_vn: str
    probabilities: Dict[str, float]
    confidence: float
    is_esg: bool  # True if not Non-ESG
    
    def to_dict(self) -> Dict:
        return {
            "predicted_label": self.predicted_label,
            "predicted_label_vn": self.predicted_label_vn,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "is_esg": self.is_esg
        }


class ESGTopicClassifier(nn.Module):
    """
    PhoBERT-based classifier for ESG topic detection.
    
    Architecture:
    - Encoder: vinai/phobert-base-v2 (768-dim)
    - Dropout: 0.3
    - Classifier: Linear(768 → 512) → ReLU → Dropout → Linear(512 → 256) → ReLU → Dropout → Linear(256 → 6)
    
    Optimized for paragraph-level input with full fine-tuning enabled by default.
    
    Usage:
        model = ESGTopicClassifier()
        logits = model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_classes: int = 6,
        dropout_rate: float = 0.3,
        freeze_bert_layers: int = 0  # Default: full fine-tuning for paragraph understanding
    ):
        """
        Initialize ESG Topic Classifier.
        
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of ESG topics (default 6)
            dropout_rate: Dropout probability
            freeze_bert_layers: Number of BERT layers to freeze (0 = full fine-tuning)
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        # Load PhoBERT
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size  # 768
        
        # Freeze layers if specified
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        # Classification head - deeper MLP for better paragraph understanding
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Lower dropout in final layer
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _freeze_bert_layers(self, num_layers: int):
        """Freeze first N layers of BERT."""
        # Freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            
        Returns:
            Logits, shape (batch, num_classes)
        """
        # Get BERT output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get probability distribution over classes."""
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


class ESGTopicClassifierInference:
    """
    Inference wrapper for ESG Topic Classifier.
    
    Handles tokenization and converts outputs to ESGClassificationResult.
    Optimized for paragraph-level input with max_length=512.
    
    Usage:
        classifier = ESGTopicClassifierInference()
        result = classifier.predict("Ngân hàng cam kết giảm phát thải carbon...")
    """
    
    def __init__(
        self,
        model: Optional[ESGTopicClassifier] = None,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-loaded ESGTopicClassifier model
            model_path: Path to saved model checkpoint
            device: "cpu", "cuda", or "auto"
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        
        # Load model
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            # Initialize new model (untrained)
            self.model = ESGTopicClassifier().to(self.device)
        
        self.model.eval()
    
    def _load_model(self, path: str) -> ESGTopicClassifier:
        """Load model from checkpoint."""
        model = ESGTopicClassifier()
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        max_length: int = MAX_SEQ_LENGTH
    ) -> ESGClassificationResult:
        """
        Classify ESG topic for a single text (sentence or paragraph).
        
        Args:
            text: Text in Vietnamese (can be sentence or paragraph)
            max_length: Maximum sequence length (default 512 for paragraphs)
            
        Returns:
            ESGClassificationResult with prediction
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Get predictions
        probs = self.model.predict_proba(input_ids, attention_mask)
        probs = probs.cpu().numpy()[0]
        
        # Get predicted class
        predicted_id = int(probs.argmax())
        predicted_label = ID_TO_LABEL[predicted_id]
        confidence = float(probs[predicted_id])
        
        return ESGClassificationResult(
            predicted_label=predicted_label,
            predicted_label_vn=ESG_LABELS_VN[predicted_label],
            probabilities={label: float(probs[i]) for i, label in enumerate(ESG_LABELS)},
            confidence=confidence,
            is_esg=(predicted_label != "Not_ESG_Related")
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = MAX_SEQ_LENGTH,
        batch_size: int = 32
    ) -> List[ESGClassificationResult]:
        """
        Classify ESG topics for a batch of texts.
        
        Args:
            texts: List of texts (sentences or paragraphs)
            max_length: Maximum sequence length (default 512)
            batch_size: Batch size for inference
            
        Returns:
            List of ESGClassificationResult
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            # Get predictions
            probs = self.model.predict_proba(input_ids, attention_mask)
            probs = probs.cpu().numpy()
            
            # Convert to results
            for j, prob in enumerate(probs):
                predicted_id = int(prob.argmax())
                predicted_label = ID_TO_LABEL[predicted_id]
                
                results.append(ESGClassificationResult(
                    predicted_label=predicted_label,
                    predicted_label_vn=ESG_LABELS_VN[predicted_label],
                    probabilities={label: float(prob[k]) for k, label in enumerate(ESG_LABELS)},
                    confidence=float(prob[predicted_id]),
                    is_esg=(predicted_label != "Not_ESG_Related")
                ))
        
        return results


def create_esg_classifier(
    freeze_bert_layers: int = 0,
    dropout_rate: float = 0.3
) -> ESGTopicClassifier:
    """
    Factory function to create ESG Topic Classifier.
    
    Args:
        freeze_bert_layers: Number of BERT layers to freeze (default 0 = full fine-tuning)
        dropout_rate: Dropout rate
        
    Returns:
        ESGTopicClassifier model ready for training
    """
    return ESGTopicClassifier(
        freeze_bert_layers=freeze_bert_layers,
        dropout_rate=dropout_rate
    )


def load_esg_classifier_from_huggingface(
    model_id: str = HUGGINGFACE_ESG_MODEL,
    device: str = "auto"
):
    """
    Load pre-trained ESG classifier from HuggingFace.
    
    Args:
        model_id: HuggingFace model ID (default: huypham71/esgify_vn_class_weights)
        device: "cpu", "cuda", or "auto"
        
    Returns:
        Tuple of (model, tokenizer) ready for inference
    
    Usage:
        model, tokenizer = load_esg_classifier_from_huggingface()
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required")
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


class HuggingFaceESGClassifierInference:
    """
    Inference wrapper using pre-trained HuggingFace model.
    
    Loads model from huypham71/esgify_vn_class_weights.
    Optimized for paragraph-level input with max_length=512.
    
    Usage:
        classifier = HuggingFaceESGClassifierInference()
        result = classifier.predict("Ngân hàng cam kết giảm phát thải carbon...")
    """
    
    def __init__(
        self,
        model_id: str = HUGGINGFACE_ESG_MODEL,
        device: str = "auto"
    ):
        """
        Initialize with HuggingFace model.
        
        Args:
            model_id: HuggingFace model ID
            device: "cpu", "cuda", or "auto"
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model, self.tokenizer = load_esg_classifier_from_huggingface(model_id, str(self.device))
        
        # Get label mapping from model config
        if hasattr(self.model.config, 'id2label'):
            self.id2label = self.model.config.id2label
        else:
            self.id2label = ID_TO_LABEL
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        max_length: int = MAX_SEQ_LENGTH
    ) -> ESGClassificationResult:
        """
        Classify ESG topic for a single text (sentence or paragraph).
        
        Args:
            text: Text in Vietnamese
            max_length: Maximum sequence length (default 512 for paragraphs)
            
        Returns:
            ESGClassificationResult with prediction
        """
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        predicted_id = int(probs.argmax())
        predicted_label = self.id2label.get(predicted_id, ID_TO_LABEL[predicted_id])
        confidence = float(probs[predicted_id])
        
        return ESGClassificationResult(
            predicted_label=predicted_label,
            predicted_label_vn=ESG_LABELS_VN.get(predicted_label, predicted_label),
            probabilities={self.id2label.get(i, str(i)): float(probs[i]) for i in range(len(probs))},
            confidence=confidence,
            is_esg=(predicted_label != "Not_ESG_Related")
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = MAX_SEQ_LENGTH,
        batch_size: int = 32
    ) -> List[ESGClassificationResult]:
        """
        Classify ESG topics for a batch of texts.
        """
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
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for prob in probs:
                predicted_id = int(prob.argmax())
                predicted_label = self.id2label.get(predicted_id, ID_TO_LABEL[predicted_id])
                
                results.append(ESGClassificationResult(
                    predicted_label=predicted_label,
                    predicted_label_vn=ESG_LABELS_VN.get(predicted_label, predicted_label),
                    probabilities={self.id2label.get(j, str(j)): float(prob[j]) for j in range(len(prob))},
                    confidence=float(prob[predicted_id]),
                    is_esg=(predicted_label != "Not_ESG_Related")
                ))
        
        return results
