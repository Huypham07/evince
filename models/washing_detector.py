"""
EVINCE: Washing Detector with Attention

Deep Learning model for ESG-washing detection with explainability via attention.
Optimized for paragraph-level input with max_length=512 and full fine-tuning.

Washing Types:
- NOT_WASHING: Genuine ESG claim
- VAGUE_COMMITMENT: Cam kết mơ hồ
- SELECTIVE_DISCLOSURE: Tiết lộ chọn lọc  
- SYMBOLIC_ACTION: Hành động biểu tượng
- DECOUPLING: Tách rời lời nói-hành động
- MISLEADING_METRICS: Số liệu gây hiểu lầm
- FUTURE_DEFLECTION: Trì hoãn tương lai

Reference:
- Lagasio (2024): ESGSI - ESG Washing Severity Index
- A3CG (Ong et al. 2025): Aspect-Action Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed.")


# Maximum sequence length for paragraph-level input
MAX_SEQ_LENGTH = 512

# Washing Type Labels
WASHING_LABELS = [
    "NOT_WASHING",
    "VAGUE_COMMITMENT",
    "SELECTIVE_DISCLOSURE",
    "SYMBOLIC_ACTION",
    "DECOUPLING",
    "MISLEADING_METRICS",
    "FUTURE_DEFLECTION"
]

WASHING_LABEL_TO_ID = {label: i for i, label in enumerate(WASHING_LABELS)}
WASHING_ID_TO_LABEL = {i: label for i, label in enumerate(WASHING_LABELS)}

# Vietnamese names
WASHING_LABELS_VN = {
    "NOT_WASHING": "Không washing",
    "VAGUE_COMMITMENT": "Cam kết mơ hồ",
    "SELECTIVE_DISCLOSURE": "Tiết lộ chọn lọc",
    "SYMBOLIC_ACTION": "Hành động biểu tượng",
    "DECOUPLING": "Tách rời lời nói-hành động",
    "MISLEADING_METRICS": "Số liệu gây hiểu lầm",
    "FUTURE_DEFLECTION": "Trì hoãn tương lai"
}


@dataclass
class WashingDetectionResult:
    """Result from washing detection."""
    is_washing: bool
    washing_type: str
    washing_type_vn: str
    probabilities: Dict[str, float]
    confidence: float
    attention_weights: Optional[np.ndarray] = None
    highlighted_tokens: Optional[List[Tuple[str, float]]] = None
    
    def to_dict(self) -> Dict:
        return {
            "is_washing": self.is_washing,
            "washing_type": self.washing_type,
            "washing_type_vn": self.washing_type_vn,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "highlighted_tokens": self.highlighted_tokens
        }


class WashingDetector(nn.Module):
    """
    End-to-end Deep Learning washing detector with attention for explainability.
    
    Architecture:
    - Encoder: PhoBERT-base (768-dim)
    - Attention: Multi-head self-attention (8 heads)
    - Classifier: MLP head with 7 washing types
    
    Optimized for paragraph-level input with full fine-tuning enabled by default.
    The attention weights can be used to explain which tokens contribute
    most to the washing detection decision.
    
    Usage:
        model = WashingDetector()
        logits, attn_weights = model(input_ids, attention_mask)
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_classes: int = 7,
        num_attention_heads: int = 8,
        dropout_rate: float = 0.3,
        freeze_bert_layers: int = 0  # Default: full fine-tuning for paragraph understanding
    ):
        """
        Initialize Washing Detector.
        
        Args:
            model_name: HuggingFace model identifier
            num_classes: Number of washing types (default 7)
            num_attention_heads: Number of attention heads for explainability
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
        
        # Self-attention layer for explainability
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization after attention
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Classification head - deeper MLP for better paragraph understanding
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
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: Classification logits, shape (batch, num_classes)
            attn_weights: Attention weights, shape (batch, seq_len, seq_len)
        """
        # Get BERT output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Create key padding mask for attention
        # True means the position should be ignored (padding)
        key_padding_mask = ~attention_mask.bool()
        
        # Self-attention
        attn_output, attn_weights = self.attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True  # Average over attention heads
        )
        
        # Residual connection + layer norm
        hidden_states = self.layer_norm(hidden_states + attn_output)
        
        # Use [CLS] token for classification
        pooled_output = hidden_states[:, 0, :]  # (batch, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits, attn_weights
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get probability distribution over classes."""
        logits, _ = self.forward(input_ids, attention_mask, return_attention=False)
        return F.softmax(logits, dim=-1)


class WashingDetectorInference:
    """
    Inference wrapper for Washing Detector.
    
    Provides:
    - Single and batch prediction
    - Token-level attention highlighting for explainability
    
    Optimized for paragraph-level input with max_length=512.
    
    Usage:
        detector = WashingDetectorInference()
        result = detector.predict("Ngân hàng cam kết phát triển bền vững...")
    """
    
    def __init__(
        self,
        model: Optional[WashingDetector] = None,
        model_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-loaded WashingDetector model
            model_path: Path to saved model checkpoint
            device: "cpu", "cuda", or "auto"
        """
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
            self.model = WashingDetector().to(self.device)
        
        self.model.eval()
    
    def _load_model(self, path: str) -> WashingDetector:
        """Load model from checkpoint."""
        model = WashingDetector()
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    def _get_highlighted_tokens(
        self,
        tokens: List[str],
        attention_weights: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get top-K tokens with highest attention weights.
        
        Args:
            tokens: List of token strings
            attention_weights: Attention weights for [CLS] token
            top_k: Number of top tokens to return
            
        Returns:
            List of (token, attention_score) tuples
        """
        # Get attention from [CLS] token to all other tokens
        cls_attention = attention_weights[0, :]  # First row is [CLS] attention
        
        # Get top-K indices
        top_indices = np.argsort(cls_attention)[-top_k:][::-1]
        
        highlighted = []
        for idx in top_indices:
            if idx < len(tokens):
                highlighted.append((tokens[idx], float(cls_attention[idx])))
        
        return highlighted
    
    @torch.no_grad()
    def predict(
        self,
        text: str,
        max_length: int = MAX_SEQ_LENGTH,
        return_attention: bool = True
    ) -> WashingDetectionResult:
        """
        Detect washing for a single text (sentence or paragraph).
        
        Args:
            text: Text in Vietnamese (can be sentence or paragraph)
            max_length: Maximum sequence length (default 512 for paragraphs)
            return_attention: Whether to return highlighted tokens
            
        Returns:
            WashingDetectionResult with prediction and explanation
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
        
        # Get predictions with attention
        logits, attn_weights = self.model(
            input_ids, 
            attention_mask, 
            return_attention=return_attention
        )
        
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get predicted class
        predicted_id = int(probs.argmax())
        predicted_label = WASHING_ID_TO_LABEL[predicted_id]
        confidence = float(probs[predicted_id])
        
        # Get highlighted tokens
        highlighted_tokens = None
        attn_np = None
        if return_attention and attn_weights is not None:
            attn_np = attn_weights.cpu().numpy()[0]
            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[0].cpu().numpy()
            )
            highlighted_tokens = self._get_highlighted_tokens(tokens, attn_np)
        
        return WashingDetectionResult(
            is_washing=(predicted_label != "NOT_WASHING"),
            washing_type=predicted_label,
            washing_type_vn=WASHING_LABELS_VN[predicted_label],
            probabilities={label: float(probs[i]) for i, label in enumerate(WASHING_LABELS)},
            confidence=confidence,
            attention_weights=attn_np,
            highlighted_tokens=highlighted_tokens
        )
    
    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        max_length: int = MAX_SEQ_LENGTH,
        batch_size: int = 32
    ) -> List[WashingDetectionResult]:
        """
        Detect washing for a batch of texts.
        
        Args:
            texts: List of texts (sentences or paragraphs)
            max_length: Maximum sequence length (default 512)
            batch_size: Batch size for inference
            
        Returns:
            List of WashingDetectionResult
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
            
            # No attention for batch (too memory intensive)
            logits, _ = self.model(input_ids, attention_mask, return_attention=False)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            for prob in probs:
                predicted_id = int(prob.argmax())
                predicted_label = WASHING_ID_TO_LABEL[predicted_id]
                
                results.append(WashingDetectionResult(
                    is_washing=(predicted_label != "NOT_WASHING"),
                    washing_type=predicted_label,
                    washing_type_vn=WASHING_LABELS_VN[predicted_label],
                    probabilities={label: float(prob[k]) for k, label in enumerate(WASHING_LABELS)},
                    confidence=float(prob[predicted_id]),
                    attention_weights=None,
                    highlighted_tokens=None
                ))
        
        return results


def create_washing_detector(
    freeze_bert_layers: int = 0,
    num_attention_heads: int = 8,
    dropout_rate: float = 0.3
) -> WashingDetector:
    """
    Factory function to create Washing Detector.
    
    Args:
        freeze_bert_layers: Number of BERT layers to freeze (default 0 = full fine-tuning)
        num_attention_heads: Number of attention heads
        dropout_rate: Dropout rate
        
    Returns:
        WashingDetector model ready for training
    """
    return WashingDetector(
        freeze_bert_layers=freeze_bert_layers,
        num_attention_heads=num_attention_heads,
        dropout_rate=dropout_rate
    )
