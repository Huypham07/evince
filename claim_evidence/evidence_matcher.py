"""
EVINCE: Claim-Evidence Cross-Encoder

Cross-encoder architecture for scoring claim-evidence pairs.
More accurate than bi-encoder for matching but slower.

Input: [CLS] claim [SEP] evidence [SEP]
Output: matching_score (0-1)

Reference:
- Cross-Encoder architecture from Sentence-BERT (Reimers et al. 2019)
- A3CG (Ong et al. 2025): Aspect-Action matching
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


@dataclass
class ClaimEvidenceResult:
    """Result from claim-evidence matching."""
    claim: str
    evidence: str
    match_score: float
    is_supporting: bool  # True if score > threshold
    
    def to_dict(self) -> Dict:
        return {
            "claim": self.claim,
            "evidence": self.evidence,
            "match_score": self.match_score,
            "is_supporting": self.is_supporting
        }


class ClaimEvidenceMatcher(nn.Module):
    """
    Cross-encoder to score how well an evidence supports a claim.
    
    Architecture:
    - Encoder: PhoBERT-base
    - Input: Concatenated [CLS] claim [SEP] evidence [SEP]
    - Output: Matching score (0-1) via sigmoid
    
    This model learns to understand semantic relationships between
    ESG claims and their potential supporting evidence.
    
    Usage:
        model = ClaimEvidenceMatcher()
        score = model(input_ids, attention_mask)  # -> (batch, 1)
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        dropout_rate: float = 0.2,
        freeze_bert_layers: int = 6
    ):
        """
        Initialize Claim-Evidence Matcher.
        
        Args:
            model_name: HuggingFace model identifier
            dropout_rate: Dropout probability
            freeze_bert_layers: Number of BERT layers to freeze
        """
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        
        if freeze_bert_layers > 0:
            self._freeze_bert_layers(freeze_bert_layers)
        
        # Scoring head
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.scorer.modules():
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
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs for [CLS] claim [SEP] evidence [SEP]
            attention_mask: Attention mask
            
        Returns:
            Matching scores, shape (batch, 1)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        scores = self.scorer(pooled)
        return scores


class ClaimEvidenceMatcherInference:
    """
    Inference wrapper for Claim-Evidence Matcher.
    
    Provides:
    - Score single claim-evidence pair
    - Find best evidence for a claim from a pool
    - Batch scoring for efficiency
    
    Usage:
        matcher = ClaimEvidenceMatcherInference()
        score = matcher.score("claim text", "evidence text")
        best_evidence = matcher.find_best_evidence(claim, evidence_pool, top_k=3)
    """
    
    def __init__(
        self,
        model: Optional[ClaimEvidenceMatcher] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        match_threshold: float = 0.6
    ):
        """
        Initialize inference engine.
        
        Args:
            model: Pre-loaded model
            model_path: Path to saved checkpoint
            device: "cpu", "cuda", or "auto"
            match_threshold: Threshold for considering evidence as supporting
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.match_threshold = match_threshold
        
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = ClaimEvidenceMatcher().to(self.device)
        
        self.model.eval()
    
    def _load_model(self, path: str) -> ClaimEvidenceMatcher:
        model = ClaimEvidenceMatcher()
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)
    
    @torch.no_grad()
    def score(
        self,
        claim: str,
        evidence: str,
        max_length: int = 256
    ) -> ClaimEvidenceResult:
        """
        Score a claim-evidence pair.
        
        Args:
            claim: The ESG claim text
            evidence: The potential supporting evidence
            max_length: Maximum sequence length
            
        Returns:
            ClaimEvidenceResult with matching score
        """
        # Tokenize with pair encoding
        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        score = self.model(input_ids, attention_mask)
        score = float(score.cpu().numpy()[0, 0])
        
        return ClaimEvidenceResult(
            claim=claim,
            evidence=evidence,
            match_score=score,
            is_supporting=(score >= self.match_threshold)
        )
    
    @torch.no_grad()
    def score_batch(
        self,
        claims: List[str],
        evidences: List[str],
        max_length: int = 256,
        batch_size: int = 32
    ) -> List[ClaimEvidenceResult]:
        """
        Score multiple claim-evidence pairs.
        
        Args:
            claims: List of claim texts
            evidences: List of evidence texts (same length as claims)
            max_length: Maximum sequence length
            batch_size: Batch size
            
        Returns:
            List of ClaimEvidenceResult
        """
        assert len(claims) == len(evidences), "claims and evidences must have same length"
        
        results = []
        
        for i in range(0, len(claims), batch_size):
            batch_claims = claims[i:i + batch_size]
            batch_evidences = evidences[i:i + batch_size]
            
            encoding = self.tokenizer(
                batch_claims,
                batch_evidences,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            scores = self.model(input_ids, attention_mask)
            scores = scores.cpu().numpy()[:, 0]
            
            for j, score in enumerate(scores):
                results.append(ClaimEvidenceResult(
                    claim=batch_claims[j],
                    evidence=batch_evidences[j],
                    match_score=float(score),
                    is_supporting=(score >= self.match_threshold)
                ))
        
        return results
    
    @torch.no_grad()
    def find_best_evidence(
        self,
        claim: str,
        evidence_pool: List[str],
        top_k: int = 3,
        max_length: int = 256
    ) -> List[ClaimEvidenceResult]:
        """
        Find the best matching evidence for a claim from a pool.
        
        Args:
            claim: The ESG claim text
            evidence_pool: List of potential evidence sentences
            top_k: Number of top results to return
            max_length: Maximum sequence length
            
        Returns:
            Top-K ClaimEvidenceResult sorted by score (descending)
        """
        if not evidence_pool:
            return []
        
        # Create pairs: same claim with each evidence
        claims = [claim] * len(evidence_pool)
        
        results = self.score_batch(claims, evidence_pool, max_length)
        
        # Sort by score descending
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        return results[:top_k]
    
    @torch.no_grad()
    def compute_verification_score(
        self,
        claim: str,
        evidence_pool: List[str],
        max_length: int = 256
    ) -> Tuple[float, List[ClaimEvidenceResult]]:
        """
        Compute verification score for a claim based on evidence pool.
        
        The verification score indicates how well-supported a claim is.
        
        Args:
            claim: The ESG claim text
            evidence_pool: List of potential evidence sentences
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (verification_score, top_evidence_list)
        """
        if not evidence_pool:
            return 0.0, []
        
        top_evidence = self.find_best_evidence(claim, evidence_pool, top_k=5, max_length=max_length)
        
        if not top_evidence:
            return 0.0, []
        
        # Verification score = weighted average of top-K evidence scores
        # Higher weight for best matches
        weights = [0.4, 0.25, 0.15, 0.12, 0.08][:len(top_evidence)]
        total_weight = sum(weights)
        
        verification_score = sum(
            w * e.match_score for w, e in zip(weights, top_evidence)
        ) / total_weight
        
        return verification_score, top_evidence


def create_claim_evidence_matcher(
    freeze_bert_layers: int = 6,
    dropout_rate: float = 0.2
) -> ClaimEvidenceMatcher:
    """
    Factory function to create Claim-Evidence Matcher.
    
    Args:
        freeze_bert_layers: Number of BERT layers to freeze
        dropout_rate: Dropout rate
        
    Returns:
        ClaimEvidenceMatcher model
    """
    return ClaimEvidenceMatcher(
        freeze_bert_layers=freeze_bert_layers,
        dropout_rate=dropout_rate
    )
