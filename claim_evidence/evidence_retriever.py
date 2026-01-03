"""
EVINCE: Evidence Retriever

Semantic search to retrieve relevant evidence for ESG claims.
Uses PhoBERT embeddings for bi-encoder retrieval (faster than cross-encoder).

Two-stage retrieval:
1. Bi-encoder: Fast candidate retrieval using embedding similarity
2. Cross-encoder: Re-rank top candidates for accuracy

Reference:
- Sentence-BERT (Reimers et al. 2019)
- Dense Passage Retrieval (Karpukhin et al. 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Result from evidence retrieval."""
    evidence_text: str
    similarity_score: float
    evidence_index: int
    
    def to_dict(self) -> Dict:
        return {
            "evidence_text": self.evidence_text,
            "similarity_score": self.similarity_score,
            "evidence_index": self.evidence_index
        }


class EvidenceRetriever:
    """
    Bi-encoder based evidence retriever using PhoBERT embeddings.
    
    Uses mean pooling of token embeddings for sentence representation.
    Retrieves evidence using cosine similarity.
    
    Usage:
        retriever = EvidenceRetriever()
        retriever.index_evidence(evidence_sentences)
        results = retriever.retrieve(claim, top_k=5)
    """
    
    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        device: str = "auto"
    ):
        """
        Initialize Evidence Retriever.
        
        Args:
            model_name: HuggingFace model identifier
            device: "cpu", "cuda", or "auto"
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.encoder.eval()
        
        # Evidence index
        self.evidence_texts: List[str] = []
        self.evidence_embeddings: Optional[torch.Tensor] = None
    
    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling of token embeddings, respecting attention mask.
        
        Args:
            model_output: Last hidden state from BERT (batch, seq, hidden)
            attention_mask: Attention mask (batch, seq)
            
        Returns:
            Sentence embeddings (batch, hidden)
        """
        # Expand attention mask for broadcasting
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        
        # Sum embeddings, weighted by mask
        sum_embeddings = torch.sum(model_output * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def encode(
        self,
        texts: List[str],
        max_length: int = 256,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
            
        Returns:
            Embeddings tensor (num_texts, hidden_size)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoding = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def index_evidence(
        self,
        evidence_texts: List[str],
        max_length: int = 256,
        batch_size: int = 32
    ):
        """
        Index evidence sentences for retrieval.
        
        Args:
            evidence_texts: List of evidence sentences
            max_length: Maximum sequence length
            batch_size: Batch size for encoding
        """
        self.evidence_texts = evidence_texts
        self.evidence_embeddings = self.encode(evidence_texts, max_length, batch_size)
    
    @torch.no_grad()
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_length: int = 256,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Retrieve top-K evidence for a query/claim.
        
        Args:
            query: Query text (claim)
            top_k: Number of results to return
            max_length: Maximum sequence length
            min_score: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult sorted by score
        """
        if self.evidence_embeddings is None or len(self.evidence_texts) == 0:
            raise ValueError("No evidence indexed. Call index_evidence first.")
        
        # Encode query
        query_embedding = self.encode([query], max_length)[0]  # (hidden,)
        
        # Compute cosine similarity
        similarities = torch.matmul(self.evidence_embeddings, query_embedding)
        similarities = similarities.cpu().numpy()
        
        # Get top-K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                results.append(RetrievalResult(
                    evidence_text=self.evidence_texts[idx],
                    similarity_score=score,
                    evidence_index=int(idx)
                ))
        
        return results
    
    @torch.no_grad()
    def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        max_length: int = 256
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve evidence for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
            max_length: Maximum sequence length
            
        Returns:
            List of result lists, one per query
        """
        if self.evidence_embeddings is None or len(self.evidence_texts) == 0:
            raise ValueError("No evidence indexed. Call index_evidence first.")
        
        # Encode all queries
        query_embeddings = self.encode(queries, max_length)  # (num_queries, hidden)
        
        # Compute all similarities at once
        similarities = torch.matmul(query_embeddings, self.evidence_embeddings.T)
        similarities = similarities.cpu().numpy()  # (num_queries, num_evidence)
        
        all_results = []
        for q_idx in range(len(queries)):
            query_sims = similarities[q_idx]
            top_indices = np.argsort(query_sims)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append(RetrievalResult(
                    evidence_text=self.evidence_texts[idx],
                    similarity_score=float(query_sims[idx]),
                    evidence_index=int(idx)
                ))
            all_results.append(results)
        
        return all_results
    
    def clear_index(self):
        """Clear the evidence index."""
        self.evidence_texts = []
        self.evidence_embeddings = None


class TwoStageRetriever:
    """
    Two-stage retrieval: Bi-encoder + Cross-encoder re-ranking.
    
    1. Bi-encoder retrieves top-K candidates (fast)
    2. Cross-encoder re-ranks candidates (accurate)
    
    This provides both speed and accuracy for production use.
    """
    
    def __init__(
        self,
        bi_encoder: Optional[EvidenceRetriever] = None,
        cross_encoder = None,  # ClaimEvidenceMatcherInference
        device: str = "auto"
    ):
        """
        Initialize two-stage retriever.
        
        Args:
            bi_encoder: EvidenceRetriever for first stage
            cross_encoder: ClaimEvidenceMatcherInference for re-ranking
            device: Device for models
        """
        self.bi_encoder = bi_encoder or EvidenceRetriever(device=device)
        self.cross_encoder = cross_encoder  # Optional
    
    def index_evidence(self, evidence_texts: List[str], **kwargs):
        """Index evidence in bi-encoder."""
        self.bi_encoder.index_evidence(evidence_texts, **kwargs)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        first_stage_k: int = 20,
        use_reranking: bool = True
    ) -> List[RetrievalResult]:
        """
        Two-stage retrieval.
        
        Args:
            query: Query text
            top_k: Final number of results
            first_stage_k: Number of candidates from bi-encoder
            use_reranking: Whether to use cross-encoder re-ranking
            
        Returns:
            List of RetrievalResult
        """
        # Stage 1: Bi-encoder retrieval
        candidates = self.bi_encoder.retrieve(query, top_k=first_stage_k)
        
        if not use_reranking or self.cross_encoder is None:
            return candidates[:top_k]
        
        # Stage 2: Cross-encoder re-ranking
        candidate_texts = [c.evidence_text for c in candidates]
        reranked = self.cross_encoder.find_best_evidence(query, candidate_texts, top_k=top_k)
        
        # Convert to RetrievalResult with original indices
        text_to_idx = {c.evidence_text: c.evidence_index for c in candidates}
        
        results = []
        for r in reranked:
            results.append(RetrievalResult(
                evidence_text=r.evidence,
                similarity_score=r.match_score,
                evidence_index=text_to_idx.get(r.evidence, -1)
            ))
        
        return results
