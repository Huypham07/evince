"""
EVINCE: Document Analyzer

Document-level ESG-washing analysis through claim-evidence linking.

This is the main component that orchestrates:
1. Sentence classification (CLAIM/EVIDENCE/CONTEXT/NON-ESG)
2. Claim-evidence matching
3. Verification score calculation
4. Document Washing Index computation

Reference:
- ESGSI (Lagasio 2024): Document-level washing index
- A3CG (Ong et al. 2025): Claim-action linking
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

from .sentence_classifier import SentenceClassifierInference, SentenceType
from .evidence_retriever import EvidenceRetriever, RetrievalResult
from .evidence_matcher import ClaimEvidenceMatcherInference, ClaimEvidenceResult


@dataclass
class ClaimAnalysis:
    """Analysis result for a single claim."""
    claim_text: str
    claim_index: int
    verification_score: float
    supporting_evidence: List[ClaimEvidenceResult]
    washing_risk: str  # "LOW", "MEDIUM", "HIGH"
    
    def to_dict(self) -> Dict:
        return {
            "claim_text": self.claim_text,
            "claim_index": self.claim_index,
            "verification_score": self.verification_score,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "washing_risk": self.washing_risk
        }


@dataclass
class DocumentAnalysisResult:
    """Complete analysis result for a document."""
    document_id: str
    bank: str
    year: int
    report_type: str
    
    # Sentence breakdown
    total_sentences: int
    num_claims: int
    num_evidence: int
    num_context: int
    num_non_esg: int
    
    # Claim analysis
    claims: List[ClaimAnalysis]
    
    # Document-level scores
    document_washing_index: float  # 0-1, higher = more washing
    average_verification_score: float  # 0-1, higher = better supported
    
    # Summary
    high_risk_claims: int
    medium_risk_claims: int
    low_risk_claims: int
    
    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "bank": self.bank,
            "year": self.year,
            "report_type": self.report_type,
            "total_sentences": self.total_sentences,
            "num_claims": self.num_claims,
            "num_evidence": self.num_evidence,
            "num_context": self.num_context,
            "num_non_esg": self.num_non_esg,
            "document_washing_index": self.document_washing_index,
            "average_verification_score": self.average_verification_score,
            "high_risk_claims": self.high_risk_claims,
            "medium_risk_claims": self.medium_risk_claims,
            "low_risk_claims": self.low_risk_claims,
            "claims": [c.to_dict() for c in self.claims]
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        risk_level = "LOW" if self.document_washing_index < 0.3 else \
                     "MEDIUM" if self.document_washing_index < 0.5 else "HIGH"
        
        return f"""
Document Analysis Summary
=========================
Bank: {self.bank} | Year: {self.year} | Report: {self.report_type}

Sentence Breakdown:
- Total sentences: {self.total_sentences}
- Claims: {self.num_claims}
- Evidence: {self.num_evidence}
- Context: {self.num_context}
- Non-ESG: {self.num_non_esg}

Washing Analysis:
- Document Washing Index: {self.document_washing_index:.3f} ({risk_level} RISK)
- Average Verification Score: {self.average_verification_score:.3f}
- High Risk Claims: {self.high_risk_claims}
- Medium Risk Claims: {self.medium_risk_claims}
- Low Risk Claims: {self.low_risk_claims}
"""


class DocumentAnalyzer:
    """
    Analyzes ESG-washing at document level through claim-evidence linking.
    
    Pipeline:
    1. Classify all sentences into CLAIM/EVIDENCE/CONTEXT/NON-ESG
    2. For each CLAIM, retrieve and score supporting EVIDENCE
    3. Calculate verification score for each claim
    4. Aggregate to Document Washing Index
    
    Usage:
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(sentences, bank="BIDV", year=2023)
    """
    
    def __init__(
        self,
        sentence_classifier: Optional[SentenceClassifierInference] = None,
        evidence_retriever: Optional[EvidenceRetriever] = None,
        evidence_matcher: Optional[ClaimEvidenceMatcherInference] = None,
        device: str = "auto"
    ):
        """
        Initialize Document Analyzer.
        
        Args:
            sentence_classifier: Classifier for sentence types
            evidence_retriever: Bi-encoder retriever
            evidence_matcher: Cross-encoder for scoring
            device: Device for models
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Initialize components (lazy loading if not provided)
        self.sentence_classifier = sentence_classifier
        self.evidence_retriever = evidence_retriever
        self.evidence_matcher = evidence_matcher
        
        # Washing risk thresholds
        self.high_risk_threshold = 0.3
        self.medium_risk_threshold = 0.6
    
    def _ensure_components(self):
        """Lazy load components if not initialized."""
        if self.sentence_classifier is None:
            self.sentence_classifier = SentenceClassifierInference(device=self.device)
        if self.evidence_retriever is None:
            self.evidence_retriever = EvidenceRetriever(device=self.device)
        if self.evidence_matcher is None:
            self.evidence_matcher = ClaimEvidenceMatcherInference(device=self.device)
    
    def _classify_sentences(
        self,
        sentences: List[str]
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Classify all sentences by type.
        
        Returns:
            Dict mapping type to list of (index, sentence) tuples
        """
        self._ensure_components()
        
        results = self.sentence_classifier.predict_batch(sentences)
        
        classified = defaultdict(list)
        for i, (sentence, result) in enumerate(zip(sentences, results)):
            classified[result.sentence_type].append((i, sentence))
        
        return dict(classified)
    
    def _analyze_claim(
        self,
        claim_idx: int,
        claim_text: str,
        evidence_sentences: List[str],
        top_k: int = 5
    ) -> ClaimAnalysis:
        """
        Analyze a single claim against evidence pool.
        
        Args:
            claim_idx: Index of claim in original document
            claim_text: The claim text
            evidence_sentences: Pool of evidence sentences
            top_k: Number of evidence to retrieve
            
        Returns:
            ClaimAnalysis with verification score
        """
        if not evidence_sentences:
            return ClaimAnalysis(
                claim_text=claim_text,
                claim_index=claim_idx,
                verification_score=0.0,
                supporting_evidence=[],
                washing_risk="HIGH"
            )
        
        # Use cross-encoder for accurate matching
        verification_score, top_evidence = self.evidence_matcher.compute_verification_score(
            claim_text,
            evidence_sentences
        )
        
        # Determine risk level
        if verification_score < self.high_risk_threshold:
            risk = "HIGH"
        elif verification_score < self.medium_risk_threshold:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        return ClaimAnalysis(
            claim_text=claim_text,
            claim_index=claim_idx,
            verification_score=verification_score,
            supporting_evidence=top_evidence,
            washing_risk=risk
        )
    
    def _compute_document_washing_index(
        self,
        claim_analyses: List[ClaimAnalysis]
    ) -> Tuple[float, float]:
        """
        Compute Document Washing Index from claim analyses.
        
        Formula:
        DocWashingIndex = sum((1 - verification_score_i) * weight_i) / sum(weight_i)
        
        Higher index = more washing (claims not well-supported)
        
        Returns:
            Tuple of (washing_index, average_verification_score)
        """
        if not claim_analyses:
            return 0.0, 1.0  # No claims = no washing, but also no verification
        
        # All claims weighted equally for now
        # Future: weight by claim importance
        total_verification = sum(c.verification_score for c in claim_analyses)
        avg_verification = total_verification / len(claim_analyses)
        
        # Washing index is inverse of verification
        washing_index = 1.0 - avg_verification
        
        return washing_index, avg_verification
    
    def analyze_document(
        self,
        sentences: List[str],
        bank: str = "unknown",
        year: int = 0,
        report_type: str = "BCTN",
        document_id: Optional[str] = None,
        top_k_evidence: int = 5
    ) -> DocumentAnalysisResult:
        """
        Analyze a complete document for ESG-washing.
        
        Args:
            sentences: List of sentences from the document
            bank: Bank name
            year: Report year
            report_type: Type of report (BCTN, BCTDML, etc.)
            document_id: Optional document identifier
            top_k_evidence: Number of evidence to retrieve per claim
            
        Returns:
            DocumentAnalysisResult with full analysis
        """
        self._ensure_components()
        
        if document_id is None:
            document_id = f"{bank}_{year}_{report_type}"
        
        # Step 1: Classify all sentences
        classified = self._classify_sentences(sentences)
        
        claims = classified.get("CLAIM", [])
        evidence = classified.get("EVIDENCE", [])
        context = classified.get("CONTEXT", [])
        non_esg = classified.get("NON_ESG", [])
        
        evidence_texts = [text for _, text in evidence]
        
        # Step 2: Index evidence for retrieval
        if evidence_texts:
            self.evidence_retriever.index_evidence(evidence_texts)
        
        # Step 3: Analyze each claim
        claim_analyses = []
        for claim_idx, claim_text in claims:
            analysis = self._analyze_claim(
                claim_idx, claim_text, evidence_texts, top_k_evidence
            )
            claim_analyses.append(analysis)
        
        # Step 4: Compute document-level metrics
        washing_index, avg_verification = self._compute_document_washing_index(claim_analyses)
        
        # Count risk levels
        high_risk = sum(1 for c in claim_analyses if c.washing_risk == "HIGH")
        medium_risk = sum(1 for c in claim_analyses if c.washing_risk == "MEDIUM")
        low_risk = sum(1 for c in claim_analyses if c.washing_risk == "LOW")
        
        return DocumentAnalysisResult(
            document_id=document_id,
            bank=bank,
            year=year,
            report_type=report_type,
            total_sentences=len(sentences),
            num_claims=len(claims),
            num_evidence=len(evidence),
            num_context=len(context),
            num_non_esg=len(non_esg),
            claims=claim_analyses,
            document_washing_index=washing_index,
            average_verification_score=avg_verification,
            high_risk_claims=high_risk,
            medium_risk_claims=medium_risk,
            low_risk_claims=low_risk
        )
    
    def analyze_multiple_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[DocumentAnalysisResult]:
        """
        Analyze multiple documents.
        
        Args:
            documents: List of dicts with keys:
                - sentences: List[str]
                - bank: str
                - year: int
                - report_type: str (optional)
                
        Returns:
            List of DocumentAnalysisResult
        """
        results = []
        
        for doc in documents:
            result = self.analyze_document(
                sentences=doc["sentences"],
                bank=doc.get("bank", "unknown"),
                year=doc.get("year", 0),
                report_type=doc.get("report_type", "BCTN")
            )
            results.append(result)
        
        return results


def create_document_analyzer(device: str = "auto") -> DocumentAnalyzer:
    """
    Factory function to create Document Analyzer.
    
    Args:
        device: Device for models
        
    Returns:
        DocumentAnalyzer instance
    """
    return DocumentAnalyzer(device=device)
