"""
EVINCE: Claim-Evidence Module

Document-level analysis for ESG-washing detection through claim-evidence linking.

Components:
- SentenceClassifier: Classify sentences as CLAIM/EVIDENCE/CONTEXT/NON-ESG
- EvidenceMatcher: Cross-encoder for claim-evidence pair scoring
- EvidenceRetriever: Retrieve relevant evidence for claims
- DocumentAnalyzer: Document-level washing analysis
"""

from .sentence_classifier import (
    SentenceClassifier,
    SentenceClassifierInference,
    SentenceType,
    SENTENCE_TYPES,
    SentenceClassificationResult
)

from .evidence_matcher import (
    ClaimEvidenceMatcher,
    ClaimEvidenceMatcherInference,
    ClaimEvidenceResult,
    create_claim_evidence_matcher
)

from .evidence_retriever import (
    EvidenceRetriever,
    TwoStageRetriever,
    RetrievalResult
)

from .document_analyzer import (
    DocumentAnalyzer,
    DocumentAnalysisResult,
    ClaimAnalysis,
    create_document_analyzer
)

__all__ = [
    # Sentence Classifier
    "SentenceClassifier",
    "SentenceClassifierInference",
    "SentenceType",
    "SENTENCE_TYPES",
    "SentenceClassificationResult",
    # Evidence Matcher
    "ClaimEvidenceMatcher",
    "ClaimEvidenceMatcherInference",
    "ClaimEvidenceResult",
    "create_claim_evidence_matcher",
    # Evidence Retriever
    "EvidenceRetriever",
    "TwoStageRetriever",
    "RetrievalResult",
    # Document Analyzer
    "DocumentAnalyzer",
    "DocumentAnalysisResult",
    "ClaimAnalysis",
    "create_document_analyzer"
]
