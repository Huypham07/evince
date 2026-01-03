"""
EVINCE: Evaluation Package

Contains:
- Classification metrics (accuracy, F1, precision, recall)
- Calibration metrics (ECE)
- Document-level metrics
"""

from .metrics import (
    ClassificationMetrics,
    compute_accuracy,
    compute_precision_recall_f1,
    compute_confusion_matrix,
    compute_classification_metrics,
    compute_ece,
    compute_cohens_kappa,
    print_classification_report
)

__all__ = [
    "ClassificationMetrics",
    "compute_accuracy",
    "compute_precision_recall_f1",
    "compute_confusion_matrix",
    "compute_classification_metrics",
    "compute_ece",
    "compute_cohens_kappa",
    "print_classification_report"
]
