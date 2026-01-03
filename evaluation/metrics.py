"""
EVINCE: Evaluation Metrics

Metrics for ESG-washing detection evaluation:
- Classification metrics (accuracy, F1, precision, recall)
- Document-level metrics (Document Washing Index correlation)
- Calibration metrics (ECE)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class_f1: Dict[str, float]
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "per_class_f1": self.per_class_f1
        }


def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Compute accuracy."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if y_true else 0.0


def compute_precision_recall_f1(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """
    Compute per-class precision, recall, F1.
    
    Returns:
        Tuple of (precision_dict, recall_dict, f1_dict)
    """
    precision = {}
    recall = {}
    f1 = {}
    
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precision[c] = p
        recall[c] = r
        f1[c] = f
    
    return precision, recall, f1


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int
) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None
) -> ClassificationMetrics:
    """
    Compute all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of label names
        
    Returns:
        ClassificationMetrics dataclass
    """
    num_classes = max(max(y_true), max(y_pred)) + 1
    
    accuracy = compute_accuracy(y_true, y_pred)
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred, num_classes)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    
    # Macro averages
    macro_precision = np.mean(list(precision.values()))
    macro_recall = np.mean(list(recall.values()))
    macro_f1 = np.mean(list(f1.values()))
    
    # Per-class F1 with names
    if label_names:
        per_class_f1 = {label_names[c]: f1[c] for c in range(num_classes)}
    else:
        per_class_f1 = {str(c): f1[c] for c in range(num_classes)}
    
    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_class_f1=per_class_f1,
        confusion_matrix=cm
    )


def compute_ece(
    y_true: List[int],
    y_probs: np.ndarray,
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Measures how well predicted probabilities match actual accuracies.
    Lower is better.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (N x num_classes)
        num_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    # Get predicted class and confidence
    y_pred = np.argmax(y_probs, axis=1)
    confidences = np.max(y_probs, axis=1)
    accuracies = (y_pred == np.array(y_true)).astype(float)
    
    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    
    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            avg_confidence = np.mean(confidences[in_bin])
            avg_accuracy = np.mean(accuracies[in_bin])
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return ece


def compute_cohens_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """
    Compute Cohen's Kappa coefficient.
    
    Measures agreement between predictions and labels,
    accounting for chance agreement.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Kappa coefficient (-1 to 1, higher is better)
    """
    n = len(y_true)
    
    # Observed agreement
    po = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    
    # Expected agreement by chance
    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    pe = sum(
        (true_counts[c] / n) * (pred_counts[c] / n)
        for c in set(y_true) | set(y_pred)
    )
    
    # Kappa
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def print_classification_report(
    metrics: ClassificationMetrics,
    title: str = "Classification Report"
):
    """Print formatted classification report."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Accuracy:        {metrics.accuracy:.4f}")
    print(f"Macro F1:        {metrics.macro_f1:.4f}")
    print(f"Macro Precision: {metrics.macro_precision:.4f}")
    print(f"Macro Recall:    {metrics.macro_recall:.4f}")
    print(f"\nPer-class F1:")
    for label, f1 in metrics.per_class_f1.items():
        print(f"  {label}: {f1:.4f}")
    print(f"{'='*50}\n")
