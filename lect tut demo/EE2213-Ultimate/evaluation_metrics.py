import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_regression_metrics(y_true, y_pred, sample_weight=None) -> Dict[str, float]:
    """Return standard regression metrics using sklearn.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_pred : array-like, shape (n_samples,)
    sample_weight : optional sample weights

    Returns
    -------
    dict with keys: 'mse', 'mae'
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mse": float(mean_squared_error(y_true, y_pred, sample_weight=sample_weight)),
        "mae": float(mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)),
    }


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else float("nan")


def compute_binary_classification_metrics(
    y_true,
    y_pred,
    positive_label=1,
) -> Dict[str, float]:
    """Binary metrics: confusion matrix and derived rates.

    Accepts arbitrary label values; set positive_label to choose the positive class.

    Returns dict with:
    - tp, fn, fp, tn
    - accuracy, precision, recall
    - tpr (recall), fnr, tnr, fpr
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Determine negative label as any label not equal to positive_label
    # Use confusion_matrix with explicit label order [neg, pos]
    neg_label = None
    for v in np.unique(y_true):
        if v != positive_label:
            neg_label = v
            break
    if neg_label is None:
        # Degenerate case: only positive labels present
        neg_label = 0 if positive_label != 0 else -1

    cm = confusion_matrix(y_true, y_pred, labels=[neg_label, positive_label])
    # cm layout [[tn, fp], [fn, tp]] with our labels order
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    # Use zero_division=0 to avoid exceptions; we also compute rates manually
    prec = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    # Derived rates
    tpr = _safe_div(tp, tp + fn)  # same as recall
    fnr = _safe_div(fn, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, tn + fp)

    return {
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "tpr": float(tpr),
        "fnr": float(fnr),
        "tnr": float(tnr),
        "fpr": float(fpr),
    }


def compute_multiclass_metrics(y_true, y_pred) -> Dict[str, object]:
    """Basic multi-class summary: confusion matrix, overall accuracy,
    macro-averaged precision and recall.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    # macro averages treat all classes equally
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "confusion_matrix": cm,
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "classes": np.unique(y_true),
    }


def print_regression_metrics(metrics: Dict[str, float], decimals: int = 4) -> None:
    np.set_printoptions(precision=decimals, suppress=True)
    print(f"MSE: {metrics['mse']:.{decimals}f}")
    print(f"MAE: {metrics['mae']:.{decimals}f}")


def print_binary_classification_metrics(metrics: Dict[str, float], decimals: int = 4) -> None:
    np.set_printoptions(precision=decimals, suppress=True)
    print("Confusion Matrix (rows: actual class, columns: predicted class. Diagonal are correct predictions):")
    print("Confusion Matrix (TP, FN, FP, TN):")
    print(f"  TP: {metrics['tp']:.{decimals}f}  FN: {metrics['fn']:.{decimals}f}")
    print(f"  FP: {metrics['fp']:.{decimals}f}  TN: {metrics['tn']:.{decimals}f}")
    print(f"Accuracy:  {metrics['accuracy']:.{decimals}f}")
    print(f"Precision: {metrics['precision']:.{decimals}f}")
    print(f"Recall:    {metrics['recall']:.{decimals}f}")
    print(f"TPR:       {metrics['tpr']:.{decimals}f}")
    print(f"FNR:       {metrics['fnr']:.{decimals}f}")
    print(f"TNR:       {metrics['tnr']:.{decimals}f}")
    print(f"FPR:       {metrics['fpr']:.{decimals}f}")


def print_multiclass_metrics(summary: Dict[str, object], decimals: int = 4) -> None:
    np.set_printoptions(precision=decimals, suppress=True)
    print("Confusion Matrix:")
    print(summary["confusion_matrix"])  # matrix prints nicely with np options
    print(f"Accuracy:  {summary['accuracy']:.{decimals}f}")
    print(f"Precision (macro): {summary['precision_macro']:.{decimals}f}")
    print(f"Recall (macro):    {summary['recall_macro']:.{decimals}f}")
