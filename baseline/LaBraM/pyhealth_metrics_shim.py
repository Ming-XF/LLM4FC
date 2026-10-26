"""Minimal reimplementation of the pyhealth.metrics entry points used by LaBraM.

pyhealth==1.1.4 (the version pinned in requirements.txt) does not build on
Python 3.12, so the two functions the engine calls are reimplemented here with
numpy + sklearn.  Only the metrics the LLM4FC baselines use are computed.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, cohen_kappa_score,
                             f1_score, roc_auc_score)


def binary_metrics_fn(target, preds, metrics=None, threshold=0.5):
    target = np.asarray(target).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    y_pred = (preds > threshold).astype(int)
    metrics = metrics or ['accuracy']
    results = {}
    if 'roc_auc' in metrics:
        try:
            results['roc_auc'] = float(roc_auc_score(target, preds))
        except ValueError:
            results['roc_auc'] = 0.0
    if 'pr_auc' in metrics:
        try:
            results['pr_auc'] = float(average_precision_score(target, preds))
        except ValueError:
            results['pr_auc'] = 0.0
    if 'accuracy' in metrics:
        results['accuracy'] = float(accuracy_score(target, y_pred))
    if 'balanced_accuracy' in metrics:
        results['balanced_accuracy'] = float(balanced_accuracy_score(target, y_pred))
    if 'f1' in metrics:
        results['f1'] = float(f1_score(target, y_pred))
    return results


def multiclass_metrics_fn(target, preds, metrics=None):
    target = np.asarray(target).reshape(-1)
    preds = np.asarray(preds)
    y_pred = preds.argmax(-1) if preds.ndim > 1 else preds
    metrics = metrics or ['accuracy']
    results = {}
    if 'accuracy' in metrics:
        results['accuracy'] = float(accuracy_score(target, y_pred))
    if 'balanced_accuracy' in metrics:
        results['balanced_accuracy'] = float(balanced_accuracy_score(target, y_pred))
    if 'cohen_kappa' in metrics:
        results['cohen_kappa'] = float(cohen_kappa_score(target, y_pred))
    if 'f1_weighted' in metrics:
        results['f1_weighted'] = float(f1_score(target, y_pred, average='weighted'))
    return results
