"""Evaluation metrics for Business Entity Resolution Challenge.

Calculates Macro-averaged F_0.5 score across all Source 1 entities:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Singletons (Source 1 entities with no true matches):
    - Score is 1.0 if predicted set is empty.
    - Score is 0.0 if any match is predicted.
"""

from typing import Dict, Set, Iterable, Tuple


def compute_entity_f05(
    true_matches: Set[str],
    pred_matches: Set[str]
) -> Tuple[float, float, float]:
    """Compute (f05, precision, recall) for a single Source 1 entity."""
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Singleton case (no true matches)
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0  # Predicted false positives on singleton

    # Entity with matches, but model predicted nothing
    if n_pred == 0:
        return 0.0, 1.0, 0.0

    # Entity with matches and model made predictions
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom

    return f05, precision, recall


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Compute Macro F_0.5, Macro Precision, Macro Recall, and Singleton Accuracy.

    Args:
        ground_truth: mapping of source1_entity_id -> set of true matching entity_ids.
        predictions: mapping of source1_entity_id -> set of predicted matching entity_ids.

    Returns:
        dict with macro_f05, macro_precision, macro_recall, singleton_acc, total_entities.
    """
    total = len(ground_truth)
    if total == 0:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "singleton_acc": 0.0, "count": 0}

    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    singleton_count = 0
    singleton_correct = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05, prec, rec = compute_entity_f05(true_set, pred_set)
        total_f05 += f05
        total_prec += prec
        total_rec += rec

        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1

    return {
        "macro_f05": total_f05 / total,
        "macro_precision": total_prec / total,
        "macro_recall": total_rec / total,
        "singleton_acc": (singleton_correct / singleton_count) if singleton_count > 0 else 1.0,
        "total_entities": total,
        "singletons": singleton_count,
    }
