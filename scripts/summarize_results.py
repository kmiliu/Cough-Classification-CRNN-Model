"""Recompute aggregate metrics from committed predictions using the standard library."""
import argparse
import csv
import hashlib
import json
import math
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / 'output/crnn_model_results_5/test_predictions_f1opt_v2.csv'


def summarize(path=DEFAULT_PREDICTIONS):
    path = Path(path)
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {'y_true', 'y_pred_prob', 'y_pred_label'}:
            raise ValueError('Expected only y_true, y_pred_prob, y_pred_label columns')
        rows = [(int(r['y_true']), float(r['y_pred_prob']), int(r['y_pred_label'])) for r in reader]
    if not rows or any(y not in (0, 1) or pred not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1 for y, score, pred in rows):
        raise ValueError('Predictions must be nonempty, finite, binary-labeled probabilities')
    positives = sum(y for y, _, _ in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        raise ValueError('Both classes are required for ROC-AUC')
    tn = sum(y == 0 and pred == 0 for y, _, pred in rows)
    fp = sum(y == 0 and pred == 1 for y, _, pred in rows)
    fn = sum(y == 1 and pred == 0 for y, _, pred in rows)
    tp = sum(y == 1 and pred == 1 for y, _, pred in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives
    negative_below = 0
    pair_credit = 0.0
    for _, group in groupby(sorted(rows, key=lambda r: r[1]), key=lambda r: r[1]):
        group = list(group)
        p = sum(y for y, _, _ in group)
        n = len(group) - p
        pair_credit += p * (negative_below + n / 2)
        negative_below += n
    cum_tp = cum_fp = 0
    previous_recall, previous_precision = 0.0, 1.0
    pr_auc = 0.0
    for _, group in groupby(sorted(rows, key=lambda r: r[1], reverse=True), key=lambda r: r[1]):
        group = list(group)
        cum_tp += sum(y for y, _, _ in group)
        cum_fp += sum(1 - y for y, _, _ in group)
        current_recall = cum_tp / positives
        current_precision = cum_tp / (cum_tp + cum_fp)
        pr_auc += (current_recall - previous_recall) * (current_precision + previous_precision) / 2
        previous_recall, previous_precision = current_recall, current_precision
    return {
        'source': path.name,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows': len(rows), 'positive_rows': positives, 'negative_rows': negatives,
        'roc_auc': pair_credit / (positives * negatives),
        'pr_auc_trapezoidal': pr_auc,
        'positive_precision': precision, 'positive_recall': recall,
        'positive_f1': 2 * tp / (2 * tp + fp + fn),
        'accuracy': (tp + tn) / len(rows),
        'confusion_matrix_actual_by_predicted': [[tn, fp], [fn, tp]],
        'interpretation': 'Historical predictions; labels use a threshold selected on the same test set. No independent evaluation or clinical validation is claimed.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(summarize(args.predictions), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
