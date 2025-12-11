"""Evaluation and metrics module."""
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def create_predictions_table(predictions, y_test):
    """Create full predictions table with ground truth.

    Args:
        predictions: DataFrame from cascade prediction
        y_test: True labels

    Returns:
        DataFrame with index, ground_truth, stage1_pred, stage2_pred, final_pred
    """
    results = pd.DataFrame({
        'index': y_test.index,
        'ground_truth': y_test.values,
        'stage1_pred': predictions['stage1_pred'].values,
        'stage2_pred': predictions['stage2_pred'].values,
        'final_pred': predictions['final_pred'].values
    })

    results = results.sort_values('index').reset_index(drop=True)
    return results


def calculate_accuracy(predictions, y_test):
    """Calculate test accuracy."""
    accuracy = accuracy_score(y_test, predictions['final_pred'])
    return accuracy


def calculate_confusion_matrix(predictions, y_test):
    """Calculate confusion matrix for final predictions.

    Args:
        predictions: DataFrame with final_pred column
        y_test: True labels

    Returns:
        cm: Confusion matrix array
        labels: Class labels in order
    """
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, predictions['final_pred'], labels=labels)
    return cm, labels
