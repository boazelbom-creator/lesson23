"""Visualization module for creating plots."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import io
import base64


def create_confusion_matrix_plot(cm, labels):
    """Create confusion matrix heatmap and return as base64 string.

    Args:
        cm: Confusion matrix array
        labels: Class labels

    Returns:
        Base64 encoded image string
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Final Predictions')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


def create_pca_plot(X_train_scaled, X_test_scaled, y_test, predictions):
    """Create PCA 2D scatter plot and return as base64 string.

    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_test: True test labels
        predictions: DataFrame with final_pred column

    Returns:
        Base64 encoded image string
    """
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train_scaled)

    X_test_pca = pca.transform(X_test_scaled)

    correct = y_test == predictions['final_pred']

    species_colors = {
        'setosa': '#1f77b4',
        'versicolor': '#ff7f0e',
        'virginica': '#2ca02c'
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for species in sorted(y_test.unique()):
        mask = y_test == species

        correct_mask = mask & correct
        incorrect_mask = mask & ~correct

        if correct_mask.any():
            ax.scatter(X_test_pca[correct_mask, 0],
                      X_test_pca[correct_mask, 1],
                      c=species_colors[species],
                      marker='o',
                      s=100,
                      label=f'{species} (correct)',
                      edgecolors='black',
                      linewidths=1)

        if incorrect_mask.any():
            ax.scatter(X_test_pca[incorrect_mask, 0],
                      X_test_pca[incorrect_mask, 1],
                      c=species_colors[species],
                      marker='x',
                      s=150,
                      label=f'{species} (misclassified)',
                      linewidths=3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA 2D Scatter Plot - Test Set')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_base64
