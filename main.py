#!/usr/bin/env python3
"""Main script for Iris Two-Stage SVM Cascade."""
import os
from src.data_prep import (
    load_iris_data,
    split_data,
    standardize_features,
    prepare_stage1_labels,
    prepare_stage2_data,
    get_merged_classes
)
from src.models import TwoStageSVM
from src.evaluation import (
    create_predictions_table,
    calculate_accuracy,
    calculate_confusion_matrix
)
from src.visualizations import (
    create_confusion_matrix_plot,
    create_pca_plot
)
from src.report_generator import generate_html_report


def main():
    """Run the complete two-stage SVM pipeline."""
    print("=" * 60)
    print("Iris Two-Stage SVM Cascade")
    print("=" * 60)

    singleton_class = 'setosa'
    C_stage1 = 1.0
    C_stage2 = 1.0

    print("\n[1/8] Loading Iris dataset...")
    df = load_iris_data()
    print(f"  Loaded {len(df)} samples with {len(df.columns)-1} features")

    print("\n[2/8] Splitting data (75% train / 25% test)...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    print("\n[3/8] Standardizing features...")
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    print("  StandardScaler fitted on training data")

    print("\n[4/8] Training Stage 1 SVM (Singleton vs Merged)...")
    y_train_stage1 = prepare_stage1_labels(y_train, singleton_class)
    model = TwoStageSVM(C_stage1, C_stage2, singleton_class)
    model.train_stage1(X_train_scaled, y_train_stage1)
    print(f"  Stage 1 trained: {singleton_class} vs Non-{singleton_class}")

    print("\n[5/8] Training Stage 2 SVM (Split merged pair)...")
    X_train_merged, y_train_merged = prepare_stage2_data(
        X_train_scaled, y_train, singleton_class
    )
    model.train_stage2(X_train_merged, y_train_merged)
    merged_classes = get_merged_classes(singleton_class)
    print(f"  Stage 2 trained: {merged_classes[0]} vs {merged_classes[1]}")

    print("\n[6/8] Making predictions on test set...")
    predictions = model.predict_cascade(X_test_scaled)
    predictions_table = create_predictions_table(predictions, y_test)
    accuracy = calculate_accuracy(predictions, y_test)
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n[7/8] Generating visualizations...")
    cm, labels = calculate_confusion_matrix(predictions, y_test)
    cm_base64 = create_confusion_matrix_plot(cm, labels)
    pca_base64 = create_pca_plot(
        X_train_scaled, X_test_scaled, y_test, predictions
    )
    print("  Confusion matrix and PCA plot created")

    print("\n[8/8] Generating HTML report...")
    os.makedirs('output', exist_ok=True)
    output_path = os.path.abspath('output/report.html')

    generate_html_report(
        output_path=output_path,
        model=model,
        scaler=scaler,
        feature_names=list(X_train.columns),
        predictions_table=predictions_table,
        accuracy=accuracy,
        cm_base64=cm_base64,
        pca_base64=pca_base64,
        singleton_class=singleton_class,
        merged_classes=merged_classes
    )

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"\nReport generated: {output_path}")
    print("\nOpen the report in your web browser to view results.")


if __name__ == '__main__':
    main()
