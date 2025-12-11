"""HTML report generation module."""
from datetime import datetime
import sys
import sklearn
import pandas as pd
from src.html_template import get_html_template


def generate_html_report(
    output_path,
    model,
    scaler,
    feature_names,
    predictions_table,
    accuracy,
    cm_base64,
    pca_base64,
    singleton_class,
    merged_classes
):
    """Generate self-contained HTML report.

    Args:
        output_path: Path to save HTML file
        model: TwoStageSVM instance
        scaler: StandardScaler instance
        feature_names: List of feature names
        predictions_table: DataFrame with predictions
        accuracy: Test accuracy value
        cm_base64: Base64 encoded confusion matrix image
        pca_base64: Base64 encoded PCA plot image
        singleton_class: Name of singleton class
        merged_classes: List of merged class names
    """
    w1, b1, margin1 = model.get_stage1_params()
    w2, b2, margin2 = model.get_stage2_params()

    w1_orig, b1_orig = model.back_transform_params(w1, b1, scaler)
    w2_orig, b2_orig = model.back_transform_params(w2, b2, scaler)

    stage1_params_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight (Standardized)': w1,
        'Weight (Original)': w1_orig
    })

    stage2_params_df = pd.DataFrame({
        'Feature': feature_names,
        'Weight (Standardized)': w2,
        'Weight (Original)': w2_orig
    })

    template = get_html_template()
    html_content = template.format(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        python_version=sys.version.split()[0],
        sklearn_version=sklearn.__version__,
        c_stage1=model.C_stage1,
        c_stage2=model.C_stage2,
        singleton=singleton_class,
        merged0=merged_classes[0],
        merged1=merged_classes[1],
        stage1_table=stage1_params_df.to_html(index=False, float_format='%.6f'),
        b1=b1,
        b1_orig=b1_orig,
        margin1=margin1,
        stage2_table=stage2_params_df.to_html(index=False, float_format='%.6f'),
        b2=b2,
        b2_orig=b2_orig,
        margin2=margin2,
        accuracy=accuracy,
        accuracy_pct=accuracy * 100,
        correct_count=int(accuracy * len(predictions_table)),
        total_count=len(predictions_table),
        predictions_table=predictions_table.to_html(index=False),
        cm_img=cm_base64,
        pca_img=pca_base64
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
