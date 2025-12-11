"""HTML template for the report."""


def get_html_template():
    """Return the HTML template string."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Iris Two-Stage SVM Cascade Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0 0 10px 0;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            background-color: #e7f3ff;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 10px 10px 10px 0;
            font-weight: bold;
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .metadata {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Iris Two-Stage SVM Cascade Report</h1>
        <p class="metadata">Generated: {timestamp}</p>
        <p class="metadata">Python {python_version} | scikit-learn {sklearn_version}</p>
    </div>

    <div class="section">
        <h2>Training Configuration</h2>
        <div class="metric">Kernel: Linear</div>
        <div class="metric">Stage 1 C: {c_stage1}</div>
        <div class="metric">Stage 2 C: {c_stage2}</div>
        <div class="metric">Scaling: StandardScaler</div>
        <div class="metric">Split: 75/25 Stratified</div>
        <div class="metric">Random State: 42</div>
        <p><strong>Class Grouping:</strong></p>
        <ul>
            <li>Singleton: {singleton}</li>
            <li>Merged: {merged0} ∪ {merged1}</li>
        </ul>
    </div>

    <div class="section">
        <h2>Stage 1 — Hyperplane & Margin (Singleton vs Merged)</h2>
        {stage1_table}
        <div class="metric">Bias (Standardized): {b1:.6f}</div>
        <div class="metric">Bias (Original): {b1_orig:.6f}</div>
        <div class="metric">Margin: {margin1:.6f}</div>
    </div>

    <div class="section">
        <h2>Stage 2 — Hyperplane & Margin ({merged0} vs {merged1})</h2>
        {stage2_table}
        <div class="metric">Bias (Standardized): {b2:.6f}</div>
        <div class="metric">Bias (Original): {b2_orig:.6f}</div>
        <div class="metric">Margin: {margin2:.6f}</div>
    </div>

    <div class="section">
        <h2>Test Set Accuracy</h2>
        <div class="metric">Accuracy: {accuracy:.4f} ({accuracy_pct:.2f}%)</div>
        <p>Correct predictions: {correct_count} / {total_count}</p>
    </div>

    <div class="section">
        <h2>Predictions vs Ground Truth (25% Test Set)</h2>
        {predictions_table}
    </div>

    <div class="section">
        <h2>Confusion Matrix</h2>
        <div class="img-container">
            <img src="data:image/png;base64,{cm_img}" alt="Confusion Matrix">
        </div>
        <p><em>3×3 confusion matrix comparing final predictions against ground truth.</em></p>
    </div>

    <div class="section">
        <h2>PCA 2D Scatter Plot (Test Set)</h2>
        <div class="img-container">
            <img src="data:image/png;base64,{pca_img}" alt="PCA Scatter Plot">
        </div>
        <p><em>PCA projection of test set. Circles (o) indicate correct predictions,
        crosses (x) indicate misclassifications. Colors represent true species.</em></p>
    </div>
</body>
</html>"""
