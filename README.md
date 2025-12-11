# Iris Two-Stage SVM Cascade

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

The HTML report will be generated at `output/report.html`.

## Configuration

You can modify the default configuration in `main.py`:
- `singleton_class`: Which species to use as singleton (default: 'setosa')
- `C_stage1`: Regularization parameter for Stage 1 SVM (default: 1.0)
- `C_stage2`: Regularization parameter for Stage 2 SVM (default: 1.0)

## Results

### Test Set Performance
- **Accuracy: 97.37%** (37/38 correct predictions)
- Dataset split: 75% train (112 samples) / 25% test (38 samples)
- Stratified random split with seed 42

### Confusion Matrix
```
                 Predicted
                 setosa  versicolor  virginica
Actual setosa       13           0          0
Actual versicolor    0          15          1
Actual virginica     0           0          9
```

### Stage 1 SVM Parameters (Setosa vs Others)
| Parameter | Standardized | Original Scale |
|-----------|--------------|----------------|
| Bias (b₁) | -0.802497 | -0.977254 |
| Margin | 1.782642 | - |

### Stage 2 SVM Parameters (Versicolor vs Virginica)
| Parameter | Standardized | Original Scale |
|-----------|--------------|----------------|
| Bias (b₂) | -2.698949 | -7.892728 |
| Margin | 0.662820 | - |

### Model Architecture
**Stage 1 SVM** - Binary classification (Singleton vs Merged)
- Separates `setosa` from (`versicolor` ∪ `virginica`)
- Linear kernel with C=1.0

**Stage 2 SVM** - Binary classification (Split merged pair)
- Separates `versicolor` from `virginica`
- Linear kernel with C=1.0
- Only processes samples classified as non-setosa by Stage 1

### Full HTML Report
The generated HTML report (`output/report.html`) includes:
- Training configuration and hyperparameters
- Stage 1 & Stage 2 hyperplane parameters (weights, bias, margins)
- Parameters in both standardized and original feature spaces
- Complete predictions table for all test samples
- Confusion matrix heatmap visualization
- PCA 2D scatter plot showing correct/incorrect classifications
- Detailed accuracy metrics

**📊 View the Full HTML Report:**

**Option 1: GitHub Pages (Live Preview)**
🔗 **[View Report Online](https://boazelbom-creator.github.io/lesson23/)** - No download needed!

**Option 2: Download and View Locally**
1. Download [report.html](./output/report.html)
2. Open it in your web browser

**Option 3: Use HTMLPreview.github.io**
🔗 [View via HTMLPreview](https://htmlpreview.github.io/?https://github.com/boazelbom-creator/lesson23/blob/main/output/report.html)

The report includes:
- Interactive confusion matrix heatmap
- PCA scatter plot with all 38 test samples
- Complete hyperplane weight vectors for all 4 features
- Full predictions table with stage-by-stage classification results

### Features
- StandardScaler normalization fitted on training data
- Back-transformation of SVM parameters to original feature space
- Self-contained HTML report with embedded visualizations
- All Python files ≤ 150 lines (modular architecture)
