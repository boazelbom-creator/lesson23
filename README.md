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

### Model Architecture
**Stage 1 SVM** - Binary classification (Singleton vs Merged)
- Separates `setosa` from (`versicolor` ∪ `virginica`)
- Linear kernel with C=1.0

**Stage 2 SVM** - Binary classification (Split merged pair)
- Separates `versicolor` from `virginica`
- Linear kernel with C=1.0
- Only processes samples classified as non-setosa by Stage 1

### Report Contents
The generated HTML report (`output/report.html`) includes:
- Training configuration and hyperparameters
- Stage 1 & Stage 2 hyperplane parameters (weights, bias, margins)
- Parameters in both standardized and original feature spaces
- Complete predictions table for all test samples
- Confusion matrix visualization
- PCA 2D scatter plot showing correct/incorrect classifications
- Detailed accuracy metrics

### Features
- StandardScaler normalization fitted on training data
- Back-transformation of SVM parameters to original feature space
- Self-contained HTML report with embedded visualizations
- All Python files ≤ 150 lines (modular architecture)
