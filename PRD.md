# Product Requirements Document (PRD)

## Project: Iris Two-Stage SVM Cascade with HTML Report Generation

**Version:** 1.0
**Date:** 2025-12-11
**Author:** AI Assistant

---

## 1. Executive Summary

This project implements a two-stage cascade SVM classifier for the Iris dataset using scikit-learn. The system performs hierarchical classification by first separating one species from a merged group of two others (Stage 1), then splitting the merged group (Stage 2). The output is a comprehensive, self-contained HTML report containing model parameters, predictions, metrics, and visualizations.

---

## 2. Objectives

### Primary Objective
Develop a Python-based machine learning pipeline that:
- Trains a two-stage SVM cascade on the Iris dataset
- Uses a 75/25 stratified train-test split with reproducible results
- Generates a single self-contained HTML report with all analysis results

### Secondary Objectives
- Maintain clean code structure with no Python file exceeding 150 lines
- Ensure full reproducibility with fixed random seeds
- Provide interpretable model parameters (weights, bias, margins)
- Include visual analytics (confusion matrix, PCA scatter plot)

---

## 3. Technical Requirements

### 3.1 Environment Setup
- **Python Version:** 3.10+
- **Virtual Environment:** Required (e.g., `.venv`)
- **Dependencies:**
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
- **Execution:** Must run end-to-end in a fresh environment after dependency installation

### 3.2 Code Structure Constraints
- **File Size Limit:** No Python file shall exceed 150 lines
- **Modularity:** Code should be split across multiple files if needed to meet the 150-line limit
- **Entry Point:** Clear main script for execution

---

## 4. Data Requirements

### 4.1 Dataset
- **Source:** `sklearn.datasets.load_iris()`
- **Features:** 4 numerical features
  - sepal length (cm)
  - sepal width (cm)
  - petal length (cm)
  - petal width (cm)
- **Target:** 3 species classes (setosa, versicolor, virginica)
- **Format:** pandas DataFrame with string labels for species

### 4.2 Data Split
- **Split Ratio:** 75% train / 25% test
- **Strategy:** Stratified split to maintain class proportions
- **Random State:** 42 (for reproducibility)
- **Index Preservation:** Original row indices must be maintained for traceability

### 4.3 Preprocessing
- **Standardization:** StandardScaler
  - Fit on training data only
  - Transform both train and test sets
  - Maintain consistent feature order
- **Feature Names:** Preserve original Iris feature names throughout pipeline

---

## 5. Model Architecture

### 5.1 Two-Stage Cascade Design

#### Stage 1: Binary SVM (Merged vs Singleton)
**Purpose:** Separate singleton class from merged class

**Default Configuration:**
- Singleton class: Setosa
- Merged class: Non-Setosa (Versicolor ∪ Virginica)

**Model Specifications:**
- Kernel: Linear (`kernel='linear'`)
- Regularization: C=1.0 (default, configurable)
- Training data: Full training set with binary labels

**Output Requirements:**
- Weight vector (w): `coef_` in standardized space
- Bias (b): `intercept_`
- Margin: `2 / ||w||` in standardized space
- Optional: Back-transformed w and b in original feature units

#### Stage 2: Binary SVM (Split Merged Pair)
**Purpose:** Classify between the two merged species

**Default Configuration:**
- Class A: Versicolor
- Class B: Virginica

**Model Specifications:**
- Kernel: Linear (`kernel='linear'`)
- Regularization: C=1.0 (default, configurable)
- Training data: Only samples belonging to merged class from training set

**Output Requirements:**
- Weight vector (w): `coef_` in standardized space
- Bias (b): `intercept_`
- Margin: `2 / ||w||` in standardized space
- Optional: Back-transformed w and b in original feature units

### 5.2 Back-Transformation (Optional but Recommended)

Convert hyperplane parameters from standardized space to original feature units:

**Formula:**
- `w = w' / s` (element-wise division by scaler's `scale_`)
- `b = b' - w' · (μ / s)` (where μ is scaler's `mean_`)

---

## 6. Prediction Logic

### 6.1 Cascade Inference Flow
For each test sample:

1. **Stage 1 Prediction:** Classify as Singleton or Merged
2. **Decision Point:**
   - If Singleton → Final prediction = singleton species (e.g., Setosa)
   - If Merged → Proceed to Stage 2
3. **Stage 2 Prediction:** Classify within merged pair (e.g., Versicolor vs Virginica)
4. **Final Output:** Resolved three-class species prediction

### 6.2 Predictions Table Structure
Required columns:
- `index`: Original dataset row ID
- `ground_truth`: Actual species
- `stage1_pred`: Singleton vs Merged classification
- `stage2_pred`: Species within merged pair (blank if Stage 1 = Singleton)
- `final_pred`: Final resolved species

---

## 7. Evaluation Metrics

### 7.1 Required Metrics
- **Test Accuracy:** Overall accuracy of final three-class predictions
- **Confusion Matrix:** 3×3 matrix comparing final predictions vs ground truth

### 7.2 Validation Requirements
- Ensure Stage 2 trains only on merged class samples from training set
- No data leakage from test set
- All metrics computed on 25% test set only

---

## 8. Visualization Requirements

### 8.1 Confusion Matrix Heatmap
- **Type:** 3×3 heatmap
- **Axes:** Class names (setosa, versicolor, virginica)
- **Format:** Base64-embedded image in HTML
- **Library:** seaborn or matplotlib

### 8.2 PCA 2D Scatter Plot
**Data Preparation:**
- Fit PCA on standardized training features
- Transform standardized test features to 2D

**Plot Elements:**
- Points colored by ground truth species
- Visual distinction for misclassifications (different marker/outline)
- Legend mapping colors/markers to species
- Legend indicating correct vs incorrect predictions

**Format:** Base64-embedded image in HTML

---

## 9. HTML Report Specification

### 9.1 Output Format
- **File Type:** Single self-contained HTML file
- **Default Path:** `output/report.html`
- **Self-Contained:** All images embedded as base64 (no external files)

### 9.2 Report Structure and Content

#### Section 1: Header & Metadata
- Report title
- Generation timestamp
- Environment details:
  - Python version
  - scikit-learn version
  - Other relevant package versions

#### Section 2: Kernel & Training Details
- Kernel type: Linear (both stages)
- Regularization parameter C (both stages)
- Scaling method: StandardScaler
- Train/Test split configuration:
  - Ratio: 75/25
  - Strategy: Stratified
  - Random state: 42
- Class grouping configuration:
  - Singleton species
  - Merged species pair

#### Section 3: Stage 1 Model Parameters
**Title:** Stage 1 — Hyperplane & Margin (Singleton vs Merged)

**Content:**
- Table of feature-wise weights (w) with feature names
- Bias (b)
- Margin value: `2 / ||w||`
- Space: Standardized
- Optional: Back-transformed parameters in original units

#### Section 4: Stage 2 Model Parameters
**Title:** Stage 2 — Hyperplane & Margin (Within Merged Pair)

**Content:**
- Table of feature-wise weights (w) with feature names
- Bias (b)
- Margin value: `2 / ||w||`
- Space: Standardized
- Optional: Back-transformed parameters in original units

#### Section 5: Test Set Accuracy
- Overall accuracy percentage
- Number of correct predictions / total predictions

#### Section 6: Predictions Table
**Title:** Predictions vs Ground Truth (25% Test Set)

**Content:**
- Full predictions table with all required columns
- Sorted by original index
- Formatted for readability

#### Section 7: Confusion Matrix
- Embedded base64 heatmap image
- Caption explaining the matrix

#### Section 8: PCA Visualization
- Embedded base64 scatter plot
- Caption explaining markers, colors, and what they represent
- Legend description (circles = correct, crosses = misclassified, etc.)

### 9.3 HTML Requirements
- Professional styling (CSS embedded or inline)
- Responsive tables
- Clear section headers
- Readable font sizes
- Proper image sizing

---

## 10. File Handling & Output

### 10.1 Directory Management
- Ensure `output/` directory exists (create if missing)
- Write report to `output/report.html`

### 10.2 Terminal Output
- Print clear message indicating report generation success
- Display full path to generated report
- Example: `"Report successfully generated: /path/to/output/report.html"`

### 10.3 Image Embedding
- All images must be base64-encoded
- Embed directly in HTML `<img>` tags
- No external image files allowed

---

## 11. Reproducibility Requirements

### 11.1 Random State Management
- `random_state=42` for:
  - Train-test split
  - Any random initialization in models
  - PCA (if applicable)

### 11.2 Data Integrity
- Stratified splitting to preserve class distributions
- Clear separation of training and test data
- Stage 2 training restricted to merged class from training set only

### 11.3 Feature Consistency
- Maintain exact Iris feature names throughout
- Consistent feature ordering in all tables and outputs
- Clear labeling in all visualizations

---

## 12. Configuration Options (Optional)

### 12.1 Class Grouping Configuration
- **Singleton Species Selection:** CLI flag to choose which species is singleton
- **Merged Pair Selection:** Automatic (the other two species)
- **Default:** Setosa (singleton) vs Versicolor+Virginica (merged)

### 12.2 Model Hyperparameters
- **Stage 1 C:** Independent configuration
- **Stage 2 C:** Independent configuration
- **Default:** C=1.0 for both

### 12.3 Output Format
- **Primary Format:** HTML (required)
- **Alternative Format:** PDF (optional future enhancement)
- **Default:** HTML

---

## 13. Acceptance Criteria

### 13.1 Environment & Dependencies
- ✅ Uses virtual environment
- ✅ All dependencies installed via requirements file
- ✅ Runs successfully in fresh environment
- ✅ Uses scikit-learn for all ML operations

### 13.2 Model Implementation
- ✅ Two-stage cascade correctly implemented
- ✅ Stage 1: Singleton vs Merged classification
- ✅ Stage 2: Split merged pair classification
- ✅ Linear kernel for both stages
- ✅ Correct prediction routing logic

### 13.3 Data Handling
- ✅ 75/25 stratified split
- ✅ StandardScaler applied correctly (fit on train only)
- ✅ Original indices preserved
- ✅ No data leakage

### 13.4 HTML Report
- ✅ Single self-contained HTML file
- ✅ All required sections present
- ✅ Kernel details for both stages
- ✅ Weight vectors with feature names
- ✅ Margin and bias for both stages
- ✅ Predictions table with all columns
- ✅ Test accuracy displayed
- ✅ Confusion matrix heatmap embedded
- ✅ PCA scatter plot embedded
- ✅ No external files required

### 13.5 Code Quality
- ✅ No Python file exceeds 150 lines
- ✅ Clear code structure
- ✅ Proper error handling
- ✅ Informative terminal output

### 13.6 Reproducibility
- ✅ Fixed random seed (42) used throughout
- ✅ Results are reproducible across runs
- ✅ Stratified split maintains class proportions

---

## 14. Success Metrics

### 14.1 Functional Metrics
- Program executes without errors
- HTML report generates successfully
- All visualizations render correctly
- Predictions are logically consistent with cascade logic

### 14.2 Quality Metrics
- Code is modular and maintainable
- Report is professional and readable
- Model parameters are interpretable
- Documentation is clear

### 14.3 Performance Benchmarks
- Execution time: < 30 seconds for full pipeline
- HTML file size: Reasonable (< 5MB with embedded images)
- Test accuracy: Comparable to standard single-stage multi-class SVM on Iris

---

## 15. Future Enhancements (Out of Scope)

- Support for other datasets beyond Iris
- Non-linear kernels (RBF, polynomial)
- Hyperparameter tuning via grid search
- Cross-validation analysis
- PDF report generation
- Interactive HTML visualizations
- Model persistence (save/load)
- API endpoint for predictions

---

## 16. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Stage 2 trains on test data | High | Strict filtering to only use training set samples |
| Images not embedded correctly | Medium | Test base64 encoding thoroughly |
| 150-line limit too restrictive | Medium | Modularize code across multiple files |
| Reproducibility issues | High | Enforce `random_state=42` everywhere |
| HTML rendering issues | Low | Use standard HTML/CSS, test in multiple browsers |

---

## 17. Dependencies & Tools Summary

### Required Python Packages
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Development Tools
- Python 3.10+
- Virtual environment tool (venv or virtualenv)
- Text editor / IDE
- Web browser (for viewing HTML report)

---

## 18. Deliverables

1. **Python Source Code**
   - Main execution script
   - Supporting modules (if needed for 150-line limit)
   - Each file ≤ 150 lines

2. **Requirements File**
   - `requirements.txt` with pinned versions

3. **HTML Report**
   - Self-contained file at `output/report.html`
   - All content and images embedded

4. **Documentation**
   - README with setup and execution instructions
   - This PRD document

---

## 19. Timeline & Phases

### Phase 1: Setup & Data Preparation
- Environment setup
- Data loading and splitting
- Preprocessing pipeline

### Phase 2: Model Development
- Stage 1 SVM implementation
- Stage 2 SVM implementation
- Cascade prediction logic

### Phase 3: Evaluation & Metrics
- Accuracy calculation
- Confusion matrix generation
- Predictions table construction

### Phase 4: Visualization
- PCA transformation
- Scatter plot generation
- Confusion matrix heatmap

### Phase 5: Report Generation
- HTML template creation
- Content population
- Image embedding
- Final formatting

### Phase 6: Testing & Validation
- End-to-end testing
- Reproducibility verification
- Code review for 150-line limit

---

## 20. Appendix

### A. Feature Names (Iris Dataset)
1. sepal length (cm)
2. sepal width (cm)
3. petal length (cm)
4. petal width (cm)

### B. Species Names (Iris Dataset)
1. setosa
2. versicolor
3. virginica

### C. Back-Transformation Formulas

**Standardization:**
```
x_scaled = (x - μ) / s
```

**Hyperplane in standardized space:**
```
w' · x_scaled + b' = 0
```

**Hyperplane in original space:**
```
w = w' / s  (element-wise)
b = b' - w' · (μ / s)
w · x + b = 0
```

---

**End of PRD**