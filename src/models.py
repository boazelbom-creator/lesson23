"""Two-stage SVM cascade model."""
import numpy as np
from sklearn.svm import SVC


class TwoStageSVM:
    """Two-stage cascade SVM classifier."""

    def __init__(self, C_stage1=1.0, C_stage2=1.0, singleton_class='setosa'):
        """Initialize two-stage SVM.

        Args:
            C_stage1: Regularization parameter for Stage 1
            C_stage2: Regularization parameter for Stage 2
            singleton_class: The species to classify as singleton
        """
        self.C_stage1 = C_stage1
        self.C_stage2 = C_stage2
        self.singleton_class = singleton_class

        self.stage1_svm = None
        self.stage2_svm = None
        self.scaler = None
        self.feature_names = None

    def train_stage1(self, X_train, y_train_binary):
        """Train Stage 1: Singleton vs Merged."""
        self.stage1_svm = SVC(kernel='linear', C=self.C_stage1)
        self.stage1_svm.fit(X_train, y_train_binary)

    def train_stage2(self, X_train_merged, y_train_merged):
        """Train Stage 2: Split merged pair."""
        self.stage2_svm = SVC(kernel='linear', C=self.C_stage2)
        self.stage2_svm.fit(X_train_merged, y_train_merged)

    def get_stage1_params(self):
        """Get Stage 1 hyperplane parameters."""
        w = self.stage1_svm.coef_[0]
        b = self.stage1_svm.intercept_[0]
        margin = 2.0 / np.linalg.norm(w)
        return w, b, margin

    def get_stage2_params(self):
        """Get Stage 2 hyperplane parameters."""
        w = self.stage2_svm.coef_[0]
        b = self.stage2_svm.intercept_[0]
        margin = 2.0 / np.linalg.norm(w)
        return w, b, margin

    def back_transform_params(self, w_scaled, b_scaled, scaler):
        """Transform hyperplane parameters to original feature space.

        Args:
            w_scaled: Weight vector in standardized space
            b_scaled: Bias in standardized space
            scaler: StandardScaler object

        Returns:
            w_original, b_original
        """
        w_original = w_scaled / scaler.scale_
        b_original = b_scaled - np.dot(w_scaled, scaler.mean_ / scaler.scale_)
        return w_original, b_original

    def predict_cascade(self, X_test):
        """Predict using cascade logic.

        Args:
            X_test: Test features (scaled)

        Returns:
            predictions: DataFrame with stage1_pred, stage2_pred, final_pred
        """
        import pandas as pd

        stage1_pred = self.stage1_svm.predict(X_test)

        stage2_pred = []
        final_pred = []

        for i, s1_pred in enumerate(stage1_pred):
            if s1_pred == 'Singleton':
                stage2_pred.append('')
                final_pred.append(self.singleton_class)
            else:
                s2_pred = self.stage2_svm.predict(X_test.iloc[[i]])[0]
                stage2_pred.append(s2_pred)
                final_pred.append(s2_pred)

        predictions = pd.DataFrame({
            'stage1_pred': stage1_pred,
            'stage2_pred': stage2_pred,
            'final_pred': final_pred
        }, index=X_test.index)

        return predictions
