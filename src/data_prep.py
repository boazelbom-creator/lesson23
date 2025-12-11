"""Data preparation module for Iris dataset."""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_data():
    """Load Iris dataset and return as DataFrame with species names."""
    iris = load_iris()
    species_names = ['setosa', 'versicolor', 'virginica']

    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )
    df['species'] = [species_names[i] for i in iris.target]

    return df


def split_data(df, test_size=0.25, random_state=42):
    """Perform stratified train-test split."""
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def standardize_features(X_train, X_test):
    """Standardize features using StandardScaler fitted on train set."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_scaled, X_test_scaled, scaler


def prepare_stage1_labels(y, singleton_class='setosa'):
    """Create binary labels for Stage 1: singleton vs merged."""
    y_binary = y.apply(
        lambda x: 'Singleton' if x == singleton_class else 'Merged'
    )
    return y_binary


def prepare_stage2_data(X, y, singleton_class='setosa'):
    """Filter data to only merged class for Stage 2 training."""
    mask = y != singleton_class
    X_merged = X[mask]
    y_merged = y[mask]
    return X_merged, y_merged


def get_merged_classes(singleton_class='setosa'):
    """Get the two classes that form the merged group."""
    all_classes = ['setosa', 'versicolor', 'virginica']
    merged = [c for c in all_classes if c != singleton_class]
    return merged
