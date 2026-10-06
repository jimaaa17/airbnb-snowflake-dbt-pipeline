"""Abstract base class for all Airbnb predictive models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import joblib
import pandas as pd

class BaseAirbnbModel(ABC):
    """Abstract interface guaranteeing consistent model lifecycle across estimators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = None
        self.is_fitted = False

    @abstractmethod
    def build_pipeline(self):
        """Constructs the scikit-learn pipeline."""
        pass

    @abstractmethod
    def train(self, df_train: pd.DataFrame) -> Dict[str, float]:
        """Fits the pipeline on training data and returns in-sample metrics."""
        pass

    @abstractmethod
    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates pipeline against holdout test partition."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Any:
        """Generates predictions."""
        pass

    def save(self, filepath: str):
        """Persists fitted model artifact to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "config": self.config}, filepath)

    def load(self, filepath: str):
        """Loads fitted model artifact from disk."""
        artifact = joblib.load(filepath)
        self.pipeline = artifact["pipeline"]
        self.config = artifact["config"]
        self.is_fitted = True
        return self
