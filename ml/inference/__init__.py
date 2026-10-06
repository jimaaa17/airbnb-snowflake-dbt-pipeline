"""Inference and model serving module."""

from ml.inference.batch_predictor import BatchPredictor
from ml.inference.service import ModelInferenceService

__all__ = ["BatchPredictor", "ModelInferenceService"]
