"""MLflow experiment tracking, model registry governance, and SME metric logging."""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List
import pandas as pd

# Suppress verbose MLflow agent hint
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

def get_default_tracking_uri() -> str:
    """Returns absolute path SQLite tracking URI for the repository."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_db = os.path.join(project_root, "ml", "mlruns.db")
    os.makedirs(os.path.dirname(default_db), exist_ok=True)
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{default_db}")

def get_default_artifact_location() -> str:
    """Returns local artifact storage URI."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_artifacts = os.path.join(project_root, "ml", "artifacts", "mlruns")
    os.makedirs(default_artifacts, exist_ok=True)
    return f"file://{default_artifacts}"


class MLflowTracker:
    """Enterprise MLOps tracker for Snowflake-dbt ML pipelines using MLflow Model Registry."""

    def __init__(
        self,
        experiment_name: str = "airbnb_predictive_models",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ):
        self.tracking_uri = tracking_uri or get_default_tracking_uri()
        self.artifact_location = artifact_location or get_default_artifact_location()
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        # Ensure experiment exists
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            try:
                self.experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=self.artifact_location
                )
            except Exception:
                exp = mlflow.get_experiment_by_name(experiment_name)
                self.experiment_id = exp.experiment_id if exp else "0"
        else:
            self.experiment_id = self.experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        logger.info("MLflow Tracker active: URI=%s | Experiment=%s (ID=%s)",
                    self.tracking_uri, self.experiment_name, self.experiment_id)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flattens nested dictionaries for param logging."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                items.update(MLflowTracker._flatten_dict(v, key))
            elif isinstance(v, list):
                items[key] = ",".join(str(x) for x in v[:15])
            else:
                items[key] = v
        return items

    def log_and_register_model(
        self,
        run_name: str,
        model_name: str,
        pipeline: Any,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        alias: Optional[str] = "champion"
    ) -> Dict[str, Any]:
        """Logs parameters, technical & SME metrics, artifacts, and registers model in Model Registry."""
        mlflow.set_tracking_uri(self.tracking_uri)

        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id) as run:
            run_id = run.info.run_id

            # 1. Log Tags
            run_tags = {
                "project": "airbnb_snowflake_dbt_pipeline",
                "model_name": model_name,
                "framework": "scikit-learn",
                "registered": "true",
                "target_persona": "airbnb_hosts_and_sme_operators"
            }
            if tags:
                run_tags.update(tags)
            mlflow.set_tags(run_tags)

            # 2. Log Parameters
            flat_params = self._flatten_dict(parameters)
            sanitized_params = {k[:250]: str(v)[:450] for k, v in flat_params.items()}
            mlflow.log_params(sanitized_params)

            # 3. Log Numeric & SME Business Metrics
            numeric_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, bool)):
                    numeric_metrics[k] = float(v)
                elif isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float, bool)):
                            numeric_metrics[f"{k}_{sub_k}"] = float(sub_v)
            mlflow.log_metrics(numeric_metrics)

            # 4. Save Detailed Metrics JSON Artifact
            metrics_artifact_path = os.path.join("ml", "artifacts", f"{run_name}_metrics.json")
            os.makedirs(os.path.dirname(metrics_artifact_path), exist_ok=True)
            with open(metrics_artifact_path, "w") as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(metrics_artifact_path, artifact_path="evaluation")

            # 5. Log Model to MLflow using cloudpickle
            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
                registered_model_name=model_name,
                signature=signature,
                input_example=input_example
            )

            # 6. Retrieve Registered Version & Set Model Alias
            assigned_version = None
            try:
                versions = self.client.search_model_versions(f"name = '{model_name}'")
                if versions:
                    latest_v = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
                    assigned_version = latest_v.version
                    if alias:
                        self.client.set_registered_model_alias(model_name, alias, str(assigned_version))
                        logger.info("Assigned alias '@%s' to model '%s' version %s", alias, model_name, assigned_version)
            except Exception as e:
                logger.warning("Could not set registry alias for %s: %s", model_name, e)

            return {
                "run_id": run_id,
                "model_name": model_name,
                "version": assigned_version,
                "alias": alias,
                "model_uri": f"models:/{model_name}@{alias}" if (alias and assigned_version) else model_info.model_uri,
                "metrics": numeric_metrics
            }

    def load_registered_model(self, model_name: str, alias_or_version: str = "champion") -> Any:
        """Loads a model directly from the MLflow Model Registry via alias (@champion) or version number."""
        mlflow.set_tracking_uri(self.tracking_uri)
        if str(alias_or_version).isdigit():
            model_uri = f"models:/{model_name}/{alias_or_version}"
        else:
            model_uri = f"models:/{model_name}@{alias_or_version}"

        logger.info("Loading registered model from URI: %s", model_uri)
        return mlflow.sklearn.load_model(model_uri)

    def get_registered_models_summary(self) -> List[Dict[str, Any]]:
        """Returns summary of all models currently registered in the registry."""
        models = self.client.search_registered_models()
        summary = []
        for m in models:
            versions = self.client.search_model_versions(f"name = '{m.name}'")
            summary.append({
                "model_name": m.name,
                "latest_version": max([int(v.version) for v in versions]) if versions else None,
                "all_versions": [v.version for v in versions],
                "aliases": dict(m.aliases),
                "tags": dict(m.tags)
            })
        return summary


class ExperimentTracker:
    """Backwards-compatible tracker integrating MLflow Model Registry with legacy disk JSONs."""

    def __init__(self, experiment_name: str = "airbnb_predictive_models", base_dir: str = "ml/experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.mlflow_tracker = MLflowTracker(experiment_name=experiment_name)

    def log_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        artifact_path: str
    ) -> Dict[str, Any]:
        """Records run to legacy JSON and mirrors to MLflow."""
        run_record = {
            "experiment": self.experiment_name,
            "run_name": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": parameters,
            "metrics": metrics,
            "artifact_path": artifact_path
        }

        # 1. Write legacy JSON
        run_file = os.path.join(self.base_dir, f"{run_name}_{int(time.time())}.json")
        with open(run_file, "w") as f:
            json.dump(run_record, f, indent=2)

        # 2. Mirror to MLflow
        try:
            mlflow.set_tracking_uri(self.mlflow_tracker.tracking_uri)
            with mlflow.start_run(run_name=run_name, experiment_id=self.mlflow_tracker.experiment_id):
                mlflow.log_params(self.mlflow_tracker._flatten_dict(parameters))
                num_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, bool))}
                mlflow.log_metrics(num_metrics)
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
        except Exception as e:
            logger.warning("MLflow background logging failed: %s", e)

        return run_record
