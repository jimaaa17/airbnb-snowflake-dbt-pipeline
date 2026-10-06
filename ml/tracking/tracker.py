"""Lightweight experiment tracking and metadata logging."""

import os
import json
import time
from typing import Dict, Any

class ExperimentTracker:
    """Logs experiment parameters, evaluation metrics, and artifacts."""

    def __init__(self, experiment_name: str, base_dir: str = "ml/experiments"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def log_run(
        self,
        run_name: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        artifact_path: str
    ) -> Dict[str, Any]:
        """Records an experiment run to disk."""
        run_record = {
            "experiment": self.experiment_name,
            "run_name": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": parameters,
            "metrics": metrics,
            "artifact_path": artifact_path
        }

        run_file = os.path.join(self.base_dir, f"{run_name}_{int(time.time())}.json")
        with open(run_file, "w") as f:
            json.dump(run_record, f, indent=2)

        return run_record
