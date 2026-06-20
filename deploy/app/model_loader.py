import os
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

_model = None
_model_info = {}


def load_model(model_uri: str | None = None):
    global _model, _model_info

    if model_uri is None:
        model_uri = os.getenv("MODEL_URI")
    if model_uri is None:
        artifacts_root = Path("mlartifacts/1/models")
        if not artifacts_root.exists():
            artifacts_root = Path("/app/mlartifacts/1/models")
        if artifacts_root.exists():
            stages = sorted(artifacts_root.iterdir(), reverse=True)
            if stages:
                model_uri = str(stages[0] / "artifacts")
            else:
                raise RuntimeError("No model artifact directories found")
        else:
            raise RuntimeError(
                "No MODEL_URI set and no mlartifacts directory found"
            )

    logger.info("Loading model from %s", model_uri)
    _model = mlflow.sklearn.load_model(model_uri)
    _model_info = {"uri": model_uri}
    return _model


def get_model():
    if _model is None:
        load_model()
    return _model


def get_model_info():
    return _model_info
