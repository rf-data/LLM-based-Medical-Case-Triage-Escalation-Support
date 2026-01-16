# src/utils/session.py
from pathlib import Path
from typing import Dict, Any
import json

import src.utils.general_helper as gh


class Session:
    def __init__(self):
        # --- runtime / env ---
        self.env = None
        self.branch = None
        self.root = gh.find_project_root()
        self.venv = None
        self.url_repo = None

        # --- experiment context ---
        self.mode: str | None = None
        self.experiment_name: str | None = None
        self.artifacts_location: str | None = None
        self.namespace: str | None = None
        self.now: str | None = None

        self.tags: Dict[str, Any] = {}
        self.parameters: Dict[str, Any] = {}

        # --- model / prompt snapshot ---
        self.prompt: str | None = None
        self.allowed_values: str | None = None
        self.json_scheme: Dict[str, Any] | None = None

        self.env_loaded = False

    # ---------- configuration ----------
    def load_config(self, config: Dict[str, Any]):
        """Load experiment-relevant configuration into session."""
        self.mode = config.get("mode", None)
        self.namespace = config.get("namespace", None)
        self.experiment_name = config.get("experiment_name", None)
        self.artifacts_location = config.get("artifacts_location", None)
        self.now = config.get("now", None)

        self.tags.update(config.get("tag", {}))
        self.parameters.update(config.get("parameter", {}))

        self.prompt = config.get("prompt", None)
        self.allowed_values = config.get("allowed_values", None)
        self.json_scheme = config.get("json_scheme", None)

    # ---------- persistence ----------
    def save_session(self):
        if self.root is None:
            raise RuntimeError("Session.root must be set before saving session.")
        file_path = Path(self.root) / ".env.session"

        state= {
            "SESSION_ENV": self.env if self.env is not None else None,
            "SESSION_BRANCH": self.branch if self.branch is not None else None,
            "SESSION_ROOT": str(self.root) if self.root is not None else None,
            "SESSION_DATA": str(self.data) if self.data is not None else None,
            "SESSION_VENV": str(self.venv) if self.venv is not None else None,
            "SESSION_URL_REPO": self.url_repo if self.url_repo is not None else None,

            "SESSION_EXPERIMENT": self.experiment_name if self.experiment_name is not None else None,
            "SESSION_ARTIFACTS": self.artifacts_location if self.artifacts_location is not None else None,
            "SESSION_MODE": self.mode if self.mode is not None else None,
            "SESSION_NAMESPACE": self.namespace if self.namespace is not None else None,
        }

        with open(file_path, "w") as f:
            for key, value in state.items():
                f.write(f"{key}={value}\n")
    
    def save_snapshot(self):
        """Save full experiment snapshot (for MLflow / debugging)."""
        if self.root is None:
            raise RuntimeError("Session.root must be set before saving snapshot.")

        snapshot_path = Path(self.root) / "session_snapshot.json"

        snapshot = {
            "experiment_name": self.experiment_name,
            "artifacts_location": self.artifacts_location,
            "mode": self.mode,
            "tags": self.tags,
            "parameters": self.parameters,
            "prompt": self.prompt,
            "allowed_values": self.allowed_values,
            "json_scheme": self.json_scheme,
        }

        gh.save_dict(snapshot_path, snapshot)
        

session = Session()


class Session:
    

    # ---------- configuration ----------
    def load_config(self, config: Dict[str, Any]):
        """Load experiment-relevant configuration into session."""
        self.experiment_name = config.get("experiment_name")
        self.artifacts_location = config.get("artifacts_location")

        self.tags.update(config.get("tag", {}))
        self.parameters.update(config.get("parameter", {}))

        self.prompt = config.get("prompt")
        self.allowed_values = config.get("allowed_values")
        self.json_scheme = config.get("json_scheme")

    # ---------- persistence ----------
    

    def save_snapshot(self):
        """Save full experiment snapshot (for MLflow / debugging)."""
        if self.root is None:
            raise RuntimeError("Session.root must be set before saving snapshot.")

        snapshot_path = Path(self.root) / "session_snapshot.json"

        snapshot = {
            "experiment_name": self.experiment_name,
            "artifacts_location": self.artifacts_location,
            "tags": self.tags,
            "parameters": self.parameters,
            "prompt": self.prompt,
            "allowed_values": self.allowed_values,
            "json_scheme": self.json_scheme,
        }

        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)
