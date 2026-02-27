"""ML model versioning and registry.

Uses only the Python standard library (json, pathlib, datetime, logging)
so it works without any external dependencies.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REGISTRY_FILE = "registry.json"
_METADATA_FILE = "metadata.json"


class ModelRegistry:
    """File-based ML model registry for versioning and discovery.

    Directory layout::

        registry_root/
            registry.json          # index of all registered models
            <model_name>/
                <version>/
                    metadata.json  # version metadata
                    model.*        # model artifact(s) (managed externally)

    """

    def __init__(self, registry_root: str = "models/registry") -> None:
        """Initialise the registry.

        Args:
            registry_root: Root directory for all model artefacts.
        """
        self.root = Path(registry_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.root / _REGISTRY_FILE
        self._registry: Dict[str, Any] = self._load_registry()

    # ------------------------------------------------------------------
    # Internal I/O helpers
    # ------------------------------------------------------------------

    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry index from disk (or return empty dict)."""
        if self._registry_path.exists():
            with self._registry_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save_registry(self) -> None:
        """Persist the in-memory registry index to disk."""
        with self._registry_path.open("w", encoding="utf-8") as fh:
            json.dump(self._registry, fh, indent=2, default=str)

    def _model_dir(self, name: str, version: str) -> Path:
        """Return the directory path for a specific model version."""
        return self.root / name / version

    def _next_version(self, name: str) -> str:
        """Compute the next semantic version for *name*.

        Returns:
            Version string like ``"v1"``, ``"v2"``, …
        """
        existing = self._registry.get(name, {}).get("versions", [])
        if not existing:
            return "v1"
        last = sorted(existing, key=lambda v: int(v.lstrip("v")))[-1]
        return f"v{int(last.lstrip('v')) + 1}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        artifact_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Register a new model version.

        Args:
            name: Logical model name (e.g. ``"lstm_btcusdt"``).
            artifact_path: Path to the model artifact file/directory to
                copy into the registry.  May be *None* if the artifact is
                managed externally.
            metadata: Arbitrary key-value metadata (hyperparams, metrics,
                etc.).
            version: Explicit version string; auto-incremented if *None*.
            tags: Optional list of tag strings (e.g. ``["production"]``).

        Returns:
            The assigned version string.
        """
        ver = version or self._next_version(name)
        model_dir = self._model_dir(name, ver)
        model_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(tz=timezone.utc).isoformat()
        entry: Dict[str, Any] = {
            "name": name,
            "version": ver,
            "registered_at": now,
            "metadata": metadata or {},
            "tags": tags or [],
            "artifact": None,
        }

        if artifact_path is not None:
            src = Path(artifact_path)
            if src.is_file():
                dest = model_dir / src.name
                shutil.copy2(src, dest)
                entry["artifact"] = str(dest.relative_to(self.root))
                logger.info("Copied artifact %s → %s", src, dest)
            elif src.is_dir():
                dest = model_dir / src.name
                shutil.copytree(src, dest, dirs_exist_ok=True)
                entry["artifact"] = str(dest.relative_to(self.root))
                logger.info("Copied artifact directory %s → %s", src, dest)
            else:
                logger.warning("Artifact path '%s' not found; skipping copy.", artifact_path)

        # Write per-version metadata
        meta_path = model_dir / _METADATA_FILE
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2, default=str)

        # Update registry index
        if name not in self._registry:
            self._registry[name] = {"versions": [], "latest": None}
        self._registry[name]["versions"].append(ver)
        self._registry[name]["latest"] = ver
        self._save_registry()

        logger.info("Registered model '%s' version '%s'.", name, ver)
        return ver

    def get_metadata(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve metadata for a model version.

        Args:
            name: Model name.
            version: Version string; defaults to the latest version.

        Returns:
            Metadata dictionary.

        Raises:
            KeyError: If *name* or *version* is not found in the registry.
        """
        ver = version or self._registry.get(name, {}).get("latest")
        if ver is None:
            raise KeyError(f"Model '{name}' is not registered.")
        meta_path = self._model_dir(name, ver) / _METADATA_FILE
        if not meta_path.exists():
            raise KeyError(f"Metadata for '{name}' version '{ver}' not found.")
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def list_models(self) -> List[str]:
        """Return all registered model names.

        Returns:
            Sorted list of model names.
        """
        return sorted(self._registry.keys())

    def list_versions(self, name: str) -> List[str]:
        """Return all versions registered for *name*.

        Args:
            name: Model name.

        Returns:
            List of version strings in registration order.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' is not registered.")
        return list(self._registry[name]["versions"])

    def get_latest_version(self, name: str) -> str:
        """Return the latest version string for *name*.

        Args:
            name: Model name.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' is not registered.")
        return self._registry[name]["latest"]

    def get_artifact_path(self, name: str, version: Optional[str] = None) -> Optional[Path]:
        """Return the full path to the registered model artifact.

        Args:
            name: Model name.
            version: Version; defaults to latest.

        Returns:
            :class:`pathlib.Path` to the artifact, or *None* if not set.
        """
        meta = self.get_metadata(name, version)
        artifact_rel = meta.get("artifact")
        if artifact_rel is None:
            return None
        return self.root / artifact_rel

    def tag_version(self, name: str, version: str, tags: List[str]) -> None:
        """Add tags to an existing model version.

        Args:
            name: Model name.
            version: Version to tag.
            tags: Tags to add (duplicates ignored).
        """
        meta = self.get_metadata(name, version)
        existing = set(meta.get("tags", []))
        existing.update(tags)
        meta["tags"] = sorted(existing)
        meta_path = self._model_dir(name, version) / _METADATA_FILE
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        logger.info("Tagged '%s' version '%s' with %s.", name, version, tags)

    def delete_version(self, name: str, version: str) -> None:
        """Delete a model version from the registry.

        Args:
            name: Model name.
            version: Version to delete.

        Raises:
            KeyError: If *name* or *version* is not found.
        """
        if name not in self._registry:
            raise KeyError(f"Model '{name}' is not registered.")
        versions = self._registry[name]["versions"]
        if version not in versions:
            raise KeyError(f"Version '{version}' not found for model '{name}'.")

        # Remove artefact directory
        model_dir = self._model_dir(name, version)
        if model_dir.exists():
            shutil.rmtree(model_dir)

        versions.remove(version)
        if self._registry[name]["latest"] == version:
            self._registry[name]["latest"] = versions[-1] if versions else None
        if not versions:
            del self._registry[name]

        self._save_registry()
        logger.info("Deleted model '%s' version '%s'.", name, version)

    def search(self, tag: Optional[str] = None,
               name_prefix: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the registry by tag or name prefix.

        Args:
            tag: Return only versions carrying this tag.
            name_prefix: Return only models whose name starts with this
                prefix.

        Returns:
            List of metadata dicts matching the query.
        """
        results: List[Dict[str, Any]] = []
        for model_name, info in self._registry.items():
            if name_prefix and not model_name.startswith(name_prefix):
                continue
            for ver in info["versions"]:
                try:
                    meta = self.get_metadata(model_name, ver)
                except KeyError:
                    continue
                if tag is None or tag in meta.get("tags", []):
                    results.append(meta)
        return results
