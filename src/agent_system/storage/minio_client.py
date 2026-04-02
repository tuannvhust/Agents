"""MinIO client — handles file upload/download for agent artifacts."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO

from agent_system.config import get_settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """Thin wrapper around the official MinIO Python SDK.

    Provides high-level helpers for the agent system:
      - Upload bytes or local files
      - Download objects to bytes or local paths
      - List and delete objects
      - Presigned URL generation for temporary access
    """

    def __init__(self) -> None:
        self._cfg = get_settings().minio
        self._client = self._build_client()
        self._ensure_bucket()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_client(self):  # type: ignore[return]
        try:
            from minio import Minio

            return Minio(
                endpoint=self._cfg.endpoint,
                access_key=self._cfg.access_key,
                secret_key=self._cfg.secret_key,
                secure=self._cfg.secure,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialise MinIO client: %s", exc)
            return None

    def _ensure_bucket(self) -> None:
        if self._client is None:
            return
        try:
            if not self._client.bucket_exists(self._cfg.bucket):
                self._client.make_bucket(self._cfg.bucket)
                logger.info("Created MinIO bucket: %s", self._cfg.bucket)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not ensure MinIO bucket exists: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def upload_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes; returns the object name (key)."""
        self._require_client()
        stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=self._cfg.bucket,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        logger.debug("Uploaded %d bytes → %s/%s", len(data), self._cfg.bucket, object_name)
        return object_name

    def upload_file(self, object_name: str, file_path: str | Path) -> str:
        """Upload a local file; returns the object name."""
        self._require_client()
        self._client.fput_object(
            bucket_name=self._cfg.bucket,
            object_name=object_name,
            file_path=str(file_path),
        )
        logger.debug("Uploaded file %s → %s/%s", file_path, self._cfg.bucket, object_name)
        return object_name

    def download_bytes(self, object_name: str) -> bytes:
        """Download an object and return its raw bytes."""
        self._require_client()
        response = self._client.get_object(self._cfg.bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def download_file(self, object_name: str, dest_path: str | Path) -> Path:
        """Download an object to a local path."""
        self._require_client()
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(self._cfg.bucket, object_name, str(dest))
        return dest

    def presigned_url(self, object_name: str, expires_seconds: int = 3600) -> str:
        """Generate a presigned download URL valid for ``expires_seconds``."""
        from datetime import timedelta

        self._require_client()
        return self._client.presigned_get_object(
            self._cfg.bucket,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )

    def list_objects(self, prefix: str = "") -> list[str]:
        self._require_client()
        objects = self._client.list_objects(self._cfg.bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def delete_object(self, object_name: str) -> None:
        self._require_client()
        self._client.remove_object(self._cfg.bucket, object_name)
        logger.debug("Deleted %s/%s", self._cfg.bucket, object_name)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError("MinIO client is not available. Check your configuration.")
