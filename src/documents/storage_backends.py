from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ArtifactUploader(Protocol):
    async def upload(self, local_path: Path, *, content_type: str | None = None) -> str:
        """Upload local_path and return a publicly accessible URL."""
        ...


@dataclass
class NoopArtifactUploader:
    """Artifact uploader that simply returns the local path for testing."""

    async def upload(self, local_path: Path, *, content_type: str | None = None) -> str:
        return str(local_path)


@dataclass
class S3ArtifactUploader:
    """Upload artifacts to an S3-compatible object storage."""

    bucket: str
    prefix: str = ""
    region_name: str | None = None
    endpoint_url: str | None = None
    public_base_url: str | None = None
    acl: str | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)

    async def upload(self, local_path: Path, *, content_type: str | None = None) -> str:
        return await asyncio.to_thread(self._upload_sync, local_path, content_type)

    def _upload_sync(self, local_path: Path, content_type: str | None) -> str:
        try:
            import boto3  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("boto3 is required for S3ArtifactUploader") from exc

        client = boto3.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        )

        key = self._build_key(local_path)
        extra_args: dict[str, Any] = dict(self.extra_args)
        if content_type:
            extra_args.setdefault("ContentType", content_type)
        if self.acl:
            extra_args.setdefault("ACL", self.acl)

        kwargs: dict[str, Any] = {}
        if extra_args:
            kwargs["ExtraArgs"] = extra_args

        client.upload_file(str(local_path), self.bucket, key, **kwargs)
        return self._build_public_url(key)

    def _build_key(self, local_path: Path) -> str:
        safe_prefix = self.prefix.strip("/")
        unique = uuid.uuid4().hex
        filename = f"{unique}_{local_path.name}"
        if safe_prefix:
            return f"{safe_prefix}/{filename}"
        return filename

    def _build_public_url(self, key: str) -> str:
        if self.public_base_url:
            base = self.public_base_url.rstrip("/")
            return f"{base}/{key}"

        if self.endpoint_url:
            base = self.endpoint_url.rstrip("/")
            return f"{base}/{self.bucket}/{key}"

        region = self.region_name or "us-east-1"
        if region == "us-east-1":
            return f"https://{self.bucket}.s3.amazonaws.com/{key}"
        return f"https://{self.bucket}.s3.{region}.amazonaws.com/{key}"


__all__ = [
    "ArtifactUploader",
    "NoopArtifactUploader",
    "S3ArtifactUploader",
]
