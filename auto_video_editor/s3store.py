"""S3 access for the watch mode: listing, and transfers that are verified end to end.

Downloads land in a temporary file and are checked against the object's size (and MD5 when the
ETag is a plain MD5) before they are renamed. Uploads send ``Content-MD5`` so the server refuses a
corrupted body, then the stored object is read back and its size compared.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ENV_NAMES = {"endpoint": "AVE_S3_ENDPOINT", "bucket": "AVE_S3_BUCKET",
             "access_key": "AVE_S3_ACCESS_KEY", "secret_key": "AVE_S3_SECRET_KEY"}
DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_CHUNK = 1024 * 1024


class TransferError(RuntimeError):
    """A download or upload whose content does not match what S3 reports."""


def _mask(value: str) -> str:
    return "***" if value else ""


@dataclass
class S3Config:
    endpoint: str
    bucket: str
    access_key: str = field(repr=False)
    secret_key: str = field(repr=False)
    region: str = "ru-1"

    def __repr__(self) -> str:
        return (f"S3Config(endpoint={self.endpoint!r}, bucket={self.bucket!r}, "
                f"access_key={_mask(self.access_key)!r}, secret_key={_mask(self.secret_key)!r}, "
                f"region={self.region!r})")

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "S3Config":
        """Read ``AVE_S3_*`` from the environment, then from ``env_file`` (default: ``<repo>/.env``)."""
        values = _read_env_file(Path(env_file) if env_file else DEFAULT_ENV_FILE)
        resolved = {}
        for name, variable in ENV_NAMES.items():
            value = os.environ.get(variable) or values.get(variable, "")
            if not value.strip():
                raise RuntimeError(f"S3 is not configured: set {variable} in the environment or in "
                                   f"{env_file or DEFAULT_ENV_FILE}.")
            resolved[name] = value.strip()
        region = os.environ.get("AVE_S3_REGION") or values.get("AVE_S3_REGION") or "ru-1"
        return cls(region=region.strip(), **resolved)


def _read_env_file(path: Path) -> Dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass
class ObjectInfo:
    key: str
    etag: str
    size: int
    last_modified: Optional[datetime] = None
    metadata: Dict[str, str] = field(default_factory=dict)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


class S3Store:
    def __init__(self, config: S3Config, client=None) -> None:
        self.config = config
        self.bucket = config.bucket
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3", endpoint_url=config.endpoint, region_name=config.region,
                aws_access_key_id=config.access_key, aws_secret_access_key=config.secret_key,
                config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
            )
        self.client = client

    def list(self, prefix: str) -> List[ObjectInfo]:
        result = []
        for page in self.client.get_paginator("list_objects_v2").paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                result.append(ObjectInfo(item["Key"], item["ETag"].strip('"'), int(item["Size"]),
                                         item.get("LastModified")))
        return result

    def head(self, key: str) -> Optional[ObjectInfo]:
        from botocore.exceptions import ClientError

        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return ObjectInfo(key, response["ETag"].strip('"'), int(response["ContentLength"]),
                          response.get("LastModified"), dict(response.get("Metadata") or {}))

    def download(self, info: ObjectInfo, dest: Path) -> None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=info.key)
            digest = hashlib.md5()
            with open(tmp, "wb") as out:
                for chunk in response["Body"].iter_chunks(_CHUNK):
                    out.write(chunk)
                    digest.update(chunk)
            size = tmp.stat().st_size
            if size != info.size:
                raise TransferError(f"{info.key}: downloaded {size} bytes, expected {info.size}")
            if info.etag and "-" not in info.etag and digest.hexdigest() != info.etag.lower():
                raise TransferError(f"{info.key}: MD5 {digest.hexdigest()} does not match ETag {info.etag}")
            os.replace(tmp, dest)
        finally:
            if tmp.exists():
                tmp.unlink()

    def upload(self, path: Path, key: str, metadata: Dict[str, str],
               content_type: str = "video/mp4") -> ObjectInfo:
        path = Path(path)
        md5 = file_md5(path)
        with open(path, "rb") as body:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=body, Metadata=metadata,
                                   ContentType=content_type,
                                   ContentMD5=base64.b64encode(bytes.fromhex(md5)).decode("ascii"))
        info = self.head(key)
        size = path.stat().st_size
        if info is None or info.size != size:
            raise TransferError(f"{key}: stored {info.size if info else 'nothing'}, uploaded {size} bytes")
        return info

    def copy(self, src: str, dst: str) -> None:
        self.client.copy_object(Bucket=self.bucket, Key=dst, CopySource={"Bucket": self.bucket, "Key": src},
                                MetadataDirective="COPY")

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)

    def put_json(self, key: str, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json",
                               ContentMD5=base64.b64encode(hashlib.md5(body).digest()).decode("ascii"))

    def get_json(self, key: str) -> Optional[dict]:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if str(exc.response.get("Error", {}).get("Code", "")) in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        try:
            return json.loads(response["Body"].read().decode("utf-8"))
        except ValueError:
            return None
