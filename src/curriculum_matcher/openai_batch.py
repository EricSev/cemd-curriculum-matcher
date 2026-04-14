from __future__ import annotations

import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any
from urllib import error, request


OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_ENV_LOADED = False


def _load_repo_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        _ENV_LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)

    _ENV_LOADED = True


def _api_key() -> str:
    _load_repo_env()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")
    return api_key


def _default_headers() -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {_api_key()}",
    }
    organization = os.getenv("OPENAI_ORG_ID", "").strip()
    project = os.getenv("OPENAI_PROJECT", "").strip()
    if organization:
        headers["OpenAI-Organization"] = organization
    if project:
        headers["OpenAI-Project"] = project
    return headers


def _json_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = _default_headers()
    headers["Content-Type"] = "application/json"
    req = request.Request(f"{OPENAI_API_BASE}{path}", data=data, headers=headers, method=method)
    try:
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc


def _multipart_request(path: str, fields: dict[str, str], file_field: str, file_path: str | Path) -> dict[str, Any]:
    boundary = f"----OpenAIBatch{uuid.uuid4().hex}"
    file_path = Path(file_path)
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    parts: list[bytes] = []
    for name, value in fields.items():
        parts.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    file_bytes = file_path.read_bytes()
    parts.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    body = b"".join(parts)

    headers = _default_headers()
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    req = request.Request(f"{OPENAI_API_BASE}{path}", data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc


def upload_batch_file(file_path: str | Path) -> dict[str, Any]:
    return _multipart_request("/files", {"purpose": "batch"}, "file", file_path)


def create_batch(
    *,
    input_file_id: str,
    endpoint: str = "/v1/responses",
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    return _json_request("POST", "/batches", payload)


def retrieve_batch(batch_id: str) -> dict[str, Any]:
    return _json_request("GET", f"/batches/{batch_id}")


def cancel_batch(batch_id: str) -> dict[str, Any]:
    return _json_request("POST", f"/batches/{batch_id}/cancel")


def download_file_content(file_id: str) -> bytes:
    headers = _default_headers()
    req = request.Request(f"{OPENAI_API_BASE}/files/{file_id}/content", headers=headers, method="GET")
    try:
        with request.urlopen(req) as response:
            return response.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc
