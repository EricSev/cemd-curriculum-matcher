from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import PurePosixPath
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

VENDOR_DOMAINS = (
    "savvas.com",
    "hmhco.com",
    "mheducation.com",
    "benchmarkeducation.com",
    "greatminds.org",
    "lexialearning.com",
    "edmentum.com",
    "ixl.com",
)


def classify_url(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    if not parsed.scheme or not parsed.netloc:
        return "invalid"
    if "docs.google.com" in domain:
        return "google_doc"
    if path.endswith(".pdf"):
        return "pdf"
    if any(domain.endswith(vendor) for vendor in VENDOR_DOMAINS):
        return "vendor_page"
    if any(token in path for token in ("board", "procurement", "bid", "rfp", "agenda")):
        return "procurement_or_board_doc"
    if domain.endswith(".k12.ca.us") or domain.endswith(".org") or domain.endswith(".edu"):
        return "district_or_org_page"
    return "web_page"


def extract_title_from_html(text: str) -> str:
    match = TITLE_RE.search(text)
    if not match:
        return ""
    title = unescape(TAG_RE.sub(" ", match.group(1)))
    return WHITESPACE_RE.sub(" ", title).strip()


def extract_visible_text(text: str, *, max_chars: int = 600) -> str:
    stripped = unescape(TAG_RE.sub(" ", text))
    stripped = WHITESPACE_RE.sub(" ", stripped).strip()
    return stripped[:max_chars]


def document_filename(url: str) -> str:
    parsed = urlparse(url)
    return PurePosixPath(parsed.path).name


def lightweight_fetch(url: str, *, timeout_seconds: int = 8, max_bytes: int = 65536) -> dict[str, Any]:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return {
            "fetch_status": "invalid_url",
            "http_status": "",
            "content_type": "",
            "final_url": url,
            "page_title": "",
            "visible_text_excerpt": "",
        }
    request = Request(
        url,
        headers={"User-Agent": "cemd-curriculum-matcher/0.1"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get("Content-Type", "")
            final_url = response.geturl()
            payload = response.read(max_bytes)
            text = payload.decode("utf-8", errors="ignore")
            result = {
                "fetch_status": "success",
                "http_status": getattr(response, "status", 200),
                "content_type": content_type,
                "final_url": final_url,
                "page_title": extract_title_from_html(text) if "html" in content_type else "",
                "visible_text_excerpt": extract_visible_text(text) if "html" in content_type else "",
            }
            return result
    except HTTPError as exc:
        return {
            "fetch_status": "http_error",
            "http_status": exc.code,
            "content_type": "",
            "final_url": url,
            "page_title": "",
            "visible_text_excerpt": "",
        }
    except (URLError, TimeoutError, ValueError):
        return {
            "fetch_status": "network_error",
            "http_status": "",
            "content_type": "",
            "final_url": url,
            "page_title": "",
            "visible_text_excerpt": "",
        }


@dataclass(frozen=True)
class UrlFieldAudit:
    field_name: str
    url: str
    url_type: str
    domain: str
    document_filename: str
    fetch_status: str
    http_status: Any
    content_type: str
    final_url: str
    page_title: str
    visible_text_excerpt: str


def audit_url_field(
    field_name: str,
    value: Any,
    *,
    fetch_live: bool,
    timeout_seconds: int = 8,
    max_bytes: int = 65536,
) -> UrlFieldAudit:
    url = "" if pd.isna(value) else str(value).strip()
    if not url:
        return UrlFieldAudit(
            field_name=field_name,
            url="",
            url_type="missing",
            domain="",
            document_filename="",
            fetch_status="missing",
            http_status="",
            content_type="",
            final_url="",
            page_title="",
            visible_text_excerpt="",
        )

    parsed = urlparse(url)
    metadata = {
        "fetch_status": "not_fetched",
        "http_status": "",
        "content_type": "",
        "final_url": url,
        "page_title": "",
        "visible_text_excerpt": "",
    }
    if fetch_live:
        metadata = lightweight_fetch(
            url, timeout_seconds=timeout_seconds, max_bytes=max_bytes
        )
    return UrlFieldAudit(
        field_name=field_name,
        url=url,
        url_type=classify_url(url),
        domain=parsed.netloc.lower(),
        document_filename=document_filename(url),
        fetch_status=str(metadata["fetch_status"]),
        http_status=metadata["http_status"],
        content_type=str(metadata["content_type"]),
        final_url=str(metadata["final_url"]),
        page_title=str(metadata["page_title"]),
        visible_text_excerpt=str(metadata["visible_text_excerpt"]),
    )


def normalize_evidence_text(*parts: Any) -> str:
    combined = " ".join(str(part) for part in parts if part)
    combined = combined.lower()
    combined = re.sub(r"[^a-z0-9\s]+", " ", combined)
    combined = WHITESPACE_RE.sub(" ", combined).strip()
    return combined


def audit_row_urls(
    row: dict[str, Any] | pd.Series,
    *,
    fetch_live: bool,
    timeout_seconds: int = 8,
    max_bytes: int = 65536,
) -> list[UrlFieldAudit]:
    return [
        audit_url_field(
            "adopted_curriculum_source_url",
            row.get("adopted_curriculum_source_url"),
            fetch_live=fetch_live,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
        ),
        audit_url_field(
            "source_document_link",
            row.get("source_document_link"),
            fetch_live=fetch_live,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
        ),
    ]
