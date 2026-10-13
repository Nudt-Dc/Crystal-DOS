"""Data-use-gated AFLOW metadata inventory for external confirmation.

This module cannot contact AFLOW unless either a locally archived written
authorization or a frozen public-terms basis passes validation. It retrieves
metadata only; raw DOS files, model training, inference, and metric access are
outside this stage.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable
from urllib.request import Request, urlopen
import uuid


SCHEMA_VERSION = "aflow_confirmatory_metadata_inventory_v2"
AUTH_SCHEMA_VERSION = "aflow_data_use_authorization_v1"
PUBLIC_TERMS_SCHEMA_VERSION = "aflow_public_terms_basis_v1"
QUERY_PREFIX = (
    "https://aflow.org/API/aflux/?"
    "ael_bulk_modulus_vrh(*),ael_shear_modulus_vrh(*),"
    "$files(*'DOSCAR.static.xz'*),positions_fractional(*),"
    "auid,compound,aurl"
)
POSITION_KEY = re.compile(r"^([1-9][0-9]*) of ([1-9][0-9]*)$")
REQUIRED_RECORD_FIELDS = {
    "ael_bulk_modulus_vrh",
    "ael_shear_modulus_vrh",
    "auid",
    "compound",
    "aurl",
    "positions_fractional",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_authorization(record_path: Path | str) -> dict[str, Any]:
    """Validate an archived positive written response before any network call."""
    path = Path(record_path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "decision",
        "systematic_selective_retrieval_allowed",
        "received_at_utc",
        "sender_email",
        "message_file",
        "message_sha256",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("authorization record is missing required fields")
    if record["schema_version"] != AUTH_SCHEMA_VERSION:
        raise ValueError("authorization record has an unsupported schema version")
    if record["decision"] != "authorized":
        raise PermissionError("AFLOW written authorization is not positive")
    if record["systematic_selective_retrieval_allowed"] is not True:
        raise PermissionError("AFLOW selective retrieval is not explicitly allowed")
    if not isinstance(record["sender_email"], str) or "@" not in record["sender_email"]:
        raise ValueError("authorization sender email is invalid")
    try:
        received = datetime.fromisoformat(str(record["received_at_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("authorization receipt time is invalid") from exc
    if received.tzinfo is None:
        raise ValueError("authorization receipt time must include a timezone")
    message_path = Path(record["message_file"])
    if not message_path.is_absolute():
        message_path = path.parent / message_path
    message_path = message_path.resolve()
    if not message_path.is_file():
        raise FileNotFoundError("archived authorization message is missing")
    actual_hash = sha256_file(message_path)
    if actual_hash != str(record["message_sha256"]).lower():
        raise ValueError("archived authorization message hash mismatch")
    normalized = dict(record)
    normalized["authorization_record_file"] = str(path)
    normalized["message_file"] = str(message_path)
    normalized["message_sha256"] = actual_hash
    normalized["basis_type"] = "written_authorization"
    normalized["basis_record_file"] = str(path)
    normalized["evidence_file"] = str(message_path)
    normalized["evidence_sha256"] = actual_hash
    return normalized


def validate_public_terms_basis(record_path: Path | str) -> dict[str, Any]:
    """Validate the frozen basis for internal selective non-commercial use."""
    path = Path(record_path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "decision",
        "scientific_academic_noncommercial_use",
        "selective_retrieval_only",
        "whole_repository_download",
        "raw_aflow_redistribution_allowed",
        "recorded_at_utc",
        "terms_source_url",
        "terms_snapshot_file",
        "terms_snapshot_sha256",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("public-terms basis is missing required fields")
    if record["schema_version"] != PUBLIC_TERMS_SCHEMA_VERSION:
        raise ValueError("public-terms basis has an unsupported schema version")
    if record["decision"] != "allowed_for_internal_noncommercial_selective_retrieval":
        raise PermissionError("public-terms basis does not allow this retrieval")
    if record["scientific_academic_noncommercial_use"] is not True:
        raise PermissionError("scientific non-commercial scope is not affirmed")
    if record["selective_retrieval_only"] is not True:
        raise PermissionError("retrieval is not constrained to the selected cohort")
    if record["whole_repository_download"] is not False:
        raise PermissionError("whole-repository download is prohibited")
    if record["raw_aflow_redistribution_allowed"] is not False:
        raise PermissionError("raw AFLOW redistribution must remain disabled")
    source_url = str(record["terms_source_url"])
    if not source_url.startswith("https://aflow.org/"):
        raise ValueError("public terms source must be an official AFLOW URL")
    try:
        recorded = datetime.fromisoformat(
            str(record["recorded_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("public-terms record time is invalid") from exc
    if recorded.tzinfo is None:
        raise ValueError("public-terms record time must include a timezone")
    snapshot_path = Path(record["terms_snapshot_file"])
    if not snapshot_path.is_absolute():
        snapshot_path = path.parent / snapshot_path
    snapshot_path = snapshot_path.resolve()
    if not snapshot_path.is_file():
        raise FileNotFoundError("public terms snapshot is missing")
    actual_hash = sha256_file(snapshot_path)
    if actual_hash != str(record["terms_snapshot_sha256"]).lower():
        raise ValueError("public terms snapshot hash mismatch")
    normalized = dict(record)
    normalized["basis_type"] = "public_terms"
    normalized["basis_record_file"] = str(path)
    normalized["evidence_file"] = str(snapshot_path)
    normalized["evidence_sha256"] = actual_hash
    return normalized


def validate_data_use_basis(record_path: Path | str) -> dict[str, Any]:
    path = Path(record_path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    schema_version = value.get("schema_version") if isinstance(value, dict) else None
    if schema_version == AUTH_SCHEMA_VERSION:
        return validate_authorization(path)
    if schema_version == PUBLIC_TERMS_SCHEMA_VERSION:
        return validate_public_terms_basis(path)
    raise ValueError("data-use basis has an unsupported schema version")


def build_query(page: int, page_size: int) -> str:
    if page < 1:
        raise ValueError("page must be positive")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be in [1, 100]")
    return f"{QUERY_PREFIX},paging({page},{page_size})"


def parse_page(payload: bytes) -> tuple[int, list[tuple[int, dict[str, Any]]]]:
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("AFLUX page must be a non-empty object")
    total: int | None = None
    parsed: list[tuple[int, dict[str, Any]]] = []
    for key, record in value.items():
        match = POSITION_KEY.fullmatch(str(key))
        if match is None or not isinstance(record, dict):
            raise ValueError("AFLUX page has an invalid position key or record")
        position, candidate_total = map(int, match.groups())
        if total is None:
            total = candidate_total
        elif candidate_total != total:
            raise ValueError("AFLUX page contains inconsistent totals")
        if not REQUIRED_RECORD_FIELDS.issubset(record):
            raise ValueError("AFLUX metadata record is missing required fields")
        if not str(record["auid"]).startswith("aflow:"):
            raise ValueError("AFLUX metadata record has an invalid AUID")
        parsed.append((position, record))
    positions = [position for position, _ in parsed]
    if len(positions) != len(set(positions)):
        raise ValueError("AFLUX page contains duplicate positions")
    return int(total), sorted(parsed)


def fetch_page(url: str, *, timeout_seconds: int = 60) -> tuple[bytes, dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "CrystalDOS-authorized-inventory/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read(), {
            "http_status": int(response.status),
            "content_type": response.headers.get("Content-Type"),
            "date": response.headers.get("Date"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }


def materialize_inventory(
    data_use_record: Path | str,
    output_dir: Path | str,
    *,
    page_size: int = 100,
    request_delay_seconds: float = 1.0,
    timeout_seconds: int = 60,
    fetcher: Callable[..., tuple[bytes, dict[str, Any]]] = fetch_page,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Retrieve the permitted metadata cohort with raw page hashes and sealing."""
    data_use_basis = validate_data_use_basis(data_use_record)
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be in [1, 100]")
    if request_delay_seconds < 1.0:
        raise ValueError("request_delay_seconds must be at least 1.0")
    if not 1 <= timeout_seconds <= 120:
        raise ValueError("timeout_seconds must be in [1, 120]")

    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite inventory: {destination}")
    staging = destination.with_name(
        f"{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seen_auids: set[str] = set()
    expected_position = 1
    total_count: int | None = None
    try:
        (staging / "pages").mkdir(parents=True)
        page = 1
        while total_count is None or len(records) < total_count:
            url = build_query(page, page_size)
            payload, headers = fetcher(url, timeout_seconds=timeout_seconds)
            page_total, parsed = parse_page(payload)
            if total_count is None:
                total_count = page_total
            elif page_total != total_count:
                raise ValueError("AFLUX total changed during inventory retrieval")
            positions = [position for position, _ in parsed]
            required_positions = list(
                range(expected_position, expected_position + len(parsed))
            )
            if positions != required_positions:
                raise ValueError("AFLUX page positions are not contiguous")
            for _, record in parsed:
                auid = str(record["auid"])
                if auid in seen_auids:
                    raise ValueError("AFLUX inventory contains a duplicate AUID")
                seen_auids.add(auid)
                records.append(record)
            filename = f"page_{page:04d}.json"
            page_path = staging / "pages" / filename
            page_path.write_bytes(payload)
            receipts.append({
                "page": page,
                "query_url": url,
                "record_count": len(parsed),
                "first_position": positions[0],
                "last_position": positions[-1],
                "response_bytes": len(payload),
                "response_sha256": sha256_bytes(payload),
                "response_headers": headers,
                "snapshot_file": f"pages/{filename}",
            })
            expected_position += len(parsed)
            if len(records) < total_count:
                sleeper(request_delay_seconds)
            page += 1
        if len(records) != total_count or expected_position != total_count + 1:
            raise ValueError("AFLUX inventory is incomplete")

        records_path = staging / "candidate_records.jsonl"
        records_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
            newline="\n",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "complete": True,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_use_basis": {
                "basis_type": data_use_basis["basis_type"],
                "record_file": data_use_basis["basis_record_file"],
                "record_sha256": sha256_file(data_use_basis["basis_record_file"]),
                "evidence_file": data_use_basis["evidence_file"],
                "evidence_sha256": data_use_basis["evidence_sha256"],
                "raw_aflow_redistribution_allowed": data_use_basis.get(
                    "raw_aflow_redistribution_allowed", False
                ),
            },
            "query_prefix": QUERY_PREFIX,
            "page_size": page_size,
            "request_delay_seconds": request_delay_seconds,
            "record_count": len(records),
            "unique_auid_count": len(seen_auids),
            "page_receipts": receipts,
            "candidate_records_sha256": sha256_file(records_path),
            "raw_dos_downloaded": False,
            "model_training_performed": False,
            "model_inference_performed": False,
            "model_metrics_read": False,
        }
        (staging / "inventory_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        staging.replace(destination)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
