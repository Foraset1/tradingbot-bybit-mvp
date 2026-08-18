"""Deterministic archive acceptance for audited market-data partitions.

The strict audit remains the source of truth. Daily archival may preserve a
structurally valid partition whose only findings are kline gaps, but downstream
research must then prove continuity for every feature and label window.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, cast

ARCHIVE_ACCEPTANCE_POLICY: Final = "continuity_aware_v1"
ARCHIVE_ALLOWED_WARNING_CODES: Final = frozenset({"kline_gap"})
ARCHIVE_QUALITY_CLEAN: Final = "clean"
ARCHIVE_QUALITY_GAPPED: Final = "gapped"


@dataclass(frozen=True, slots=True)
class ArchiveAcceptance:
    """Validated decision separating preservation from research eligibility."""

    policy: str | None
    ok: bool
    quality_status: str
    observed_warning_codes: tuple[str, ...]
    warning_items: int
    warning_occurrences: int
    reasons: tuple[str, ...]

    @property
    def requires_continuity_filter(self) -> bool:
        return self.quality_status == ARCHIVE_QUALITY_GAPPED

    def to_dict(self) -> dict[str, object]:
        if self.policy is None:
            raise ValueError("legacy acceptance has no serializable policy block")
        return {
            "policy": self.policy,
            "ok": self.ok,
            "quality_status": self.quality_status,
            "allowed_warning_codes": sorted(ARCHIVE_ALLOWED_WARNING_CODES),
            "observed_warning_codes": list(self.observed_warning_codes),
            "warning_items": self.warning_items,
            "warning_occurrences": self.warning_occurrences,
            "training_requires_continuity_filter": self.requires_continuity_filter,
            "reasons": list(self.reasons),
        }

    def quality_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy or "legacy_strict_v1",
            "status": self.quality_status,
            "warning_codes": list(self.observed_warning_codes),
            "warning_items": self.warning_items,
            "warning_occurrences": self.warning_occurrences,
            "training_requires_continuity_filter": self.requires_continuity_filter,
        }


def _array(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"audit report {key} must be an array")
    return value


def assess_archive_payload(payload: Mapping[str, object]) -> ArchiveAcceptance:
    """Assess an audit report without trusting a stored acceptance decision."""

    reasons: list[str] = []
    errors = _array(payload, "errors")
    warnings = _array(payload, "warnings")
    partial_files = _array(payload, "partial_files")
    missing = _array(payload, "missing_expected_streams")
    short = _array(payload, "short_streams")

    warning_codes: set[str] = set()
    warning_occurrences = 0
    for index, raw_warning in enumerate(warnings):
        if not isinstance(raw_warning, dict):
            raise ValueError(f"audit warning {index} must be an object")
        warning = cast(dict[str, Any], raw_warning)
        code = warning.get("code")
        count = warning.get("count")
        if not isinstance(code, str) or not code:
            raise ValueError(f"audit warning {index} has an invalid code")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"audit warning {index} has an invalid count")
        warning_codes.add(code)
        warning_occurrences += count

    readiness_raw = payload.get("readiness")
    if not isinstance(readiness_raw, dict):
        raise ValueError("audit report readiness must be an object")
    readiness = cast(dict[str, Any], readiness_raw)
    readiness_reasons = readiness.get("reasons")
    if not isinstance(readiness_reasons, list) or not all(
        isinstance(item, str) for item in readiness_reasons
    ):
        raise ValueError("audit readiness.reasons must be a string array")

    if readiness.get("strict") is not True:
        reasons.append("strict_audit_required")
    if errors:
        reasons.append("validation_errors")
    if partial_files or payload.get("partial_file_count") != 0:
        reasons.append("partial_files_present")
    if missing:
        reasons.append("missing_expected_streams")
    if short:
        reasons.append("minimum_duration_not_met")
    unsupported = warning_codes - ARCHIVE_ALLOWED_WARNING_CODES
    if unsupported:
        reasons.append("unsupported_warning_codes")
    if payload.get("file_count") in (None, 0):
        reasons.append("empty_file_manifest")

    expected_readiness_reasons = ["strict_warnings"] if warnings else []
    expected_ready = not warnings
    if (
        readiness.get("ok") is not expected_ready
        or readiness_reasons != expected_readiness_reasons
    ):
        reasons.append("inconsistent_strict_readiness")

    quality = ARCHIVE_QUALITY_GAPPED if warnings else ARCHIVE_QUALITY_CLEAN
    return ArchiveAcceptance(
        policy=ARCHIVE_ACCEPTANCE_POLICY,
        ok=not reasons,
        quality_status=quality,
        observed_warning_codes=tuple(sorted(warning_codes)),
        warning_items=len(warnings),
        warning_occurrences=warning_occurrences,
        reasons=tuple(reasons),
    )


def attach_archive_acceptance(payload: dict[str, object]) -> ArchiveAcceptance:
    """Attach the deterministic policy decision to an audit payload in place."""

    acceptance = assess_archive_payload(payload)
    payload["archive_acceptance"] = acceptance.to_dict()
    return acceptance


def read_archive_acceptance(payload: Mapping[str, object]) -> ArchiveAcceptance:
    """Validate a stored decision, while retaining legacy strict-clean support."""

    expected = assess_archive_payload(payload)
    stored = payload.get("archive_acceptance")
    if stored is None:
        if not expected.ok or expected.quality_status != ARCHIVE_QUALITY_CLEAN:
            raise ValueError("gapped audit requires a validated archive_acceptance block")
        return ArchiveAcceptance(
            policy=None,
            ok=True,
            quality_status=ARCHIVE_QUALITY_CLEAN,
            observed_warning_codes=(),
            warning_items=0,
            warning_occurrences=0,
            reasons=(),
        )
    if not isinstance(stored, dict):
        raise ValueError("archive_acceptance must be an object")
    if cast(dict[str, Any], stored) != expected.to_dict():
        raise ValueError("archive_acceptance does not match the audit findings")
    if not expected.ok:
        raise ValueError(
            "audit report is not accepted by the continuity-aware archive policy"
        )
    return expected
