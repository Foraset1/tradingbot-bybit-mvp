from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tradingbot.research.contracts import (
    FEATURE_SCHEMA,
    LABEL_SCHEMA,
    MICROSTRUCTURE_RESEARCH_PROFILE,
    PRICE_FEATURE_SCHEMA,
    PRICE_LABEL_SCHEMA,
    PRICE_RESEARCH_PROFILE,
    RESEARCH_SCHEMA_VERSION,
)
from tradingbot.research.evaluation_contracts import (
    DIRECT_FEATURE_COLUMNS,
    LOG1P_FEATURE_COLUMNS,
    OUTCOME_NAMES,
    PRICE_DIRECT_FEATURE_COLUMNS,
    PRICE_LOG1P_FEATURE_COLUMNS,
    EvaluationError,
    PreparedData,
    ResearchDataset,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label).lower()
    if len(text) != 64:
        raise EvaluationError(f"{label} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise EvaluationError(f"{label} must be a SHA-256 digest") from exc
    return text


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{label} is unreadable: {path}") from exc
    return _object(parsed, label)


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    path = PurePosixPath(_string(value, label))
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise EvaluationError(f"{label} is not a safe relative path")
    return path


def _schema_manifest(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def validate_research_dataset(dataset_path: str | Path) -> ResearchDataset:
    """Verify every immutable research output before it is used for modeling."""

    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise EvaluationError(f"research dataset directory does not exist: {root}")
    manifest = _load_json(root / "manifest.json", "research manifest")
    if manifest.get("research_schema_version") != RESEARCH_SCHEMA_VERSION:
        raise EvaluationError("research dataset uses an unsupported schema version")
    raw_profile = manifest.get("research_profile", MICROSTRUCTURE_RESEARCH_PROFILE)
    research_profile = _string(raw_profile, "research_profile")
    if research_profile == MICROSTRUCTURE_RESEARCH_PROFILE:
        feature_schema = FEATURE_SCHEMA
        label_schema = LABEL_SCHEMA
        expected_feature_rule = "received_at_ns <= decision_at_ns"
        expected_label_rule = (
            "decision_at_ns < trade.received_at_ns <= label_end_ns"
        )
    elif research_profile == PRICE_RESEARCH_PROFILE:
        feature_schema = PRICE_FEATURE_SCHEMA
        label_schema = PRICE_LABEL_SCHEMA
        expected_feature_rule = "available_at_ns <= decision_at_ns"
        expected_label_rule = (
            "decision_at_ns < trade_bar_1s.available_at_ns <= label_end_ns"
        )
    else:
        raise EvaluationError(f"unsupported research profile: {research_profile}")
    dataset_id = _string(manifest.get("research_dataset_id"), "research_dataset_id")
    if root.name != dataset_id:
        raise EvaluationError("research dataset ID does not match its directory")
    input_fingerprint = _sha256(manifest.get("input_fingerprint"), "input_fingerprint")
    output_fingerprint = _sha256(
        manifest.get("output_fingerprint"), "output_fingerprint"
    )

    causality = _object(manifest.get("causality"), "causality")
    if causality.get("feature_rule") != expected_feature_rule:
        raise EvaluationError("research feature causality contract is missing")
    if causality.get("label_rule") != expected_label_rule:
        raise EvaluationError("research label causality contract is missing")
    if causality.get("execution_labels_included") is not False:
        raise EvaluationError("research dataset unexpectedly claims execution labels")
    if causality.get("maker_fill_claimed") is not False:
        raise EvaluationError("research dataset unexpectedly claims maker fills")

    schemas = _object(manifest.get("schemas"), "schemas")
    if schemas.get("features") != _schema_manifest(feature_schema):
        raise EvaluationError("research feature schema manifest is inconsistent")
    if schemas.get("labels") != _schema_manifest(label_schema):
        raise EvaluationError("research label schema manifest is inconsistent")

    source = _object(manifest.get("source"), "source")
    source_dataset_id = _string(source.get("dataset_id"), "source.dataset_id")
    source_output_fingerprint = _sha256(
        source.get("output_fingerprint"), "source.output_fingerprint"
    )
    raw_symbols = source.get("symbols")
    if (
        not isinstance(raw_symbols, list)
        or not raw_symbols
        or not all(isinstance(symbol, str) and symbol for symbol in raw_symbols)
        or len(raw_symbols) != len(set(raw_symbols))
    ):
        raise EvaluationError("source.symbols must be a unique non-empty string list")
    symbols = tuple(cast(list[str], raw_symbols))

    manifest_copy_name = _string(source.get("manifest_copy"), "source.manifest_copy")
    if manifest_copy_name != "source-manifest.json":
        raise EvaluationError("source manifest copy has an unexpected name")
    source_manifest_path = root / manifest_copy_name
    source_manifest_sha = _sha256(
        source.get("manifest_sha256"), "source.manifest_sha256"
    )
    if not source_manifest_path.is_file() or _sha256_file(source_manifest_path) != (
        source_manifest_sha
    ):
        raise EvaluationError("source-manifest.json failed SHA-256 validation")
    source_manifest = _load_json(source_manifest_path, "source manifest copy")
    if source_manifest.get("dataset_id") != source_dataset_id:
        raise EvaluationError("source dataset ID differs from its copied manifest")
    if source_manifest.get("output_fingerprint") != source_output_fingerprint:
        raise EvaluationError("source fingerprint differs from its copied manifest")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise EvaluationError("research manifest.files must be a non-empty array")
    descriptors: list[dict[str, object]] = []
    paths_by_table: dict[str, list[Path]] = {"features": [], "labels": []}
    rows_by_table: Counter[str] = Counter()
    seen_paths: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        item = _object(raw_file, f"files[{index}]")
        relative = _safe_relative_path(item.get("path"), f"files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise EvaluationError(f"duplicate research output path: {relative_text}")
        seen_paths.add(relative_text)
        table_name = _string(item.get("table"), f"files[{index}].table")
        if table_name not in paths_by_table:
            raise EvaluationError(f"unsupported research table: {table_name}")
        symbol = _string(item.get("symbol"), f"files[{index}].symbol")
        if symbol not in symbols:
            raise EvaluationError(f"research file contains unknown symbol: {symbol}")
        date = _string(item.get("date"), f"files[{index}].date")
        rows = _nonnegative_int(item.get("rows"), f"files[{index}].rows")
        size = _nonnegative_int(item.get("bytes"), f"files[{index}].bytes")
        digest = _sha256(item.get("sha256"), f"files[{index}].sha256")
        expected_prefix = f"table={table_name}/symbol={symbol}/date={date}/"
        if not relative_text.startswith(expected_prefix):
            raise EvaluationError(f"research partition path is inconsistent: {relative}")
        actual = root.joinpath(*relative.parts).resolve()
        if not actual.is_relative_to(root) or not actual.is_file():
            raise EvaluationError(f"research output file is missing: {relative}")
        expected_schema = feature_schema if table_name == "features" else label_schema
        if actual.stat().st_size != size or _sha256_file(actual) != digest:
            raise EvaluationError(f"research output file is corrupted: {relative}")
        try:
            parquet = pq.ParquetFile(actual)
        except (OSError, pa.ArrowInvalid) as exc:
            raise EvaluationError(
                f"research output file is corrupted: {relative}"
            ) from exc
        if parquet.metadata.num_rows != rows or not parquet.schema_arrow.equals(
            expected_schema, check_metadata=True
        ):
            raise EvaluationError(f"research output file is corrupted: {relative}")
        descriptor: dict[str, object] = {
            "path": relative_text,
            "table": table_name,
            "symbol": symbol,
            "date": date,
            "rows": rows,
            "bytes": size,
            "sha256": digest,
        }
        descriptors.append(descriptor)
        paths_by_table[table_name].append(actual)
        rows_by_table[table_name] += rows

    if len(descriptors) != manifest.get("output_file_count"):
        raise EvaluationError("research output_file_count is inconsistent")
    if _sha256_json(sorted(descriptors, key=lambda item: cast(str, item["path"]))) != (
        output_fingerprint
    ):
        raise EvaluationError("research output_fingerprint is inconsistent")
    output_rows = _object(manifest.get("output_rows"), "output_rows")
    expected_rows = {
        key: _nonnegative_int(value, f"output_rows.{key}")
        for key, value in output_rows.items()
    }
    if dict(sorted(rows_by_table.items())) != dict(sorted(expected_rows.items())):
        raise EvaluationError("research output row totals are inconsistent")
    if rows_by_table["features"] <= 0 or rows_by_table["labels"] <= 0:
        raise EvaluationError("research dataset contains no modelable rows")

    return ResearchDataset(
        root=root,
        research_dataset_id=dataset_id,
        research_profile=research_profile,
        source_dataset_id=source_dataset_id,
        input_fingerprint=input_fingerprint,
        output_fingerprint=output_fingerprint,
        symbols=symbols,
        feature_paths=tuple(sorted(paths_by_table["features"])),
        label_paths=tuple(sorted(paths_by_table["labels"])),
        feature_rows=rows_by_table["features"],
        label_rows=rows_by_table["labels"],
        manifest=manifest,
    )


def _read_tables(paths: tuple[Path, ...], columns: list[str]) -> pa.Table:
    tables = [pq.ParquetFile(path).read(columns=columns) for path in paths]
    if not tables:
        raise EvaluationError("research dataset has no Parquet inputs")
    return pa.concat_tables(tables)


def _count_true(values: pa.Array | pa.ChunkedArray) -> int:
    result = pc.sum(pc.cast(values, pa.int64())).as_py()
    return 0 if result is None else int(result)


def _float_array(table: pa.Table, name: str) -> NDArray[np.float64]:
    values = pc.cast(table.column(name).combine_chunks(), pa.float64())
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)


def _int_array(
    table: pa.Table, name: str, *, null_value: int | None = None
) -> NDArray[np.int64]:
    values: pa.Array | pa.ChunkedArray = table.column(name).combine_chunks()
    if null_value is not None:
        values = pc.fill_null(values, pa.scalar(null_value, type=pa.int64()))
    return np.asarray(
        pc.cast(values, pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64
    )


def prepare_evaluation_data(
    dataset: ResearchDataset, *, horizon_minutes: int
) -> PreparedData:
    """Join one market horizon to causal features and create a bounded float32 matrix."""

    if horizon_minutes not in {5, 15, 30, 60}:
        raise EvaluationError("horizon_minutes must be one of 5, 15, 30, or 60")
    direct_feature_columns: tuple[str, ...]
    log1p_feature_columns: tuple[str, ...]
    if dataset.research_profile == PRICE_RESEARCH_PROFILE:
        direct_feature_columns = PRICE_DIRECT_FEATURE_COLUMNS
        log1p_feature_columns = PRICE_LOG1P_FEATURE_COLUMNS
    else:
        direct_feature_columns = DIRECT_FEATURE_COLUMNS
        log1p_feature_columns = LOG1P_FEATURE_COLUMNS
    model_columns = list(
        dict.fromkeys(direct_feature_columns + log1p_feature_columns)
    )
    feature_columns = [
        "decision_id",
        "symbol",
        "decision_at_ns",
        *model_columns,
    ]
    label_columns = [
        "decision_id",
        "symbol",
        "decision_at_ns",
        "side",
        "horizon_minutes",
        "label_end_ns",
        "stop_distance_bps",
        "take_profit_distance_bps",
        "outcome",
        "hit_at_ns",
        "outcome_return_bps",
    ]
    features = _read_tables(dataset.feature_paths, feature_columns)
    labels = _read_tables(dataset.label_paths, label_columns)
    labels = labels.filter(
        pc.equal(labels.column("horizon_minutes"), pa.scalar(horizon_minutes))
    )
    if labels.num_rows == 0:
        raise EvaluationError(f"research dataset has no {horizon_minutes}m labels")

    ambiguous = pc.equal(labels.column("outcome"), pa.scalar("AMBIGUOUS"))
    excluded_ambiguous = _count_true(ambiguous)
    recognized = pc.is_in(
        labels.column("outcome"), value_set=pa.array(OUTCOME_NAMES, type=pa.string())
    )
    unpriced = pc.is_null(labels.column("outcome_return_bps"))
    excluded_unpriced = _count_true(pc.and_(recognized, unpriced))
    usable = pc.and_(recognized, pc.invert(unpriced))
    labels = labels.filter(usable)
    if labels.num_rows == 0:
        raise EvaluationError("research dataset has no resolved, priced labels")

    labels = labels.rename_columns(
        [
            "decision_id",
            "label_symbol",
            "label_decision_at_ns",
            "side",
            "horizon_minutes",
            "label_end_ns",
            "stop_distance_bps",
            "take_profit_distance_bps",
            "outcome",
            "hit_at_ns",
            "outcome_return_bps",
        ]
    )
    joined = labels.join(features, keys="decision_id", join_type="inner")
    if joined.num_rows != labels.num_rows:
        raise EvaluationError("not every model label has exactly one feature row")
    same_symbol = pc.all(
        pc.equal(joined.column("label_symbol"), joined.column("symbol"))
    ).as_py()
    same_decision = pc.all(
        pc.equal(
            joined.column("label_decision_at_ns"), joined.column("decision_at_ns")
        )
    ).as_py()
    if same_symbol is not True or same_decision is not True:
        raise EvaluationError("feature/label identity mismatch after decision_id join")
    order = pc.sort_indices(
        joined,
        sort_keys=[
            ("decision_at_ns", "ascending"),
            ("symbol", "ascending"),
            ("side", "ascending"),
        ],
    )
    joined = joined.take(order)

    rows = joined.num_rows
    symbols_raw = np.asarray(
        joined.column("symbol").combine_chunks().to_pylist(), dtype=np.str_
    )
    symbol_lookup = {symbol: index for index, symbol in enumerate(dataset.symbols)}
    try:
        symbol_codes = np.fromiter(
            (symbol_lookup[value] for value in symbols_raw),
            dtype=np.int16,
            count=rows,
        )
    except KeyError as exc:
        raise EvaluationError(f"joined data contains an unknown symbol: {exc}") from exc
    sides_raw = np.asarray(
        joined.column("side").combine_chunks().to_pylist(), dtype=np.str_
    )
    if np.any(~np.isin(sides_raw, np.asarray(["LONG", "SHORT"]))):
        raise EvaluationError("joined data contains an unsupported side")
    side_codes = np.where(sides_raw == "LONG", 1, -1).astype(np.int8)

    outcomes = joined.column("outcome").combine_chunks()
    y = np.asarray(
        pc.index_in(outcomes, value_set=pa.array(OUTCOME_NAMES)).to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int64,
    )
    if np.any((y < 0) | (y >= len(OUTCOME_NAMES))):
        raise EvaluationError("joined data contains an unsupported outcome")

    feature_names = (
        *direct_feature_columns,
        *(f"log1p_{name}" for name in log1p_feature_columns),
        "stop_distance_bps",
        "take_profit_distance_bps",
        "side_direction",
        *(f"symbol_{symbol}" for symbol in dataset.symbols),
    )
    x = np.empty((rows, len(feature_names)), dtype=np.float32)
    column_index = 0
    for name in direct_feature_columns:
        values = _float_array(joined, name)
        if np.any(np.isinf(values)):
            raise EvaluationError(f"model feature contains infinity: {name}")
        x[:, column_index] = values.astype(np.float32)
        column_index += 1
    for name in log1p_feature_columns:
        values = _float_array(joined, name)
        if np.any(np.isinf(values)) or np.any(values[~np.isnan(values)] < 0):
            raise EvaluationError(f"log1p model feature is invalid: {name}")
        x[:, column_index] = np.log1p(values).astype(np.float32)
        column_index += 1
    stop_distance = _float_array(joined, "stop_distance_bps")
    take_profit_distance = _float_array(joined, "take_profit_distance_bps")
    if (
        np.any(~np.isfinite(stop_distance))
        or np.any(stop_distance <= 0)
        or np.any(~np.isfinite(take_profit_distance))
        or np.any(take_profit_distance <= 0)
    ):
        raise EvaluationError("barrier distances are invalid")
    x[:, column_index] = stop_distance.astype(np.float32)
    column_index += 1
    x[:, column_index] = take_profit_distance.astype(np.float32)
    column_index += 1
    x[:, column_index] = side_codes.astype(np.float32)
    column_index += 1
    for symbol_code in range(len(dataset.symbols)):
        x[:, column_index] = (symbol_codes == symbol_code).astype(np.float32)
        column_index += 1
    if column_index != x.shape[1]:
        raise AssertionError("feature matrix width does not match its contract")

    decision_at_ns = _int_array(joined, "decision_at_ns")
    label_end_ns = _int_array(joined, "label_end_ns")
    if np.any(label_end_ns <= decision_at_ns):
        raise EvaluationError("labels must end strictly after their decision")
    gross_returns = _float_array(joined, "outcome_return_bps")
    if np.any(~np.isfinite(gross_returns)):
        raise EvaluationError("priced label returns must be finite")
    if dataset.research_profile == PRICE_RESEARCH_PROFILE:
        funding_rate = np.full(rows, np.nan, dtype=np.float64)
        minutes_to_funding = np.full(rows, np.nan, dtype=np.float64)
    else:
        funding_rate = _float_array(joined, "funding_rate")
        minutes_to_funding = _float_array(joined, "minutes_to_funding")
    if np.any(np.isinf(funding_rate)) or np.any(np.isinf(minutes_to_funding)):
        raise EvaluationError("funding features contain infinity")

    return PreparedData(
        x=x,
        y=y,
        feature_names=tuple(feature_names),
        decision_ids=np.asarray(
            joined.column("decision_id").combine_chunks().to_pylist(), dtype=np.str_
        ),
        decision_at_ns=decision_at_ns,
        label_end_ns=label_end_ns,
        hit_at_ns=_int_array(joined, "hit_at_ns", null_value=-1),
        symbol_codes=symbol_codes,
        symbols=dataset.symbols,
        side_codes=side_codes,
        outcome_return_bps=gross_returns,
        stop_distance_bps=stop_distance,
        take_profit_distance_bps=take_profit_distance,
        funding_rate=funding_rate,
        minutes_to_funding=minutes_to_funding,
        excluded_ambiguous_rows=excluded_ambiguous,
        excluded_unpriced_rows=excluded_unpriced,
    )
