from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray

from tradingbot.research.contracts import (
    EXECUTION_FEATURE_SCHEMA,
    EXECUTION_LABEL_SCHEMA,
    EXECUTION_RESEARCH_PROFILE,
    EXECUTION_RESEARCH_SCHEMA_VERSION,
)
from tradingbot.research.evaluation_contracts import (
    DIRECT_FEATURE_COLUMNS,
    LOG1P_FEATURE_COLUMNS,
    EvaluationError,
)
from tradingbot.research.execution_evaluation_contracts import (
    EXECUTION_OUTCOME_NAMES,
    FILL_NAMES,
    FILL_TO_INDEX,
    ExecutionPreparedData,
    ExecutionResearchDataset,
)

LOGGER = logging.getLogger(__name__)


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


def _partition_symbol(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("symbol="):
            return part.removeprefix("symbol=")
    return None


def validate_execution_research_dataset(
    dataset_path: str | Path,
) -> ExecutionResearchDataset:
    """Verify the immutable V3 dataset and every Parquet input before modeling."""

    root = Path(dataset_path).expanduser().resolve()
    if not root.is_dir():
        raise EvaluationError(
            f"execution research dataset directory does not exist: {root}"
        )
    manifest = _load_json(root / "manifest.json", "execution research manifest")
    if (
        manifest.get("execution_research_schema_version")
        != EXECUTION_RESEARCH_SCHEMA_VERSION
    ):
        raise EvaluationError("unsupported execution research schema version")
    if manifest.get("research_profile") != EXECUTION_RESEARCH_PROFILE:
        raise EvaluationError("unsupported execution research profile")
    dataset_id = _string(
        manifest.get("execution_dataset_id"), "execution_dataset_id"
    )
    if root.name != dataset_id:
        raise EvaluationError("execution dataset ID does not match its directory")
    input_fingerprint = _sha256(
        manifest.get("input_fingerprint"), "input_fingerprint"
    )
    output_fingerprint = _sha256(
        manifest.get("output_fingerprint"), "output_fingerprint"
    )

    scope = _object(manifest.get("scope"), "scope")
    required_scope = {
        "real_exchange_orders_observed": False,
        "maker_fill_is_proxy": True,
        "partial_fills_retained": True,
        "eligible_for_fill_model_training": True,
        "eligible_for_profitability_conclusion": False,
    }
    for key, expected in required_scope.items():
        if scope.get(key) is not expected:
            raise EvaluationError(f"execution scope contract is invalid: {key}")

    causality = _object(manifest.get("causality"), "causality")
    if causality.get("feature_rule") != "received_at_ns <= decision_at_ns":
        raise EvaluationError("execution feature causality contract is missing")
    if causality.get("partial_fill_class") != "PARTIAL_FILL":
        raise EvaluationError("execution partial-fill contract is missing")
    if causality.get("no_fill_class") != "NO_FILL":
        raise EvaluationError("execution no-fill contract is missing")

    schemas = _object(manifest.get("schemas"), "schemas")
    if schemas.get("features") != _schema_manifest(EXECUTION_FEATURE_SCHEMA):
        raise EvaluationError("execution feature schema manifest is inconsistent")
    if schemas.get("execution_labels") != _schema_manifest(
        EXECUTION_LABEL_SCHEMA
    ):
        raise EvaluationError("execution label schema manifest is inconsistent")

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
    raw_dates = source.get("partition_dates")
    if (
        not isinstance(raw_dates, list)
        or not raw_dates
        or not all(isinstance(value, str) for value in raw_dates)
    ):
        raise EvaluationError("source.partition_dates must be a non-empty date list")
    partition_dates = tuple(cast(list[str], raw_dates))
    try:
        parsed_dates = tuple(date.fromisoformat(value) for value in partition_dates)
    except ValueError as exc:
        raise EvaluationError("source.partition_dates contains an invalid date") from exc
    if len(set(parsed_dates)) != len(parsed_dates) or any(
        right != left + timedelta(days=1)
        for left, right in zip(parsed_dates, parsed_dates[1:], strict=False)
    ):
        raise EvaluationError("source.partition_dates must be consecutive and unique")

    source_copy = root / _string(
        source.get("manifest_copy"), "source.manifest_copy"
    )
    source_copy_sha = _sha256(
        source.get("manifest_sha256"), "source.manifest_sha256"
    )
    if (
        not source_copy.is_file()
        or not source_copy.resolve().is_relative_to(root)
        or _sha256_file(source_copy) != source_copy_sha
    ):
        raise EvaluationError("execution source-manifest.json failed validation")

    parameters = _object(manifest.get("parameters"), "parameters")
    parameter_fingerprint = _sha256(
        parameters.get("fingerprint"), "parameters.fingerprint"
    )
    parameter_payload = {
        key: value for key, value in parameters.items() if key != "fingerprint"
    }
    if _sha256_json(parameter_payload) != parameter_fingerprint:
        raise EvaluationError("execution parameter fingerprint is inconsistent")
    builder = _object(manifest.get("builder"), "builder")
    package_version = _string(
        builder.get("package_version"), "builder.package_version"
    )
    pyarrow_version = _string(
        builder.get("pyarrow_version"), "builder.pyarrow_version"
    )
    numpy_version = _string(builder.get("numpy_version"), "builder.numpy_version")
    expected_input_fingerprint = _sha256_json(
        {
            "execution_research_schema_version": (
                EXECUTION_RESEARCH_SCHEMA_VERSION
            ),
            "research_profile": EXECUTION_RESEARCH_PROFILE,
            "package_version": package_version,
            "pyarrow_version": pyarrow_version,
            "numpy_version": numpy_version,
            "source_dataset_id": source_dataset_id,
            "source_manifest_sha256": source_copy_sha,
            "source_output_fingerprint": source_output_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
        }
    )
    if input_fingerprint != expected_input_fingerprint:
        raise EvaluationError("execution input_fingerprint is inconsistent")
    expected_dataset_id = (
        f"execution-research-v{EXECUTION_RESEARCH_SCHEMA_VERSION}-"
        f"{input_fingerprint[:16]}"
    )
    if dataset_id != expected_dataset_id:
        raise EvaluationError("execution dataset ID does not match input_fingerprint")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise EvaluationError("execution manifest.files must be a non-empty array")
    paths_by_table: dict[str, list[Path]] = {
        "features": [],
        "execution_labels": [],
    }
    rows_by_table: Counter[str] = Counter()
    descriptors: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for index, raw_file in enumerate(raw_files):
        item = _object(raw_file, f"files[{index}]")
        relative = _safe_relative_path(item.get("path"), f"files[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise EvaluationError(
                f"duplicate execution output path: {relative_text}"
            )
        seen_paths.add(relative_text)
        table_name = _string(item.get("table"), f"files[{index}].table")
        if table_name not in paths_by_table:
            raise EvaluationError(f"unsupported execution table: {table_name}")
        symbol = _string(item.get("symbol"), f"files[{index}].symbol")
        if symbol not in symbols:
            raise EvaluationError(f"execution file has unknown symbol: {symbol}")
        partition = _string(item.get("date"), f"files[{index}].date")
        if partition not in partition_dates:
            raise EvaluationError(f"execution file has unknown date: {partition}")
        rows = _nonnegative_int(item.get("rows"), f"files[{index}].rows")
        size = _nonnegative_int(item.get("bytes"), f"files[{index}].bytes")
        digest = _sha256(item.get("sha256"), f"files[{index}].sha256")
        expected_prefix = f"table={table_name}/symbol={symbol}/date={partition}/"
        if not relative_text.startswith(expected_prefix):
            raise EvaluationError(
                f"execution partition path is inconsistent: {relative}"
            )
        actual = root.joinpath(*relative.parts).resolve()
        if not actual.is_relative_to(root) or not actual.is_file():
            raise EvaluationError(f"execution output file is missing: {relative}")
        expected_schema = (
            EXECUTION_FEATURE_SCHEMA
            if table_name == "features"
            else EXECUTION_LABEL_SCHEMA
        )
        if actual.stat().st_size != size or _sha256_file(actual) != digest:
            raise EvaluationError(f"execution output file is corrupted: {relative}")
        try:
            parquet = pq.ParquetFile(actual)
        except (OSError, pa.ArrowInvalid) as exc:
            raise EvaluationError(
                f"execution output file is corrupted: {relative}"
            ) from exc
        if parquet.metadata.num_rows != rows or not parquet.schema_arrow.equals(
            expected_schema, check_metadata=True
        ):
            raise EvaluationError(f"execution output file is corrupted: {relative}")
        descriptor: dict[str, object] = {
            "path": relative_text,
            "table": table_name,
            "symbol": symbol,
            "date": partition,
            "rows": rows,
            "bytes": size,
            "sha256": digest,
        }
        descriptors.append(descriptor)
        paths_by_table[table_name].append(actual)
        rows_by_table[table_name] += rows
        completed = index + 1
        if completed == len(raw_files) or completed % 25 == 0:
            LOGGER.info(
                "Validated execution input %d/%d Parquet files",
                completed,
                len(raw_files),
            )

    if len(descriptors) != manifest.get("output_file_count"):
        raise EvaluationError("execution output_file_count is inconsistent")
    if _sha256_json(sorted(descriptors, key=lambda item: cast(str, item["path"]))) != (
        output_fingerprint
    ):
        raise EvaluationError("execution output_fingerprint is inconsistent")
    output_rows = _object(manifest.get("output_rows"), "output_rows")
    expected_rows = {
        key: _nonnegative_int(value, f"output_rows.{key}")
        for key, value in output_rows.items()
    }
    if dict(sorted(rows_by_table.items())) != dict(sorted(expected_rows.items())):
        raise EvaluationError("execution output row totals are inconsistent")
    if rows_by_table["features"] <= 0 or rows_by_table["execution_labels"] <= 0:
        raise EvaluationError("execution dataset contains no modelable rows")

    return ExecutionResearchDataset(
        root=root,
        execution_dataset_id=dataset_id,
        source_dataset_id=source_dataset_id,
        input_fingerprint=input_fingerprint,
        output_fingerprint=output_fingerprint,
        symbols=symbols,
        partition_dates=partition_dates,
        feature_paths=tuple(sorted(paths_by_table["features"])),
        label_paths=tuple(sorted(paths_by_table["execution_labels"])),
        feature_rows=rows_by_table["features"],
        label_rows=rows_by_table["execution_labels"],
        manifest=manifest,
    )


def _read_tables(paths: tuple[Path, ...], columns: list[str]) -> pa.Table:
    tables = [pq.ParquetFile(path).read(columns=columns) for path in paths]
    if not tables:
        raise EvaluationError("execution dataset has no Parquet inputs")
    return pa.concat_tables(tables)


def _read_selected_labels(
    paths: tuple[Path, ...],
    columns: list[str],
    *,
    horizon_minutes: int,
    order_notional_usdt: float,
) -> pa.Table:
    tables: list[pa.Table] = []
    for path in paths:
        table = pq.ParquetFile(path).read(columns=columns)
        selected = pc.and_(
            pc.equal(
                table.column("horizon_minutes"),
                pa.scalar(horizon_minutes, type=pa.int32()),
            ),
            pc.equal(
                table.column("order_notional_usdt"),
                pa.scalar(order_notional_usdt, type=pa.float64()),
            ),
        )
        table = table.filter(selected)
        if table.num_rows:
            tables.append(table)
    if not tables:
        raise EvaluationError(
            "execution dataset has no labels for the selected horizon/notional"
        )
    return pa.concat_tables(tables)


def _count_true(values: pa.Array | pa.ChunkedArray) -> int:
    result = pc.sum(pc.cast(values, pa.int64())).as_py()
    return 0 if result is None else int(result)


def _float_array(table: pa.Table, name: str) -> NDArray[np.float64]:
    values = pc.cast(table.column(name).combine_chunks(), pa.float64())
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float64)


def _int_array(
    table: pa.Table,
    name: str,
    *,
    null_value: int | None = None,
) -> NDArray[np.int64]:
    values: pa.Array | pa.ChunkedArray = table.column(name).combine_chunks()
    if null_value is not None:
        values = pc.fill_null(values, pa.scalar(null_value, type=pa.int64()))
    return np.asarray(
        pc.cast(values, pa.int64()).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )


def prepare_execution_evaluation_data(
    dataset: ExecutionResearchDataset,
    *,
    horizon_minutes: int,
    order_notional_usdt: float,
) -> ExecutionPreparedData:
    """Join one pre-registered V3 scenario to causal decision-time features."""

    parameters = _object(dataset.manifest.get("parameters"), "parameters")
    raw_horizons = parameters.get("position_horizons_minutes")
    raw_notionals = parameters.get("order_notionals_usdt")
    if (
        not isinstance(raw_horizons, list)
        or horizon_minutes not in raw_horizons
        or not isinstance(raw_notionals, list)
        or order_notional_usdt not in raw_notionals
    ):
        raise EvaluationError(
            "selected horizon/notional was not pre-registered in the V3 dataset"
        )
    if not np.isfinite(order_notional_usdt) or order_notional_usdt <= 0:
        raise EvaluationError("order_notional_usdt must be positive and finite")

    model_columns = list(
        dict.fromkeys(DIRECT_FEATURE_COLUMNS + LOG1P_FEATURE_COLUMNS)
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
        "order_notional_usdt",
        "activation_delay_ms",
        "entry_window_end_ns",
        "post_only_valid",
        "fill_status",
        "fill_fraction",
        "full_fill_at_ns",
        "time_to_full_fill_ms",
        "stop_distance_bps",
        "take_profit_distance_bps",
        "position_end_ns",
        "outcome",
        "hit_at_ns",
        "outcome_return_bps",
    ]
    features = _read_tables(dataset.feature_paths, feature_columns)
    labels = _read_selected_labels(
        dataset.label_paths,
        label_columns,
        horizon_minutes=horizon_minutes,
        order_notional_usdt=order_notional_usdt,
    )
    LOGGER.info(
        "Loaded execution scenario H%dm / %.12g USDT: %d features, %d labels",
        horizon_minutes,
        order_notional_usdt,
        features.num_rows,
        labels.num_rows,
    )

    recognized_fill = pc.is_in(
        labels.column("fill_status"), value_set=pa.array(FILL_NAMES)
    )
    full_fill = pc.equal(labels.column("fill_status"), pa.scalar("FULL_FILL"))
    recognized_outcome = pc.is_in(
        labels.column("outcome"), value_set=pa.array(EXECUTION_OUTCOME_NAMES)
    )
    unpriced = pc.is_null(labels.column("outcome_return_bps"))
    excluded_ambiguous = _count_true(
        pc.and_(full_fill, pc.invert(recognized_outcome))
    )
    excluded_unpriced = _count_true(
        pc.and_(full_fill, pc.and_(recognized_outcome, unpriced))
    )
    usable_full = pc.and_(recognized_outcome, pc.invert(unpriced))
    usable = pc.and_(
        recognized_fill,
        pc.or_(pc.invert(full_fill), usable_full),
    )
    labels = labels.filter(usable)
    if labels.num_rows == 0:
        raise EvaluationError("execution dataset has no usable selected labels")

    labels = labels.rename_columns(
        [
            "decision_id",
            "label_symbol",
            "label_decision_at_ns",
            "side",
            "horizon_minutes",
            "order_notional_usdt",
            "activation_delay_ms",
            "entry_window_end_ns",
            "post_only_valid",
            "fill_status",
            "fill_fraction",
            "full_fill_at_ns",
            "time_to_full_fill_ms",
            "stop_distance_bps",
            "take_profit_distance_bps",
            "position_end_ns",
            "outcome",
            "hit_at_ns",
            "outcome_return_bps",
        ]
    )
    joined = labels.join(features, keys="decision_id", join_type="inner")
    if joined.num_rows != labels.num_rows:
        raise EvaluationError("not every execution label has exactly one feature row")
    if pc.all(pc.equal(joined.column("label_symbol"), joined.column("symbol"))).as_py() is not True:
        raise EvaluationError("execution feature/label symbol mismatch")
    if pc.all(
        pc.equal(
            joined.column("label_decision_at_ns"),
            joined.column("decision_at_ns"),
        )
    ).as_py() is not True:
        raise EvaluationError("execution feature/label timestamp mismatch")

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
        raise EvaluationError(f"execution data has an unknown symbol: {exc}") from exc
    sides_raw = np.asarray(
        joined.column("side").combine_chunks().to_pylist(), dtype=np.str_
    )
    if np.any(~np.isin(sides_raw, np.asarray(["LONG", "SHORT"]))):
        raise EvaluationError("execution data has an unsupported side")
    side_codes = np.where(sides_raw == "LONG", 1, -1).astype(np.int8)

    fills = joined.column("fill_status").combine_chunks()
    fill_y = np.asarray(
        pc.index_in(fills, value_set=pa.array(FILL_NAMES)).to_numpy(
            zero_copy_only=False
        ),
        dtype=np.int64,
    )
    if np.any((fill_y < 0) | (fill_y >= len(FILL_NAMES))):
        raise EvaluationError("execution data has an unsupported fill status")

    outcomes_raw = joined.column("outcome").combine_chunks()
    outcome_y = np.full(rows, -1, dtype=np.int64)
    full_mask = fill_y == FILL_TO_INDEX["FULL_FILL"]
    if np.any(full_mask):
        full_outcomes = outcomes_raw.filter(pa.array(full_mask))
        encoded = np.asarray(
            pc.index_in(
                full_outcomes,
                value_set=pa.array(EXECUTION_OUTCOME_NAMES),
            ).to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if np.any((encoded < 0) | (encoded >= len(EXECUTION_OUTCOME_NAMES))):
            raise EvaluationError("full-fill data has an unsupported outcome")
        outcome_y[full_mask] = encoded

    feature_names = (
        *DIRECT_FEATURE_COLUMNS,
        *(f"log1p_{name}" for name in LOG1P_FEATURE_COLUMNS),
        "stop_distance_bps",
        "take_profit_distance_bps",
        "side_direction",
        *(f"symbol_{symbol}" for symbol in dataset.symbols),
    )
    x = np.empty((rows, len(feature_names)), dtype=np.float32)
    column_index = 0
    for name in DIRECT_FEATURE_COLUMNS:
        values = _float_array(joined, name)
        if np.any(np.isinf(values)):
            raise EvaluationError(f"execution feature contains infinity: {name}")
        x[:, column_index] = values.astype(np.float32)
        column_index += 1
    for name in LOG1P_FEATURE_COLUMNS:
        values = _float_array(joined, name)
        if np.any(np.isinf(values)) or np.any(values[~np.isnan(values)] < 0):
            raise EvaluationError(f"execution log1p feature is invalid: {name}")
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
        raise EvaluationError("execution barrier distances are invalid")
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
        raise AssertionError("execution feature matrix width is inconsistent")

    decision_ids = np.asarray(
        pc.cast(joined.column("decision_id").combine_chunks(), pa.binary()).to_pylist(),
        dtype=np.bytes_,
    )
    if len(decision_ids) > 1:
        duplicate_candidate = (decision_ids[1:] == decision_ids[:-1]) & (
            side_codes[1:] == side_codes[:-1]
        )
        if np.any(duplicate_candidate):
            raise EvaluationError("execution candidate rows are duplicated")

    decision_at_ns = _int_array(joined, "decision_at_ns")
    entry_window_end_ns = _int_array(joined, "entry_window_end_ns")
    position_end_ns = _int_array(joined, "position_end_ns", null_value=-1)
    hit_at_ns = _int_array(joined, "hit_at_ns", null_value=-1)
    full_fill_at_ns = _int_array(joined, "full_fill_at_ns", null_value=-1)
    label_end_ns = np.where(
        full_mask, position_end_ns, entry_window_end_ns
    ).astype(np.int64)
    if np.any(entry_window_end_ns <= decision_at_ns):
        raise EvaluationError("entry windows must end after their decision")
    if np.any(label_end_ns <= decision_at_ns):
        raise EvaluationError("execution labels must end after their decision")
    if np.any(full_mask & (full_fill_at_ns <= decision_at_ns)):
        raise EvaluationError("full fills must occur after their decision")
    if np.any(full_mask & (position_end_ns <= full_fill_at_ns)):
        raise EvaluationError("full-fill position windows are invalid")
    timeout_mask = full_mask & (
        outcome_y == EXECUTION_OUTCOME_NAMES.index("TIMEOUT")
    )
    barrier_mask = full_mask & ~timeout_mask
    if np.any(
        barrier_mask
        & ((hit_at_ns < full_fill_at_ns) | (hit_at_ns > position_end_ns))
    ):
        raise EvaluationError("execution barrier timestamps are invalid")
    if np.any(timeout_mask & (hit_at_ns >= 0)):
        raise EvaluationError("execution timeout rows must not have a hit timestamp")

    fill_fraction = _float_array(joined, "fill_fraction")
    if np.any(~np.isfinite(fill_fraction)) or np.any(
        (fill_fraction < 0) | (fill_fraction > 1)
    ):
        raise EvaluationError("execution fill fractions are invalid")
    expected_fraction = np.where(
        fill_y == FILL_TO_INDEX["NO_FILL"],
        0.0,
        np.where(fill_y == FILL_TO_INDEX["FULL_FILL"], 1.0, fill_fraction),
    )
    if np.any(np.abs(fill_fraction - expected_fraction) > 1e-9):
        raise EvaluationError("execution fill fraction contradicts fill status")

    outcome_return_bps = _float_array(joined, "outcome_return_bps")
    if np.any(full_mask & ~np.isfinite(outcome_return_bps)):
        raise EvaluationError("full-fill outcome returns must be finite")
    funding_rate = _float_array(joined, "funding_rate")
    minutes_to_funding = _float_array(joined, "minutes_to_funding")
    if np.any(np.isinf(funding_rate)) or np.any(np.isinf(minutes_to_funding)):
        raise EvaluationError("execution funding features contain infinity")
    activation_delay_ms = _float_array(joined, "activation_delay_ms")
    time_to_full_fill_ms = _float_array(joined, "time_to_full_fill_ms")
    if np.any(~np.isfinite(activation_delay_ms)) or np.any(activation_delay_ms < 0):
        raise EvaluationError("execution activation delays are invalid")
    if np.any(full_mask & ~np.isfinite(time_to_full_fill_ms)):
        raise EvaluationError("full-fill timing must be finite")
    post_only_valid = np.asarray(
        joined.column("post_only_valid").combine_chunks().to_numpy(
            zero_copy_only=False
        ),
        dtype=np.bool_,
    )
    if np.any(
        ~post_only_valid & (fill_y != FILL_TO_INDEX["NO_FILL"])
    ):
        raise EvaluationError("invalid PostOnly candidates cannot be filled")

    return ExecutionPreparedData(
        x=x,
        feature_names=tuple(feature_names),
        decision_ids=decision_ids,
        decision_at_ns=decision_at_ns,
        label_end_ns=label_end_ns,
        entry_window_end_ns=entry_window_end_ns,
        position_end_ns=position_end_ns,
        hit_at_ns=hit_at_ns,
        full_fill_at_ns=full_fill_at_ns,
        symbol_codes=symbol_codes,
        symbols=dataset.symbols,
        side_codes=side_codes,
        fill_y=fill_y,
        outcome_y=outcome_y,
        fill_fraction=fill_fraction,
        outcome_return_bps=outcome_return_bps,
        stop_distance_bps=stop_distance,
        take_profit_distance_bps=take_profit_distance,
        funding_rate=funding_rate,
        minutes_to_funding=minutes_to_funding,
        activation_delay_ms=activation_delay_ms,
        time_to_full_fill_ms=time_to_full_fill_ms,
        post_only_valid=post_only_valid,
        horizon_minutes=horizon_minutes,
        order_notional_usdt=order_notional_usdt,
        excluded_ambiguous_full_fills=excluded_ambiguous,
        excluded_unpriced_full_fills=excluded_unpriced,
    )
