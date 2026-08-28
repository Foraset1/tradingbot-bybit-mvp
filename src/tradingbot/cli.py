from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tradingbot.config import AppConfig, ConfigError, load_config
from tradingbot.data.audit import audit_dataset
from tradingbot.health import health_snapshot, health_worker, write_health
from tradingbot.market.bybit_public import BybitPublicCollector, CollectorStats
from tradingbot.market.records import MarketRecord
from tradingbot.market.topics import build_public_topics
from tradingbot.storage.jsonl import SegmentedJsonlWriter

LOGGER = logging.getLogger(__name__)


def _default_config_path() -> str:
    return os.getenv("TRADINGBOT_CONFIG", str(Path.cwd() / "config" / "tradingbot.toml"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TradingBot Bybit market-data tools")
    parser.add_argument("--config", default=_default_config_path(), help="Path to TOML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-config", help="Validate configuration and risk invariants")
    subparsers.add_parser("show-topics", help="Print public WebSocket subscription topics")
    collect = subparsers.add_parser("collect", help="Start read-only public market collection")
    collect.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Stop automatically after N seconds (useful for smoke tests)",
    )
    audit = subparsers.add_parser(
        "audit-data", help="Audit completed raw JSONL segments and print a JSON report"
    )
    audit.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root (defaults to storage.root from the configuration)",
    )
    audit.add_argument(
        "--minimum-duration-seconds",
        type=float,
        default=0.0,
        help="Minimum required duration of every expected stream",
    )
    audit.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings, including gaps and partial files, as not ready",
    )
    audit.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally write the same JSON report atomically to this path",
    )
    audit.add_argument(
        "--partition-date",
        default=None,
        help="Only audit one UTC partition (YYYY-MM-DD)",
    )
    dataset = subparsers.add_parser(
        "build-dataset",
        help="Build deterministic canonical Parquet from a successful strict audit",
    )
    dataset.add_argument(
        "--audit-report",
        type=Path,
        required=True,
        help="Strict audit report v2 used as the immutable source manifest",
    )
    dataset.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Source root override (defaults to dataset_root in the audit report)",
    )
    dataset.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Dataset parent directory (defaults to a datasets sibling of storage.root)",
    )
    research = subparsers.add_parser(
        "build-research",
        help="Build causal features and market labels from canonical Parquet",
    )
    research_source = research.add_mutually_exclusive_group(required=True)
    research_source.add_argument(
        "--dataset",
        type=Path,
        help="Canonical dataset directory containing manifest.json",
    )
    research_source.add_argument(
        "--catalog",
        type=Path,
        help="Daily archive catalog.json used as a multi-day canonical source",
    )
    research.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Research dataset parent directory (defaults to research beside storage.root)",
    )
    execution_research = subparsers.add_parser(
        "build-execution-research",
        help="Build conservative maker-fill and post-fill labels from live microstructure",
    )
    execution_source = execution_research.add_mutually_exclusive_group(required=True)
    execution_source.add_argument(
        "--dataset",
        type=Path,
        help="Canonical dataset directory containing manifest.json",
    )
    execution_source.add_argument(
        "--catalog",
        type=Path,
        help="Daily archive catalog.json used as a multi-day canonical source",
    )
    execution_research.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Execution dataset parent (defaults to execution-research beside storage.root)",
    )
    execution_research.add_argument(
        "--horizon-minutes",
        type=int,
        action="append",
        choices=(15, 30),
        default=None,
        help="Repeat to override the pre-registered 15/30 minute horizons",
    )
    execution_research.add_argument(
        "--order-notional-usdt",
        type=float,
        action="append",
        default=None,
        help="Repeat to override the 50/100/250/500 USDT reference sizes",
    )
    execution_research.add_argument(
        "--submission-latency-ms",
        type=int,
        default=None,
        help="Decision-to-submission latency (default: 250 ms)",
    )
    execution_research.add_argument(
        "--activation-max-delay-ms",
        type=int,
        default=None,
        help="Maximum wait for the first observable activation book (default: 2500 ms)",
    )
    execution_research.add_argument(
        "--entry-ttl-seconds",
        type=int,
        default=None,
        help="Maker order lifetime from the decision timestamp (default: 30 s)",
    )
    execution_research.add_argument(
        "--queue-ahead-multiplier",
        type=float,
        default=None,
        help="Conservative visible queue multiplier, at least 1.0 (default: 1.0)",
    )
    price_research = subparsers.add_parser(
        "build-price-research",
        help="Build causal price-only research from official Bybit history",
    )
    price_research.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Verified /data/history/catalog.json",
    )
    price_research.add_argument(
        "--from-date",
        required=True,
        help="First imported UTC day to select (YYYY-MM-DD, inclusive)",
    )
    price_research.add_argument(
        "--to-date",
        required=True,
        help="Last imported UTC day to select (YYYY-MM-DD, inclusive)",
    )
    price_research.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Research dataset parent directory (defaults to research beside storage.root)",
    )
    evaluation = subparsers.add_parser(
        "run-backtest",
        help="Run causal baselines, LightGBM, and a conditional-entry market backtest",
    )
    evaluation.add_argument(
        "--research-dataset",
        type=Path,
        required=True,
        help="Verified research dataset directory containing manifest.json",
    )
    evaluation.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Evaluation parent directory (defaults to evaluations beside storage.root)",
    )
    evaluation.add_argument(
        "--horizon-minutes",
        type=int,
        choices=(15, 30, 60),
        default=None,
        help="Pre-registered barrier horizon override (15, 30, or 60 minutes)",
    )
    execution_evaluation = subparsers.add_parser(
        "run-execution-backtest",
        help=(
            "Fit separate maker-fill/outcome models and replay one order across "
            "all pairs"
        ),
    )
    execution_evaluation.add_argument(
        "--execution-dataset",
        type=Path,
        required=True,
        help="Verified execution-research-v1 directory containing manifest.json",
    )
    execution_evaluation.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Execution evaluation parent directory "
            "(defaults to execution-evaluations beside storage.root)"
        ),
    )
    execution_evaluation.add_argument(
        "--horizon-minutes",
        type=int,
        choices=(15, 30),
        default=15,
        help="One pre-registered execution horizon to evaluate (default: 15)",
    )
    execution_evaluation.add_argument(
        "--order-notional-usdt",
        type=float,
        default=50.0,
        help=(
            "One pre-registered maker-order size to evaluate "
            "(default: 50 USDT)"
        ),
    )
    shadow_bundle = subparsers.add_parser(
        "build-shadow-bundle",
        help=(
            "Freeze one authenticated execution-evaluation fold for read-only "
            "live shadowing; never retrains"
        ),
    )
    shadow_bundle.add_argument(
        "--execution-evaluation",
        type=Path,
        required=True,
        help="Verified execution-backtest-v1 directory",
    )
    shadow_bundle.add_argument(
        "--execution-dataset",
        type=Path,
        required=True,
        help="Exact execution-research-v1 directory used by that evaluation",
    )
    shadow_bundle.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Bundle parent directory (defaults to shadow-bundles beside storage.root)",
    )
    shadow_bundle.add_argument(
        "--allow-engineering-only",
        "--allow-technical-smoke",
        dest="allow_technical_smoke",
        action="store_true",
        help=(
            "Explicitly allow an unreviewed engineering-only bundle; "
            "--allow-technical-smoke is a compatibility alias"
        ),
    )
    validate_shadow = subparsers.add_parser(
        "validate-shadow-bundle",
        help="Verify every frozen Shadow Mode artifact and print its safety scope",
    )
    validate_shadow.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="shadow-bundle-v1 directory containing manifest.json",
    )
    shadow = subparsers.add_parser(
        "shadow",
        help="Observe public Bybit data and journal model decisions without orders",
    )
    shadow.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Verified immutable shadow-bundle-v1 directory",
    )
    shadow.add_argument(
        "--shadow-root",
        type=Path,
        default=None,
        help="Journal parent directory (defaults to shadow beside storage.root)",
    )
    shadow.add_argument(
        "--run-id",
        default=None,
        help="Optional stable run ID for one restartable journal",
    )
    shadow.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="Stop automatically after N seconds (useful for smoke tests)",
    )
    shadow.add_argument(
        "--processing-delay-ms",
        type=int,
        default=750,
        help="Wait after each UTC decision timestamp before causal scoring (default: 750)",
    )
    archive = subparsers.add_parser(
        "archive-day",
        help=(
            "Audit and archive one completed UTC day as immutable Parquet; "
            "preserve explicitly marked kline gaps"
        ),
    )
    archive.add_argument(
        "--date",
        default=None,
        help="UTC partition to archive (YYYY-MM-DD; defaults to yesterday)",
    )
    archive.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Raw root (defaults to storage.root)",
    )
    archive.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root (defaults to archive.root)",
    )
    archive.add_argument(
        "--minimum-duration-seconds",
        type=float,
        default=None,
        help="Minimum coverage per expected stream",
    )
    archive.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally write the result JSON atomically to this path",
    )
    retention = subparsers.add_parser(
        "plan-retention",
        help="Verify archives and print a raw-retention dry-run; never deletes files",
    )
    retention.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Raw root (defaults to storage.root)",
    )
    retention.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root (defaults to archive.root)",
    )
    retention.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help="Number of complete raw days to retain",
    )
    retention.add_argument(
        "--as-of-date",
        default=None,
        help="UTC policy date (YYYY-MM-DD; defaults to today)",
    )
    retention.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally write the dry-run JSON atomically to this path",
    )
    apply_retention = subparsers.add_parser(
        "apply-retention",
        help="Revalidate and delete only files from one explicitly approved plan",
    )
    apply_retention.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Saved plan-retention JSON to verify and apply",
    )
    apply_retention.add_argument(
        "--confirm-plan-fingerprint",
        required=True,
        help="Exact plan_fingerprint copied from the reviewed plan",
    )
    apply_retention.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Raw root (defaults to storage.root)",
    )
    apply_retention.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Archive root (defaults to archive.root)",
    )
    apply_retention.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Required durable JSON receipt path outside raw/archive roots",
    )
    history = subparsers.add_parser(
        "import-history",
        help="Stream official Bybit daily trades into compact 1s/1m Parquet bars",
    )
    history.add_argument(
        "--from-date",
        required=True,
        help="First completed UTC day to import (YYYY-MM-DD, inclusive)",
    )
    history.add_argument(
        "--to-date",
        required=True,
        help="Last completed UTC day to import (YYYY-MM-DD, inclusive)",
    )
    history.add_argument(
        "--symbol",
        dest="symbols",
        action="append",
        default=None,
        help="Import one configured symbol; repeat to select several (default: all)",
    )
    history.add_argument(
        "--history-root",
        type=Path,
        default=None,
        help="History root (defaults to history.root)",
    )
    history.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally write the range result JSON atomically to this path",
    )
    return parser


async def _storage_worker(
    queue: asyncio.Queue[MarketRecord],
    writer: SegmentedJsonlWriter,
    stop_event: asyncio.Event,
    flush_seconds: float,
) -> None:
    try:
        while not stop_event.is_set() or not queue.empty():
            try:
                record = await asyncio.wait_for(queue.get(), timeout=flush_seconds)
            except TimeoutError:
                writer.flush_if_due()
                continue
            try:
                writer.write(record)
            finally:
                queue.task_done()
            for _ in range(499):
                try:
                    record = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    writer.write(record)
                finally:
                    queue.task_done()
            writer.flush_if_due()
    finally:
        writer.flush_if_due(force=True)
        writer.close()


async def _put_record(
    queue: asyncio.Queue[MarketRecord], stats: CollectorStats, record: MarketRecord
) -> None:
    was_full = queue.full()
    stats.observe_queue(queue.qsize(), was_full=was_full)
    if was_full:
        LOGGER.warning(
            "Storage queue reached capacity (%d); applying collector backpressure",
            queue.maxsize,
        )
    await queue.put(record)
    stats.observe_queue(queue.qsize(), was_full=False)


async def _stop_after(seconds: float, stop_event: asyncio.Event) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        stop_event.set()


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(signum, lambda _signum, _frame: stop_event.set())


async def _run_collection(config: AppConfig, run_seconds: float | None) -> None:
    if run_seconds is not None and run_seconds <= 0:
        raise ValueError("--run-seconds must be positive")

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    queue: asyncio.Queue[MarketRecord] = asyncio.Queue(maxsize=config.storage.queue_maxsize)
    stats = CollectorStats()
    writer = SegmentedJsonlWriter(config.storage)

    async def sink(record: MarketRecord) -> None:
        await _put_record(queue, stats, record)

    collector = BybitPublicCollector(config, sink, stats)
    LOGGER.info("Starting read-only collector for %s", ", ".join(config.bybit.symbols))
    LOGGER.info("Writing normalized data to %s", config.storage.root)
    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(collector.run(stop_event), name="bybit-public-collector")
            tasks.create_task(
                _storage_worker(queue, writer, stop_event, config.storage.flush_seconds),
                name="jsonl-storage",
            )
            tasks.create_task(
                health_worker(
                    stats,
                    queue,
                    config.storage.health_path,
                    config.storage.root,
                    config.storage.min_free_bytes,
                    stop_event,
                ),
                name="health-writer",
            )
            if run_seconds is not None:
                tasks.create_task(
                    _stop_after(run_seconds, stop_event), name="automatic-stop"
                )
    except Exception as exc:
        stats.mark_fatal_error(f"Collector task group failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        stop_event.set()
        stats.mark_stopped()
        try:
            write_health(
                config.storage.health_path,
                health_snapshot(stats, queue.qsize(), config.storage.root),
            )
        except OSError:
            LOGGER.exception("Could not write final collector health file")
        LOGGER.info("Collector stopped; emitted %d records", stats.records_emitted)


def _run_data_audit(
    config: AppConfig,
    root: Path | None,
    minimum_duration_seconds: float,
    strict: bool,
    output: Path | None,
    partition_date: str | None,
) -> None:
    report = audit_dataset(
        root=config.storage.root if root is None else root,
        symbols=config.bybit.symbols,
        kline_intervals=config.market.kline_intervals,
        minimum_duration_seconds=minimum_duration_seconds,
        strict=strict,
        scratch_dir=config.storage.health_path.parent,
        partition_date=partition_date,
    )
    payload = report.to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        write_health(output.expanduser().resolve(), payload)
    print(rendered, end="")
    if not report.ok:
        raise SystemExit(1)


def _config_summary(config: AppConfig) -> dict[str, object]:
    return {
        "config": str(config.source_path),
        "public_ws_url": config.bybit.public_ws_url,
        "symbols": list(config.bybit.symbols),
        "topics": len(build_public_topics(config)),
        "data_root": str(config.storage.root),
        "health_path": str(config.storage.health_path),
        "stale_connection_seconds": config.bybit.stale_connection_seconds,
        "min_free_bytes": config.storage.min_free_bytes,
        "archive": {
            "root": str(config.archive.root),
            "raw_retention_days": config.archive.raw_retention_days,
            "daily_minimum_duration_seconds": (
                config.archive.daily_minimum_duration_seconds
            ),
        },
        "history": {
            "root": str(config.history.root),
            "public_base_url": config.history.public_base_url,
            "assumed_latency_ms": config.history.assumed_latency_ms,
            "maximum_consecutive_trade_free_minutes": (
                config.history.maximum_missing_minutes
            ),
            "profile": "price_futures_v1",
            "retains_individual_trades": False,
        },
        "risk": {
            "max_notional_fraction": config.risk.max_notional_fraction,
            "target_risk_fraction": config.risk.target_risk_fraction,
            "max_planned_risk_fraction": config.risk.max_planned_risk_fraction,
            "rolling_24h_loss_fraction": config.risk.rolling_24h_loss_fraction,
            "max_open_positions": config.risk.max_open_positions,
            "max_hold_seconds": config.risk.max_hold_seconds,
        },
        "evaluation": {
            "horizon_minutes": config.evaluation.horizon_minutes,
            "embargo_minutes": config.evaluation.embargo_minutes,
            "minimum_train_days": config.evaluation.minimum_train_days,
            "test_days": config.evaluation.test_days,
            "acceptance_minimum_days": config.evaluation.acceptance_minimum_days,
            "calibration_days": config.evaluation.calibration_days,
            "minimum_calibration_rows": (
                config.evaluation.minimum_calibration_rows
            ),
            "minimum_symbol_coverage_fraction": (
                config.evaluation.minimum_symbol_coverage_fraction
            ),
            "logistic_max_training_rows": (
                config.evaluation.logistic_max_training_rows
            ),
            "training_threads": config.evaluation.training_threads,
        },
        "mode": "public-read-only",
    }


def _run_build_dataset(
    config: AppConfig,
    audit_report: Path,
    root: Path | None,
    output_root: Path | None,
) -> None:
    try:
        from tradingbot.data.canonical import (
            DatasetBuildError,
            build_canonical_dataset,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow" or (exc.name or "").startswith("pyarrow."):
            LOGGER.error(
                "Dataset support is not installed; install the project with "
                "the [dataset] extra"
            )
            raise SystemExit(1) from exc
        raise

    destination = (
        config.storage.root.parent / "datasets" if output_root is None else output_root
    )
    try:
        result = build_canonical_dataset(
            audit_report=audit_report,
            source_root=root,
            output_root=destination,
            minimum_free_bytes=config.storage.min_free_bytes,
        )
    except DatasetBuildError as exc:
        LOGGER.error("Dataset build rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_build_research(
    config: AppConfig,
    dataset: Path | None,
    catalog: Path | None,
    output_root: Path | None,
) -> None:
    try:
        from tradingbot.research.builder import (
            build_research_dataset,
            build_research_dataset_from_catalog,
        )
        from tradingbot.research.contracts import ResearchBuildError
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "pyarrow"} or (exc.name or "").startswith(
            ("numpy.", "pyarrow.")
        ):
            LOGGER.error(
                "Research support is not installed; install the project with "
                "the [research] extra"
            )
            raise SystemExit(1) from exc
        raise

    destination = (
        config.storage.root.parent / "research"
        if output_root is None
        else output_root
    )
    try:
        if catalog is not None:
            result = build_research_dataset_from_catalog(
                archive_catalog=catalog,
                output_root=destination,
                minimum_free_bytes=config.storage.min_free_bytes,
            )
        elif dataset is not None:
            result = build_research_dataset(
                canonical_dataset=dataset,
                output_root=destination,
                minimum_free_bytes=config.storage.min_free_bytes,
            )
        else:  # guarded by argparse, retained for direct callers
            raise ResearchBuildError("one research source must be selected")
    except ResearchBuildError as exc:
        LOGGER.error("Research build rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_build_execution_research(
    config: AppConfig,
    dataset: Path | None,
    catalog: Path | None,
    output_root: Path | None,
    horizons: list[int] | None,
    order_notionals: list[float] | None,
    submission_latency_ms: int | None,
    activation_max_delay_ms: int | None,
    entry_ttl_seconds: int | None,
    queue_ahead_multiplier: float | None,
) -> None:
    try:
        from tradingbot.research.contracts import (
            ExecutionResearchParameters,
            ResearchBuildError,
        )
        from tradingbot.research.execution_builder import (
            build_execution_research_dataset,
            build_execution_research_dataset_from_catalog,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "pyarrow"} or (exc.name or "").startswith(
            ("numpy.", "pyarrow.")
        ):
            LOGGER.error(
                "Execution research support is not installed; install the project "
                "with the [research] extra"
            )
            raise SystemExit(1) from exc
        raise

    parameters = ExecutionResearchParameters()
    if horizons is not None:
        parameters = replace(
            parameters, position_horizons_minutes=tuple(horizons)
        )
    if order_notionals is not None:
        parameters = replace(
            parameters, order_notionals_usdt=tuple(order_notionals)
        )
    if submission_latency_ms is not None:
        parameters = replace(
            parameters, submission_latency_ms=submission_latency_ms
        )
    if activation_max_delay_ms is not None:
        parameters = replace(
            parameters, activation_max_delay_ms=activation_max_delay_ms
        )
    if entry_ttl_seconds is not None:
        parameters = replace(parameters, entry_ttl_seconds=entry_ttl_seconds)
    if queue_ahead_multiplier is not None:
        parameters = replace(
            parameters, queue_ahead_multiplier=queue_ahead_multiplier
        )

    destination = (
        config.storage.root.parent / "execution-research"
        if output_root is None
        else output_root
    )
    try:
        if catalog is not None:
            result = build_execution_research_dataset_from_catalog(
                archive_catalog=catalog,
                output_root=destination,
                parameters=parameters,
                minimum_free_bytes=config.storage.min_free_bytes,
            )
        elif dataset is not None:
            result = build_execution_research_dataset(
                canonical_dataset=dataset,
                output_root=destination,
                parameters=parameters,
                minimum_free_bytes=config.storage.min_free_bytes,
            )
        else:  # guarded by argparse, retained for direct callers
            raise ResearchBuildError("one execution research source must be selected")
    except ResearchBuildError as exc:
        LOGGER.error("Execution research build rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_build_price_research(
    config: AppConfig,
    catalog: Path,
    start_date: str,
    end_date: str,
    output_root: Path | None,
) -> None:
    try:
        from tradingbot.research.contracts import ResearchBuildError
        from tradingbot.research.price_history_builder import (
            build_price_research_dataset,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"numpy", "pyarrow"} or (exc.name or "").startswith(
            ("numpy.", "pyarrow.")
        ):
            LOGGER.error(
                "Price research support is not installed; install the project with "
                "the [research] extra"
            )
            raise SystemExit(1) from exc
        raise

    destination = (
        config.storage.root.parent / "research" if output_root is None else output_root
    )
    try:
        result = build_price_research_dataset(
            history_catalog=catalog,
            output_root=destination,
            start_date=start_date,
            end_date=end_date,
            minimum_free_bytes=config.storage.min_free_bytes,
        )
    except ResearchBuildError as exc:
        LOGGER.error("Price research build rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_backtest(
    config: AppConfig,
    research_dataset: Path,
    output_root: Path | None,
    horizon_minutes: int | None,
) -> None:
    try:
        from tradingbot.research.evaluation_contracts import EvaluationError
        from tradingbot.research.evaluator import run_offline_evaluation
    except ModuleNotFoundError as exc:
        if exc.name in {"lightgbm", "numpy", "pyarrow", "sklearn"} or (
            exc.name or ""
        ).startswith(("lightgbm.", "numpy.", "pyarrow.", "sklearn.")):
            LOGGER.error(
                "Backtest support is not installed; install the project with "
                "the [research] extra"
            )
            raise SystemExit(1) from exc
        raise

    if horizon_minutes is not None:
        config = replace(
            config,
            evaluation=replace(
                config.evaluation,
                horizon_minutes=horizon_minutes,
            ),
        )
    destination = (
        config.storage.root.parent / "evaluations"
        if output_root is None
        else output_root
    )
    try:
        result = run_offline_evaluation(
            research_dataset=research_dataset,
            output_root=destination,
            config=config,
            minimum_free_bytes=config.storage.min_free_bytes,
        )
    except EvaluationError as exc:
        LOGGER.error("Offline evaluation rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_execution_backtest(
    config: AppConfig,
    execution_dataset: Path,
    output_root: Path | None,
    horizon_minutes: int,
    order_notional_usdt: float,
) -> None:
    try:
        from tradingbot.research.evaluation_contracts import EvaluationError
        from tradingbot.research.execution_evaluator import (
            run_execution_evaluation,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"lightgbm", "numpy", "pyarrow", "sklearn"} or (
            exc.name or ""
        ).startswith(("lightgbm.", "numpy.", "pyarrow.", "sklearn.")):
            LOGGER.error(
                "Execution backtest support is not installed; install the project "
                "with the [research] extra"
            )
            raise SystemExit(1) from exc
        raise

    config = replace(
        config,
        evaluation=replace(
            config.evaluation,
            horizon_minutes=horizon_minutes,
        ),
    )
    destination = (
        config.storage.root.parent / "execution-evaluations"
        if output_root is None
        else output_root
    )
    try:
        result = run_execution_evaluation(
            execution_dataset=execution_dataset,
            output_root=destination,
            config=config,
            order_notional_usdt=order_notional_usdt,
            minimum_free_bytes=config.storage.min_free_bytes,
        )
    except EvaluationError as exc:
        LOGGER.error("Execution-aware evaluation rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_build_shadow_bundle(
    config: AppConfig,
    execution_evaluation: Path,
    execution_dataset: Path,
    output_root: Path | None,
    allow_technical_smoke: bool,
) -> None:
    try:
        from tradingbot.shadow.bundle import (
            ShadowBundleError,
            build_shadow_bundle,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"lightgbm", "numpy", "pyarrow"} or (
            exc.name or ""
        ).startswith(("lightgbm.", "numpy.", "pyarrow.")):
            LOGGER.error(
                "Shadow bundle support is not installed; install the project with "
                "the [research] extra"
            )
            raise SystemExit(1) from exc
        raise
    destination = (
        config.storage.root.parent / "shadow-bundles"
        if output_root is None
        else output_root
    )
    try:
        bundle = build_shadow_bundle(
            execution_evaluation=execution_evaluation,
            execution_dataset=execution_dataset,
            output_root=destination,
            allow_technical_smoke=allow_technical_smoke,
        )
    except ShadowBundleError as exc:
        LOGGER.error("Shadow bundle build rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))


def _run_validate_shadow_bundle(bundle_path: Path) -> None:
    try:
        from tradingbot.shadow.bundle import (
            ShadowBundleError,
            validate_shadow_bundle,
        )
        from tradingbot.shadow.model import ShadowScorer
    except ModuleNotFoundError as exc:
        if exc.name in {"lightgbm", "numpy", "pyarrow"} or (
            exc.name or ""
        ).startswith(("lightgbm.", "numpy.", "pyarrow.")):
            LOGGER.error(
                "Shadow validation support is not installed; install the project "
                "with the [research] extra"
            )
            raise SystemExit(1) from exc
        raise
    try:
        bundle = validate_shadow_bundle(bundle_path)
        ShadowScorer(bundle)
    except ShadowBundleError as exc:
        LOGGER.error("Shadow bundle validation rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))


def _run_shadow(
    config: AppConfig,
    bundle_path: Path,
    shadow_root: Path | None,
    run_id: str | None,
    run_seconds: float | None,
    processing_delay_ms: int,
) -> None:
    try:
        from tradingbot.shadow.bundle import ShadowBundleError
        from tradingbot.shadow.runtime import run_shadow_mode
    except ModuleNotFoundError as exc:
        if exc.name in {"lightgbm", "numpy", "pyarrow"} or (
            exc.name or ""
        ).startswith(("lightgbm.", "numpy.", "pyarrow.")):
            LOGGER.error(
                "Shadow runtime support is not installed; install the project with "
                "the [research] extra"
            )
            raise SystemExit(1) from exc
        raise
    destination = (
        config.storage.root.parent / "shadow"
        if shadow_root is None
        else shadow_root
    )
    try:
        result = asyncio.run(
            run_shadow_mode(
                config=config,
                bundle_path=bundle_path,
                shadow_root=destination,
                run_id=run_id,
                run_seconds=run_seconds,
                processing_delay_ms=processing_delay_ms,
            )
        )
    except ShadowBundleError as exc:
        LOGGER.error("Shadow Mode rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


def _run_archive_day(
    config: AppConfig,
    partition_date: str | None,
    raw_root: Path | None,
    archive_root: Path | None,
    minimum_duration_seconds: float | None,
    output: Path | None,
) -> None:
    try:
        from tradingbot.data.archive import (
            ARCHIVE_DAY_SCHEMA_VERSION,
            ArchiveError,
            archive_day,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow" or (exc.name or "").startswith("pyarrow."):
            LOGGER.error(
                "Archive support is not installed; install the project with "
                "the [dataset] extra"
            )
            raise SystemExit(1) from exc
        raise

    selected_date = (
        (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
        if partition_date is None
        else partition_date
    )
    try:
        result = archive_day(
            raw_root=config.storage.root if raw_root is None else raw_root,
            archive_root=config.archive.root if archive_root is None else archive_root,
            symbols=config.bybit.symbols,
            kline_intervals=config.market.kline_intervals,
            partition_date=selected_date,
            minimum_duration_seconds=(
                config.archive.daily_minimum_duration_seconds
                if minimum_duration_seconds is None
                else minimum_duration_seconds
            ),
            minimum_free_bytes=config.storage.min_free_bytes,
            scratch_dir=config.storage.health_path.parent,
        )
    except (ArchiveError, ValueError) as exc:
        LOGGER.error("Daily archive rejected: %s", exc)
        payload: dict[str, object] = {
            "archive_day_schema_version": ARCHIVE_DAY_SCHEMA_VERSION,
            "ok": False,
            "partition_date": selected_date,
            "error": str(exc),
        }
        if isinstance(exc, ArchiveError):
            payload.update(exc.details)
        if output is not None:
            write_health(output.expanduser().resolve(), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    payload = result.to_dict()
    if output is not None:
        write_health(output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def _run_retention_plan(
    config: AppConfig,
    raw_root: Path | None,
    archive_root: Path | None,
    retention_days: int | None,
    as_of_date: str | None,
    output: Path | None,
) -> None:
    from tradingbot.data.archive import ArchiveError, plan_raw_retention

    try:
        plan = plan_raw_retention(
            raw_root=config.storage.root if raw_root is None else raw_root,
            archive_root=config.archive.root if archive_root is None else archive_root,
            retention_days=(
                config.archive.raw_retention_days
                if retention_days is None
                else retention_days
            ),
            as_of_date=as_of_date,
        )
    except ArchiveError as exc:
        LOGGER.error("Retention plan rejected: %s", exc)
        raise SystemExit(1) from exc
    payload = plan.to_dict()
    if output is not None:
        write_health(output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if plan.blockers:
        raise SystemExit(1)


def _retention_apply_summary(
    payload: dict[str, object], output: Path
) -> dict[str, object]:
    keys = (
        "retention_apply_schema_version",
        "ok",
        "mode",
        "status",
        "deletion_performed",
        "plan_fingerprint",
        "delete_before_date",
        "planned_file_count",
        "planned_bytes",
        "deleted_file_count",
        "deleted_bytes",
        "current_partition_date",
        "error",
        "failed_path",
        "receipt_fingerprint",
    )
    summary = {key: payload.get(key) for key in keys if key in payload}
    summary["receipt_path"] = output.expanduser().resolve().as_posix()
    return summary


def _run_retention_apply(
    config: AppConfig,
    plan_path: Path,
    confirmed_plan_fingerprint: str,
    raw_root: Path | None,
    archive_root: Path | None,
    output: Path,
) -> None:
    from tradingbot.data.archive import (
        RETENTION_APPLY_SCHEMA_VERSION,
        ArchiveError,
        apply_raw_retention,
    )

    selected_output = output.expanduser().resolve()
    selected_raw = (
        config.storage.root if raw_root is None else raw_root
    ).expanduser().resolve()
    selected_archive = (
        config.archive.root if archive_root is None else archive_root
    ).expanduser().resolve()
    output_existed_before = selected_output.exists()
    output_is_control_safe = not (
        selected_output in (selected_raw, selected_archive)
        or selected_output.is_relative_to(selected_raw)
        or selected_output.is_relative_to(selected_archive)
    )
    try:
        payload = apply_raw_retention(
            raw_root=selected_raw,
            archive_root=selected_archive,
            plan_path=plan_path,
            confirmed_plan_fingerprint=confirmed_plan_fingerprint,
            receipt_path=selected_output,
        )
    except ArchiveError as exc:
        LOGGER.error("Retention apply rejected or failed: %s", exc)
        payload = {
            "retention_apply_schema_version": RETENTION_APPLY_SCHEMA_VERSION,
            "ok": False,
            "mode": "apply",
            "status": "rejected",
            "deletion_performed": False,
            "error": str(exc),
        }
        payload.update(exc.details)
        if output_existed_before:
            payload["receipt_write_skipped"] = "output_already_existed"
        elif not output_is_control_safe:
            payload["receipt_write_skipped"] = "output_inside_data_roots"
        else:
            write_health(selected_output, payload)
        print(
            json.dumps(
                _retention_apply_summary(payload, selected_output),
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1) from exc
    print(
        json.dumps(
            _retention_apply_summary(payload, selected_output),
            indent=2,
            sort_keys=True,
        )
    )


def _run_history_import(
    config: AppConfig,
    start_date: str,
    end_date: str,
    symbols: Sequence[str] | None,
    history_root: Path | None,
    output: Path | None,
) -> None:
    try:
        from tradingbot.data.bybit_history import (
            HistoryImportError,
            import_bybit_history_range,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow" or (exc.name or "").startswith("pyarrow."):
            LOGGER.error(
                "History import support is not installed; install the project with "
                "the [dataset] extra"
            )
            raise SystemExit(1) from exc
        raise

    selected_symbols = config.bybit.symbols if symbols is None else tuple(symbols)
    unknown = sorted(set(selected_symbols) - set(config.bybit.symbols))
    if unknown:
        LOGGER.error("History symbols are not configured: %s", ", ".join(unknown))
        raise SystemExit(1)
    destination = (
        config.history.root if history_root is None else history_root
    ).expanduser().resolve()
    for label, protected in (
        ("storage.root", config.storage.root),
        ("archive.root", config.archive.root),
    ):
        if destination.is_relative_to(protected) or protected.is_relative_to(
            destination
        ):
            LOGGER.error("History root must not overlap %s", label)
            raise SystemExit(1)
    try:
        result = import_bybit_history_range(
            history_root=destination,
            start_date=start_date,
            end_date=end_date,
            symbols=selected_symbols,
            public_base_url=config.history.public_base_url,
            assumed_latency_ms=config.history.assumed_latency_ms,
            request_timeout_seconds=config.history.request_timeout_seconds,
            download_attempts=config.history.download_attempts,
            maximum_missing_minutes=config.history.maximum_missing_minutes,
            minimum_free_bytes=config.storage.min_free_bytes,
        )
    except HistoryImportError as exc:
        LOGGER.error("Official history import rejected: %s", exc)
        payload: dict[str, object] = {
            "ok": False,
            "dataset_profile": "price_futures_v1",
            "start_date": start_date,
            "end_date": end_date,
            "symbols": list(selected_symbols),
            "error": str(exc),
        }
        if output is not None:
            write_health(output.expanduser().resolve(), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1) from exc
    payload = result.to_dict()
    if output is not None:
        write_health(output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))

    if args.command == "validate-config":
        print(json.dumps(_config_summary(config), indent=2, sort_keys=True))
        return
    if args.command == "show-topics":
        print("\n".join(build_public_topics(config)))
        return
    if args.command == "collect":
        asyncio.run(_run_collection(config, args.run_seconds))
        return
    if args.command == "audit-data":
        _run_data_audit(
            config,
            args.root,
            args.minimum_duration_seconds,
            args.strict,
            args.output,
            args.partition_date,
        )
        return
    if args.command == "build-dataset":
        _run_build_dataset(
            config,
            args.audit_report,
            args.root,
            args.output_root,
        )
        return
    if args.command == "build-research":
        _run_build_research(
            config,
            args.dataset,
            args.catalog,
            args.output_root,
        )
        return
    if args.command == "build-execution-research":
        _run_build_execution_research(
            config,
            args.dataset,
            args.catalog,
            args.output_root,
            args.horizon_minutes,
            args.order_notional_usdt,
            args.submission_latency_ms,
            args.activation_max_delay_ms,
            args.entry_ttl_seconds,
            args.queue_ahead_multiplier,
        )
        return
    if args.command == "build-price-research":
        _run_build_price_research(
            config,
            args.catalog,
            args.from_date,
            args.to_date,
            args.output_root,
        )
        return
    if args.command == "run-backtest":
        _run_backtest(
            config,
            args.research_dataset,
            args.output_root,
            args.horizon_minutes,
        )
        return
    if args.command == "run-execution-backtest":
        _run_execution_backtest(
            config,
            args.execution_dataset,
            args.output_root,
            args.horizon_minutes,
            args.order_notional_usdt,
        )
        return
    if args.command == "build-shadow-bundle":
        _run_build_shadow_bundle(
            config,
            args.execution_evaluation,
            args.execution_dataset,
            args.output_root,
            args.allow_technical_smoke,
        )
        return
    if args.command == "validate-shadow-bundle":
        _run_validate_shadow_bundle(args.bundle)
        return
    if args.command == "shadow":
        _run_shadow(
            config,
            args.bundle,
            args.shadow_root,
            args.run_id,
            args.run_seconds,
            args.processing_delay_ms,
        )
        return
    if args.command == "archive-day":
        _run_archive_day(
            config,
            args.date,
            args.root,
            args.archive_root,
            args.minimum_duration_seconds,
            args.output,
        )
        return
    if args.command == "plan-retention":
        _run_retention_plan(
            config,
            args.root,
            args.archive_root,
            args.retention_days,
            args.as_of_date,
            args.output,
        )
        return
    if args.command == "apply-retention":
        _run_retention_apply(
            config,
            args.plan,
            args.confirm_plan_fingerprint,
            args.root,
            args.archive_root,
            args.output,
        )
        return
    if args.command == "import-history":
        _run_history_import(
            config,
            args.from_date,
            args.to_date,
            args.symbols,
            args.history_root,
            args.output,
        )
        return
    parser.error(f"Unknown command: {args.command}")
