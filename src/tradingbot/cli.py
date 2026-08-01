from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
from collections.abc import Sequence
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
    research.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Canonical dataset directory containing manifest.json",
    )
    research.add_argument(
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
) -> None:
    report = audit_dataset(
        root=config.storage.root if root is None else root,
        symbols=config.bybit.symbols,
        kline_intervals=config.market.kline_intervals,
        minimum_duration_seconds=minimum_duration_seconds,
        strict=strict,
        scratch_dir=config.storage.health_path.parent,
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
    dataset: Path,
    output_root: Path | None,
) -> None:
    try:
        from tradingbot.research.builder import build_research_dataset
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
        result = build_research_dataset(
            canonical_dataset=dataset,
            output_root=destination,
            minimum_free_bytes=config.storage.min_free_bytes,
        )
    except ResearchBuildError as exc:
        LOGGER.error("Research build rejected: %s", exc)
        raise SystemExit(1) from exc
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def _run_backtest(
    config: AppConfig,
    research_dataset: Path,
    output_root: Path | None,
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
            args.output_root,
        )
        return
    if args.command == "run-backtest":
        _run_backtest(config, args.research_dataset, args.output_root)
        return
    parser.error(f"Unknown command: {args.command}")
