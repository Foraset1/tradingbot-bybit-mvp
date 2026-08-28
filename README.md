# TradingBot — Bybit futures research foundation

Безопасная read-only основа проекта: сборщик публичных рыночных данных Bybit,
канонический Parquet-слой, воспроизводимые causal features/market labels,
execution-aware proxy labels, offline-оценка локальной модели и безопасный read-only
Shadow Mode. Эта версия рассчитывает и журналирует live-кандидаты только по публичным
данным, **не запрашивает API-ключи и не умеет отправлять ордера**.

Поддерживаемые USDT perpetual пары:

- `BTCUSDT`
- `ETHUSDT`
- `SOLUSDT`
- `XRPUSDT`
- `BNBUSDT`
- `LINKUSDT`

## Зафиксированные правила будущей торговой системы

| Правило | Значение MVP |
|---|---:|
| Биржа и рынок | Bybit, USDT perpetual futures |
| Одновременные позиции | максимум 1 на все пары |
| Максимальное время позиции | 60 минут |
| Мягкий выход | с 55-й минуты |
| Принудительный выход | с 59-й минуты |
| Вход | `PostOnly`, maker |
| Take-profit | по возможности maker, `reduce-only` |
| Stop-loss | защитный условный market-ордер, `reduce-only` |
| Лимит позиции | не более 5% equity по номиналу |
| Бюджет риска сделки | 0,5% equity, фактический риск может быть меньше |
| Жёсткий плановый риск сделки | не более 0,7% equity с издержками |
| Стоп торговли | 1% потерь equity за скользящие 24 часа |
| Новости и внешняя LLM | не входят в MVP |

Перед live-режимом значение «5% депозита» нужно подтвердить: сейчас оно намеренно
трактуется консервативно как **5% equity по номиналу позиции**, а не как 5% маржи,
умноженные на плечо.

Подробные требования, риск-формулы и этапы находятся в
[`docs/MVP_SPEC.md`](docs/MVP_SPEC.md).

## Что уже реализовано

- подписка на Bybit V5 public WebSocket;
- стакан глубиной 50 с восстановлением `snapshot` + `delta` и записью состояния раз в секунду;
- публичные сделки, ticker и закрытые свечи 1/5/15 минут;
- единый стабильный формат записей и разбиение JSONL по типу, символу и дате;
- ротация файлов, восстановление незавершённых сегментов после сбоя;
- reconnect с exponential backoff, heartbeat и watchdog замолчавшего рыночного потока;
- health по каждой WebSocket-сессии и topic, контроль очереди, задержки и свободного диска;
- `schema_version`, источник и session ID в каждой новой записи;
- потоковый audit JSONL с coverage, gap/duplicate checks и SHA-256 fingerprint входов;
- атомарный канонический Parquet dataset с повторной SHA-256 проверкой raw-сегментов;
- неизменяемый суточный Parquet-архив и детерминированный каталог UTC-разделов;
- dry-run и отдельный подтверждаемый retention apply с повторной сверкой архива, raw и SHA-256;
- типизированные orderbook, trades, ticker и kline таблицы с ZSTD-сжатием;
- causal last-write-wins для ревизий закрытых свечей и versioned dataset manifest;
- минутная UTC decision grid с проверкой `received_at_ns <= decision_at_ns`;
- признаки цены, волатильности, стакана, order flow, OI/funding и режима BTC;
- отдельные triple-barrier labels 5/15/30/60 минут без ложного `maker fill`;
- отдельный профиль `execution_microstructure_v1` с `PostOnly` activation check,
  консервативной видимой очередью и классами `NO_FILL/PARTIAL_FILL/FULL_FILL`;
- post-fill TP/SL/TIMEOUT labels для горизонтов 15/30 минут и номиналов
  50/100/250/500 USDT;
- посуточная execution-сборка с максимум тремя UTC-разделами одного символа в памяти;
- атомарный versioned research dataset с проверяемыми fingerprint и provenance;
- purged walk-forward split с 60-минутным embargo и явным режимом короткого smoke-test;
- class-prior и ограниченный равномерной по времени выборкой logistic baseline, а также
  детерминированный глобальный LightGBM на всех fit-строках;
- backtest, выбирающий одну пару/сторону и учитывающий комиссии, slippage, funding,
  лимит номинала, одну позицию и rolling 24h loss gate;
- versioned evaluation report, модели и Parquet-ledger сделок с SHA-256 проверкой;
- неизменяемый Shadow bundle с точными моделями последнего проверенного fold, калибровкой,
  execution estimates и полным SHA-256 manifest;
- отдельный публичный live Shadow Mode с 61-минутным causal warm-up, единым one-position
  lock, rolling loss gate и консервативным proxy settlement;
- single-writer hash-chain журнал решений, безопасное восстановление и атомарный health;
- жёсткий отказ Shadow Mode при наличии trading credential env vars;
- строгая проверка конфигурации и заранее зафиксированных risk-инвариантов;
- контейнерный запуск и автоматические проверки.

## Быстрый старт

Нужен Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

tradingbot validate-config
tradingbot show-topics
tradingbot collect --run-seconds 90
tradingbot audit-data
```

Сборщик пишет данные в `data/raw`, а состояние — в
`runtime/collector-health.json`. Публичный поток не требует ключей Bybit.
Порядок одночасовой и 24-часовой проверки описан в
[`docs/SOAK_TEST.md`](docs/SOAK_TEST.md).
Построение проверенного Parquet-слоя описано в
[`docs/DATASET.md`](docs/DATASET.md).
Ежедневный архив, каталог и проверка retention описаны в
[`docs/DAILY_ARCHIVE.md`](docs/DAILY_ARCHIVE.md).
Bootstrap бесплатной официальной истории сделок Bybit описан в
[`docs/HISTORICAL_BOOTSTRAP.md`](docs/HISTORICAL_BOOTSTRAP.md).
Контракт causal features и market labels описан в
[`docs/FEATURES_AND_LABELS.md`](docs/FEATURES_AND_LABELS.md).
Консервативная модель maker fill V3 и команды её запуска описаны в
[`docs/EXECUTION_RESEARCH.md`](docs/EXECUTION_RESEARCH.md).
Запуск baseline, LightGBM и cost-aware backtest описан в
[`docs/RESEARCH_BACKTEST.md`](docs/RESEARCH_BACKTEST.md).
Сборка frozen model bundle и постоянный публичный Shadow Mode описаны в
[`docs/SHADOW_MODE.md`](docs/SHADOW_MODE.md).
Пошаговое production-развёртывание на Ubuntu Server 24.04 описано в
[`docs/UBUNTU_DEPLOYMENT.md`](docs/UBUNTU_DEPLOYMENT.md).

Полная локальная проверка:

```bash
make check
```

## Docker Compose

```bash
cp .env.example .env
docker compose config --environment
docker compose up --build -d
docker compose logs -f collector
```

Остановка:

```bash
docker compose down
```

Production-профиль рассчитан на VM с 6 vCPU, 10 GB RAM и диском 100 GB. Collector по
умолчанию ограничен 4 CPU и 6 GB через `.env`, поэтому значения можно уменьшить без правки
Compose-файла. Данные и health-файл сохраняются в именованных Docker volumes `market-data` и
`collector-runtime`.

Offline evaluation также поддерживает физический узел с 4 CPU / 8 GB RAM: запуски идут
строго последовательно с лимитом 2 CPU / 6 GB, освобождением Arrow-буферов, компактными ID и
детерминированным cap в 500 000 строк только для logistic baseline. Primary LightGBM и
365-дневные temporal folds не сокращаются.

Стаканы и сделки могут давать несколько гигабайт необжатых данных в сутки. В локальном
90-секундном smoke-test было записано 2 835 799 байт — грубая экстраполяция около
2,71 GB/сутки до учёта рыночного режима. На диске 100 GB целевое окно raw — 7 дней;
завершённые UTC-дни сохраняются в существенно меньшем Parquet-архиве. Disk guard оставляет
минимум 15 GiB свободного места и останавливает сбор вместо заполнения файловой системы.
Очистка разделена на проверяемый dry-run и отдельную команду применения, требующую
сохранённый план, его точный fingerprint и повторную проверку перед каждым удалением.

Официальные суточные trade archives можно потоково преобразовать в отдельный компактный
профиль `price_futures_v1` без хранения gzip и отдельных тиков:

```bash
tradingbot import-history \
  --from-date 2026-08-01 \
  --to-date 2026-08-02
```

Команда строит 1s/1m trade-bars и проверяемый `/data/history/catalog.json`. Отдельный
price-only research dataset создаётся без выдуманных book/funding/OI-признаков:

```bash
tradingbot build-price-research \
  --catalog data/history/catalog.json \
  --from-date 2026-05-10 \
  --to-date 2026-08-07 \
  --output-root data/research
```

Профиль `price_futures_research_v1` совместим с `run-backtest`, но не моделирует стакан,
spread, funding, maker queue или partial fills. Минуты без публичных сделок сохраняются как
явные gaps; защитный лимит применяется к самому длинному непрерывному trade-free интервалу,
а не к общему количеству разрозненных пустых минут за сутки.

## Формат данных

Пример пути:

```text
data/raw/orderbook/BTCUSDT/2026/07/20/part-....jsonl
```

Каждая строка — отдельный JSON-объект:

```json
{
  "exchange_ts_ms": 1784500000000,
  "kind": "orderbook",
  "payload": {
    "asks": [["118001.0", "0.42"]],
    "bids": [["118000.9", "0.31"]],
    "matching_engine_ts_ms": 1784500000000,
    "sequence": 123,
    "update_id": 456
  },
  "received_at_ns": 1784500000000000000,
  "schema_version": 1,
  "session_id": "1784500000000-1",
  "source": "bybit",
  "symbol": "BTCUSDT"
}
```

`exchange_ts_ms` — время формирования feed-сообщения биржей. Event time остаётся внутри
payload (`T` у сделки, `start/end` у свечи, matching-engine timestamp у стакана).
`received_at_ns` — локальный момент доступности записи и обязательный causal key: при построении
признаков запрещено использовать записи, полученные после времени решения.

Активный файл имеет суффикс `.jsonl.partial`. При штатной остановке он атомарно
переименовывается в `.jsonl`; после аварии при следующем старте сохраняется как
`*-recovered.jsonl`.

Bybit может прислать более позднюю версию уже закрытой свечи с тем же
`symbol/interval/start`, но обновлёнными OHLCV. Raw-слой намеренно сохраняет обе версии.
Audit report v2 отдельно считает точные повторные доставки в `duplicate_klines` и
изменившиеся версии в `kline_revisions`. Канонический Parquet-слой выбирает запись с
максимальным `received_at_ns`; разные payload с одинаковым `received_at_ns` остаются ошибкой,
которую нельзя разрешать автоматически.

После успешного строгого audit:

```bash
tradingbot build-dataset \
  --audit-report runtime/24-hour-audit.json \
  --root data/raw \
  --output-root data/datasets
```

После этого research-слой строится только из конкретного канонического dataset:

```bash
tradingbot build-research \
  --dataset data/datasets/canonical-v1-<input-fingerprint> \
  --output-root data/research
```

Для длительного сбора обычный market-label builder использует только каталог
последовательных суточных архивов со статусом `clean`:

```bash
tradingbot build-research \
  --catalog data/archive/catalog.json \
  --output-root data/research
```

Отдельный V3 dataset для консервативной оценки `PostOnly` и maker queue строится из того
же live-каталога:

```bash
tradingbot build-execution-research \
  --catalog data/archive/catalog.json \
  --output-root data/execution-research
```

Официальный price-only архив не подходит для этой команды, потому что не содержит
локальных L50 snapshots и `received_at_ns`.
V3 может безопасно использовать дни со статусом `gapped`: он не восстанавливает пропуски,
а исключает feature/entry/post-fill окна, пересекающие kline gap, слишком большой разрыв
стакана или смену WebSocket-сессии.

Из одного неизменяемого V3 dataset запускается отдельная execution-aware оценка. Один запуск
фиксирует ровно один горизонт и один reference-номинал, обучает независимые fill и post-fill
модели и затем выбирает не больше одного maker-ордера среди всех пар в каждую минуту:

```bash
tradingbot run-execution-backtest \
  --execution-dataset data/execution-research/execution-research-v1-<fingerprint> \
  --output-root data/execution-evaluations \
  --horizon-minutes 15 \
  --order-notional-usdt 50
```

Сравниваются class-prior, logistic и LightGBM, а для обучаемых моделей — raw и отдельно
калиброванные вероятности. `PARTIAL_FILL` не превращается в полную позицию: остаток отменяется,
а исполненная доля закрывается taker-ордером в симуляции.

Техническая offline-оценка строится из неизменяемого research dataset:

```bash
tradingbot run-backtest \
  --research-dataset data/research/research-v1-<input-fingerprint> \
  --output-root data/evaluations \
  --horizon-minutes 60
```

## Границы текущей версии

В проекте пока намеренно нет приватного API Bybit, ключей, плеча, исполнения ордеров,
автоматической торговли или обещания доходности. Исследовательская policy уже формирует
кандидатов в Shadow Mode, но они существуют только в симуляционном журнале. Baseline,
LightGBM и purged walk-forward backtest реализованы. Evaluation V2 добавляет отдельное
purged calibration-окно, coverage gate
по символам, ablation календарных признаков и diagnostics selection bias. История короче
44 дней запускает только технический 70/30 smoke-test; следующий зафиксированный model review
требует 365 завершённых UTC дней и минимум трёх временных folds.
Даже тогда нельзя делать вывод о реальной доходности. V3 строит консервативные proxy labels
`NO_FILL/PARTIAL_FILL/FULL_FILL`, а execution-aware evaluator отдельно моделирует fill и
post-fill outcome, учитывает maker/taker fee, funding, slippage, partial unwind и правило одной
позиции на все пары. Но публичные L50 snapshots не показывают реальный order acknowledgement,
точное место ордера, hidden liquidity и все отмены внутри очереди. Read-only Shadow Mode уже
может проверить live causal pipeline, но короткий dataset остаётся `engineering_only` и
`eligible_for_trading=false`. Следующие gates — 365-дневный V3 walk-forward и калибровка
proxy по Bybit Demo fills. Только после них возможен отдельный paper/testnet этап.

Это исследовательское ПО, а не финансовая рекомендация. Даже хорошо протестированная
модель может потерять деньги из-за смены режима рынка, ошибок исполнения или сбоя биржи.

## Используемые контракты Bybit V5

Контракты сверены с официальной документацией Bybit 20 июля 2026 года:

- [WebSocket connection и heartbeat](https://bybit-exchange.github.io/docs/v5/ws/connect)
- [Orderbook snapshot/delta](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook)
- [Public trades](https://bybit-exchange.github.io/docs/v5/websocket/public/trade)
- [Ticker](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)
- [Kline](https://bybit-exchange.github.io/docs/v5/websocket/public/kline)
