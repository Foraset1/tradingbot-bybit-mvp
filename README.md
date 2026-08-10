# TradingBot — Bybit futures research foundation

Безопасная read-only основа проекта: сборщик публичных рыночных данных Bybit,
канонический Parquet-слой, воспроизводимый набор causal features/market labels и offline
оценка локальной модели. Эта версия **не принимает live-решений, не запрашивает API-ключи
и не умеет отправлять ордера**.

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
- безопасный dry-run retention: raw разрешён к очистке только после сверки архива и SHA-256;
- типизированные orderbook, trades, ticker и kline таблицы с ZSTD-сжатием;
- causal last-write-wins для ревизий закрытых свечей и versioned dataset manifest;
- минутная UTC decision grid с проверкой `received_at_ns <= decision_at_ns`;
- признаки цены, волатильности, стакана, order flow, OI/funding и режима BTC;
- отдельные triple-barrier labels 5/15/30/60 минут без ложного `maker fill`;
- атомарный versioned research dataset с проверяемыми fingerprint и provenance;
- purged walk-forward split с 60-минутным embargo и явным режимом короткого smoke-test;
- class-prior и logistic baseline, а также детерминированный глобальный LightGBM;
- backtest, выбирающий одну пару/сторону и учитывающий комиссии, slippage, funding,
  лимит номинала, одну позицию и rolling 24h loss gate;
- versioned evaluation report, модели и Parquet-ledger сделок с SHA-256 проверкой;
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
Запуск baseline, LightGBM и cost-aware backtest описан в
[`docs/RESEARCH_BACKTEST.md`](docs/RESEARCH_BACKTEST.md).
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

Стаканы и сделки могут давать несколько гигабайт необжатых данных в сутки. В локальном
90-секундном smoke-test было записано 2 835 799 байт — грубая экстраполяция около
2,71 GB/сутки до учёта рыночного режима. На диске 100 GB целевое окно raw — 7 дней;
завершённые UTC-дни сохраняются в существенно меньшем Parquet-архиве. Disk guard оставляет
минимум 15 GiB свободного места и останавливает сбор вместо заполнения файловой системы.
Версия 0.6.0 строит только проверяемый dry-run очистки и ещё не удаляет файлы.

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

Для длительного сбора используется каталог последовательных суточных архивов:

```bash
tradingbot build-research \
  --catalog data/archive/catalog.json \
  --output-root data/research
```

Техническая offline-оценка строится из неизменяемого research dataset:

```bash
tradingbot run-backtest \
  --research-dataset data/research/research-v1-<input-fingerprint> \
  --output-root data/evaluations
```

## Границы текущей версии

В проекте пока намеренно нет приватного API Bybit, ключей, плеча, исполнения ордеров,
торговой стратегии или обещания доходности. Baseline, LightGBM и purged walk-forward backtest
уже реализованы. История короче 44 дней запускает только технический 70/30 smoke-test;
первый осмысленный обзор модели разрешён после 90 дней и минимум трёх временных folds.
Даже тогда нельзя делать вывод о реальной доходности до отдельного maker execution simulator.
Текущих секундных L50 snapshots и public trades достаточно для первой модели движения на
5–60 минут, но недостаточно для точного положения maker-ордера в очереди. `NO_FILL` и
partial fills будут отдельным этапом simulator с более детальными изменениями стакана и
калибровкой по demo fills.

Это исследовательское ПО, а не финансовая рекомендация. Даже хорошо протестированная
модель может потерять деньги из-за смены режима рынка, ошибок исполнения или сбоя биржи.

## Используемые контракты Bybit V5

Контракты сверены с официальной документацией Bybit 20 июля 2026 года:

- [WebSocket connection и heartbeat](https://bybit-exchange.github.io/docs/v5/ws/connect)
- [Orderbook snapshot/delta](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook)
- [Public trades](https://bybit-exchange.github.io/docs/v5/websocket/public/trade)
- [Ticker](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)
- [Kline](https://bybit-exchange.github.io/docs/v5/websocket/public/kline)
