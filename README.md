# TradingBot — Bybit futures research foundation

Первая безопасная версия проекта: read-only сборщик публичных рыночных данных Bybit
для последующего обучения и проверки локальной модели. Эта версия **не принимает торговых
решений, не запрашивает API-ключи и не умеет отправлять ордера**.

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
Пошаговое production-развёртывание на Ubuntu Server 24.04 описано в
[`docs/UBUNTU_DEPLOYMENT.md`](docs/UBUNTU_DEPLOYMENT.md).

Полная локальная проверка:

```bash
make check
```

## Docker Compose

```bash
docker compose up --build -d
docker compose logs -f collector
```

Остановка:

```bash
docker compose down
```

Данные и health-файл сохраняются в именованных Docker volumes `market-data` и
`collector-runtime`. Для постоянного сбора заранее нужно настроить мониторинг свободного
места: стаканы и сделки могут давать несколько гигабайт необжатых данных в сутки,
особенно во время высокой волатильности. В локальном 90-секундном smoke-test было записано
2 835 799 байт — грубая экстраполяция около 2,71 GB/сутки до учёта рыночного режима.

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

## Границы текущей версии

В проекте пока намеренно нет приватного API Bybit, ключей, плеча, исполнения ордеров,
торговой стратегии или обещания доходности. Следующий этап — накопить и проверить
датасет, построить признаки и честный walk-forward backtest с комиссиями,
проскальзыванием стопа и funding. Текущих секундных L50 snapshots и public trades достаточно
для первой модели движения на 5–60 минут, но недостаточно для точного положения maker-ордера
в очереди. `NO_FILL` и partial fills будут отдельным этапом simulator с более детальными
изменениями стакана и калибровкой по demo fills.

Это исследовательское ПО, а не финансовая рекомендация. Даже хорошо протестированная
модель может потерять деньги из-за смены режима рынка, ошибок исполнения или сбоя биржи.

## Используемые контракты Bybit V5

Контракты сверены с официальной документацией Bybit 20 июля 2026 года:

- [WebSocket connection и heartbeat](https://bybit-exchange.github.io/docs/v5/ws/connect)
- [Orderbook snapshot/delta](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook)
- [Public trades](https://bybit-exchange.github.io/docs/v5/websocket/public/trade)
- [Ticker](https://bybit-exchange.github.io/docs/v5/websocket/public/ticker)
- [Kline](https://bybit-exchange.github.io/docs/v5/websocket/public/kline)
