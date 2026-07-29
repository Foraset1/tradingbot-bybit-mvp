# Канонический Dataset stage

Этот этап преобразует прошедший строгий аудит immutable snapshot из JSONL в
типизированный Parquet. Он остаётся полностью read-only относительно рынка и не имеет
торговых полномочий.

Этот слой остаётся каноническим рыночным входом. Признаки, decision grid и labels
`TP_FIRST` / `SL_FIRST` / `TIMEOUT` строятся поверх него отдельной командой, чтобы raw
нормализация не смешивалась с параметрами исследовательского эксперимента. Их контракт
описан в [`FEATURES_AND_LABELS.md`](FEATURES_AND_LABELS.md).

## Входной gate

Команда принимает только audit report schema v2, для которого одновременно выполнено:

- `readiness.ok=true`;
- `readiness.strict=true`;
- нет errors, warnings, `.partial`, missing и short streams;
- fingerprint отчёта совпадает с его списком файлов.

Builder не сканирует произвольные новые файлы. Он читает только пути из audit report и во
время конвертации повторно проверяет размер, число строк, число wrapper-записей и SHA-256
каждого сегмента. Поэтому продолжающий работать collector не может незаметно изменить вход
уже зафиксированной сборки.

## Запуск

В контейнере:

```bash
docker compose run --rm --no-deps collector \
  python -m tradingbot build-dataset \
  --audit-report /app/runtime/24-hour-audit.json \
  --root /data/raw \
  --output-root /data/datasets
```

Если `--root` не указан, используется `dataset_root` из audit report. Если
`--output-root` не указан, используется каталог `datasets` рядом с настроенным
`storage.root`.

Свободного места перед стартом должно хватать как минимум на полный размер входных JSONL
плюс настроенный disk reserve. Это консервативная проверка: ZSTD Parquet обычно будет
существенно меньше raw JSONL.

## Результат

Имя каталога детерминировано версией схемы и fingerprint входа:

```text
/data/datasets/canonical-v1-dfd2a620552d79b9/
├── manifest.json
├── source-audit.json
└── market/
    ├── kind=orderbook/symbol=BTCUSDT/date=2026-07-27/part-00000.parquet
    ├── kind=ticker/symbol=BTCUSDT/date=2026-07-27/part-00000.parquet
    ├── kind=trades/symbol=BTCUSDT/date=2026-07-27/part-00000.parquet
    └── kind=kline_1/symbol=BTCUSDT/date=2026-07-27/part-00000.parquet
```

Запись идёт сначала в уникальный временный каталог. `manifest.json` создаётся последним,
после чего весь каталог атомарно переименовывается. Ошибка не оставляет частично готовый
dataset под финальным именем.

Повторный запуск с тем же входом не переписывает файлы: он сверяет SHA-256 существующего
набора и возвращает `reused=true`. Изменённый или неполный существующий набор считается
ошибкой.

## Канонизация

- orderbook и ticker: одна типизированная строка на raw wrapper;
- public trades: массив сделки разворачивается в одну строку на trade ID;
- kline: одна строка на `(symbol, interval, start_ms)`;
- для нескольких версий kline выбирается payload с максимальным `received_at_ns`;
- разные payload с одинаковым `received_at_ns` являются неразрешимой ошибкой;
- raw JSONL никогда не изменяется.

`received_at_ns` остаётся обязательным causal key. Даже финальную версию свечи запрещено
использовать в историческом решении раньше времени, когда эта версия фактически пришла
collector-у.

Партиция `date` строится по UTC-дате `exchange_ts_ms`, то есть по дате доступного биржевого
сообщения, а не по локальному времени сервера.

## Типизированные таблицы

Основные поля:

- общие: schema version, source, session ID, exchange timestamp, causal receive timestamp и
  provenance исходной строки;
- orderbook: update/sequence, matching-engine timestamp и отдельные массивы price/size для
  bid/ask;
- trades: event timestamp, trade ID, side, price, size, sequence и flags;
- ticker: bid/ask, last/index/mark, open interest, funding и 24h показатели;
- kline: interval, start/end, OHLCV, turnover и SHA-256 выбранного payload.

Parquet создаётся PyArrow 25.0.0, формат 2.6, ZSTD level 3, со statistics, page index и page
checksums. Версии и полные схемы записаны в `manifest.json`.

## Что хранить

До завершения воспроизводимого backtest нельзя удалять:

- исходный audited snapshot;
- соответствующий audit report;
- `source-audit.json`;
- `manifest.json`.

Fingerprint входа связывает raw snapshot, audit и канонический dataset. Fingerprint выхода
связывает список, размеры и SHA-256 всех Parquet-файлов.
