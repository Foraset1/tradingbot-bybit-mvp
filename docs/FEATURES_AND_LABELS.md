# Causal features и market labels

Этот этап преобразует принятый канонический Parquet dataset в две физически раздельные
таблицы:

- `features` — сведения, реально доступные на момент исторического решения;
- `labels` — будущее движение рынка после решения.

Система остаётся полностью read-only. Здесь нет API-ключей, ордеров, позиции, оценки
maker fill или решения «торговать».

Существуют два несовместимых по входным данным, но совместимых с offline evaluator
research-профиля:

- `microstructure_research_v1` — live orderbook/trades/ticker/kline с локальными
  `received_at_ns`;
- `price_futures_research_v1` — официальные 1s/1m trade-bars с консервативными
  `available_at_ns`, без стакана, spread, funding и OI.

Второй профиль строится командой `build-price-research`; отсутствующие признаки не входят
в его Arrow-схему и модель, а не заполняются искусственными нулями.

## Входной gate

`build-research` принимает либо один dataset, созданный `build-dataset`, либо
`catalog.json` последовательных суточных архивов. Перед чтением данных команда повторно
проверяет:

- `source-audit.json`;
- ID и schema version канонического dataset;
- размер, число строк и SHA-256 каждого Parquet-файла;
- входной и выходной fingerprint канонического manifest.

Произвольный набор Parquet-файлов без прошедшего gate не принимается.
Для archive catalog дополнительно требуются последовательные UTC-дни и одинаковый набор
символов во всех дневных datasets.

## Запуск

В контейнере:

```bash
docker compose run --rm --no-deps collector \
  python -m tradingbot build-research \
  --dataset /data/datasets/canonical-v1-dfd2a620552d79b9 \
  --output-root /data/research
```

После длительного сбора предпочтительным источником является суточный каталог:

```bash
docker compose run --rm --no-deps collector \
  python -m tradingbot build-research \
  --catalog /data/archive/catalog.json \
  --output-root /data/research
```

Команда обрабатывает пары по очереди, поэтому пиковое потребление памяти зависит от
наиболее активной пары, а не от суммы всех шести. VM с 6 vCPU и 10 GB RAM для текущего
24-часового snapshot имеет достаточный запас. Collector может продолжать запись raw JSONL:
research builder читает только immutable canonical dataset.

## Decision grid и причинность

Стандартная сетка версии 1:

- шаг — 60 секунд;
- точки выровнены по UTC epoch;
- решение принимается на 5-й секунде минуты;
- стакан и ticker не старше 2,5 секунды;
- требуется непрерывная история 61 закрытой минутной свечи.

Для каждой feature-строки действует жёсткое правило:

```text
source.received_at_ns <= decision_at_ns
```

Event timestamp биржи не заменяет момент доступности. Поздно полученная свеча, сделка или
ревизия не может попасть в более раннее решение. В output сохраняются causal timestamps
последнего использованного стакана, ticker, свечи и сделки, чтобы это ограничение можно было
проверить независимо.

Кандидатная точка пропускается, если стакан/ticker устарел, L50 неполон, книга пересечена,
нет непрерывной часовой истории свечей либо нужная версия свечи ещё не была получена.
Причины и количества записываются по каждой паре в `quality_by_symbol`.

## Признаки версии 1

Основные группы:

- returns 1/3/5/15/60 минут;
- realised volatility 5/15/60 минут, ATR14 и минутный range;
- best bid/ask, spread, mid и microprice;
- глубина и imbalance L1/L5/L10/L25/L50;
- число сделок, объём, notional, signed flow и trade return на окнах
  5/30/60 секунд и 5/15 минут;
- mark-index basis, funding, open interest и изменение OI;
- циклические признаки часа UTC и дня недели;
- режим BTC и относительные returns альткоина к BTC.

Все расстояния и returns нормализованы в долях или basis points; абсолютные цена, OI и
ликвидность также сохранены. Это позволяет сначала обучать одну глобальную модель с
идентификатором символа, а затем сравнить её с отдельными моделями.

## Рыночные labels

Для каждого решения формируются отдельные LONG и SHORT labels на горизонтах
5/15/30/60 минут. Entry reference — mid-price последнего допустимого L50 snapshot.

Ожидаемая волатильность горизонта вычисляется только из прошлых 60 минут:

```text
sigma_1m = realised_volatility_60m / sqrt(60)
stop_bps = clip(sigma_1m * sqrt(horizon_minutes) * 10000, 10, 250)
take_profit_bps = 1.5 * stop_bps
```

Будущие public trades упорядочиваются по:

```text
received_at_ns → event_ts_ms → sequence
```

Результат:

- `TP_FIRST` — первой пересечена take-profit граница;
- `SL_FIRST` — первой пересечена stop граница;
- `TIMEOUT` — ни одна граница не пересечена до конца горизонта;
- `AMBIGUOUS` — TP и SL нельзя упорядочить по доступному causal key.

Label создаётся только когда trade stream покрывает весь горизонт. Последний неполный хвост
не превращается в ложный `TIMEOUT`; число пропусков записывается в manifest.

Labels описывают движение публичного рынка. Они **не доказывают**, что `PostOnly` entry был
исполнен, не моделируют очередь, partial fill или реальное stop slippage. `NO_FILL` остаётся
execution label будущего simulator stage.

Комиссии maker/taker, adverse selection, funding и stress slippage применяются на следующем
этапе backtest до присвоения кандидату решения «торговать». Они намеренно не зашиты в
рыночный класс, чтобы разные cost-сценарии можно было сравнить без перестройки признаков.

## Результат и воспроизводимость

Пример структуры:

```text
/data/research/research-v1-<input-fingerprint>/
├── manifest.json
├── source-manifest.json
├── table=features/symbol=BTCUSDT/date=YYYY-MM-DD/part-00000.parquet
└── table=labels/symbol=BTCUSDT/date=YYYY-MM-DD/part-00000.parquet
```

ID зависит от canonical fingerprint, параметров, schema/package и зафиксированных версий
PyArrow/NumPy. В manifest находятся:

- полный source provenance;
- параметры и их fingerprint;
- обе Arrow-схемы;
- quality counters;
- распределение labels;
- SHA-256 каждого output-файла и общий output fingerprint.

Запись идёт во временный каталог, `manifest.json` создаётся последним, затем каталог
атомарно переименовывается. Повторный запуск с тем же входом не переписывает данные, а
проверяет все файлы и возвращает `reused=true`.

## Ограничение текущего snapshot

Принятые 24 часа нужны для end-to-end проверки сборщика и pipeline. Они не покрывают
достаточное число рыночных режимов, недель, funding cycles и стресс-событий. На этом наборе
нельзя честно оценить прибыльность или принять модель. Для стадии LightGBM/walk-forward
нужно продолжать сбор и отдельно получить длительную историю доступных market streams.
