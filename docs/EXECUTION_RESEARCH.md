# Execution-aware research V3

Этот read-only этап строит из накопленной live-микроструктуры отдельный неизменяемый
dataset для оценки исполнения `PostOnly`-входа. Он отвечает на два разных вопроса:

1. был бы лимитный maker-ордер полностью, частично или совсем не исполнен;
2. только после полного исполнения — что произошло раньше: TP, SL или временная граница.

Команда ничего не отправляет на Bybit, не требует API-ключей и не разрешает live-торговлю.
Результат остаётся приближённой моделью исполнения, потому что публичный поток не сообщает
реальное место нашего ордера в очереди.

## Источник данных

Профиль `execution_microstructure_v1` принимает только:

- один canonical dataset, созданный `build-dataset`; или
- `/data/archive/catalog.json` с последовательными завершёнными UTC-днями live-сбора.

Официальный price-only архив Bybit для этого этапа недостаточен: в нём нет локальных
`received_at_ns`, L50-стакана, spread и видимой очереди. Он остаётся полезным для модели
движения цены V2, но не может подтвердить maker fill.

Перед сборкой повторно проверяются canonical manifests, fingerprints и состав обязательных
таблиц `orderbook`, `trades`, `ticker`, `kline_1` для всех шести пар.

Каталог может содержать дни с `quality.status: gapped`, если их единственная проблема —
зафиксированные `kline_gap`. V3 не интерполирует стакан, сделки или свечи и не считает весь
такой день пригодным автоматически. Пригодность доказывается отдельно для каждого окна.

## Зафиксированные предположения schema v1

| Параметр | Значение |
|---|---:|
| Decision grid | раз в минуту, на 5-й секунде |
| Стороны | `LONG` и `SHORT` |
| Entry LONG | best bid в момент решения |
| Entry SHORT | best ask в момент решения |
| Submission latency | 250 ms |
| Максимальное ожидание activation book | 2 500 ms |
| Entry TTL | 30 секунд от решения |
| Номиналы | 50 / 100 / 250 / 500 USDT |
| Горизонты после полного fill | 15 / 30 минут |
| Видимая очередь впереди | `1.0 ×` размер уровня при activation |
| Cancellations | не продвигают наш ордер |
| Hidden liquidity | не моделируется |
| Максимальный разрыв orderbook | 90 000 ms |

Параметры записываются в manifest и входят в ID dataset. Изменение любого значения создаёт
другой immutable dataset, а не перезаписывает старый.

## Причинность и модель очереди

Feature-строка использует только записи с:

```text
received_at_ns <= decision_at_ns
```

После виртуальной задержки выбирается первый стакан, реально полученный не раньше времени
submission. На нём повторно проверяется `PostOnly`:

- LONG должен оставаться строго ниже activation ask;
- SHORT должен оставаться строго выше activation bid;
- пересекающий spread ордер получает `NO_FILL` с причиной
  `post_only_would_cross_at_observed_activation`.

Если цена сохранилась в книге, объём уровня считается очередью впереди. Если цена оказалась
внутри spread, очередь впереди равна нулю. Отмены впереди намеренно игнорируются — это
консервативнее, чем автоматически считать их нашим продвижением.

Далее только публичные агрессивные сделки противоположной стороны уменьшают очередь.
Block trades и RPI prints исключаются и из queue depletion, и из post-fill TP/SL proxy,
потому что они не доказывают исполнение в видимой очереди обычного `PostOnly`-ордера:

- для LONG — сделки `Sell` ровно по entry price;
- для SHORT — сделки `Buy` ровно по entry price;
- сделка сквозь цену лимита означает полный fill;
- достигнутый после очереди, но меньший полного размера объём сохраняется как
  `PARTIAL_FILL`;
- отсутствие доступного объёма до TTL — `NO_FILL`.

TP/SL начинает отсчитываться только после события полного fill. При одинаковом
`received_at_ns` дальнейшие сделки упорядочиваются по `event_ts_ms` и `sequence`. Если
точный порядок всё равно нельзя доказать, строка получает `AMBIGUOUS`.

## Фильтр непрерывности

До выпуска строки V3 проверяет три независимых интервала:

1. историю признаков до `decision_at_ns`;
2. maker-entry от activation snapshot до конца TTL;
3. после `FULL_FILL` — весь TP/SL horizon.

Каждый интервал принимается только когда:

- orderbook остаётся в одной наблюдаемой `session_id`;
- до начала и конца окна есть достаточно свежие снимки;
- между соседними снимками нет разрыва больше `maximum_continuity_gap_ms`;
- присутствует каждая минутная свеча, пересекающая окно.

Небезопасное окно исключается, но остальные интервалы того же UTC-дня остаются доступными.
Причины и количества исключений записываются в `quality_by_symbol`, включая
`skipped_discontinuous_feature_window`,
`entry_scenarios_skipped_discontinuous_window` и
`labels_skipped_discontinuous_position_<horizon>m`. Исходные gapped-даты перечисляются в
`source.gapped_partition_dates`.

## Выходные таблицы

Dataset имеет вид:

```text
/data/execution-research/execution-research-v1-<fingerprint>/
  manifest.json
  source-manifest.json
  table=features/symbol=BTCUSDT/date=YYYY-MM-DD/part-00000.parquet
  table=execution_labels/symbol=BTCUSDT/date=YYYY-MM-DD/part-00000.parquet
  ...
```

`execution_labels` хранит отдельно:

- исходный номинал и размер в базовом активе;
- activation book и результат `PostOnly`-проверки;
- видимую и требуемую очередь;
- `NO_FILL`, `PARTIAL_FILL` или `FULL_FILL`;
- момент и цену полного исполнения;
- post-fill outcome `TP_FIRST`, `SL_FIRST`, `TIMEOUT` или `AMBIGUOUS`;
- timestamps, sequence, расстояния барьеров и причину разрешения.

Manifest содержит SHA-256 всех Parquet-файлов, source/parameter/output fingerprints,
счётчики качества по каждой паре, исходные gapped-даты и распределения fill/outcome.

## Ограничение памяти

Каталог обрабатывается по одному UTC-дню. Для каждого символа одновременно загружаются
не более трёх разделов: предыдущий день для causal history, текущий день и следующий день
для незавершённого TTL/горизонта. Это позволяет запускать V3 на физическом сервере
4 CPU / 8 GB RAM без загрузки всей многодневной истории в память. Пары также обрабатываются
последовательно.

## Запуск на сервере

Сначала обновить код и образ. Конкретный commit следует сверить с опубликованным PR:

```bash
cd /opt/tradingbot
git fetch origin
git switch main
git pull --ff-only
docker compose build collector
docker compose up -d collector
docker compose ps
```

Создать каталог отчётов пользователя `foraset1`:

```bash
sudo install -d -o foraset1 -g foraset1 /home/foraset1/tradingbot-reports
```

Сборку удобно запустить в `tmux`, поскольку она может идти несколько часов:

```bash
tmux new -s execution-v3
```

Внутри `tmux`:

```bash
set +e
cd /opt/tradingbot

REPORT_DIR=/home/foraset1/tradingbot-reports
STAMP=$(date -u +%Y%m%d-%H%M%S)
REPORT="$REPORT_DIR/execution-v3-$STAMP-build.json"
LOG="$REPORT_DIR/execution-v3-$STAMP.log"

docker compose run --rm --no-deps collector \
  python -m tradingbot build-execution-research \
  --catalog /data/archive/catalog.json \
  --output-root /data/execution-research \
  > "$REPORT" 2> "$LOG"

STATUS=$?
sudo chown foraset1:foraset1 "$REPORT" "$LOG"
echo "exit=$STATUS"
jq . "$REPORT"
grep -Ei 'warning|error|traceback' "$LOG" || true
docker compose ps
```

При `exit=0` сохранить полный manifest с распределениями fill/outcome:

```bash
DATASET=$(jq -r '.dataset_path' "$REPORT")
MANIFEST="$REPORT_DIR/execution-v3-$STAMP-manifest.json"

docker compose run --rm --no-deps collector \
  sh -c "cat '$DATASET/manifest.json'" \
  > "$MANIFEST"
sudo chown foraset1:foraset1 "$MANIFEST"

jq '{execution_dataset_id,source,processing,output_rows,fill_statuses,execution_outcomes,quality_by_symbol}' \
  "$MANIFEST"
```

Отсоединиться, не прерывая процесс: `Ctrl+B`, затем `D`. Вернуться:

```bash
tmux attach -t execution-v3
```

Повтор той же команды безопасен: при совпадающих source и параметрах проверенный dataset
будет переиспользован с `"reused": true`.

## Как интерпретировать результат

Этот dataset разрешено использовать для обучения fill-модели и последующего
execution-aware backtest. Его **нельзя** трактовать как доказательство прибыльности:

- реальный order acknowledgement и фактическая queue position не наблюдаются;
- модификации/отмены книги между снимками не восстанавливают очередь полностью;
- hidden/iceberg liquidity неизвестна;
- шаг количества, минимальный ордер, комиссии, funding и slippage выхода ещё должны быть
  применены в simulator;
- partial fill пока является отдельным классом, а не автоматически полноценной позицией.

Следующий gate после успешной сборки: проверить coverage/fill distributions, затем обучить
отдельную модель вероятности `FULL_FILL` и объединить её с post-fill EV. До этого V2 H15
остаётся только исследовательским кандидатом, а H30/H60 не допускаются к торговле.
