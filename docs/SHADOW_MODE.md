# Read-only Shadow Mode

Shadow Mode подключает **зафиксированную** пару LightGBM-моделей к отдельному публичному
WebSocket Bybit, рассчитывает решения на минутной UTC-сетке и записывает их результат по
консервативному public-data execution proxy. Он не содержит приватного REST/WebSocket API,
не принимает API-ключи и не отправляет ордера.

Текущие 28 дней V3-данных дают только `technical_smoke`. Поэтому первый запуск предназначен
для проверки live-пайплайна, causal parity, стабильности модели и полноты журналов. Он не
является подтверждением прибыльности и не разрешает переход к реальным сделкам.

## Защитный контракт

- bundle копирует точные LightGBM-файлы **последнего уже проверенного temporal fold**;
- `build-shadow-bundle` не обучает и не дообучает модель;
- SHA-256 проверяет evaluation report, исходные модели, bundle и его manifest;
- любой ещё не прошедший model-review bundle требует явного
  `--allow-engineering-only` (`--allow-technical-smoke` сохранён как alias);
- в bundle навсегда записаны `eligible_for_trading=false` и `order_submission=false`;
- runtime прекращает запуск, если видит `BYBIT_API_KEY`, `BYBIT_API_SECRET` или другой
  известный trading-key env var;
- runtime принимает только production linear public endpoint Bybit и зафиксированный
  профиль L50/1s snapshots, соответствующий обучающим признакам;
- конфигурация обязана содержать ровно тот же уникальный набор символов, а их порядок
  может отличаться; внутри модели сохраняется канонический порядок bundle;
- после reconnect все live-буферы очищаются и начинается новый 61-минутный warm-up;
- одновременно допускается только один выбранный кандидат на все шесть пар;
- каждое решение сохраняется в single-writer hash-chain JSONL;
- неразрешимый или неоднозначный public proxy получает нулевой результат, а не выдуманный
  fill/PnL.

Public L50 snapshots всё равно не показывают реальную позицию заявки в очереди, exchange
acknowledgement, hidden liquidity и фактическую задержку отправки. Поэтому `FULL_FILL` в
журнале означает только консервативную оценку по видимому потоку, не исполнение на Bybit.

## 1. Собрать неизменяемый bundle

На сервере выбрать **один** execution-aware эксперимент и точный V3 dataset, из которого он
был построен. Одновременно запускать H15 и H30 нельзя: два независимых процесса не разделят
единый one-position lock.

Для текущего H15 technical smoke:

```bash
cd /opt/tradingbot
REPORT_DIR=/home/foraset1/tradingbot-reports
EXECUTION_DATASET=/data/execution-research/execution-research-v1-4c5b78cf97f6a138
EXECUTION_EVALUATION=/data/execution-evaluations/execution-backtest-v1-4b8e1aeb875a78ef
STAMP=$(date -u +%Y%m%d-%H%M%S)

docker compose run --rm --no-deps collector \
  python -m tradingbot build-shadow-bundle \
  --execution-evaluation "$EXECUTION_EVALUATION" \
  --execution-dataset "$EXECUTION_DATASET" \
  --output-root /data/shadow-bundles \
  --allow-engineering-only \
  > "$REPORT_DIR/shadow-bundle-h15-$STAMP.json" \
  2> "$REPORT_DIR/shadow-bundle-h15-$STAMP.log"

STATUS=$?
chown foraset1:foraset1 \
  "$REPORT_DIR/shadow-bundle-h15-$STAMP.json" \
  "$REPORT_DIR/shadow-bundle-h15-$STAMP.log"
echo "exit=$STATUS"
jq . "$REPORT_DIR/shadow-bundle-h15-$STAMP.json"
```

Ожидаются `exit=0`, `data_mode=technical_smoke`, `engineering_only=true` и
`reused=false` при первой сборке. Повторный запуск проверяет все файлы и возвращает тот же
bundle с `reused=true`.

Сохранить путь:

```bash
BUNDLE=$(jq -r '.bundle_path' \
  "$REPORT_DIR/shadow-bundle-h15-$STAMP.json")
echo "$BUNDLE"

docker compose run --rm --no-deps collector \
  python -m tradingbot validate-shadow-bundle \
  --bundle "$BUNDLE"
```

`validate-shadow-bundle` дополнительно загружает обе модели и сверяет число признаков.

## 2. Настроить постоянный Compose-процесс

Записать точный путь и стабильный ID в `/opt/tradingbot/.env`:

```dotenv
TRADINGBOT_SHADOW_BUNDLE=/data/shadow-bundles/shadow-bundle-v1-<fingerprint>
TRADINGBOT_SHADOW_RUN_ID=shadow-h15-n50-v1
TRADINGBOT_SHADOW_CPUS=1.0
TRADINGBOT_SHADOW_MEMORY=2g
```

Для физического узла 4 CPU / 8 GB разумный суммарный профиль:

```dotenv
TRADINGBOT_COLLECTOR_CPUS=2.5
TRADINGBOT_COLLECTOR_MEMORY=4g
TRADINGBOT_SHADOW_CPUS=1.0
TRADINGBOT_SHADOW_MEMORY=2g
```

Не добавлять в `.env` API key/secret. Стандартный Compose-профиль вообще не передаёт такие
переменные контейнеру; если ключ всё же будет явно внедрён через `-e` или изменённый Compose,
Shadow Mode завершится с ошибкой до открытия WebSocket.

Проверить Compose и запустить профиль:

```bash
cd /opt/tradingbot
docker compose config --quiet
docker compose --profile shadow up -d shadow
docker compose --profile shadow ps
docker compose logs --tail=100 shadow
```

Shadow использует отдельное публичное WebSocket-соединение с теми же 36 topics. Постоянный
collector продолжает независимо записывать raw-данные.

## 3. Warm-up и health

После старта требуется около **61 минуты** непрерывной одной WebSocket-сессии: модели нужны
закрытые 1m candles, trades, ticker и L50 orderbook за полный feature window. До завершения
warm-up каждую минуту будет journal event с `skip_reason=warmup_or_data_quality`.

```bash
RUN_ID=shadow-h15-n50-v1

docker compose exec -T shadow sh -c \
  "cat /data/shadow/$RUN_ID/health.json" | jq .

docker compose exec -T shadow sh -c \
  "tail -5 /data/shadow/$RUN_ID/events-*.jsonl"
```

Health обязан показывать:

- `status=running`;
- `scope.order_submission=false`;
- `engineering_only=true` для текущего короткого dataset;
- `disk.above_guard=true`;
- текущие `session_id`, `last_market_record_at_ns` и `last_decision_at_ns`;
- не больше одного `pending_decision_id`.

## 4. Что записывается

Каталог `/data/shadow/<run-id>/` содержит:

- `run.json` — неизменяемая связь с bundle fingerprint и read-only scope;
- `events-YYYY-MM-DD.jsonl` — hash-chain журнал;
- `health.json` — атомарный текущий снимок.

Основные события:

- `run_started` / `run_resumed`;
- `websocket_session_reset`;
- `decision_cycle` с causal features, всеми 12 кандидатами, EV и причиной пропуска;
- `candidate_selected`;
- `candidate_settled` с public execution proxy и simulated PnL;
- `candidate_unresolved` для неполного/неоднозначного окна;
- `run_stopped`.

Выбранный кандидат блокирует глобальный слот на entry TTL, полный H15/H30 и 90 секунд
settlement grace. Это намеренно консервативнее исторического replay: live-процесс не
освобождает слот досрочно по предполагаемому TP/SL.

## 5. Остановка и перезапуск

```bash
cd /opt/tradingbot
docker compose stop -t 30 shadow
docker compose --profile shadow up -d shadow
```

Тот же `TRADINGBOT_SHADOW_RUN_ID` продолжает проверенную hash-chain. Кандидат, который был
активен при остановке или reconnect, фиксируется как `candidate_unresolved` с нулевым
simulated return — он не восстанавливается по неполному live-окну.

Для нового эксперимента нужен новый bundle и новый run ID. Нельзя заменять bundle под уже
существующим ID: runtime отвергнет несовпадающий fingerprint.

## 6. Когда считать этап завершённым

Минимальная инженерная проверка Shadow Mode:

1. не менее 14 непрерывных дней без повреждения журнала;
2. отсутствие crash-loop, необъяснимых session resets и систематических data-quality skips;
3. causal feature parity на выбранных решениях против offline replay;
4. стабильные распределения fill/outcome probabilities по символам и сторонам;
5. ручная сверка proxy settlements на нескольких десятках кандидатов;
6. 365 завершённых UTC дней V3 и минимум три walk-forward folds для следующего model review.

Даже после этого Shadow Mode не становится торговым сервисом. Следующий отдельный этап —
Bybit Demo с минимальными разрешениями, exchange acknowledgements и измерением реальных
PostOnly fill/latency. Он потребует нового кода и отдельного явного подтверждения.
