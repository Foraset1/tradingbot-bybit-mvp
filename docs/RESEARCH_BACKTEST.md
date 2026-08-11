# Offline evaluation V2 и conditional-entry backtest

Этот этап проверяет, содержит ли price/microstructure dataset причинный сигнал на горизонте
15, 30 или 60 минут. Контур полностью read-only: Bybit API key не нужен, ордера не создаются,
токены OpenAI не расходуются.

## Почему появился V2

Первый 90-дневный эксперимент V1 был отклонён:

- LightGBM: 613 сделок, `-3.0471%`, profit factor `0.6285`;
- logistic: 25 сделок, `-0.238%`;
- predicted EV имел отрицательную связь с фактическим результатом;
- модель переоценивала вероятность TP и недооценивала SL;
- даже те же сделки без комиссий оставались немного отрицательными.

Этот результат является development evidence, а не финальным holdout. Нельзя повторно
подбирать параметры по тем же 90 дням и объявлять улучшение.

## Контракт V2

Один запуск:

1. проверяет manifest, SHA-256 и Parquet-схемы;
2. для price-history вычисляет coverage каждой пары по полной UTC minute-grid;
3. до обучения исключает пары с coverage ниже `0.95` (BTC обязан пройти gate);
4. строит expanding walk-forward folds без перемешивания времени;
5. внутри каждого fold выделяет отдельные `fit → purge → calibration → purge → test` окна;
6. обучает class-prior, logistic regression и LightGBM;
7. сравнивает два заранее заданных feature profile: `full` и `no_calendar`;
8. сохраняет raw и причинно calibrated вероятности;
9. в каждую минуту выбирает максимум одну пару/сторону по expected net bps;
10. соблюдает одну позицию, комиссии, slippage, funding и 24-часовой loss limit;
11. записывает breakdown по fold/symbol/side/outcome/EV-bin/EV-decile;
12. отдельно показывает expected-vs-actual gap и max-candidate selection bias;
13. атомарно сохраняет модели, сделки, report и проверяемый manifest.

Калибратор перебирает фиксированную сетку temperature/prior shrinkage только на calibration
окне. Test не используется ни для обучения модели, ни для выбора калибровки.

## Pre-registered матрица

Матрица для следующего исследования фиксируется до просмотра результатов:

| Измерение | Значения |
|---|---|
| Horizon | 15, 30, 60 минут |
| Features | `full`, `no_calendar` |
| Probability | `raw`, `calibrated` |
| Models | logistic, LightGBM (+ class-prior baseline) |
| Primary candidate | `lightgbm_full_calibrated` |
| Symbol coverage | минимум 95% |
| История для review | минимум 365 завершённых UTC дней |

Каждый horizon создаёт отдельный immutable `backtest-v2-*`. Все три результата должны быть
сохранены; нельзя оставить только лучший.

## 365-дневный dataset

Зафиксированный диапазон: `2025-08-08` — `2026-08-07` включительно (365 UTC дней).
Импорт повторно использует уже готовые 90 дней и скачивает только отсутствующие partitions.
Перед запуском должно быть не менее 30 GiB свободного места:

```bash
cd /opt/tradingbot
docker compose exec collector sh -c 'df -h /data; du -sh /data/history /data/research /data/evaluations 2>/dev/null || true'
```

Импорт следует запускать в `tmux`, последовательно, не останавливая collector:

```bash
tmux new -s bybit-history-365d

set +e
cd /opt/tradingbot
REPORT_DIR=/home/foraset1/tradingbot-reports
install -d -m 0750 -o foraset1 -g foraset1 "$REPORT_DIR"
REPORT="$REPORT_DIR/history-import-365d-2026-08-07.json"
LOG="$REPORT_DIR/history-import-365d-2026-08-07.log"

docker compose run --rm --no-deps collector \
  python -m tradingbot import-history \
  --from-date 2025-08-08 \
  --to-date 2026-08-07 \
  > "$REPORT" 2> "$LOG"

STATUS=$?
chown foraset1:foraset1 "$REPORT" "$LOG"
echo "exit=$STATUS"
jq . "$REPORT"
grep -Ei 'warning|error|traceback' "$LOG" || true
docker compose ps
```

Отсоединиться: `Ctrl-b`, затем `d`. Вернуться: `tmux attach -t bybit-history-365d`.

После успешного импорта строится immutable research dataset:

```bash
set +e
cd /opt/tradingbot
REPORT_DIR=/home/foraset1/tradingbot-reports
REPORT="$REPORT_DIR/price-research-365d-2026-08-07.json"
LOG="$REPORT_DIR/price-research-365d-2026-08-07.log"

docker compose run --rm --no-deps collector \
  python -m tradingbot build-price-research \
  --catalog /data/history/catalog.json \
  --from-date 2025-08-08 \
  --to-date 2026-08-07 \
  --output-root /data/research \
  > "$REPORT" 2> "$LOG"

STATUS=$?
chown foraset1:foraset1 "$REPORT" "$LOG"
echo "exit=$STATUS"
jq . "$REPORT"
grep -Ei 'warning|error|traceback' "$LOG" || true
```

## Запуск трёх V2 экспериментов

Запуски выполняются последовательно: два параллельных LightGBM процесса могут мешать
collector и превысить 10 GiB RAM.

```bash
set +e
cd /opt/tradingbot
REPORT_DIR=/home/foraset1/tradingbot-reports
PRICE_RESULT="$REPORT_DIR/price-research-365d-2026-08-07.json"
RESEARCH_DATASET=$(jq -er '.dataset_path' "$PRICE_RESULT") || exit 1

for HORIZON in 15 30 60; do
  RESULT="$REPORT_DIR/backtest-v2-price-365d-h${HORIZON}-build.json"
  LOG="$REPORT_DIR/backtest-v2-price-365d-h${HORIZON}.log"
  TRADINGBOT_COLLECTOR_MEMORY=8g docker compose run --rm --no-deps collector \
    python -m tradingbot run-backtest \
    --research-dataset "$RESEARCH_DATASET" \
    --output-root /data/evaluations \
    --horizon-minutes "$HORIZON" \
    > "$RESULT" 2> "$LOG"
  STATUS=$?
  chown foraset1:foraset1 "$RESULT" "$LOG"
  echo "horizon=$HORIZON exit=$STATUS"
  [ "$STATUS" -eq 0 ] || break
  jq . "$RESULT"
done
```

Эту команду также следует выполнять в отдельной `tmux`-сессии. Она может работать несколько
часов. Collector должен оставаться `healthy`.

## Результаты

```text
/data/evaluations/backtest-v2-<input-fingerprint>/
├── manifest.json
├── report.json
├── models/
│   ├── fold-01-lightgbm-full.txt
│   └── fold-01-lightgbm-no_calendar.txt
└── trades/
    ├── class_prior.parquet
    ├── lightgbm_full_raw.parquet
    ├── lightgbm_full_calibrated.parquet
    ├── lightgbm_no_calendar_raw.parquet
    └── ...
```

`report.json` содержит:

- eligibility/exclusion и coverage каждой пары;
- границы fit/calibration/test и доказательство purge;
- raw/calibrated log loss, Brier, ECE и частоты классов;
- cost-aware backtest и zero-cost same-trades comparison;
- breakdown по fold, symbol, side, outcome, EV bins и deciles;
- predicted-vs-observed вероятности на выбранных сделках;
- Pearson/Spearman expected-vs-actual;
- число конкурирующих кандидатов и преимущество победителя над вторым;
- ограничения price-only и `eligible_for_profitability_conclusion: false`.

Для копирования малых JSON каждого horizon:

```bash
REPORT_DIR=/home/foraset1/tradingbot-reports
for HORIZON in 15 30 60; do
  BUILD="$REPORT_DIR/backtest-v2-price-365d-h${HORIZON}-build.json"
  RESULT_PATH=$(jq -er '.experiment_path' "$BUILD") || exit 1
  docker compose run --rm --no-deps collector \
    sh -c "cat '$RESULT_PATH/report.json'" \
    > "$REPORT_DIR/backtest-v2-price-365d-h${HORIZON}-report.json"
  docker compose run --rm --no-deps collector \
    sh -c "cat '$RESULT_PATH/manifest.json'" \
    > "$REPORT_DIR/backtest-v2-price-365d-h${HORIZON}-manifest.json"
done
chown foraset1:foraset1 "$REPORT_DIR"/backtest-v2-price-365d-*.json
```

## Ограничения интерпретации

Даже положительный V2 не доказывает реальную доходность:

- price-only profile не содержит стакан, spread, funding history и open interest;
- conditional entry предполагает исполнение maker-ордера;
- `NO_FILL`, queue position и partial fills не моделируются;
- выбор лучшего кандидата создаёт зависимые наблюдения;
- после этой development-матрицы нужен новый untouched future/shadow holdout;
- затем нужны maker execution simulator и Demo/Shadow Mode.

Новый private API или live trading этим этапом не включается.
