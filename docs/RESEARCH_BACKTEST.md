# Offline evaluation и conditional-entry backtest

Этот этап проверяет, содержат ли причинные рыночные признаки полезный сигнал на горизонте
60 минут. Он остаётся полностью read-only: Bybit API key не нужен, ордера не создаются,
токены OpenAI не расходуются.

## Что запускается

Одна команда выполняет воспроизводимый конвейер:

1. проверяет manifest, SHA-256 и Parquet-схемы research dataset;
2. оставляет разрешённые 60-минутные labels и связывает их с features по `decision_id`;
3. строит expanding walk-forward folds без перемешивания времени;
4. удаляет из train labels, пересекающие границу test, и добавляет embargo 60 минут;
5. сравнивает class-prior baseline, logistic regression и LightGBM;
6. выбирает максимум одну пару/сторону на каждую минуту;
7. симулирует максимум одну позицию с комиссиями, slippage, funding и 24-часовым
   ограничением потерь;
8. атомарно записывает модели, сделки, отчёт и проверяемый manifest.

Все параметры входят в fingerprint эксперимента. Повторный запуск с теми же данными,
конфигурацией и версиями библиотек проверяет существующие файлы и возвращает
`reused: true`.

## Запуск

Локально:

```bash
tradingbot run-backtest \
  --research-dataset data/research/research-v1-<fingerprint> \
  --output-root data/evaluations
```

На сервере из Docker volume:

```bash
cd /opt/tradingbot

REPORT_DIR=/home/foraset1/tradingbot-reports
sudo install -d -m 0750 -o foraset1 -g foraset1 "$REPORT_DIR"
umask 027

RESEARCH_DATASET=$(
  sudo docker compose run --rm --no-deps \
    --entrypoint python collector \
    -c "from pathlib import Path; paths=list(Path('/data/research').glob('research-v1-*')); assert paths, 'research dataset not found'; print(max(paths, key=lambda p: p.stat().st_mtime_ns))"
)
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot run-backtest \
  --research-dataset "$RESEARCH_DATASET" \
  --output-root /data/evaluations \
  > "$REPORT_DIR/backtest-build-result.json"

jq . "$REPORT_DIR/backtest-build-result.json"
sudo chown foraset1:foraset1 "$REPORT_DIR/backtest-build-result.json"
```

Для результата `build-price-research` путь лучше брать прямо из сохранённого JSON, не
угадывая fingerprint:

```bash
PRICE_RESULT=/home/foraset1/tradingbot-reports/price-research-90d-2026-08-07.json
RESEARCH_DATASET=$(jq -er '.dataset_path' "$PRICE_RESULT")

sudo docker compose run --rm --no-deps collector \
  python -m tradingbot run-backtest \
  --research-dataset "$RESEARCH_DATASET" \
  --output-root /data/evaluations \
  > /home/foraset1/tradingbot-reports/backtest-price-90d-2026-08-07.json
```

Для первого технического smoke-test можно использовать уже созданный snapshot. Для нового
72-часового snapshot сначала нужно повторить strict audit, `build-dataset` и
`build-research`; нельзя дописывать данные в уже созданный versioned dataset.

## Результаты

Команда создаёт каталог вида:

```text
/data/evaluations/backtest-v1-<input-fingerprint>/
├── manifest.json
├── report.json
├── models/
│   └── fold-01-lightgbm.txt
└── trades/
    ├── class_prior.parquet
    ├── lightgbm.parquet
    └── logistic.parquet
```

`report.json` содержит:

- границы train/test каждого fold;
- log loss, multiclass Brier score, accuracy и calibration error;
- cost-aware результат каждой модели, drawdown, profit factor и причины пропуска входов;
- число исключённых `AMBIGUOUS` и непросчитанных labels;
- версии Python, NumPy, PyArrow, scikit-learn и LightGBM;
- data gate и явные ограничения симуляции.

Чтобы скопировать небольшие JSON-файлы на host после выполнения команды:

```bash
REPORT_DIR=/home/foraset1/tradingbot-reports
RESULT_PATH=$(jq -r '.experiment_path' "$REPORT_DIR/backtest-build-result.json")
sudo docker compose run --rm --no-deps collector \
  sh -c "cat '$RESULT_PATH/report.json'" \
  > "$REPORT_DIR/backtest-report.json"
sudo docker compose run --rm --no-deps collector \
  sh -c "cat '$RESULT_PATH/manifest.json'" \
  > "$REPORT_DIR/backtest-manifest.json"
sudo chown foraset1:foraset1 \
  "$REPORT_DIR/backtest-report.json" \
  "$REPORT_DIR/backtest-manifest.json"
```

## Как трактовать короткую историю

При истории короче 44 дней (30 дней train + 14 дней test) создаётся один явный
`technical_smoke` split 70/30. Он проверяет только работоспособность конвейера, отсутствие
утечки между train/test и формат результатов.

Для содержательного первого обзора модели конфигурация требует минимум 90 дней и не менее
трёх walk-forward folds. Даже после этого `eligible_for_profitability_conclusion` остаётся
`false`, пока отдельный simulator не смоделирует вероятность maker fill, положение в очереди
и partial fills.

Иными словами:

- 72 часа — запускать сейчас для проверки техники;
- 44+ дня — появляются обычные walk-forward окна;
- 90+ дней — можно впервые сравнивать устойчивость рыночного сигнала;
- вывод о реальной доходности — только после execution simulator и paper/testnet.

## Зафиксированные допущения

Значения по умолчанию находятся в `[evaluation]` файла `config/tradingbot.toml`:

- maker fee: 2 bps на вход и maker TP;
- taker fee: 5,5 bps для stop/timeout;
- adverse selection входа: 1 bp;
- дополнительное stop slippage: 3 bps;
- дополнительное timeout slippage: 1 bp;
- вход разрешён при ожидаемом net return не ниже 1 bp;
- LightGBM использует 4 потока, оставляя ресурсы collector на VM с 6 vCPU.

Перед финансовой интерпретацией комиссии нужно заменить на фактический тариф аккаунта.
Текущий backtest условно предполагает, что выбранный maker-вход исполнился. Он не выдаёт
`NO_FILL`, не оценивает queue position и не заявляет реальную maker fill rate.

Для `price_futures_research_v1` дополнительно отсутствует история funding: соответствующие
значения остаются неизвестными (`NaN`) и не включаются в модель, а funding cost в таком
предварительном backtest равен нулю. Поэтому даже 90-дневный положительный результат этого
профиля нельзя трактовать как оценку чистой реальной доходности.

## Что прислать для проверки

Достаточно трёх небольших файлов:

- `/home/foraset1/tradingbot-reports/backtest-build-result.json`;
- `/home/foraset1/tradingbot-reports/backtest-report.json`;
- `/home/foraset1/tradingbot-reports/backtest-manifest.json`.

Parquet со сделками и файлы моделей передавать не нужно, пока JSON-аудит не выявит проблему.
