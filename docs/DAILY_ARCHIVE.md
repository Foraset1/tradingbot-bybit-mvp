# Суточный архив и контроль диска

Этот этап хранит каждый завершённый UTC-день как отдельный неизменяемый Parquet-набор.
Collector при этом продолжает работать: архиватор выбирает только раздел указанной даты,
а активные `.jsonl.partial` текущего дня не читает.

На фактически измеренном потоке шести пар raw занимает около 2,69 GiB в сутки, а
канонический Parquet — около 0,28 GiB в сутки. Для VM с диском 100 GB целевой режим:

- raw JSONL: 7 последних полных дней;
- Parquet-архив: все принятые дни;
- резерв свободного места: 15 GiB;
- старые research/evaluation-наборы удаляются только вручную после проверки, потому что они
  тоже занимают место.

90 дней такого Parquet-архива занимают ориентировочно 25–26 GiB. Вместе с семью днями raw
это заметно меньше 100 GB, но запас нужно контролировать командой `df -h`.

## Что создаётся

```text
/data/archive/
├── catalog.json
├── audits/date=YYYY-MM-DD/audit-v2-<fingerprint>.json
├── days/date=YYYY-MM-DD/manifest.json
└── canonical/date=YYYY-MM-DD/canonical-v1-<fingerprint>/
    ├── manifest.json
    ├── source-audit.json
    └── market/.../*.parquet
```

`days/.../manifest.json` является commit marker дня. Retention доверяет дню только после
повторной проверки audit manifest, canonical manifest, размеров, строк и SHA-256 всех
Parquet-файлов.

Архивирование и пригодность конкретного временного окна для обучения теперь разделены.
Строгий audit по-прежнему фиксирует все разрывы. Если единственные предупреждения дня —
`kline_gap`, день сохраняется без синтетического восстановления данных и получает
`quality.status: gapped`. Ошибки схемы, отсутствующие потоки, `.partial`, недостаточная
длительность и любые другие warning-коды по-прежнему блокируют архив.

## Первый ручной запуск на Ubuntu

Обновить код и образ, не останавливая collector:

```bash
cd /opt/tradingbot
git fetch origin
git switch main
git pull --ff-only
sudo docker compose build collector
sudo docker compose up -d collector
sudo docker compose ps
```

Создать каталог отчётов в домашней папке пользователя:

```bash
REPORT_DIR=/home/foraset1/tradingbot-reports
sudo install -d -m 0750 -o foraset1 -g foraset1 "$REPORT_DIR"
```

Посмотреть доступные UTC-разделы на примере ticker BTC:

```bash
sudo docker compose exec collector sh -c \
  'find /data/raw/ticker/BTCUSDT -mindepth 3 -maxdepth 3 -type d | sort'
```

Архивировать один полный день. Вместо даты подставить реально завершённый UTC-день:

```bash
DAY=2026-07-28
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot archive-day --date "$DAY" \
  > "$REPORT_DIR/archive-day-$DAY.json"
sudo chown foraset1:foraset1 "$REPORT_DIR/archive-day-$DAY.json"
jq . "$REPORT_DIR/archive-day-$DAY.json"
```

Нормальный результат содержит `ok: true`, `raw_files > 0`, `canonical_files > 0`, пути
внутри `/data/archive`, блок `quality` и `reused: false`. `quality.status` может быть
`clean` или `gapped`; второй вариант означает, что V3 исключит небезопасные окна при
построении dataset. Повторная команда обязана вернуть те же fingerprint и `reused: true`.

При отказе команда также печатает JSON (`ok: false`, `error`, `archive_acceptance`) в
stdout и записывает тот же payload через `--output`. Поэтому даже неуспешный запуск можно
передать на аудит без пустого build-файла.

День, в который collector был впервые запущен не с 00:00 UTC, можно принять отдельно с
явным снижением порога покрытия:

```bash
FIRST_DAY=2026-07-27
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot archive-day \
  --date "$FIRST_DAY" \
  --minimum-duration-seconds 0 \
  > "$REPORT_DIR/archive-day-$FIRST_DAY.json"
sudo chown foraset1:foraset1 "$REPORT_DIR/archive-day-$FIRST_DAY.json"
```

Это допустимо только для известного неполного первого дня. Для всех следующих дней остаётся
настроенный порог 82 800 секунд и строгая проверка duplicate/path/schema. Kline gaps
сохраняются как явный признак качества, но не замалчиваются.

## Архивирование накопившихся полных дней

Дни запускаются последовательно. Ошибка одного дня не должна обходиться:

```bash
for DAY in 2026-07-28 2026-07-29 2026-07-30 2026-07-31 2026-08-01; do
  REPORT="$REPORT_DIR/archive-day-$DAY.json"
  sudo docker compose run --rm --no-deps collector \
    python -m tradingbot archive-day --date "$DAY" \
    > "$REPORT"
  STATUS=$?
  sudo chown foraset1:foraset1 "$REPORT"
  jq '{ok,partition_date,reused,quality,error,archive_acceptance}' "$REPORT"
  [ "$STATUS" -eq 0 ] || break
done
```

Список дат нужно скорректировать по фактическому календарю сервера. Текущий UTC-день
архиватор отклоняет специально.

## Dry-run retention

Следующая команда ничего не удаляет:

```bash
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot plan-retention \
  > "$REPORT_DIR/retention-plan.json"
sudo chown foraset1:foraset1 "$REPORT_DIR/retention-plan.json"
jq '{mode,delete_before_date,candidate_file_count,candidate_bytes,blocker_count,safe_to_apply,partial_file_count}' \
  "$REPORT_DIR/retention-plan.json"
```

Правильные свойства отчёта:

- `mode` равно `dry_run`;
- `deletion_performed` равно `false`;
- активные файлы текущего дня только перечислены и не являются кандидатами;
- старый raw-файл становится кандидатом только при точном совпадении размера и SHA-256 с
  принятым audit;
- перед этим повторно проверяются все Parquet-файлы соответствующего дня;
- любая поздняя запись, изменение raw или повреждение Parquet создаёт blocker.

Пока кандидатов нет (сбор идёт меньше семи дней), `safe_to_apply` будет `false`, но при
`blocker_count: 0` это нормальный результат: удалять просто нечего.

## Подтверждаемое применение retention

`apply-retention` является отдельной командой и не может запускаться одним флагом у
dry-run. Перед применением оператор должен сохранить и проверить полный JSON-план. Команда
требует этот файл, точный `plan_fingerprint` и новый путь для неизменяемого отчёта.

Перед первым удалением команда под archive lock повторяет полную проверку:

- схема, внутренние счётчики и fingerprint сохранённого плана;
- совпадение настроенных `/data/raw` и `/data/archive` с планом;
- неизменность `catalog_fingerprint`;
- manifest и SHA-256 всех Parquet-файлов затрагиваемых дней;
- точное совпадение текущего списка старых raw-файлов со списком кандидатов;
- размер и SHA-256 каждого raw-файла;
- отсутствие blocker, path traversal, symlink и уже существующего receipt.

После общего preflight каждый файл ещё раз проверяется непосредственно перед `unlink`.
Последние семь дней, новые файлы и `.jsonl.partial` не входят в список и не удаляются.
Receipt атомарно создаётся до первого удаления, обновляется после каждого полностью
обработанного UTC-дня и содержит полный список фактически удалённых файлов.

Запускать применение нужно в `tmux`. План и receipt находятся в домашнем каталоге, поэтому
он монтируется в одноразовый контейнер отдельно от `/data`:

```bash
cd /opt/tradingbot
REPORT_DIR=/home/foraset1/tradingbot-reports
PLAN="$REPORT_DIR/retention-plan-20260822-181700.json"
PLAN_NAME=$(basename "$PLAN")
FINGERPRINT=$(jq -er '.plan_fingerprint' "$PLAN") || exit 1
STAMP=$(date -u +%Y%m%d-%H%M%S)
RECEIPT="$REPORT_DIR/retention-apply-$STAMP.json"
SUMMARY="$REPORT_DIR/retention-apply-$STAMP-summary.json"
LOG="$REPORT_DIR/retention-apply-$STAMP.log"

sudo docker compose run --rm --no-deps \
  --volume "$REPORT_DIR:/reports" \
  collector \
  python -m tradingbot apply-retention \
  --plan "/reports/$PLAN_NAME" \
  --confirm-plan-fingerprint "$FINGERPRINT" \
  --output "/reports/$(basename "$RECEIPT")" \
  > "$SUMMARY" 2> "$LOG"

STATUS=$?
sudo chown foraset1:foraset1 "$RECEIPT" "$SUMMARY" "$LOG"
echo "exit=$STATUS"
jq '{ok,status,deletion_performed,planned_file_count,deleted_file_count,planned_bytes,deleted_bytes,error,failed_path,receipt_fingerprint}' \
  "$RECEIPT"
sudo docker compose ps
sudo docker compose exec collector sh -c \
  'du -sh /data/raw /data/archive /data/history 2>/dev/null; df -h /data'
```

Нормальное завершение имеет `exit=0`, `ok=true`, `status=complete`, одинаковые planned и
deleted counts/bytes и непустой `receipt_fingerprint`. Stdout намеренно содержит только
компактную сводку; полный список находится в receipt.

Если каталог, архив или raw изменились после dry-run, команда завершается до удаления и
нужно построить новый план. Если процесс был аварийно остановлен уже после части удалений,
не следует повторно использовать тот же receipt: сохраните его, постройте новый dry-run по
оставшимся файлам и сначала проведите отдельный аудит результата. Автоматический timer для
destructive apply пока намеренно не настраивается.

## Построение общего research dataset

Каталог позволяет не объединять дневные Parquet вручную. Когда накопится нужная история:

```bash
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot build-research \
  --catalog /data/archive/catalog.json \
  --output-root /data/research \
  > "$REPORT_DIR/research-build-result-catalog.json"
sudo chown foraset1:foraset1 "$REPORT_DIR/research-build-result-catalog.json"
```

Обычный `build-research` полностью проверяет каждый дневной dataset, требует
последовательные UTC-даты с одинаковым набором символов и намеренно отклоняет каталог с
`quality.status: gapped`. Этот старый builder не умеет доказать непрерывность каждого
feature/label-окна.

Тот же каталог является источником execution-aware V3, но эта команда использует только
live orderbook/trades, а не официальный price-only history catalog:

```bash
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot build-execution-research \
  --catalog /data/archive/catalog.json \
  --output-root /data/execution-research \
  > "$REPORT_DIR/execution-v3-build-result.json" \
  2> "$REPORT_DIR/execution-v3.log"
sudo chown foraset1:foraset1 \
  "$REPORT_DIR/execution-v3-build-result.json" "$REPORT_DIR/execution-v3.log"
```

V3 обрабатывает каталог посуточно с соседними UTC-разделами и подробно описан в
[`EXECUTION_RESEARCH.md`](EXECUTION_RESEARCH.md).

Именно V3 разрешено передавать каталог с gapped-днями. Он не заполняет пропуски, а удаляет
только решения и labels, чьи feature, maker-entry или post-fill окна пересекают разрыв,
смену WebSocket-сессии либо отсутствующую минутную свечу. Список таких исходных дней
записывается в `source.gapped_partition_dates` manifest.
