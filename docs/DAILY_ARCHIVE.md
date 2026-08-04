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

Нормальный результат содержит `raw_files > 0`, `canonical_files > 0`, пути внутри
`/data/archive` и `reused: false`. Повторная команда обязана вернуть те же fingerprint и
`reused: true`.

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
настроенный порог 82 800 секунд и строгая проверка gap/duplicate/path/schema.

## Архивирование накопившихся полных дней

Дни запускаются последовательно. Ошибка одного дня не должна обходиться:

```bash
for DAY in 2026-07-28 2026-07-29 2026-07-30 2026-07-31 2026-08-01; do
  sudo docker compose run --rm --no-deps collector \
    python -m tradingbot archive-day --date "$DAY" \
    > "$REPORT_DIR/archive-day-$DAY.json" || break
  sudo chown foraset1:foraset1 "$REPORT_DIR/archive-day-$DAY.json"
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

В версии 0.5.0 намеренно отсутствует флаг `--apply`. Сначала нужны 2–3 успешных суточных
архива и сохранённые dry-run отчёты. После их аудита добавляется отдельное подтверждаемое
удаление и systemd timer; до этого raw остаётся нетронутым.

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

Research builder полностью проверяет каждый дневной dataset и принимает только
последовательные UTC-даты с одинаковым набором символов. Для окончательной оценки модели
по-прежнему требуется не менее 90 дней; до этого результат остаётся техническим smoke-test.
