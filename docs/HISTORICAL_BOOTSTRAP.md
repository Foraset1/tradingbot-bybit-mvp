# Bootstrap официальной истории Bybit

Версия 0.6.0 добавляет отдельный профиль `price_futures_v1` для бесплатной официальной
истории Bybit. Он дополняет, но не заменяет live-сборщик.

## Что импортируется

Источник для каждой пары и UTC-даты:

```text
https://public.bybit.com/trading/<SYMBOL>/<SYMBOL><YYYY-MM-DD>.csv.gz
```

Импортёр потоково читает публичные сделки и сохраняет:

- `trade_bar_1s` — OHLC, объём, turnover, buy/sell volume и число сделок за секунду;
- `trade_bar_1m` — те же поля за минуту;
- SHA-256 скачанного gzip, HTTP ETag/Last-Modified и исходное число сделок;
- SHA-256, размер и число строк каждого Parquet-файла;
- отдельные symbol/day manifests и общий `catalog.json`.

Исходный gzip и отдельные тики после агрегации не сохраняются. Это принципиально для VM
с диском 100 GB: 2 августа 2026 года шесть исходных архивов занимали около 89,6 MB и
содержали 2 572 162 сделки, а итоговые 1s/1m Parquet — около 10,1 MB.

Это только наблюдение одного дня, не гарантия постоянного коэффициента сжатия. Перед
загрузкой года сначала измеряется 7–30 дней на самом сервере.

## Чего в этом профиле нет

В официальном trade archive отсутствуют:

- исторический стакан L2 и позиция maker-ордера в очереди;
- ticker snapshots;
- локальный момент получения события;
- funding и open interest;
- точный порядок событий внутри одной секунды после агрегации.

Эти поля не заполняются нулями и не синтезируются. Для исторических баров
`available_at_ns` равен окончанию бара плюс настроенный консервативный лаг (по умолчанию
1 секунда), а manifest содержит
`timestamp_basis: bar_end_plus_assumed_latency`.

Обычный `build-research` по-прежнему ожидает полный microstructure-профиль. Для
`/data/history/catalog.json` используется отдельная команда `build-price-research`: она
не подменяет отсутствующие поля нулями и создаёт профиль `price_futures_research_v1`.
Импорт и построение research dataset можно безопасно выполнять параллельно с live-сбором.

## Защита данных

- импорт разрешён только для завершённых UTC-дней и официального HTTPS-host
  `public.bybit.com`;
- день становится видимым как `day=YYYY-MM-DD` только после завершения всех выбранных пар;
- незавершённый день остаётся в `.staging` и продолжается при повторном запуске;
- повторный запуск не доверяет имени каталога, а сверяет все Parquet SHA-256 и row counts;
- исходный CSV обязан иметь неубывающие timestamps, правильный symbol, положительные
  price/size и уникальные соседние trade IDs;
- синтетические бары запрещены;
- минуты без сделок остаются явными пропусками и не превращаются в синтетические свечи;
- архив отклоняется, если один непрерывный период без сделок превышает 5 минут; общее
  количество разрозненных пустых минут сохраняется как метрика качества;
- disk guard сохраняет минимум 15 GiB свободного места;
- глобальный lock не даёт запустить два импортёра одновременно.

Глобальная дедупликация всех trade IDs намеренно не выполняется: временная SQLite-копия
миллионов UUID увеличила бы расход диска и время импорта. Это ограничение записано в каждом
manifest. Последовательные повторы, нарушения времени и gzip CRC всё равно отклоняются.

## Обновление сервера

После слияния ветки в `main`:

```bash
cd /opt/tradingbot
git fetch origin
git switch main
git pull --ff-only
git rev-parse --short HEAD

sudo docker compose build collector
sudo docker compose up -d collector
sudo docker compose ps
sudo docker compose exec collector python -m tradingbot validate-config \
  | jq '.history'
```

Ожидаются:

```json
{
  "assumed_latency_ms": 1000,
  "maximum_consecutive_trade_free_minutes": 5,
  "profile": "price_futures_v1",
  "public_base_url": "https://public.bybit.com/trading",
  "retains_individual_trades": false,
  "root": "/data/history"
}
```

Collector должен остаться `healthy`.

В `config/tradingbot.toml` для совместимости manifest-схемы v1 этот предел пока хранится под
историческим ключом `maximum_missing_minutes`. Его фактическая семантика — максимальное число
**последовательных** минут без публичных сделок. Значение не ограничивает общую сумму отдельных
пустых минут за сутки.

## Первый реальный день

Берём позавчера: так меньше риск, что биржа ещё не опубликовала вчерашний файл. Все отчёты
сохраняются в домашней папке пользователя `foraset1`, не в `/root`.

```bash
set +e
cd /opt/tradingbot

REPORT_DIR=/home/foraset1/tradingbot-reports
DAY=$(date -u -d '2 days ago' +%F)
REPORT="$REPORT_DIR/history-import-$DAY.json"
LOG="$REPORT_DIR/history-import-$DAY.log"

sudo install -d -m 0750 -o foraset1 -g foraset1 "$REPORT_DIR"

sudo docker compose run --rm --no-deps collector \
  python -m tradingbot import-history \
  --from-date "$DAY" \
  --to-date "$DAY" \
  > "$REPORT" 2> "$LOG"
STATUS=$?

sudo chown foraset1:foraset1 "$REPORT" "$LOG"
echo "exit=$STATUS"
jq . "$REPORT"
tail -n 40 "$LOG"
docker compose ps
```

Успешный результат имеет `days: 1`, `imported_days: 1`, `output_rows > 0` и
`catalog_path: /data/history/catalog.json`. Создание временного Compose-контейнера попадает
в `.log`, а stdout остаётся чистым JSON.

Повторить ту же команду. Второй результат должен иметь `reused_days: 1` и не скачивать
архив повторно — это одновременно глубокая проверка сохранённых файлов.

## Пилот на 7 дней

После успешного дня:

```bash
set +e
cd /opt/tradingbot

REPORT_DIR=/home/foraset1/tradingbot-reports
END_DATE=$(date -u -d '2 days ago' +%F)
START_DATE=$(date -u -d "$END_DATE - 6 days" +%F)
REPORT="$REPORT_DIR/history-import-7d-$END_DATE.json"
LOG="$REPORT_DIR/history-import-7d-$END_DATE.log"

sudo docker compose run --rm --no-deps collector \
  python -m tradingbot import-history \
  --from-date "$START_DATE" \
  --to-date "$END_DATE" \
  > "$REPORT" 2> "$LOG"
STATUS=$?

sudo chown foraset1:foraset1 "$REPORT" "$LOG"
echo "exit=$STATUS"
jq . "$REPORT"
tail -n 40 "$LOG"
```

Команда коммитит и каталогизирует каждый день отдельно. Если сеть оборвётся на пятом дне,
первые четыре останутся готовыми; повтор той же команды проверит/переиспользует их и
продолжит незавершённый день.

Для длительного запуска рекомендуется `tmux`, чтобы закрытие SSH не остановило процесс:

```bash
sudo apt update
sudo apt install -y tmux
sudo tmux new -s bybit-history
```

Внутри `tmux` выполнить команду импорта. Отключиться: `Ctrl-b`, затем `d`; вернуться:

```bash
sudo tmux attach -t bybit-history
```

## Проверка размера и каталога

```bash
cd /opt/tradingbot

sudo docker compose run --rm --no-deps collector \
  sh -c 'du -sh /data/history; df -h /data'

sudo docker compose run --rm --no-deps collector \
  sh -c 'cat /data/history/catalog.json' \
  > /home/foraset1/tradingbot-reports/history-catalog.json

sudo chown foraset1:foraset1 \
  /home/foraset1/tradingbot-reports/history-catalog.json
jq '{entry_count, catalog_fingerprint}' \
  /home/foraset1/tradingbot-reports/history-catalog.json
```

После 7 дней расширяем окно до 30 дней и снова измеряем `du -sh`. Год или два года не
запускаются вслепую: live raw/archive продолжают расти, а версия 0.6.0 пока только строит
dry-run плана удаления старого raw. При приближении к 15 GiB свободного места импорт
остановится безопасно.

## Построение price-only research dataset

После успешного импорта выбранного диапазона запускается отдельный причинный builder.
Диапазон указывается явно, поэтому последующее добавление дней в `catalog.json` не меняет
уже созданный dataset:

```bash
set +e
cd /opt/tradingbot

REPORT_DIR=/home/foraset1/tradingbot-reports
REPORT="$REPORT_DIR/price-research-90d-2026-08-07.json"
LOG="$REPORT_DIR/price-research-90d-2026-08-07.log"

sudo docker compose run --rm --no-deps collector \
  python -m tradingbot build-price-research \
  --catalog /data/history/catalog.json \
  --from-date 2026-05-10 \
  --to-date 2026-08-07 \
  --output-root /data/research \
  > "$REPORT" 2> "$LOG"

STATUS=$?
sudo chown foraset1:foraset1 "$REPORT" "$LOG"
echo "exit=$STATUS"
jq . "$REPORT"
grep -Ei 'warning|error|traceback' "$LOG" || true
```

Builder повторно проверяет SHA-256 каждого выбранного history-файла, использует только
бары с `available_at_ns <= decision_at_ns`, строит признаки по 1m/1s bars и labels
5/15/30/60 минут. Одновременное пересечение TP и SL внутри одной секундной свечи получает
`AMBIGUOUS`, потому что порядок исходных сделок после агрегации неизвестен.

Результат можно передать существующему `run-backtest`. В отчёте профиль будет явно указан
как price-only; funding, стакан, spread, maker queue и partial fills останутся недоступными.

Следующие шаги после первого 90-дневного backtest:

1. проверить walk-forward folds и стабильность результата отдельно по каждой паре;
2. при достаточном свободном месте расширить price history до 12–24 месяцев;
3. сравнить price-only baseline с моделью на собственном live L2;
4. отдельно построить maker execution simulator и затем перейти к Demo/Shadow Mode.

Даже секундные bars не моделируют maker queue. Переход к Demo/Shadow Mode и отдельный
execution simulator остаются обязательными.

## Зафиксированное расширение V2 до 365 дней

90-дневный price-only backtest V1 завершён и отклонён: после затрат обе модели показали
отрицательный результат, а predicted EV не калибровался к фактическому исходу. Этот период
теперь считается development-историей.

Следующий immutable диапазон — `2025-08-08`–`2026-08-07` включительно. Команды импорта,
построения research dataset и последовательного запуска горизонтов 15/30/60 минут находятся
в [`RESEARCH_BACKTEST.md`](RESEARCH_BACKTEST.md). Импорт переиспользует уже существующие дни;
перед стартом требуется минимум 30 GiB свободного места. Все отчёты пишутся только в
`/home/foraset1/tradingbot-reports`, а тяжёлые операции запускаются последовательно в `tmux`.
