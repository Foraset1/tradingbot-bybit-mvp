# Постоянный collector на Ubuntu Server 24.04

Эта инструкция разворачивает текущую read-only версию: публичные WebSocket-потоки Bybit,
нормализацию, JSONL-хранилище, health telemetry и аудит данных. API-ключи, отправка ордеров и
рабочая торговая модель в эту версию ещё не входят.

Профиль постоянного MVP: Ubuntu Server 24.04 LTS x86_64, 6 vCPU, 10 GB RAM, 100 GB NVMe,
статический IPv4 и не менее 100 Mbit/s. Collector получает лимит 4 CPU и 6 GB RAM, оставляя
ресурсы Ubuntu, Docker, health-проверкам и будущим сервисам.

Значения вынесены в `.env`, поэтому тот же Compose-файл можно использовать на более слабой
или более мощной VM. Виртуальный диск лучше считать расширяемым: CPU и RAM уменьшаются
настройками VM и `.env`, а существующий диск безопаснее не сжимать. Если позднее потребуется
меньший диск, создаётся новая VM и переносятся только необходимые данные.

При наблюдаемом объёме около 2,71 GB/сутки диск 100 GB не подходит для бессрочного хранения
JSONL. Целевое локальное окно raw — 7 дней с суточным Parquet-архивированием; disk guard
сохраняет не менее 15 GiB свободного места. Версия 0.5.0 формирует проверяемый dry-run
retention, но сама старые данные ещё не удаляет. Порядок запуска описан в
[`DAILY_ARCHIVE.md`](DAILY_ARCHIVE.md).

## 1. Войти и проверить сервер

Подключиться из локального терминала:

```bash
ssh <user>@<server-ip>
```

Проверить параметры:

```bash
lsb_release -ds
uname -m
nproc
free -h
df -h / /var/lib/docker 2>/dev/null || df -h /
```

Ожидаются Ubuntu 24.04, архитектура `x86_64`, 6 CPU, около 10 GB RAM и 100 GB
NVMe-диска.

## 2. Обновить ОС и установить базовые пакеты

```bash
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt full-upgrade -y
sudo apt install -y ca-certificates curl git jq chrony ufw unattended-upgrades
sudo systemctl enable --now chrony
sudo timedatectl set-timezone UTC
sudo chronyc makestep
```

Проверить часы:

```bash
timedatectl status
chronyc -N tracking
chronyc -N sources
```

Перед сбором обучающих данных обязательны `System clock synchronized: yes`, активный NTP и
небольшой системный offset. Если сервер фильтрует исходящий трафик, для Ubuntu NTS нужны
`4460/tcp` и `123/udp`.

## 3. Настроить firewall

Сначала узнать публичный IP компьютера, с которого будет выполняться администрирование. Не
закрывать текущую SSH-сессию, пока вход во второй сессии не будет проверен.

Для постоянного IP администратора:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from <admin-public-ip> to any port 22 proto tcp
sudo ufw enable
sudo ufw status verbose
```

Если SSH использует другой порт, заменить `22`. Если IP администратора динамический, на первом
этапе можно выполнить `sudo ufw allow OpenSSH`, а затем сузить правило. Collector не публикует
никаких входящих портов.

## 4. Установить Docker Engine и Compose plugin

На чистом сервере добавить официальный Docker apt-репозиторий:

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
  docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
sudo docker compose version
```

Команды ниже намеренно используют `sudo docker`: членство в группе `docker` фактически даёт
пользователю административные полномочия на хосте.

## 5. Передать проект

Предпочтительный вариант — приватный Git-репозиторий:

```bash
sudo install -d -o "$USER" -g "$USER" /opt/tradingbot
git clone <repository-url> /opt/tradingbot
cd /opt/tradingbot
```

Если Git-репозитория пока нет, создать архив на Windows в PowerShell:

```powershell
tar.exe -czf tradingbot-mvp.tar.gz -C D:\Projects\TradingBot `
  Dockerfile compose.yaml pyproject.toml README.md config docs src
scp .\tradingbot-mvp.tar.gz <user>@<server-ip>:/tmp/
```

Затем на сервере:

```bash
sudo install -d -o "$USER" -g "$USER" /opt/tradingbot
tar -xzf /tmp/tradingbot-mvp.tar.gz -C /opt/tradingbot
cd /opt/tradingbot
```

Убедиться, что присутствуют `Dockerfile`, `compose.yaml`, `config/tradingbot.toml` и `src/`.

## 6. Собрать и проверить конфигурацию

```bash
cd /opt/tradingbot
cp .env.example .env
docker compose config --environment
sudo docker compose config --quiet
sudo docker compose build --pull
sudo docker compose run --rm collector python -m tradingbot validate-config
sudo docker compose run --rm collector python -m tradingbot show-topics
```

`show-topics` должен показать 36 тем: шесть пар и по шесть потоков на пару.

Основной профиль в `.env`:

```dotenv
TRADINGBOT_COLLECTOR_CPUS=4.0
TRADINGBOT_COLLECTOR_MEMORY=6g
TRADINGBOT_MIN_FREE_BYTES=16106127360
```

Для другой VM меняются только эти значения. Например, для узла 4 vCPU / 8 GB:

```dotenv
TRADINGBOT_COLLECTOR_CPUS=2.5
TRADINGBOT_COLLECTOR_MEMORY=4g
TRADINGBOT_MIN_FREE_BYTES=10737418240
```

После изменения обязательно выполнить `sudo docker compose config --quiet` и
`sudo docker compose up -d`.

## 7. Выполнить 90-секундный smoke-test

```bash
sudo docker compose run --rm \
  -e TRADINGBOT_DATA_ROOT=/data/soak/smoke \
  -e TRADINGBOT_HEALTH_PATH=/app/runtime/smoke-health.json \
  collector python -m tradingbot collect --run-seconds 90
```

Проверить финальный health:

```bash
sudo docker compose run --rm --entrypoint python collector -c \
  "import json; h=json.load(open('/app/runtime/smoke-health.json')); \
assert h['status']=='stopped'; assert h['subscription_confirmed']; \
assert not h['current_connection_missing_topics']; assert not h['fatal_error']; \
assert h['queue_size']==0; print(json.dumps(h,indent=2))"
```

Должны выполняться условия:

- `status = stopped`;
- `subscription_confirmed = true`;
- `current_connection_missing_topics = []`;
- `fatal_error = false`;
- `queue_size = 0`;
- `queue_full_events = 0`.

За 90 секунд свечи 5m/15m могут ещё не закрыться, поэтому полный строгий dataset audit на этом
этапе не ожидается.

## 8. Запустить постоянный collector

```bash
cd /opt/tradingbot
sudo docker compose up -d
sudo docker compose ps
sudo docker compose logs --tail=100 collector
```

Через 90 секунд `docker compose ps` должен показывать контейнер как `healthy`.

Текущий health:

```bash
sudo docker compose exec collector python -c \
  "import json; print(json.dumps(json.load(open('/app/runtime/collector-health.json')),indent=2))"
```

Текущий объём данных:

```bash
sudo docker compose exec collector du -sh /data/raw
sudo docker system df
```

Логи в реальном времени:

```bash
sudo docker compose logs -f --tail=100 collector
```

Выход из просмотра логов через `Ctrl+C` не останавливает контейнер.

## 9. Провести 24-часовую проверку

После непрерывной работы не менее 24 часов корректно остановить collector, чтобы завершить
активные `.partial`-сегменты:

```bash
cd /opt/tradingbot
sudo docker compose stop -t 30 collector
sudo docker compose run --rm collector python -m tradingbot audit-data \
  --root /data/raw \
  --strict \
  --minimum-duration-seconds 82800 \
  --output /app/runtime/24-hour-audit.json
```

Audit выводит прогресс примерно раз в 10 секунд. Полная проверка нескольких гигабайт JSONL
может занять десятки минут; отсутствие итогового JSON до завершения не означает зависание.
`Ctrl+C` прерывает аудит и даёт код `130`.

Команда должна завершиться с кодом `0`, после чего collector запускается снова:

```bash
sudo docker compose up -d
```

Если audit завершился с кодом `1`, не удалять данные. Сохранить отчёт и проверить причины:

```bash
sudo docker compose run --rm --entrypoint python collector -c \
  "print(open('/app/runtime/24-hour-audit.json').read())"
```

После успешного strict audit канонический Parquet строится из зафиксированного snapshot:

```bash
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot build-dataset \
  --audit-report /app/runtime/24-hour-audit.json \
  --root /data/raw \
  --output-root /data/datasets
```

Команда повторно проверяет SHA-256 всех входов, показывает прогресс и печатает короткий JSON
с dataset ID и fingerprint. Полный контракт описан в [`DATASET.md`](DATASET.md).

После проверки `canonical-manifest.json` causal research-слой строится из неизменяемого
каталога dataset. Для текущего принятого snapshot:

```bash
REPORT_DIR=/home/foraset1/tradingbot-reports
sudo install -d -m 0750 -o foraset1 -g foraset1 "$REPORT_DIR"
umask 027

sudo docker compose run --rm --no-deps collector \
  python -m tradingbot build-research \
  --dataset /data/datasets/canonical-v1-dfd2a620552d79b9 \
  --output-root /data/research \
  > "$REPORT_DIR/research-build-result.json"
sudo chown foraset1:foraset1 "$REPORT_DIR/research-build-result.json"
```

Collector при этом можно не останавливать: команда читает только уже зафиксированный
канонический dataset и пишет в отдельный каталог. Результат содержит новый dataset ID,
fingerprint, число feature/label строк и файлов. Повторный запуск проверяет все SHA-256 и
возвращает `reused=true`. Контракт описан в
[`FEATURES_AND_LABELS.md`](FEATURES_AND_LABELS.md).

Offline baseline/LightGBM/backtest запускается из конкретного неизменяемого research dataset:

```bash
REPORT_DIR=/home/foraset1/tradingbot-reports
RESEARCH_DATASET=$(jq -r '.dataset_path' "$REPORT_DIR/research-build-result.json")
sudo docker compose run --rm --no-deps collector \
  python -m tradingbot run-backtest \
  --research-dataset "$RESEARCH_DATASET" \
  --output-root /data/evaluations \
  > "$REPORT_DIR/backtest-build-result.json"

RESULT_PATH=$(jq -r '.experiment_path' "$REPORT_DIR/backtest-build-result.json")
sudo docker compose run --rm --no-deps collector \
  sh -c "cat '$RESULT_PATH/report.json'" \
  > "$REPORT_DIR/backtest-report.json"
sudo docker compose run --rm --no-deps collector \
  sh -c "cat '$RESULT_PATH/manifest.json'" \
  > "$REPORT_DIR/backtest-manifest.json"
sudo chown foraset1:foraset1 "$REPORT_DIR"/*.json
```

На текущих 72 часах ожидается `data_mode: technical_smoke`: это проверка всего конвейера,
а не качества или доходности модели. Обычные временные окна требуют минимум 44 дней;
первый model review — минимум 90 дней и три folds. Подробный контракт находится в
[`RESEARCH_BACKTEST.md`](RESEARCH_BACKTEST.md).

## 10. Остановка, обновление и важное предупреждение

Штатная остановка:

```bash
sudo docker compose down --timeout 30
```

Не использовать `docker compose down -v`: параметр `-v` удалит volumes с рыночными данными и
health-файлами.

Обновление из Git:

```bash
cd /opt/tradingbot
git pull --ff-only
sudo docker compose build --pull
sudo docker compose up -d
sudo docker compose ps
```

После перезагрузки сервера контейнер поднимется автоматически благодаря
`restart: unless-stopped`.

## Критерий готовности research dataset

К построению признаков и labels переходим только когда:

- `chrony` синхронизирован;
- контейнер `healthy`;
- все 36 тем наблюдались;
- нет queue overflow и постоянных reconnect;
- 24-часовой строгий audit вернул код `0`;
- прогноз дискового объёма приемлем для выбранного retention;
- суточный архив и dry-run retention проверены по [`DAILY_ARCHIVE.md`](DAILY_ARCHIVE.md);
- канонический Parquet dataset построен из неизменного audit manifest;
- causal research dataset построен из принятого canonical manifest;
- в manifest нет нулевого числа feature/label строк и проверены причины пропусков.
- technical offline evaluation завершилась, а manifest/report прошли повторную проверку.
