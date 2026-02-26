# Beget S3: Интеграция и использование

## Что такое Beget S3

Beget предоставляет S3-совместимое объектное хранилище для файлов.

**Ваш bucket:** `443c60e2203e-betacassiopeiae`
**Endpoint:** `s3.ru1.storage.beget.cloud`

## Для чего использовать S3

✅ **Backup'ы SQLite базы данных**
✅ **ML модели (PyTorch, ONNX)**
✅ **Исторические данные**
✅ **Логи**
✅ **Архивы**

❌ **НЕ для SQL запросов** (это не БД!)

## Настройка доступа

### 1. Получить ключи доступа

Зайти в панель Beget: https://cp.beget.com
→ Хранилище S3
→ Скопировать Access Key и Secret Key

### 2. Установить boto3

```bash
pip install boto3
```

### 3. Настроить переменные окружения

```bash
export BEGET_S3_ACCESS_KEY="ваш_access_key"
export BEGET_S3_SECRET_KEY="ваш_secret_key"
export BEGET_S3_BUCKET="443c60e2203e-betacassiopeiae"
```

## Использование

### Backup БД в S3

```bash
python3 scripts/backup_to_beget_s3.py \
  --db /var/lib/trading-bot/trading_bot.db \
  --bucket 443c60e2203e-betacassiopeiae \
  --access-key $BEGET_S3_ACCESS_KEY \
  --secret-key $BEGET_S3_SECRET_KEY
```

### Python код для работы с S3

```python
import boto3

# Подключение к Beget S3
s3 = boto3.client(
    's3',
    endpoint_url='https://s3.ru1.storage.beget.cloud',
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY'
)

# Загрузить файл
s3.upload_file(
    'local_file.db',
    '443c60e2203e-betacassiopeiae',
    's3_path/file.db'
)

# Скачать файл
s3.download_file(
    '443c60e2203e-betacassiopeiae',
    's3_path/file.db',
    'local_file.db'
)

# Список файлов
response = s3.list_objects_v2(
    Bucket='443c60e2203e-betacassiopeiae',
    Prefix='backups/'
)
for obj in response.get('Contents', []):
    print(obj['Key'])
```

## Конфигурация в config.yaml

```yaml
storage:
  s3:
    enabled: true
    provider: beget
    endpoint: https://s3.ru1.storage.beget.cloud
    bucket: 443c60e2203e-betacassiopeiae
    access_key: ${BEGET_S3_ACCESS_KEY}
    secret_key: ${BEGET_S3_SECRET_KEY}
    region: ru1
```

## Автоматические backup'ы

### Создать cron job

```bash
# Редактировать crontab
crontab -e

# Добавить задачу (ежедневный backup в 3:00)
0 3 * * * cd /opt/trading-bot && python3 scripts/backup_to_beget_s3.py \
  --db /var/lib/trading-bot/trading_bot.db \
  --bucket 443c60e2203e-betacassiopeiae \
  --access-key $BEGET_S3_ACCESS_KEY \
  --secret-key $BEGET_S3_SECRET_KEY
```

## Стоимость

Beget S3 тарифы (примерно):
- Хранение: ~2₽/GB/месяц
- Исходящий трафик: ~5₽/GB

Примерный расчет:
- 10 GB backup'ов: ~20₽/месяц
- Очень дешево для надежного хранения!

## Дополнительная информация

См. также:
- `scripts/backup_to_beget_s3.py` - готовый скрипт
- `S3_vs_PostgreSQL_EXPLAINED.md` - объяснение разницы
- `РЕШЕНИЕ_ВАШЕЙ_ПРОБЛЕМЫ.md` - ваш конкретный случай
