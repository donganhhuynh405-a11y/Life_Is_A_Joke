# 📘 ПОЛНАЯ ИНСТРУКЦИЯ: Подключение бота к MySQL и S3 от Beget

## 🎯 Содержание

1. [Введение и выбор БД](#введение)
2. [Что такое MySQL и S3](#что-такое)
3. [Аренда MySQL у Beget](#аренда-mysql)
4. [Настройка S3 хранилища](#настройка-s3)
5. [Миграция данных в MySQL](#миграция)
6. [Настройка автоматических backup'ов в S3](#backup)
7. [Конфигурация бота](#конфигурация)
8. [Проверка и запуск](#проверка)
9. [Типичные ошибки и решения](#ошибки)
10. [Стоимость и оптимизация](#стоимость)

---

## 📖 Введение {#введение}

### Зачем нужны MySQL и S3?

**MySQL** - это реляционная база данных, которая:
- ✅ Хранит текущие данные бота (сделки, позиции, статистику)
- ✅ Позволяет делать SQL запросы
- ✅ Обеспечивает быструю работу бота
- ✅ **Дешевле PostgreSQL** (100₽/мес vs 150₽/мес)

**S3** - это объектное хранилище, которое:
- ✅ Хранит backup'ы базы данных
- ✅ Хранит ML модели
- ✅ Хранит исторические данные и логи
- ✅ Очень дешево (~50₽/мес)

### Почему MySQL, а не PostgreSQL?

| Критерий | MySQL | PostgreSQL | Выбор |
|----------|-------|------------|-------|
| **Цена** | 100₽/мес ✅ | 150₽/мес | MySQL |
| **Для trading** | Отлично ✅ | Отлично ✅ | Равно |
| **Простота** | Проще ✅ | Сложнее | MySQL |
| **JSON** | Базовый ✅ | Продвинутый | Не важно |
| **Скорость** | Быстрее ✅ | Быстрее для сложных | MySQL |

**Рекомендация:** Берите MySQL - дешевле на 50₽/мес, все функции есть!

---

## 🔍 Что такое MySQL и S3 {#что-такое}

### MySQL - Реляционная база данных

```
MySQL:
  • Тип: Реляционная БД
  • Протокол: MySQL (порт 3306)
  • Данные: Таблицы, строки, SQL
  • Для чего: Текущие данные бота
  • SQL запросы: ✅ Да
  • Цена: 100₽/месяц
```

**Пример использования:**
```python
import pymysql

conn = pymysql.connect(
    host='mysql-12345.beget.tech',
    port=3306,
    database='trading_bot',
    user='bot_user',
    password='password'
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM trades WHERE profit > 0")
trades = cursor.fetchall()
```

### S3 - Объектное хранилище

```
S3:
  • Тип: Объектное хранилище
  • Протокол: HTTP/HTTPS (порт 443)
  • Данные: Файлы любого типа
  • Для чего: Backup'ы, модели, архивы
  • SQL запросы: ❌ Нет
  • Цена: ~50₽/месяц
```

**Пример использования:**
```python
import boto3

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.ru1.storage.beget.cloud',
    aws_access_key_id='YOUR_KEY',
    aws_secret_access_key='YOUR_SECRET'
)

# Загрузить backup
s3.upload_file('backup.db', 'bucket-name', 'backups/backup.db')
```

### ⚠️ ВАЖНО: MySQL ≠ S3

Это **РАЗНЫЕ сервисы**:
- MySQL - для SQL запросов
- S3 - для хранения файлов

Нельзя подключиться к S3 как к MySQL!

---

## 💰 Аренда MySQL у Beget {#аренда-mysql}

### Шаг 1: Зайти в панель управления

1. Перейти на https://cp.beget.com
2. Войти в аккаунт

### Шаг 2: Создать базу данных MySQL

1. В меню слева выбрать **"Базы данных"**
2. Нажать **"MySQL"**
3. Нажать **"Создать базу данных"**

### Шаг 3: Настроить параметры

```
Тариф: Начальный (100₽/месяц, 1 GB)
Имя БД: trading_bot
Пользователь: bot_user
Пароль: [придумать сложный пароль]
Регион: Москва (ru1)
```

### Шаг 4: Получить данные подключения

После создания вы получите:

```
Host: mysql-XXXXX.beget.tech
Port: 3306
Database: trading_bot
User: bot_user
Password: ваш_пароль
```

**Сохраните эти данные! Они понадобятся дальше.**

### Шаг 5: Проверить подключение

```bash
# Установить MySQL client (если нет)
apt-get install mysql-client

# Проверить подключение
mysql -h mysql-XXXXX.beget.tech \
      -P 3306 \
      -u bot_user \
      -p trading_bot

# Введите пароль
# Если подключилось - всё ОК!
```

---

## ☁️ Настройка S3 хранилища {#настройка-s3}

### Шаг 1: Получить доступ к S3

1. В панели Beget https://cp.beget.com
2. Перейти в **"Хранилище S3"**
3. Если еще нет - создать bucket

### Шаг 2: Получить ключи доступа

В разделе S3 найти:

```
Bucket name: 443c60e2203e-betacassiopeiae (или другой)
Endpoint: s3.ru1.storage.beget.cloud
Access Key ID: [скопировать]
Secret Access Key: [скопировать]
```

**Сохраните эти ключи!**

### Шаг 3: Установить boto3

```bash
cd /opt/trading-bot
pip install boto3
```

### Шаг 4: Настроить переменные окружения

```bash
# Создать файл с переменными
nano ~/.bashrc

# Добавить в конец файла:
export BEGET_S3_ACCESS_KEY="ваш_access_key"
export BEGET_S3_SECRET_KEY="ваш_secret_key"
export BEGET_S3_BUCKET="443c60e2203e-betacassiopeiae"

# Применить изменения
source ~/.bashrc
```

### Шаг 5: Проверить доступ к S3

```bash
# Создать тестовый файл
echo "test" > test.txt

# Загрузить в S3
python3 -c "
import boto3
s3 = boto3.client('s3',
    endpoint_url='https://s3.ru1.storage.beget.cloud',
    aws_access_key_id='$BEGET_S3_ACCESS_KEY',
    aws_secret_access_key='$BEGET_S3_SECRET_KEY')
s3.upload_file('test.txt', '$BEGET_S3_BUCKET', 'test.txt')
print('✅ S3 работает!')
"

# Если вывело "✅ S3 работает!" - всё ОК!
```

---

## 🔄 Миграция данных в MySQL {#миграция}

### Шаг 1: Установить драйвер MySQL

```bash
cd /opt/trading-bot
pip install pymysql
```

### Шаг 2: Проверить подключение к MySQL

```bash
python3 scripts/test_db_connection.py \
  --type mysql \
  --host mysql-XXXXX.beget.tech \
  --database trading_bot \
  --user bot_user \
  --password "ваш_пароль"
```

**Ожидаемый вывод:**
```
🔍 Testing MySQL connection...
✅ Connected successfully!
📊 Database info:
   Version: MySQL 8.0.32
   Character set: utf8mb4
✅ Connection test PASSED!
```

### Шаг 3: Сделать backup текущей БД (важно!)

```bash
# Backup SQLite БД перед миграцией
cp /var/lib/trading-bot/trading_bot.db \
   /var/lib/trading-bot/trading_bot.db.backup
```

### Шаг 4: Запустить миграцию

```bash
python3 scripts/migrate_db_direct.py \
  --source-db /var/lib/trading-bot/trading_bot.db \
  --target-type mysql \
  --target-host mysql-XXXXX.beget.tech \
  --target-database trading_bot \
  --target-user bot_user \
  --target-password "ваш_пароль"
```

**Процесс миграции:**
```
================================================================================
🚀 DATABASE MIGRATION
================================================================================
📂 Connecting to source: /var/lib/trading-bot/trading_bot.db
✅ Source database connected

📋 Found 4 tables: trades, positions, daily_stats, crypto_news

🔗 Connecting to MySQL: mysql-XXXXX.beget.tech
✅ MySQL connected

[1/4] trades
  ✅ Table trades created
  📊 Migrating trades (45,892 rows)
  [████████████████████████████] 100.0% | 45,892/45,892

[2/4] positions
  ✅ Table positions created
  📊 Migrating positions (12,543 rows)
  [████████████████████████████] 100.0% | 12,543/12,543

[3/4] daily_stats
  ✅ Table daily_stats created
  📊 Migrating daily_stats (365 rows)
  [████████████████████████████] 100.0% | 365/365

[4/4] crypto_news
  ✅ Table crypto_news created
  📊 Migrating crypto_news (8,234 rows)
  [████████████████████████████] 100.0% | 8,234/8,234

================================================================================
✅ MIGRATION COMPLETE!
================================================================================
  Tables migrated: 4
  Total records: 67,034
  Time: 45.2s
```

### Шаг 5: Проверить миграцию

```bash
python3 scripts/verify_migration.py \
  --source /var/lib/trading-bot/trading_bot.db \
  --target-type mysql \
  --target-host mysql-XXXXX.beget.tech \
  --target-database trading_bot \
  --target-user bot_user \
  --target-password "ваш_пароль"
```

**Должно показать:**
```
✅ MIGRATION VERIFIED SUCCESSFULLY!
All data migrated correctly.
```

---

## 💾 Настройка автоматических backup'ов в S3 {#backup}

### Шаг 1: Создать первый backup

```bash
python3 scripts/backup_to_beget_s3.py \
  --db /var/lib/trading-bot/trading_bot.db \
  --bucket 443c60e2203e-betacassiopeiae \
  --endpoint s3.ru1.storage.beget.cloud \
  --access-key $BEGET_S3_ACCESS_KEY \
  --secret-key $BEGET_S3_SECRET_KEY
```

**Вывод:**
```
================================================================================
🚀 BACKUP TO BEGET S3
================================================================================
📦 Creating backup: backup_20260211_120000.db
   Size: 45.2 MB
🗜️  Compressing...
   Compressed: 8.1 MB
   Ratio: 17.9%
☁️  Uploading to Beget S3...
✅ Backup uploaded successfully!
   Location: s3://443c60e2203e-betacassiopeiae/backups/backup_20260211_120000.db.gz
```

### Шаг 2: Настроить автоматические backup'ы

```bash
# Редактировать crontab
crontab -e

# Добавить задачу (ежедневный backup в 3:00 ночи)
0 3 * * * cd /opt/trading-bot && python3 scripts/backup_to_beget_s3.py \
  --db /var/lib/trading-bot/trading_bot.db \
  --bucket 443c60e2203e-betacassiopeiae \
  --endpoint s3.ru1.storage.beget.cloud \
  --access-key $BEGET_S3_ACCESS_KEY \
  --secret-key $BEGET_S3_SECRET_KEY >> /var/log/trading-bot-backup.log 2>&1
```

### Шаг 3: Проверить cron задачу

```bash
# Посмотреть список задач
crontab -l

# Проверить логи после первого запуска (на следующий день)
tail -f /var/log/trading-bot-backup.log
```

---

## ⚙️ Конфигурация бота {#конфигурация}

### Шаг 1: Обновить config.yaml

```bash
cd /opt/trading-bot
nano config.yaml
```

### Шаг 2: Настроить подключение к MySQL

Найти секцию `database` и изменить:

```yaml
database:
  # Тип базы данных
  type: mysql  # было: sqlite
  
  # MySQL настройки
  host: mysql-XXXXX.beget.tech  # ваш host
  port: 3306
  name: trading_bot
  user: bot_user
  password: ${DB_PASSWORD}  # пароль из переменной окружения
  
  # MySQL опции
  options:
    charset: utf8mb4
    collation: utf8mb4_unicode_ci
    autocommit: true
    pool_size: 5
    max_overflow: 10
    pool_recycle: 3600
```

### Шаг 3: Настроить S3 для моделей и backup'ов

Добавить секцию `storage`:

```yaml
storage:
  # S3 хранилище
  s3:
    enabled: true
    provider: beget
    endpoint: https://s3.ru1.storage.beget.cloud
    bucket: 443c60e2203e-betacassiopeiae
    access_key: ${BEGET_S3_ACCESS_KEY}
    secret_key: ${BEGET_S3_SECRET_KEY}
    region: ru1
    
    # Что хранить в S3
    use_for:
      - ml_models      # ML модели
      - backups        # Backup'ы БД
      - historical     # Исторические данные
      - logs           # Старые логи
```

### Шаг 4: Настроить переменные окружения

```bash
# Создать .env файл
nano /opt/trading-bot/.env

# Добавить:
DB_PASSWORD=ваш_пароль_mysql
BEGET_S3_ACCESS_KEY=ваш_s3_access_key
BEGET_S3_SECRET_KEY=ваш_s3_secret_key
```

### Шаг 5: Защитить .env файл

```bash
chmod 600 /opt/trading-bot/.env
chown trading-bot:trading-bot /opt/trading-bot/.env
```

---

## ✅ Проверка и запуск {#проверка}

### Шаг 1: Проверить конфигурацию

```bash
cd /opt/trading-bot

# Проверить что config.yaml валидный
python3 -c "
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
    print('✅ Config is valid')
    print(f'Database type: {config[\"database\"][\"type\"]}')
    print(f'Database host: {config[\"database\"][\"host\"]}')
    print(f'S3 enabled: {config[\"storage\"][\"s3\"][\"enabled\"]}')
"
```

### Шаг 2: Тестовый запуск бота

```bash
# Запустить в тестовом режиме
python3 -m src.main --test

# Должно показать:
# ✅ Connected to MySQL
# ✅ S3 storage available
# ✅ All systems operational
```

### Шаг 3: Запустить бота

```bash
# Если используете systemd
sudo systemctl restart trading-bot
sudo systemctl status trading-bot

# Проверить логи
tail -f /var/log/trading-bot/bot.log
```

### Шаг 4: Проверить подключение к MySQL

```bash
# В логах должно быть:
grep "Connected to MySQL" /var/log/trading-bot/bot.log
```

### Шаг 5: Проверить backup в S3

```bash
# Через несколько часов (или на следующий день)
python3 -c "
import boto3
s3 = boto3.client('s3',
    endpoint_url='https://s3.ru1.storage.beget.cloud',
    aws_access_key_id='$BEGET_S3_ACCESS_KEY',
    aws_secret_access_key='$BEGET_S3_SECRET_KEY')

response = s3.list_objects_v2(
    Bucket='$BEGET_S3_BUCKET',
    Prefix='backups/'
)

print('Backup files in S3:')
for obj in response.get('Contents', []):
    print(f'  • {obj[\"Key\"]} ({obj[\"Size\"]/1024/1024:.1f} MB)')
"
```

---

## 🐛 Типичные ошибки и решения {#ошибки}

### Ошибка 1: "Can't connect to MySQL server"

**Причина:** Неправильный host или firewall блокирует.

**Решение:**
```bash
# 1. Проверить host
ping mysql-XXXXX.beget.tech

# 2. Проверить порт
telnet mysql-XXXXX.beget.tech 3306

# 3. Проверить в панели Beget что БД создана и активна
```

### Ошибка 2: "Access denied for user"

**Причина:** Неправильный пользователь или пароль.

**Решение:**
```bash
# 1. Проверить пароль в .env файле
cat /opt/trading-bot/.env | grep DB_PASSWORD

# 2. Попробовать подключиться вручную
mysql -h mysql-XXXXX.beget.tech -u bot_user -p trading_bot

# 3. Если не работает - сбросить пароль в панели Beget
```

### Ошибка 3: "Could not translate host name" (для S3)

**Причина:** Пытаетесь использовать S3 URL как MySQL host.

**Решение:**
```
❌ НЕПРАВИЛЬНО:
--target-host s3.ru1.storage.beget.cloud

✅ ПРАВИЛЬНО:
--target-host mysql-XXXXX.beget.tech

S3 используется ТОЛЬКО для backup'ов через backup_to_beget_s3.py!
```

### Ошибка 4: "NoSuchBucket" (S3)

**Причина:** Неправильное имя bucket или нет доступа.

**Решение:**
```bash
# 1. Проверить имя bucket в панели Beget
# 2. Проверить ключи доступа
echo $BEGET_S3_ACCESS_KEY
echo $BEGET_S3_SECRET_KEY

# 3. Создать bucket если нет:
# https://cp.beget.com → Хранилище S3 → Создать bucket
```

### Ошибка 5: "Character set utf8mb4 not found"

**Причина:** Старая версия MySQL или неправильная кодировка.

**Решение:**
```yaml
# В config.yaml изменить:
database:
  options:
    charset: utf8  # вместо utf8mb4
    collation: utf8_general_ci  # вместо utf8mb4_unicode_ci
```

---

## 💰 Стоимость и оптимизация {#стоимость}

### Месячные расходы

```
MySQL (Beget):
  Тариф "Начальный": 100₽/месяц
  • 1 GB хранилище
  • Достаточно для начала
  • Можно расширить позже

S3 (Beget):
  Хранение: ~2₽/GB/месяц
  Трафик: ~5₽/GB (исходящий)
  
  Примерный расчет:
  • 10 GB backup'ов: 20₽/месяц
  • 5 GB моделей: 10₽/месяц
  • 10 GB трафика: 50₽/месяц
  • Итого: ~80₽/месяц

═══════════════════════════════
ИТОГО: ~180₽/месяц
═══════════════════════════════
```

### Оптимизация расходов

**1. Удаление старых backup'ов**

```bash
# Удалить backup'ы старше 30 дней
python3 -c "
import boto3
from datetime import datetime, timedelta

s3 = boto3.client('s3',
    endpoint_url='https://s3.ru1.storage.beget.cloud',
    aws_access_key_id='$BEGET_S3_ACCESS_KEY',
    aws_secret_access_key='$BEGET_S3_SECRET_KEY')

cutoff = datetime.now() - timedelta(days=30)

response = s3.list_objects_v2(
    Bucket='$BEGET_S3_BUCKET',
    Prefix='backups/'
)

for obj in response.get('Contents', []):
    if obj['LastModified'].replace(tzinfo=None) < cutoff:
        print(f'Deleting old backup: {obj[\"Key\"]}')
        s3.delete_object(Bucket='$BEGET_S3_BUCKET', Key=obj['Key'])

print('✅ Old backups cleaned')
"
```

**2. Сжатие данных**

Backup'ы автоматически сжимаются (gzip) - экономия ~80% места!

**3. Оптимизация БД**

```sql
-- Периодически очищать старые данные
DELETE FROM trades WHERE created_at < DATE_SUB(NOW(), INTERVAL 6 MONTH);
DELETE FROM crypto_news WHERE created_at < DATE_SUB(NOW(), INTERVAL 3 MONTH);

-- Оптимизировать таблицы
OPTIMIZE TABLE trades;
OPTIMIZE TABLE positions;
```

### Сравнение с PostgreSQL

```
PostgreSQL + S3:
  PostgreSQL: 150₽/месяц
  S3: 80₽/месяц
  ИТОГО: 230₽/месяц

MySQL + S3:
  MySQL: 100₽/месяц
  S3: 80₽/месяц
  ИТОГО: 180₽/месяц

ЭКОНОМИЯ: 50₽/месяц = 600₽/год
```

---

## 📚 Дополнительные ресурсы

### Скрипты в репозитории

- `scripts/migrate_db_direct.py` - миграция SQLite → MySQL
- `scripts/test_db_connection.py` - проверка подключения
- `scripts/backup_to_beget_s3.py` - backup в S3
- `scripts/verify_migration.py` - проверка миграции

### Документация

- `BEGET_КРАТКОЕ_ОБЪЯСНЕНИЕ.md` - краткое объяснение S3 vs БД
- `S3_vs_PostgreSQL_EXPLAINED.md` - подробное сравнение
- `BEGET_S3_INTEGRATION.md` - работа с S3
- `РЕШЕНИЕ_ВАШЕЙ_ПРОБЛЕМЫ.md` - типичные ошибки

### Полезные команды

```bash
# Проверить размер БД
mysql -h mysql-XXXXX.beget.tech -u bot_user -p -e "
SELECT 
    table_schema AS 'Database',
    ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'Size (MB)'
FROM information_schema.tables 
WHERE table_schema = 'trading_bot'
GROUP BY table_schema;
"

# Посмотреть backup'ы в S3
aws s3 ls s3://443c60e2203e-betacassiopeiae/backups/ \
    --endpoint-url https://s3.ru1.storage.beget.cloud

# Восстановить из backup
python3 scripts/restore_from_s3.py \
    --backup backups/backup_20260211_120000.db.gz \
    --output /var/lib/trading-bot/trading_bot.db
```

---

## ✅ Чеклист завершения

После выполнения всех шагов проверьте:

- [ ] MySQL БД арендована у Beget
- [ ] Данные успешно мигрированы из SQLite в MySQL
- [ ] Бот подключен к MySQL и работает
- [ ] S3 bucket настроен и доступен
- [ ] Первый backup в S3 успешно создан
- [ ] Автоматические backup'ы настроены (cron)
- [ ] config.yaml обновлен с правильными настройками
- [ ] Переменные окружения настроены (.env файл)
- [ ] Бот успешно запущен и работает
- [ ] Логи показывают успешное подключение

---

## 🎯 Итог

Теперь у вас:

✅ **MySQL база данных** - для текущих данных бота (100₽/мес)
✅ **S3 хранилище** - для backup'ов и ML моделей (~80₽/мес)
✅ **Автоматические backup'ы** - безопасность данных
✅ **Экономия 600₽/год** - по сравнению с PostgreSQL

**Общая стоимость: ~180₽/месяц**

Бот готов к работе! 🚀

---

## 📞 Поддержка

Если возникли проблемы:

1. Проверьте раздел [Типичные ошибки](#ошибки)
2. Проверьте логи: `tail -f /var/log/trading-bot/bot.log`
3. Проверьте подключение: `python3 scripts/test_db_connection.py`
4. Обратитесь в поддержку Beget: https://beget.com/ru/support

---

**Последнее обновление:** 11 февраля 2026

**Версия документа:** 1.0
