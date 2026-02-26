# Beget PostgreSQL: Гайд по подключению

## Введение

Если вам нужна полноценная PostgreSQL база данных (не S3!), вы можете арендовать её у Beget.

**Важно:** PostgreSQL - это ОТДЕЛЬНЫЙ сервис от S3!
- S3: для файлов (у вас уже есть)
- PostgreSQL: для SQL запросов (нужно арендовать)

## Как арендовать PostgreSQL у Beget

### Шаг 1: Перейти на сайт

https://cp.beget.com → "Базы данных" → "PostgreSQL"

### Шаг 2: Создать БД

Нажать "Создать базу данных"

Параметры:
- **Тариф:** от 150₽/месяц
- **Имя БД:** `trading_bot`
- **Пользователь:** `bot_user`
- **Пароль:** придумать сложный пароль

### Шаг 3: Получить данные подключения

После создания вы получите:

```
Host: pg-XXXXX.beget.tech  ← ЭТО ДРУГОЙ АДРЕС!
Port: 5432
Database: trading_bot
User: bot_user
Password: ваш_пароль
```

**ВАЖНО:** Адрес будет выглядеть как `pg-12345.beget.tech`, 
НЕ `s3.ru1.storage.beget.cloud`!

### Шаг 4: Проверить подключение

```bash
# Установить psql если нет
apt-get install postgresql-client

# Проверить подключение
psql -h pg-XXXXX.beget.tech \
     -p 5432 \
     -U bot_user \
     -d trading_bot

# Если подключилось - всё ОК!
```

## Миграция данных

После получения PostgreSQL:

```bash
python3 scripts/migrate_db_direct.py \
  --source-db /var/lib/trading-bot/trading_bot.db \
  --target-type postgresql \
  --target-host pg-XXXXX.beget.tech \
  --target-database trading_bot \
  --target-user bot_user \
  --target-password "ваш_пароль"
```

## Обновление конфигурации бота

После миграции обновить `config.yaml`:

```yaml
database:
  type: postgresql
  host: pg-XXXXX.beget.tech
  port: 5432
  name: trading_bot
  user: bot_user
  password: ${DB_PASSWORD}
```

И перезапустить бота:

```bash
sudo systemctl restart trading-bot
```

## Стоимость

Beget PostgreSQL тарифы:
- Базовый: ~150₽/месяц (1 GB)
- Стандартный: ~300₽/месяц (5 GB)
- Расширенный: ~600₽/месяц (10 GB)

## Резюме

### У вас СЕЙЧАС есть:
- S3 хранилище (s3.ru1.storage.beget.cloud)
- Для файлов и backup'ов

### Если нужна PostgreSQL:
1. Арендовать на cp.beget.com
2. Получить адрес pg-XXXXX.beget.tech
3. Мигрировать данные
4. Обновить конфигурацию

### Комбинированное решение:
- PostgreSQL: для текущих данных (~150₽/мес)
- S3: для backup'ов (~50₽/мес)
- Итого: ~200₽/мес

## Дополнительная информация

- `S3_vs_PostgreSQL_EXPLAINED.md` - разница между сервисами
- `РЕШЕНИЕ_ВАШЕЙ_ПРОБЛЕМЫ.md` - ваш конкретный случай
- `scripts/migrate_db_direct.py` - скрипт миграции
