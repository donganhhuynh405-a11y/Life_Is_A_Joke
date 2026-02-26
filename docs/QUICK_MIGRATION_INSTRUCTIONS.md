# 🚀 БЫСТРАЯ МИГРАЦИЯ БЕЗ ТЯЖЕЛЫХ ОБНОВЛЕНИЙ

## ⚠️ ВАША СИТУАЦИЯ:
- Обновления слишком тяжелые для сервера
- Нужно подключить внешнюю БД
- Нужно перенести данные без полного обновления

## ✅ РЕШЕНИЕ:
Легкие standalone скрипты для миграции данных

---

## 📋 ЧТО ДЕЛАТЬ ПРЯМО СЕЙЧАС:

### Шаг 1: Подготовка БД (если еще не готова)

Арендуйте PostgreSQL БД (рекомендуется):
- DigitalOcean Managed Database: $15/месяц
- AWS RDS: от $15/месяц
- Heroku Postgres: от $9/месяц

Или используйте MySQL/MongoDB - скрипты поддерживают все.

### Шаг 2: На вашем текущем сервере

```bash
# 1. Обновите код (только скрипты, без зависимостей)
cd /opt/trading-bot
git pull origin copilot/transfer-files-to-empty-repo

# 2. Установите ТОЛЬКО драйвер БД (5-10 MB)
pip install psycopg2-binary  # для PostgreSQL
# или
pip install pymysql  # для MySQL
# или
pip install pymongo  # для MongoDB

# Это ВСЁ! Никаких других зависимостей не нужно!
```

### Шаг 3: Настройка переменных окружения

```bash
# Создайте файл с credentials
export DB_HOST="your-db-host.com"
export DB_PORT="5432"
export DB_NAME="trading_bot"
export DB_USER="bot_user"
export DB_PASSWORD="your_secure_password"
```

### Шаг 4: Запуск миграции

```bash
# Прямая миграция (рекомендуется)
python scripts/migrate_db_direct.py \
  --source-db /var/lib/trading-bot/trading_bot.db \
  --target-type postgresql \
  --target-host $DB_HOST \
  --target-port $DB_PORT \
  --target-database $DB_NAME \
  --target-user $DB_USER \
  --target-password $DB_PASSWORD

# Процесс займет 5-15 минут в зависимости от объема данных
# Скрипт покажет прогресс
```

### Шаг 5: Проверка

```bash
# Проверить что данные перенесены корректно
python scripts/verify_migration.py \
  --source /var/lib/trading-bot/trading_bot.db \
  --target-type postgresql \
  --target-host $DB_HOST \
  --target-database $DB_NAME \
  --target-user $DB_USER \
  --target-password $DB_PASSWORD

# Скрипт проверит:
# - Количество записей
# - Схему таблиц
# - Образцы данных
# - Индексы
```

### Шаг 6: Обновление конфигурации бота

```bash
# Отредактируйте config.yaml
vi config.yaml
```

Измените секцию database:
```yaml
database:
  type: postgresql  # было sqlite
  host: your-db-host.com
  port: 5432
  name: trading_bot
  user: bot_user
  password: ${DB_PASSWORD}  # из переменной окружения
```

### Шаг 7: Перезапуск бота

```bash
# Добавьте DB_PASSWORD в environment
echo "DB_PASSWORD=your_secure_password" >> /etc/systemd/system/trading-bot.service.d/environment.conf

# Перезагрузите systemd
sudo systemctl daemon-reload

# Перезапустите бота
sudo systemctl restart trading-bot

# Проверьте статус
sudo systemctl status trading-bot

# Проверьте логи
sudo journalctl -u trading-bot -f
```

---

## 🎯 EXPECTED OUTPUT:

### Миграция (Шаг 4):
```
🔄 MIGRATING DATA...

Connection: postgresql://your-db-host.com:5432/trading_bot
Status: ✅ Connected

[1/5] Creating schema...
  ✅ Table: positions
  ✅ Table: trades
  ✅ Table: market_data
  ✅ Table: news
  ✅ Table: config

[2/5] Migrating positions (12,543 rows)...
  ████████████████████ 100%

[3/5] Migrating trades (45,892 rows)...
  ████████████████████ 100%

[4/5] Migrating market_data (1,234,567 rows)...
  ████████████████████ 100%

[5/5] Migrating news (8,234 rows)...
  ████████████████████ 100%

✅ MIGRATION COMPLETE!
Total: 1,301,236 records
Time: 8m 24s
```

### Проверка (Шаг 5):
```
🔍 VERIFYING MIGRATION...

[1/5] Row counts...
  positions:    12,543 ↔ 12,543 ✅
  trades:       45,892 ↔ 45,892 ✅
  market_data:  1,234,567 ↔ 1,234,567 ✅

[2/5] Schema validation... ✅
[3/5] Sample data... ✅
[4/5] Indexes... ✅
[5/5] Constraints... ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ MIGRATION VERIFIED SUCCESSFULLY!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Bot restart (Шаг 7):
```
● trading-bot.service - Trading Bot
   Active: active (running)
   
Feb 10 13:30:00 server bot[12345]: INFO - Database: PostgreSQL connected
Feb 10 13:30:01 server bot[12345]: INFO - Loading strategies...
Feb 10 13:30:02 server bot[12345]: INFO - Bot started successfully
```

---

## ⚠️ TROUBLESHOOTING:

### Problem: "Connection refused"
**Fix:** Проверьте что БД доступна извне и firewall открыт
```bash
telnet your-db-host.com 5432
```

### Problem: "Authentication failed"
**Fix:** Проверьте credentials
```bash
psql -h your-db-host.com -U bot_user -d trading_bot
```

### Problem: "Out of memory"
**Fix:** Уменьшите batch size
```bash
python scripts/migrate_db_direct.py --batch-size 500 ...
```

### Problem: "SSL required"
**Fix:** Добавьте SSL параметр
```bash
python scripts/migrate_db_direct.py --ssl-mode require ...
```

---

## 💡 IMPORTANT NOTES:

### После успешной миграции:

1. **НЕ удаляйте** старую SQLite БД сразу
   - Держите как backup минимум неделю
   - Убедитесь что всё работает на новой БД

2. **Мониторинг** первые дни
   - Проверяйте логи ежедневно
   - Убедитесь что торговля работает
   - Проверьте что данные записываются

3. **Backup новой БД**
   - Настройте автоматический backup
   - PostgreSQL: pg_dump ежедневно
   - Или используйте managed DB с auto-backup

---

## 📞 SUPPORT:

Если возникли проблемы:

1. **Check logs:**
```bash
# Migration logs
cat /opt/trading-bot/migration.log

# Bot logs
journalctl -u trading-bot -n 100
```

2. **Verify data:**
```bash
python scripts/verify_migration.py
```

3. **Rollback if needed:**
```bash
# Restore config.yaml to sqlite
vi config.yaml  # change back to sqlite

# Restart bot
systemctl restart trading-bot
```

---

## ✅ CHECKLIST:

- [ ] БД арендована и доступна
- [ ] Credentials записаны
- [ ] Git pull выполнен
- [ ] Драйвер БД установлен (psycopg2-binary)
- [ ] Миграция запущена
- [ ] Миграция завершена успешно
- [ ] Verification passed
- [ ] Config.yaml обновлен
- [ ] Bot перезапущен
- [ ] Логи проверены - всё работает
- [ ] Старая БД сохранена как backup

---

## 🎯 RESULT:

**Before:**
- SQLite на локальном сервере
- Ограниченное место
- Нет возможности масштабирования

**After:**
- PostgreSQL на отдельном сервере
- Неограниченное хранилище
- Готово к масштабированию
- **Без тяжелых обновлений!**

---

**Время на всю процедуру: 20-30 минут**

**Затраты на БД: $15-20/месяц**

**Выигрыш: Unlimited storage + Better performance**

**ГОТОВО!** ✅
