# Как обновить бот на сервере

## Быстрая установка (для нового сервера)

### Вариант 1: Одной командой
```bash
curl -sL https://raw.githubusercontent.com/matthew3f2eb8c4-pixel/life_is_a_joke/copilot/transfer-files-to-empty-repo/scripts/quick_install.sh | bash
```

### Вариант 2: Скачать и установить
```bash
# Скачать последнюю версию
bash <(curl -sL https://raw.githubusercontent.com/matthew3f2eb8c4-pixel/life_is_a_joke/copilot/transfer-files-to-empty-repo/scripts/download_update.sh) /opt/trading-bot

# Перейти в директорию
cd /opt/trading-bot

# Настроить конфигурацию
nano .env

# Установить зависимости
pip3 install -r requirements.txt
```

## Обновление работающего бота на сервере

### Автоматическое обновление (рекомендуется)

```bash
# Скачайте скрипт обновления
sudo bash <(curl -sL https://raw.githubusercontent.com/matthew3f2eb8c4-pixel/life_is_a_joke/copilot/transfer-files-to-empty-repo/scripts/update_bot.sh)
```

Этот скрипт автоматически:
- Клонирует репозиторий (если ещё не клонирован)
- Остановит бота
- Скачает последние изменения
- Обновит файлы
- Запустит бота заново

### Ручное обновление

```bash
# 1. Остановите бота
sudo systemctl stop trading-bot

# 2. Перейдите в директорию с ботом
cd ~/trading-bot-setup/life_is_a_joke

# 3. Скачайте последние изменения
git fetch origin
git checkout copilot/transfer-files-to-empty-repo
git pull origin copilot/transfer-files-to-empty-repo

# 4. Обновите зависимости (если нужно)
pip3 install -r requirements.txt

# 5. Скопируйте файлы в рабочую директорию
sudo rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    ./ /opt/trading-bot/

# 6. Запустите бота
sudo systemctl start trading-bot

# 7. Проверьте статус
sudo systemctl status trading-bot
sudo journalctl -u trading-bot -f
```

## Проверка версии и обновление

Используйте скрипт для проверки версии кода:

```bash
cd ~/trading-bot-setup/life_is_a_joke
./scripts/verify_and_update.sh
```

Этот скрипт проверит:
- Текущую версию кода
- Наличие критических исправлений
- Статус работающего бота

И предложит автоматически обновить код, если нужно.

## Информация о репозитории

- **URL репозитория**: https://github.com/matthew3f2eb8c4-pixel/life_is_a_joke
- **Ветка**: copilot/transfer-files-to-empty-repo

## Полезные команды

### Просмотр логов
```bash
sudo journalctl -u trading-bot -f
```

### Перезапуск бота
```bash
sudo systemctl restart trading-bot
```

### Проверка статуса
```bash
sudo systemctl status trading-bot
```

### Диагностика позиций
```bash
cd /opt/trading-bot
python3 scripts/diagnose_positions.py
```

### Проверка соединения с биржей
```bash
cd /opt/trading-bot
python3 scripts/test_connectivity.py
```

## Структура директорий

```
~/trading-bot-setup/life_is_a_joke/  - Git репозиторий (для обновлений)
/opt/trading-bot/                     - Рабочая директория бота
/var/lib/trading-bot/                 - База данных
/var/log/trading-bot/                 - Логи
```

## Устранение проблем

### Бот не запускается после обновления

1. Проверьте логи:
```bash
sudo journalctl -u trading-bot -n 100 --no-pager
```

2. Проверьте файл .env:
```bash
sudo nano /opt/trading-bot/.env
```

3. Проверьте зависимости:
```bash
cd /opt/trading-bot
pip3 install -r requirements.txt
```

### Конфликты при git pull

```bash
cd ~/trading-bot-setup/life_is_a_joke
git reset --hard origin/copilot/transfer-files-to-empty-repo
git clean -fd
```

### Откатиться к предыдущей версии

```bash
cd ~/trading-bot-setup/life_is_a_joke
git log --oneline -10  # найдите нужный коммит
git checkout <commit-hash>
sudo rsync -av --exclude='.git' ./ /opt/trading-bot/
sudo systemctl restart trading-bot
```

## Безопасность

⚠️ **Важно:**
- Всегда делайте резервную копию .env файла перед обновлением
- Проверяйте изменения в коде перед применением: `git log -p`
- Тестируйте обновления в тестовой среде, если возможно
- Мониторьте логи после обновления

## Автоматические обновления (опционально)

Для настройки автоматических обновлений (не рекомендуется для продакшена):

```bash
# Создайте cron задачу
sudo crontab -e

# Добавьте строку (обновление каждую ночь в 3:00)
0 3 * * * /home/user/trading-bot-setup/life_is_a_joke/scripts/update_bot.sh >> /var/log/trading-bot-update.log 2>&1
```
# Ручное обновление бота (как раньше)

## 📝 Команды для обновления

Если вы раньше обновляли бот вручную, теперь используйте эти команды:

### Вариант 1: Обновить из существующего репозитория

Если репозиторий уже клонирован в `~/trading-bot-setup/life_is_a_joke`:

```bash
# Перейти в директорию репозитория
cd ~/trading-bot-setup/life_is_a_joke

# Скачать последнюю версию
git pull origin copilot/transfer-files-to-empty-repo

# Скопировать файл стратегии
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/

# Перезапустить бота
sudo systemctl restart trading-bot

# Посмотреть логи
sudo journalctl -u trading-bot -f
```

### Вариант 2: Первая установка репозитория

Если репозиторий ещё не клонирован:

```bash
# Создать директорию
mkdir -p ~/trading-bot-setup
cd ~/trading-bot-setup

# Клонировать новый репозиторий
git clone https://github.com/matthew3f2eb8c4-pixel/life_is_a_joke.git

# Перейти в директорию
cd life_is_a_joke

# Переключиться на нужную ветку
git checkout copilot/transfer-files-to-empty-repo

# Скопировать файл стратегии
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/

# Перезапустить бота
sudo systemctl restart trading-bot

# Посмотреть логи
sudo journalctl -u trading-bot -f
```

### Вариант 3: Переключиться на новый репозиторий

**⚠️ Если получили ошибки, см. [SWITCH_REPO_GUIDE.md](SWITCH_REPO_GUIDE.md)**

Если у вас был старый репозиторий и нужно переключиться на новый:

```bash
# Перейти в директорию
cd ~/trading-bot-setup/life_is_a_joke

# Сбросить все локальные изменения (безопасно)
git reset --hard HEAD
git clean -fd

# Изменить удалённый репозиторий
git remote set-url origin https://github.com/matthew3f2eb8c4-pixel/life_is_a_joke.git

# Скачать ветки из нового репозитория
git fetch origin

# Переключиться на новую ветку
git checkout copilot/transfer-files-to-empty-repo

# Принудительно синхронизироваться с новой веткой
git reset --hard origin/copilot/transfer-files-to-empty-repo

# Скопировать файл стратегии
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/

# Перезапустить бота
sudo systemctl restart trading-bot

# Посмотреть логи
sudo journalctl -u trading-bot -f
```

## 🔄 Что изменилось

| Старый репозиторий | Новый репозиторий |
|-------------------|-------------------|
| `anthony87b7f58e-coder/life_is_a_joke` | `matthew3f2eb8c4-pixel/life_is_a_joke` |
| Ветка: `copilot/merge-all-branches` | Ветка: `copilot/transfer-files-to-empty-repo` |

## 📋 Пошаговая инструкция

### Шаг 1: Обновить код
```bash
cd ~/trading-bot-setup/life_is_a_joke
git pull origin copilot/transfer-files-to-empty-repo
```

### Шаг 2: Скопировать файлы
```bash
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/
```

Или скопировать все файлы:
```bash
sudo rsync -av --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    ~/trading-bot-setup/life_is_a_joke/ /opt/trading-bot/
```

### Шаг 3: Перезапустить бота
```bash
sudo systemctl restart trading-bot
```

### Шаг 4: Проверить логи
```bash
sudo journalctl -u trading-bot -f
```

Для выхода из просмотра логов нажмите `Ctrl+C`.

## 🛠️ Полезные команды

### Проверить текущий репозиторий
```bash
cd ~/trading-bot-setup/life_is_a_joke
git remote -v
```

### Проверить текущую ветку
```bash
git branch
```

### Проверить последние изменения
```bash
git log --oneline -5
```

### Проверить статус бота
```bash
sudo systemctl status trading-bot
```

### Посмотреть последние логи
```bash
sudo journalctl -u trading-bot -n 50
```

### Остановить бота
```bash
sudo systemctl stop trading-bot
```

### Запустить бота
```bash
sudo systemctl start trading-bot
```

## 🚨 Устранение проблем

**📖 Полное руководство:** [SWITCH_REPO_GUIDE.md](SWITCH_REPO_GUIDE.md)

### Ошибка при переключении репозитория

Если получили ошибки "Your local changes would be overwritten" или "divergent branches":

```bash
cd ~/trading-bot-setup/life_is_a_joke
git reset --hard HEAD
git clean -fd
git remote set-url origin https://github.com/matthew3f2eb8c4-pixel/life_is_a_joke.git
git fetch origin
git checkout copilot/transfer-files-to-empty-repo
git reset --hard origin/copilot/transfer-files-to-empty-repo
```

См. [SWITCH_REPO_GUIDE.md](SWITCH_REPO_GUIDE.md) для подробных объяснений.

### Ошибка "Already up to date" но код не обновился

```bash
cd ~/trading-bot-setup/life_is_a_joke
git fetch origin
git reset --hard origin/copilot/transfer-files-to-empty-repo
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/
sudo systemctl restart trading-bot
```

### Конфликты при git pull

```bash
cd ~/trading-bot-setup/life_is_a_joke
git stash
git pull origin copilot/transfer-files-to-empty-repo
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/
sudo systemctl restart trading-bot
```

### Бот не запускается после обновления

```bash
# Проверить логи
sudo journalctl -u trading-bot -n 100 --no-pager

# Проверить файлы
ls -la /opt/trading-bot/src/strategies/strategy_manager.py

# Проверить права
sudo chown -R tradingbot:tradingbot /opt/trading-bot

# Попробовать запустить вручную
cd /opt/trading-bot
python3 -m src.main
```

## 📚 Дополнительная информация

Для автоматического обновления используйте:
```bash
sudo bash <(curl -sL https://raw.githubusercontent.com/matthew3f2eb8c4-pixel/life_is_a_joke/copilot/transfer-files-to-empty-repo/scripts/update_bot.sh)
```

Полная документация:
- [QUICK_START_UPDATE.md](QUICK_START_UPDATE.md) - Быстрый старт
- [UPDATE_GUIDE.md](UPDATE_GUIDE.md) - Подробное руководство
- [UBUNTU_INSTALL.md](UBUNTU_INSTALL.md) - Установка на Ubuntu

## 🎯 Быстрая справка

```bash
# Обновить код
cd ~/trading-bot-setup/life_is_a_joke && git pull origin copilot/transfer-files-to-empty-repo

# Обновить файлы бота
sudo cp src/strategies/strategy_manager.py /opt/trading-bot/src/strategies/

# Перезапустить
sudo systemctl restart trading-bot

# Логи
sudo journalctl -u trading-bot -f
```

---

**💡 Совет:** Сохраните эту страницу в закладки для быстрого доступа к командам!
# Руководство по обновлению кода на сервере

## 🐛 Обновлённое исправление (версия 2)

**Проблема**: Бот не может найти баланс USDT в аккаунте Bybit и выдаёт ошибку:
```
WARNING - Could not find USDT in expected locations
INFO - Available USDbalance: $0.00  
ERROR - InsufficientFunds
```

**Причины**:
1. Старая версия кода на сервере (не обновлена после исправления)
2. Реально нулевой или очень маленький баланс USDT
3. USDT может быть заблокирован в открытых ордерах

**Решение**: 
- Обновлена логика получения баланса USDT с расширенным логированием
- Добавлена проверка всех доступных валют
- Добавлена проверка заблокированных средств
- Улучшены сообщения об ошибках

## 📋 Изменённые файлы

- `src/strategies/strategy_manager.py` - расширенная логика определения баланса с детальным логированием

## 🚀 ВАЖНО: Проверка перед обновлением

### Шаг 0: Убедитесь, что вы в правильной директории

```bash
# Найдите директорию проекта
find ~ -name "life_is_a_joke" -type d 2>/dev/null

# Или найдите по запущенному процессу
ps aux | grep "python.*main.py" | grep -v grep

# Перейдите в найденную директорию, например:
cd /home/user/life_is_a_joke
# ИЛИ
cd ~/life_is_a_joke  
# ИЛИ куда вы установили проект
```

### Шаг 1: Проверьте текущую версию

```bash
# Убедитесь что вы в git репозитории
git status

# Проверьте текущий коммит
git log --oneline -1

# Должно быть НЕ МЕНЬШЕ чем:
# cd7bdca Add quick fix summary in both languages
```

## 🚀 Обновление кода

### Вариант 1: Полное обновление (рекомендуется)

```bash
# 1. Остановите бота
sudo systemctl stop trading-bot

# 2. Сохраните изменения (если есть)
git stash

# 3. Получите последние обновления
git fetch origin copilot/merge-all-branches

# 4. Проверьте, что есть новые коммиты
git log HEAD..origin/copilot/merge-all-branches --oneline

# 5. Обновите код
git pull origin copilot/merge-all-branches

# 6. Проверьте, что обновление применилось
git log --oneline -3
# Должен быть коммит с "Enhanced balance detection logging"

# 7. Запустите бота
sudo systemctl start trading-bot

# 8. Проверьте логи
sudo journalctl -u trading-bot -f
```

### Вариант 2: Жёсткое обновление (если возникли конфликты)

```bash
# Это УДАЛИТ все локальные изменения!
sudo systemctl stop trading-bot
git fetch origin copilot/merge-all-branches
git reset --hard origin/copilot/merge-all-branches
sudo systemctl start trading-bot
```

## 🔍 Проверка обновления

После обновления в логах вы должны увидеть:

```
✅ ПРАВИЛЬНО (новая версия):
INFO - Available currencies: ['USDT', 'BTC', ...]
INFO - Non-zero balances: {'USDT': 100.5, 'BTC': 0.001}
INFO - USDT balance from balance['free']: 100.5
INFO - Available USDT balance: $100.50

❌ НЕПРАВИЛЬНО (старая версия):
WARNING - Could not find USDT in expected locations
INFO - Available USDbalance: $0.00  # Обратите внимание на опечатку "USDbalance"
```

## 📊 Диагностика после обновления

### Если баланс всё ещё $0.00

Новая версия покажет детальную информацию:

```bash
# Смотрите логи
sudo journalctl -u trading-bot -n 50

# Вы увидите:
# INFO - Available currencies: [список валют]
# INFO - Non-zero balances: {валюты с балансом > 0}
```

**Возможные причины нулевого баланса:**

1. **Реально нет USDT** - пополните счёт на Bybit
2. **USDT заблокирован** - закройте открытые ордера:
   ```
   WARNING - USDT total balance is 100 but free balance is 0 (funds may be locked in orders)
   ```
3. **Нет валют вообще** - проверьте:
   ```bash
   # Зайдите на Bybit и проверьте:
   # 1. Правильный ли API ключ
   # 2. Есть ли баланс в Unified Trading Account
   # 3. Не заблокирован ли аккаунт
   ```

## 🛠️ Альтернативный метод: Скачать файл напрямую

Если Git не работает:

```bash
# 1. Остановите бота
sudo systemctl stop trading-bot

# 2. Создайте резервную копию
cp src/strategies/strategy_manager.py src/strategies/strategy_manager.py.backup.$(date +%Y%m%d)

# 3. Скачайте обновлённый файл
wget -O src/strategies/strategy_manager.py \
  https://raw.githubusercontent.com/matthew3f2eb8c4-pixel/life_is_a_joke/copilot/transfer-files-to-empty-repo/src/strategies/strategy_manager.py

# 4. Проверьте, что файл скачался
ls -la src/strategies/strategy_manager.py

# 5. Проверьте содержимое (должен быть новый код)
grep "Available currencies:" src/strategies/strategy_manager.py

# Если команда выше ничего не вернула - файл НЕ обновился!

# 6. Запустите бота
sudo systemctl start trading-bot
```

## ❓ Часто задаваемые вопросы

### Q: Как проверить, что у меня последняя версия кода?
**A**: 
```bash
grep "Available currencies:" src/strategies/strategy_manager.py
```
Если команда возвращает результат - у вас новая версия.
Если ничего не возвращает - старая версия.

### Q: Логи показывают новые сообщения, но баланс всё равно $0.00
**A**: Значит проблема не в коде, а в реальном балансе:
1. Проверьте логи: `INFO - Available currencies: [...]` - какие валюты там?
2. Проверьте логи: `INFO - Non-zero balances: {...}` - есть ли USDT?
3. Зайдите на Bybit и проверьте Unified Trading Account
4. Убедитесь, что используете правильный API ключ
5. Проверьте, не заблокированы ли средства в открытых ордерах

### Q: Ошибка "InsufficientFunds" от Bybit
**A**: Это означает, что на счету действительно недостаточно средств. Проверьте:
1. Минимальная сумма для ордера на Bybit обычно $5-10
2. В логах теперь будет видно: `Bybit market buy: 0.001000 BTC = $99.84 USDT`
3. У вас должно быть минимум $100 USDT для торговли

### Q: Как откатиться к старой версии?
**A**: 
```bash
sudo systemctl stop trading-bot
cp src/strategies/strategy_manager.py.backup src/strategies/strategy_manager.py
sudo systemctl start trading-bot
```

## 🎯 Краткая инструкция (для опытных)

```bash
# Найти и перейти в директорию
cd $(find ~ -name "life_is_a_joke" -type d 2>/dev/null | head -1)

# Обновить
sudo systemctl stop trading-bot && \
git pull origin copilot/merge-all-branches && \
sudo systemctl start trading-bot && \
sudo journalctl -u trading-bot -f
```

## ⚠️ КРИТИЧЕСКИ ВАЖНО!

1. **Проверьте, что вы в правильной директории** - команда `git status` должна работать
2. **Проверьте после обновления** - в логах должны быть новые сообщения
3. **Если баланс $0** - проверьте реальный баланс на Bybit
4. **Минимальная сумма для торговли** - обычно $100+ USDT

## 📞 Поддержка

Если после обновления:
1. В логах появились новые сообщения "Available currencies:" - **код обновлён правильно**
2. Баланс всё равно $0 - **проблема в реальном балансе на Bybit, не в коде**
3. Старые сообщения "Available USDbalance:" - **код НЕ обновился, повторите процедуру**

---

**Дата обновления**: 2026-01-08  
**Версия**: 2.0  
**Коммит**: См. последний коммит в ветке copilot/merge-all-branches

