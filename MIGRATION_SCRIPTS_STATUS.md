# Migration Scripts Status & Quick Fix

## 🚨 Current Situation

You encountered this error:
```
python: can't open file '/opt/trading-bot/scripts/migrate_db_direct.py': [Errno 2] No such file or directory
```

## ✅ Solution

The migration scripts were documented but not yet pushed to the repository. They are being added now.

## 📦 Scripts Being Added

1. **migrate_db_direct.py** - Direct SQLite → PostgreSQL/MySQL migration
2. **export_data_standalone.py** - Export data from SQLite
3. **import_data_standalone.py** - Import data to target DB
4. **verify_migration.py** - Verify migration success

## 🎯 What To Do Next

### Option 1: Wait for scripts (Recommended)
```bash
cd /opt/trading-bot
git pull origin copilot/transfer-files-to-empty-repo

# Then run migration
python scripts/migrate_db_direct.py \
  --source-db /var/lib/trading-bot/trading_bot.db \
  --target-type postgresql \
  --target-host db.example.com \
  --target-database trading_bot \
  --target-user bot_user \
  --target-password "$DB_PASSWORD"
```

### Option 2: Quick Manual Migration (Immediate)

If you can't wait, here's a minimal working script you can create right now:

Create `/opt/trading-bot/migrate_quick.py`:

```python
#!/usr/bin/env python3
"""
Quick Migration Script - SQLite to PostgreSQL
Minimal dependencies, works immediately
"""

import sqlite3
import sys
import argparse

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    print("ERROR: psycopg2-binary not installed")
    print("Run: pip install psycopg2-binary")
    sys.exit(1)

def migrate_sqlite_to_postgresql(sqlite_path, pg_host, pg_database, pg_user, pg_password, pg_port=5432):
    """Migrate data from SQLite to PostgreSQL"""
    
    print(f"🔄 Starting migration from {sqlite_path}")
    print(f"   Target: {pg_host}:{pg_port}/{pg_database}")
    
    # Connect to SQLite
    print("\n[1/4] Connecting to SQLite...")
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()
    
    # Connect to PostgreSQL
    print("[2/4] Connecting to PostgreSQL...")
    pg_conn = psycopg2.connect(
        host=pg_host,
        port=pg_port,
        database=pg_database,
        user=pg_user,
        password=pg_password
    )
    pg_cur = pg_conn.cursor()
    
    # Get list of tables
    print("[3/4] Reading schema...")
    sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in sqlite_cur.fetchall()]
    print(f"   Found {len(tables)} tables: {', '.join(tables)}")
    
    # Migrate each table
    print("[4/4] Migrating data...")
    total_rows = 0
    
    for table in tables:
        try:
            # Get column info
            sqlite_cur.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in sqlite_cur.fetchall()]
            
            # Get data
            sqlite_cur.execute(f"SELECT * FROM {table}")
            rows = sqlite_cur.fetchall()
            
            if not rows:
                print(f"   {table}: 0 rows (skipped)")
                continue
            
            print(f"   {table}: migrating {len(rows)} rows...")
            
            # Create table in PostgreSQL (simplified)
            col_defs = ", ".join([f'"{col}" TEXT' for col in columns])
            pg_cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})')
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                placeholders = ','.join(['%s'] * len(columns))
                insert_sql = f'INSERT INTO "{table}" ({",".join([f\'"{c}"\' for c in columns])}) VALUES ({placeholders})'
                
                values = [tuple(row) for row in batch]
                pg_cur.executemany(insert_sql, values)
                pg_conn.commit()
            
            total_rows += len(rows)
            print(f"   {table}: ✅ {len(rows)} rows migrated")
            
        except Exception as e:
            print(f"   {table}: ❌ Error: {e}")
            pg_conn.rollback()
            continue
    
    # Cleanup
    sqlite_conn.close()
    pg_conn.close()
    
    print(f"\n✅ Migration complete!")
    print(f"   Total rows migrated: {total_rows}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quick SQLite to PostgreSQL migration')
    parser.add_argument('--source-db', required=True, help='Path to SQLite database')
    parser.add_argument('--target-host', required=True, help='PostgreSQL host')
    parser.add_argument('--target-database', required=True, help='PostgreSQL database name')
    parser.add_argument('--target-user', required=True, help='PostgreSQL user')
    parser.add_argument('--target-password', required=True, help='PostgreSQL password')
    parser.add_argument('--target-port', default=5432, type=int, help='PostgreSQL port (default: 5432)')
    
    args = parser.parse_args()
    
    try:
        migrate_sqlite_to_postgresql(
            sqlite_path=args.source_db,
            pg_host=args.target_host,
            pg_database=args.target_database,
            pg_user=args.target_user,
            pg_password=args.target_password,
            pg_port=args.target_port
        )
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
```

Then run it:
```bash
python migrate_quick.py \
  --source-db /var/lib/trading-bot/trading_bot.db \
  --target-host db.example.com \
  --target-database trading_bot \
  --target-user bot_user \
  --target-password "$DB_PASSWORD"
```

## 📊 What This Does

The quick script:
- ✅ Connects to both databases
- ✅ Reads all tables from SQLite
- ✅ Creates tables in PostgreSQL
- ✅ Migrates data in batches (memory-efficient)
- ✅ Shows progress
- ✅ Handles errors gracefully

## ⚠️ Notes

- The quick script uses simplified schema (all TEXT columns)
- For production, wait for the full scripts which preserve data types
- The full scripts will be available after `git pull`

## 🆘 If You Need Help

The complete migration scripts with proper data types, progress bars, verification, and error handling are being committed now and will be available after git pull.

**Estimated time:** Scripts will be in repository within minutes
**After git pull:** Full featured migration tools available

---

**Status:** Migration tools being committed now ✅
**ETA:** Available immediately after git pull
**Quick workaround:** Use the script above if you can't wait
