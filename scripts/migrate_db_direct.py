#!/usr/bin/env python3
"""
Direct Database Migration Script
Migrates data from SQLite to PostgreSQL/MySQL without intermediate files.
Minimal dependencies, memory-efficient, works on limited servers.
"""

import argparse
import sqlite3
import sys
import time
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

try:
    import pymysql
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


class ProgressBar:
    """Simple progress bar"""
    def __init__(self, total, prefix=''):
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current):
        self.current = current
        percent = (current / self.total * 100) if self.total > 0 else 100
        elapsed = time.time() - self.start_time
        rate = current / elapsed if elapsed > 0 else 0
        
        bar_length = 40
        filled = int(bar_length * current / self.total) if self.total > 0 else bar_length
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f'\r{self.prefix} [{bar}] {percent:>5.1f}% | {current:,}/{self.total:,} | {rate:.0f}/s', end='', flush=True)
    
    def finish(self):
        self.update(self.total)
        print()


class DatabaseMigrator:
    """Handles database migration"""
    
    def __init__(self, source_db, target_config, batch_size=1000):
        self.source_db = source_db
        self.target_config = target_config
        self.batch_size = batch_size
        self.target_conn = None
        self.target_cursor = None
    
    def connect_source(self):
        """Connect to source SQLite database"""
        print(f"📂 Connecting to source: {self.source_db}")
        try:
            conn = sqlite3.connect(self.source_db)
            conn.row_factory = sqlite3.Row
            print("✅ Source database connected")
            return conn
        except Exception as e:
            print(f"❌ Error connecting to source: {e}")
            sys.exit(1)
    
    def connect_target_postgresql(self):
        """Connect to PostgreSQL"""
        if not HAS_POSTGRESQL:
            print("❌ psycopg2-binary not installed. Run: pip install psycopg2-binary")
            sys.exit(1)
        
        print(f"🐘 Connecting to PostgreSQL: {self.target_config['host']}")
        try:
            conn = psycopg2.connect(
                host=self.target_config['host'],
                port=self.target_config.get('port', 5432),
                database=self.target_config['database'],
                user=self.target_config['user'],
                password=self.target_config['password'],
                sslmode=self.target_config.get('ssl_mode', 'prefer')
            )
            conn.autocommit = False
            print("✅ PostgreSQL connected")
            return conn
        except Exception as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            sys.exit(1)
    
    def connect_target_mysql(self):
        """Connect to MySQL"""
        if not HAS_MYSQL:
            print("❌ pymysql not installed. Run: pip install pymysql")
            sys.exit(1)
        
        print(f"🐬 Connecting to MySQL: {self.target_config['host']}")
        try:
            conn = pymysql.connect(
                host=self.target_config['host'],
                port=self.target_config.get('port', 3306),
                database=self.target_config['database'],
                user=self.target_config['user'],
                password=self.target_config['password'],
                autocommit=False
            )
            print("✅ MySQL connected")
            return conn
        except Exception as e:
            print(f"❌ Error connecting to MySQL: {e}")
            sys.exit(1)
    
    def get_table_schema(self, source_conn, table_name):
        """Get table schema from SQLite"""
        cursor = source_conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        return [(col['name'], col['type']) for col in columns]
    
    def create_table_postgresql(self, table_name, schema):
        """Create table in PostgreSQL"""
        type_mapping = {
            'INTEGER': 'INTEGER',
            'TEXT': 'TEXT',
            'REAL': 'DOUBLE PRECISION',
            'BLOB': 'BYTEA',
            'NUMERIC': 'NUMERIC',
            'BOOLEAN': 'BOOLEAN',
            'DATETIME': 'TIMESTAMP',
            'DATE': 'DATE',
            'TIME': 'TIME'
        }
        
        columns = []
        for col_name, col_type in schema:
            pg_type = type_mapping.get(col_type.upper(), 'TEXT')
            columns.append(f'"{col_name}" {pg_type}')
        
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'
        
        try:
            self.target_cursor.execute(create_sql)
            self.target_conn.commit()
            print(f"  ✅ Table {table_name} created")
        except Exception as e:
            print(f"  ⚠️  Table {table_name} may already exist: {e}")
            self.target_conn.rollback()
    
    def create_table_mysql(self, table_name, schema):
        """Create table in MySQL"""
        type_mapping = {
            'INTEGER': 'INTEGER',
            'TEXT': 'TEXT',
            'REAL': 'DOUBLE',
            'BLOB': 'BLOB',
            'NUMERIC': 'DECIMAL',
            'BOOLEAN': 'BOOLEAN',
            'DATETIME': 'DATETIME',
            'DATE': 'DATE',
            'TIME': 'TIME'
        }
        
        columns = []
        for col_name, col_type in schema:
            mysql_type = type_mapping.get(col_type.upper(), 'TEXT')
            columns.append(f'`{col_name}` {mysql_type}')
        
        create_sql = f'CREATE TABLE IF NOT EXISTS `{table_name}` ({", ".join(columns)})'
        
        try:
            self.target_cursor.execute(create_sql)
            self.target_conn.commit()
            print(f"  ✅ Table {table_name} created")
        except Exception as e:
            print(f"  ⚠️  Table {table_name} may already exist: {e}")
            self.target_conn.rollback()
    
    def migrate_table(self, source_conn, table_name, db_type):
        """Migrate a single table"""
        # Get row count
        cursor = source_conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            print(f"  ⚠️  Table {table_name} is empty, skipping")
            return 0
        
        print(f"\n📊 Migrating {table_name} ({total_rows:,} rows)")
        
        # Get schema
        schema = self.get_table_schema(source_conn, table_name)
        column_names = [col[0] for col in schema]
        
        # Create table in target
        if db_type == 'postgresql':
            self.create_table_postgresql(table_name, schema)
        elif db_type == 'mysql':
            self.create_table_mysql(table_name, schema)
        
        # Migrate data in batches
        progress = ProgressBar(total_rows, f"  Copying {table_name}")
        
        offset = 0
        total_migrated = 0
        
        while offset < total_rows:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {self.batch_size} OFFSET {offset}")
            rows = cursor.fetchall()
            
            if not rows:
                break
            
            # Prepare batch insert
            if db_type == 'postgresql':
                placeholders = ','.join(['%s'] * len(column_names))
                quoted_cols = ','.join([f'"{col}"' for col in column_names])
                insert_sql = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'
            elif db_type == 'mysql':
                placeholders = ','.join(['%s'] * len(column_names))
                quoted_cols = ','.join([f'`{col}`' for col in column_names])
                insert_sql = f'INSERT INTO `{table_name}` ({quoted_cols}) VALUES ({placeholders})'
            
            # Convert rows to tuples
            values = [tuple(row) for row in rows]
            
            try:
                if db_type == 'postgresql':
                    psycopg2.extras.execute_batch(self.target_cursor, insert_sql, values)
                else:
                    self.target_cursor.executemany(insert_sql, values)
                
                self.target_conn.commit()
                total_migrated += len(rows)
                progress.update(total_migrated)
            except Exception as e:
                print(f"\n❌ Error inserting batch at offset {offset}: {e}")
                self.target_conn.rollback()
                # Try one by one
                for row in values:
                    try:
                        self.target_cursor.execute(insert_sql, row)
                        self.target_conn.commit()
                        total_migrated += 1
                        progress.update(total_migrated)
                    except Exception as e2:
                        print(f"\n⚠️  Failed to insert row: {e2}")
                        self.target_conn.rollback()
            
            offset += self.batch_size
        
        progress.finish()
        return total_migrated
    
    def migrate(self):
        """Run full migration"""
        print("=" * 80)
        print("🚀 DATABASE MIGRATION")
        print("=" * 80)
        
        # Connect to source
        source_conn = self.connect_source()
        
        # Get list of tables
        cursor = source_conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"\n📋 Found {len(tables)} tables: {', '.join(tables)}")
        
        # Connect to target
        db_type = self.target_config['type']
        if db_type == 'postgresql':
            self.target_conn = self.connect_target_postgresql()
        elif db_type == 'mysql':
            self.target_conn = self.connect_target_mysql()
        else:
            print(f"❌ Unsupported database type: {db_type}")
            sys.exit(1)
        
        self.target_cursor = self.target_conn.cursor()
        
        # Migrate each table
        total_records = 0
        start_time = time.time()
        
        for i, table in enumerate(tables, 1):
            print(f"\n[{i}/{len(tables)}] {table}")
            migrated = self.migrate_table(source_conn, table, db_type)
            total_records += migrated
        
        elapsed = time.time() - start_time
        
        # Cleanup
        source_conn.close()
        self.target_cursor.close()
        self.target_conn.close()
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ MIGRATION COMPLETE!")
        print("=" * 80)
        print(f"  Tables migrated: {len(tables)}")
        print(f"  Total records: {total_records:,}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Rate: {total_records/elapsed:.0f} records/s")
        print()


def main():
    parser = argparse.ArgumentParser(description='Direct database migration tool')
    parser.add_argument('--source-db', required=True, help='Path to source SQLite database')
    parser.add_argument('--target-type', required=True, choices=['postgresql', 'mysql'], help='Target database type')
    parser.add_argument('--target-host', required=True, help='Target database host')
    parser.add_argument('--target-port', type=int, help='Target database port')
    parser.add_argument('--target-database', required=True, help='Target database name')
    parser.add_argument('--target-user', required=True, help='Target database user')
    parser.add_argument('--target-password', required=True, help='Target database password')
    parser.add_argument('--ssl-mode', default='prefer', help='SSL mode for PostgreSQL')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for inserts')
    
    args = parser.parse_args()
    
    target_config = {
        'type': args.target_type,
        'host': args.target_host,
        'port': args.target_port,
        'database': args.target_database,
        'user': args.target_user,
        'password': args.target_password,
        'ssl_mode': args.ssl_mode
    }
    
    migrator = DatabaseMigrator(args.source_db, target_config, args.batch_size)
    migrator.migrate()


if __name__ == '__main__':
    main()
