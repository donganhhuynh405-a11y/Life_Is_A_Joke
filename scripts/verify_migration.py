#!/usr/bin/env python3
"""
Migration Verification Script
Verifies data integrity after migration.
"""

import argparse
import sqlite3
import sys

try:
    import psycopg2
    HAS_POSTGRESQL = True
except ImportError:
    HAS_POSTGRESQL = False

try:
    import pymysql
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


def verify_postgresql(source_db, target_config):
    """Verify PostgreSQL migration"""
    if not HAS_POSTGRESQL:
        print("❌ psycopg2-binary not installed")
        sys.exit(1)
    
    print("🔍 VERIFICATION: SQLite → PostgreSQL")
    print("=" * 60)
    
    # Connect to source
    source_conn = sqlite3.connect(source_db)
    source_cursor = source_conn.cursor()
    
    # Connect to target
    target_conn = psycopg2.connect(
        host=target_config['host'],
        port=target_config.get('port', 5432),
        database=target_config['database'],
        user=target_config.get('user', 'postgres'),
        password=target_config.get('password', '')
    )
    target_cursor = target_conn.cursor()
    
    # Get tables from source
    source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in source_cursor.fetchall()]
    
    print(f"\n📋 Checking {len(tables)} tables...\n")
    
    all_match = True
    total_source = 0
    total_target = 0
    
    for table in tables:
        # Count source rows
        source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
        source_count = source_cursor.fetchone()[0]
        total_source += source_count
        
        # Count target rows
        try:
            target_cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
            target_count = target_cursor.fetchone()[0]
            total_target += target_count
        except:
            target_count = 0
        
        # Compare
        if source_count == target_count:
            print(f"  ✅ {table}: {source_count:,} ↔ {target_count:,}")
        else:
            print(f"  ❌ {table}: {source_count:,} ↔ {target_count:,} (MISMATCH!)")
            all_match = False
    
    source_conn.close()
    target_conn.close()
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ VERIFICATION PASSED!")
        print(f"  Total records: {total_source:,}")
        print("  All tables match!")
    else:
        print("❌ VERIFICATION FAILED!")
        print("  Some tables have mismatched row counts")
        sys.exit(1)


def verify_mysql(source_db, target_config):
    """Verify MySQL migration"""
    if not HAS_MYSQL:
        print("❌ pymysql not installed")
        sys.exit(1)
    
    print("🔍 VERIFICATION: SQLite → MySQL")
    print("=" * 60)
    
    # Connect to source
    source_conn = sqlite3.connect(source_db)
    source_cursor = source_conn.cursor()
    
    # Connect to target
    target_conn = pymysql.connect(
        host=target_config['host'],
        port=target_config.get('port', 3306),
        database=target_config['database'],
        user=target_config.get('user', 'root'),
        password=target_config.get('password', '')
    )
    target_cursor = target_conn.cursor()
    
    # Get tables from source
    source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in source_cursor.fetchall()]
    
    print(f"\n📋 Checking {len(tables)} tables...\n")
    
    all_match = True
    total_source = 0
    total_target = 0
    
    for table in tables:
        # Count source rows
        source_cursor.execute(f"SELECT COUNT(*) FROM {table}")
        source_count = source_cursor.fetchone()[0]
        total_source += source_count
        
        # Count target rows
        try:
            target_cursor.execute(f'SELECT COUNT(*) FROM `{table}`')
            target_count = target_cursor.fetchone()[0]
            total_target += target_count
        except:
            target_count = 0
        
        # Compare
        if source_count == target_count:
            print(f"  ✅ {table}: {source_count:,} ↔ {target_count:,}")
        else:
            print(f"  ❌ {table}: {source_count:,} ↔ {target_count:,} (MISMATCH!)")
            all_match = False
    
    source_conn.close()
    target_conn.close()
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ VERIFICATION PASSED!")
        print(f"  Total records: {total_source:,}")
        print("  All tables match!")
    else:
        print("❌ VERIFICATION FAILED!")
        print("  Some tables have mismatched row counts")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Verify database migration')
    parser.add_argument('--source', required=True, help='Source SQLite database')
    parser.add_argument('--target-type', required=True, choices=['postgresql', 'mysql'], help='Target database type')
    parser.add_argument('--target-host', required=True, help='Target database host')
    parser.add_argument('--target-port', type=int, help='Target database port')
    parser.add_argument('--target-database', required=True, help='Target database name')
    parser.add_argument('--target-user', help='Target database user')
    parser.add_argument('--target-password', help='Target database password')
    
    args = parser.parse_args()
    
    target_config = {
        'host': args.target_host,
        'port': args.target_port,
        'database': args.target_database,
        'user': args.target_user,
        'password': args.target_password
    }
    
    if args.target_type == 'postgresql':
        verify_postgresql(args.source, target_config)
    elif args.target_type == 'mysql':
        verify_mysql(args.source, target_config)


if __name__ == '__main__':
    main()
