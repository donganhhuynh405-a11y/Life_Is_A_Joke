#!/usr/bin/env python3
"""
Standalone Data Import Script
Imports data from JSON/SQL to PostgreSQL/MySQL/MongoDB.
"""

import argparse
import json
import gzip
import sys

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


def import_json_to_postgresql(json_path, db_config):
    """Import JSON data to PostgreSQL"""
    if not HAS_POSTGRESQL:
        print("❌ psycopg2-binary not installed")
        sys.exit(1)
    
    print(f"📥 Importing to PostgreSQL: {db_config['host']}")
    
    # Load data
    if json_path.endswith('.gz'):
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Connect
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config.get('port', 5432),
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    cursor = conn.cursor()
    
    total_imported = 0
    
    for table_name, rows in data.items():
        if not rows:
            continue
        
        print(f"  Importing {table_name}...", end=' ', flush=True)
        
        # Create table
        sample = rows[0]
        columns = []
        for key, value in sample.items():
            if isinstance(value, int):
                col_type = 'INTEGER'
            elif isinstance(value, float):
                col_type = 'DOUBLE PRECISION'
            elif isinstance(value, bool):
                col_type = 'BOOLEAN'
            else:
                col_type = 'TEXT'
            columns.append(f'"{key}" {col_type}')
        
        create_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(columns)})'
        cursor.execute(create_sql)
        conn.commit()
        
        # Insert data
        col_names = list(sample.keys())
        placeholders = ','.join(['%s'] * len(col_names))
        insert_sql = f'INSERT INTO "{table_name}" ({",".join([f\'"{col}"\' for col in col_names])}) VALUES ({placeholders})'
        
        values = [tuple(row[col] for col in col_names) for row in rows]
        psycopg2.extras.execute_batch(cursor, insert_sql, values)
        conn.commit()
        
        total_imported += len(rows)
        print(f"{len(rows):,} rows")
    
    cursor.close()
    conn.close()
    
    print(f"✅ Import complete! {total_imported:,} rows")


def import_json_to_mysql(json_path, db_config):
    """Import JSON data to MySQL"""
    if not HAS_MYSQL:
        print("❌ pymysql not installed")
        sys.exit(1)
    
    print(f"📥 Importing to MySQL: {db_config['host']}")
    
    # Load data
    if json_path.endswith('.gz'):
        with gzip.open(json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    # Connect
    conn = pymysql.connect(
        host=db_config['host'],
        port=db_config.get('port', 3306),
        database=db_config['database'],
        user=db_config['user'],
        password=db_config['password']
    )
    cursor = conn.cursor()
    
    total_imported = 0
    
    for table_name, rows in data.items():
        if not rows:
            continue
        
        print(f"  Importing {table_name}...", end=' ', flush=True)
        
        # Create table
        sample = rows[0]
        columns = []
        for key, value in sample.items():
            if isinstance(value, int):
                col_type = 'INTEGER'
            elif isinstance(value, float):
                col_type = 'DOUBLE'
            elif isinstance(value, bool):
                col_type = 'BOOLEAN'
            else:
                col_type = 'TEXT'
            columns.append(f'`{key}` {col_type}')
        
        create_sql = f'CREATE TABLE IF NOT EXISTS `{table_name}` ({", ".join(columns)})'
        cursor.execute(create_sql)
        conn.commit()
        
        # Insert data
        col_names = list(sample.keys())
        placeholders = ','.join(['%s'] * len(col_names))
        insert_sql = f'INSERT INTO `{table_name}` ({",".join([f"`{col}`" for col in col_names])}) VALUES ({placeholders})'
        
        values = [tuple(row[col] for col in col_names) for row in rows]
        cursor.executemany(insert_sql, values)
        conn.commit()
        
        total_imported += len(rows)
        print(f"{len(rows):,} rows")
    
    cursor.close()
    conn.close()
    
    print(f"✅ Import complete! {total_imported:,} rows")


def main():
    parser = argparse.ArgumentParser(description='Import data to database')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--db-type', required=True, choices=['postgresql', 'mysql'], help='Target database type')
    parser.add_argument('--host', required=True, help='Database host')
    parser.add_argument('--port', type=int, help='Database port')
    parser.add_argument('--database', required=True, help='Database name')
    parser.add_argument('--user', required=True, help='Database user')
    parser.add_argument('--password', required=True, help='Database password')
    
    args = parser.parse_args()
    
    db_config = {
        'host': args.host,
        'port': args.port,
        'database': args.database,
        'user': args.user,
        'password': args.password
    }
    
    if args.db_type == 'postgresql':
        import_json_to_postgresql(args.input, db_config)
    elif args.db_type == 'mysql':
        import_json_to_mysql(args.input, db_config)


if __name__ == '__main__':
    main()
