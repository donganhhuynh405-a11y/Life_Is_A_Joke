#!/usr/bin/env python3
"""
Standalone Data Export Script
Exports data from SQLite to JSON/SQL/CSV formats.
Minimal dependencies, memory-efficient.
"""

import argparse
import sqlite3
import json
import csv
import gzip
from pathlib import Path


def export_to_json(db_path, output_path, compress=False):
    """Export database to JSON format"""
    print(f"📤 Exporting to JSON: {output_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    data = {}
    total_rows = 0

    for table in tables:
        print(f"  Exporting {table}...", end=' ', flush=True)
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        data[table] = [dict(row) for row in rows]
        print(f"{len(rows):,} rows")
        total_rows += len(rows)

    # Write output
    if compress:
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    conn.close()

    file_size = Path(output_path).stat().st_size
    print("\n✅ Export complete!")
    print(f"  Tables: {len(tables)}")
    print(f"  Rows: {total_rows:,}")
    print(f"  Size: {file_size / 1024 / 1024:.1f} MB")


def export_to_sql(db_path, output_path, target_db='postgresql'):
    """Export database to SQL dump"""
    print(f"📤 Exporting to SQL ({target_db}): {output_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("-- SQL Dump\n")
        f.write(f"-- Target: {target_db}\n\n")

        for table in tables:
            print(f"  Exporting {table}...", end=' ', flush=True)

            # Get schema
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            # Create table statement
            if target_db == 'postgresql':
                col_defs = []
                for col in columns:
                    col_type = col['type'].upper()
                    if col_type == 'INTEGER':
                        col_type = 'INTEGER'
                    elif col_type == 'REAL':
                        col_type = 'DOUBLE PRECISION'
                    elif col_type == 'TEXT':
                        col_type = 'TEXT'
                    else:
                        col_type = 'TEXT'
                    col_defs.append(f'"{col["name"]}" {col_type}')

                f.write(f'CREATE TABLE IF NOT EXISTS "{table}" (\n  ')
                f.write(',\n  '.join(col_defs))
                f.write('\n);\n\n')

            # Export data
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()

            if rows:
                col_names = [col['name'] for col in columns]

                for row in rows:
                    values = []
                    for val in row:
                        if val is None:
                            values.append('NULL')
                        elif isinstance(val, (int, float)):
                            values.append(str(val))
                        else:
                            # Escape single quotes
                            escaped = str(val).replace("'", "''")
                            values.append(f"'{escaped}'")

                    if target_db == 'postgresql':
                        quoted_cols = ', '.join([f'"{c}"' for c in col_names])
                        f.write(
                            f'INSERT INTO "{table}" ({quoted_cols}) VALUES ({
                                ", ".join(values)});\n')
                    else:
                        quoted_cols = ', '.join([f'`{c}`' for c in col_names])
                        f.write(
                            f'INSERT INTO `{table}` ({quoted_cols}) VALUES ({
                                ", ".join(values)});\n')

            print(f"{len(rows):,} rows")
            f.write('\n')

    conn.close()
    print("✅ SQL export complete!")


def export_to_csv(db_path, output_dir):
    """Export database to CSV files (one per table)"""
    print(f"📤 Exporting to CSV: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        print(f"  Exporting {table}...", end=' ', flush=True)

        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()

        if rows:
            csv_file = output_path / f"{table}.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                for row in rows:
                    writer.writerow(dict(row))

            print(f"{len(rows):,} rows")
        else:
            print("empty")

    conn.close()
    print("✅ CSV export complete!")


def main():
    parser = argparse.ArgumentParser(description='Export SQLite database')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--output', required=True, help='Output file/directory')
    parser.add_argument(
        '--format',
        choices=[
            'json',
            'sql',
            'csv'],
        default='json',
        help='Export format')
    parser.add_argument('--compress', action='store_true', help='Compress JSON output (gzip)')
    parser.add_argument(
        '--target',
        choices=[
            'postgresql',
            'mysql'],
        default='postgresql',
        help='Target DB for SQL export')

    args = parser.parse_args()

    if args.format == 'json':
        export_to_json(args.db, args.output, args.compress)
    elif args.format == 'sql':
        export_to_sql(args.db, args.output, args.target)
    elif args.format == 'csv':
        export_to_csv(args.db, args.output)


if __name__ == '__main__':
    main()
