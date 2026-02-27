#!/usr/bin/env python3
"""
Backup SQLite database to Beget S3 storage
"""

import argparse
import boto3
import gzip
import shutil
import os
from datetime import datetime
import sys


def backup_to_s3(db_path, bucket, endpoint, access_key, secret_key, compress=True):
    """Backup database to Beget S3"""

    print("=" * 80)
    print("🚀 BACKUP TO BEGET S3")
    print("=" * 80)

    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Error: Database not found: {db_path}")
        sys.exit(1)

    # Create backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'backup_{timestamp}.db'

    print(f"\n📦 Creating backup: {backup_name}")
    print(f"   Source: {db_path}")

    # Copy database
    temp_path = f'/tmp/{backup_name}'
    shutil.copy2(db_path, temp_path)

    file_size = os.path.getsize(temp_path)
    print(f"   Size: {file_size / 1024 / 1024:.2f} MB")

    # Compress if requested
    if compress:
        print("\n🗜️  Compressing...")
        compressed_path = f'{temp_path}.gz'

        with open(temp_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(temp_path)
        temp_path = compressed_path
        backup_name = f'{backup_name}.gz'

        compressed_size = os.path.getsize(temp_path)
        print(f"   Compressed: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"   Ratio: {compressed_size / file_size * 100:.1f}%")

    # Upload to S3
    print("\n☁️  Uploading to Beget S3...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Bucket: {bucket}")

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=f'https://{endpoint}',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

        s3_key = f'backups/{backup_name}'
        s3.upload_file(temp_path, bucket, s3_key)

        print("\n✅ Backup uploaded successfully!")
        print(f"   Location: s3://{bucket}/{s3_key}")
        print(f"   URL: https://{endpoint}/{bucket}/{s3_key}")

    except Exception as e:
        print(f"\n❌ Error uploading to S3: {e}")
        sys.exit(1)

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    print("\n" + "=" * 80)
    print("✅ BACKUP COMPLETE!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Backup SQLite database to Beget S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --db /var/lib/trading-bot/trading_bot.db \\
           --bucket 443c60e2203e-betacassiopeiae \\
           --access-key YOUR_KEY \\
           --secret-key YOUR_SECRET

  %(prog)s --db trading_bot.db \\
           --bucket my-bucket \\
           --access-key KEY \\
           --secret-key SECRET \\
           --no-compress
        """
    )

    parser.add_argument(
        '--db',
        required=True,
        help='Path to SQLite database file'
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name (e.g., 443c60e2203e-betacassiopeiae)'
    )
    parser.add_argument(
        '--endpoint',
        default='s3.ru1.storage.beget.cloud',
        help='S3 endpoint (default: s3.ru1.storage.beget.cloud)'
    )
    parser.add_argument(
        '--access-key',
        required=True,
        help='S3 access key'
    )
    parser.add_argument(
        '--secret-key',
        required=True,
        help='S3 secret key'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Do not compress backup (faster but larger)'
    )

    args = parser.parse_args()

    backup_to_s3(
        args.db,
        args.bucket,
        args.endpoint,
        args.access_key,
        args.secret_key,
        compress=not args.no_compress
    )


if __name__ == '__main__':
    main()
