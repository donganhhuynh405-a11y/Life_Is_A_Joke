#!/usr/bin/env python3
"""
Find Large Files - Real Disk Usage Analysis
Finds what actually uses disk space
"""

import os
import sys
import subprocess
from typing import List, Tuple


def get_directory_size(path: str) -> int:
    """Get directory size using du command"""
    try:
        result = subprocess.run(
            ['du', '-sb', path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            size = int(result.stdout.split()[0])
            return size
    except BaseException:
        pass
    return 0


def format_size(size: int) -> str:
    """Format size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def find_large_directories(base_path: str = '/', min_size_mb: int = 100,
                           max_depth: int = 3) -> List[Tuple[str, int]]:
    """Find large directories"""
    print(f"\n🔍 Searching for directories larger than {min_size_mb}MB...")
    print(f"   Base path: {base_path}")
    print(f"   Max depth: {max_depth}")

    large_dirs = []

    try:
        # Use find to get directories up to max_depth
        result = subprocess.run(
            ['find', base_path, '-maxdepth', str(max_depth), '-type', 'd'],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            dirs = result.stdout.strip().split('\n')
            total = len(dirs)

            for i, directory in enumerate(dirs, 1):
                if i % 10 == 0:
                    print(f"   Progress: {i}/{total} directories scanned...")

                try:
                    size = get_directory_size(directory)
                    if size > min_size_mb * 1024 * 1024:
                        large_dirs.append((directory, size))
                except BaseException:
                    continue
    except Exception as e:
        print(f"   ⚠️  Error scanning directories: {e}")

    # Sort by size descending
    large_dirs.sort(key=lambda x: x[1], reverse=True)
    return large_dirs


def find_large_files(base_path: str = '/', min_size_mb: int = 50) -> List[Tuple[str, int]]:
    """Find large files"""
    print(f"\n📄 Searching for files larger than {min_size_mb}MB...")

    large_files = []

    try:
        # Use find to get large files
        result = subprocess.run(
            ['find', base_path, '-type', 'f', '-size', f'+{min_size_mb}M'],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            files = result.stdout.strip().split('\n')
            files = [f for f in files if f]

            for filepath in files:
                try:
                    size = os.path.getsize(filepath)
                    large_files.append((filepath, size))
                except BaseException:
                    continue
    except Exception as e:
        print(f"   ⚠️  Error scanning files: {e}")

    # Sort by size descending
    large_files.sort(key=lambda x: x[1], reverse=True)
    return large_files


def analyze_disk_usage(
        base_path: str = '/opt/trading-bot',
        min_dir_size: int = 100,
        min_file_size: int = 50):
    """Main analysis function"""
    print("=" * 60)
    print("🔍 DISK USAGE ANALYSIS")
    print("=" * 60)

    if not os.path.exists(base_path):
        print(f"\n❌ Path does not exist: {base_path}")
        print("\nTrying with root path /")
        base_path = '/'

    # Find large directories
    print("\n📁 TOP LARGE DIRECTORIES:")
    print("-" * 60)

    large_dirs = find_large_directories(base_path, min_dir_size, max_depth=4)

    if large_dirs:
        print(f"\nFound {len(large_dirs)} directories larger than {min_dir_size}MB:\n")
        for i, (path, size) in enumerate(large_dirs[:20], 1):
            print(f"{i:2}. {format_size(size):>12} - {path}")
    else:
        print(f"\n✓ No directories larger than {min_dir_size}MB found")

    # Find large files
    print("\n\n📄 TOP LARGE FILES:")
    print("-" * 60)

    large_files = find_large_files(base_path, min_file_size)

    if large_files:
        print(f"\nFound {len(large_files)} files larger than {min_file_size}MB:\n")
        for i, (path, size) in enumerate(large_files[:30], 1):
            print(f"{i:2}. {format_size(size):>12} - {path}")
    else:
        print(f"\n✓ No files larger than {min_file_size}MB found")

    # Recommendations
    print("\n\n💡 RECOMMENDATIONS:")
    print("-" * 60)

    recommendations = []

    # Check for Docker
    docker_dirs = [d for d, s in large_dirs if 'docker' in d.lower()]
    if docker_dirs:
        recommendations.append("🐳 Docker: docker system prune -af --volumes")

    # Check for logs
    log_dirs = [d for d, s in large_dirs if 'log' in d.lower()]
    if log_dirs:
        recommendations.append("📝 Logs: journalctl --vacuum-size=100M")
        recommendations.append("📝 Logs: find /var/log -type f -name '*.gz' -delete")

    # Check for .git
    git_dirs = [d for d, s in large_dirs if '.git' in d]
    if git_dirs:
        recommendations.append("🔧 Git: cd /opt/trading-bot && git gc --aggressive --prune=now")

    # Check for cache
    cache_files = [f for f, s in large_files if 'cache' in f.lower() or '__pycache__' in f]
    if cache_files:
        recommendations.append(
            "🗑️  Cache: find / -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null")
        recommendations.append("🗑️  Cache: pip cache purge")

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n✓ No specific recommendations at this time")

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Find large files and directories')
    parser.add_argument(
        '--path',
        default='/opt/trading-bot',
        help='Base path to scan (default: /opt/trading-bot)')
    parser.add_argument('--min-dir-size', type=int, default=100,
                        help='Minimum directory size in MB (default: 100)')
    parser.add_argument('--min-file-size', type=int, default=50,
                        help='Minimum file size in MB (default: 50)')

    args = parser.parse_args()

    try:
        analyze_disk_usage(args.path, args.min_dir_size, args.min_file_size)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
