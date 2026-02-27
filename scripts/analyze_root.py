#!/usr/bin/env python3
"""
/root Directory Analyzer
Analyzes the /root directory to find what's using 13.76GB of space
"""

import os
import subprocess
import sys


def get_size(path):
    """Get size of file or directory in bytes"""
    try:
        if os.path.isfile(path):
            return os.path.getsize(path)
        elif os.path.isdir(path):
            result = subprocess.run(
                ['du', '-sb', path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return int(result.stdout.split()[0])
    except Exception:
        pass
    return 0


def format_size(bytes_size):
    """Format bytes to human readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:6.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:6.2f} PB"


def analyze_root():
    """Analyze /root directory"""
    root_path = '/root'

    if not os.path.exists(root_path):
        print(f"❌ {root_path} does not exist!")
        return

    if not os.access(root_path, os.R_OK):
        print(f"❌ No read access to {root_path}. Run with sudo!")
        return

    print("=" * 60)
    print("📊 /root DIRECTORY ANALYSIS")
    print("=" * 60)
    print()

    # Get total size
    total_size = get_size(root_path)
    print(f"📁 Total size: {format_size(total_size)}")
    print()

    # Analyze subdirectories
    print("─" * 60)
    print("📂 TOP SUBDIRECTORIES:")
    print("─" * 60)
    print()

    dirs = []
    try:
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            if os.path.isdir(item_path):
                size = get_size(item_path)
                dirs.append((size, item_path))
    except PermissionError as e:
        print(f"⚠️  Permission denied: {e}")

    # Sort by size
    dirs.sort(reverse=True)

    # Print top 20 directories
    for i, (size, path) in enumerate(dirs[:20], 1):
        print(f" {i:2d}. {format_size(size):>12s} - {path}")

    if not dirs:
        print("⚠️  No subdirectories found or no access")

    print()

    # Find large files
    print("─" * 60)
    print("📄 LARGE FILES (>50MB):")
    print("─" * 60)
    print()

    large_files = []
    try:
        for root, dirs, files in os.walk(root_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    if size > 50 * 1024 * 1024:  # >50MB
                        large_files.append((size, file_path))
                except BaseException:
                    pass
    except Exception as e:
        print(f"⚠️  Error scanning files: {e}")

    large_files.sort(reverse=True)

    if large_files:
        for i, (size, path) in enumerate(large_files[:30], 1):
            print(f" {i:2d}. {format_size(size):>12s} - {path}")
    else:
        print("✓ No files larger than 50MB found")

    print()

    # Recommendations
    print("─" * 60)
    print("💡 RECOMMENDATIONS:")
    print("─" * 60)
    print()

    recommendations = []

    # Check common cache directories
    cache_dirs = {
        '/root/.cache': 'All caches',
        '/root/.npm': 'npm cache',
        '/root/.pip': 'pip cache',
        '/root/.cargo': 'Cargo/Rust cache',
        '/root/.local/share/Trash': 'Trash',
    }

    for cache_dir, name in cache_dirs.items():
        if os.path.exists(cache_dir):
            size = get_size(cache_dir)
            if size > 100 * 1024 * 1024:  # >100MB
                recommendations.append({
                    'priority': '🔴' if size > 1024 * 1024 * 1024 else '🟡',
                    'path': cache_dir,
                    'name': name,
                    'size': size,
                    'command': f"sudo rm -rf {cache_dir}/*"
                })

    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['priority']} {rec['path']} ({format_size(rec['size'])})")
            print(f"   {rec['name']}")
            print(f"   Clean: {rec['command']}")
            print()

    # General cleanup commands
    print("🧹 GENERAL CLEANUP COMMANDS:")
    print()
    print("# Clear all caches (safe):")
    print("sudo rm -rf /root/.cache/*")
    print()
    print("# Clear npm cache:")
    print("sudo npm cache clean --force 2>/dev/null")
    print()
    print("# Clear pip cache:")
    print("sudo pip cache purge 2>/dev/null")
    print()
    print("# Clear trash:")
    print("sudo rm -rf /root/.local/share/Trash/*")
    print()
    print("# Find and remove core dumps:")
    print("sudo find /root -name 'core.*' -delete")
    print()

    # Calculate potential savings
    total_cache = sum(rec['size'] for rec in recommendations)
    if total_cache > 0:
        print(f"💾 Potential space to free: {format_size(total_cache)}")
        print()

    print("=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    if os.geteuid() != 0:
        print("⚠️  This script should be run with sudo for full access to /root")
        print("   Run: sudo python3 scripts/analyze_root.py")
        print()

    try:
        analyze_root()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
