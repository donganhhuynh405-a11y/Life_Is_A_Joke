#!/usr/bin/env python3
"""
System-wide Disk Usage Analyzer
Finds what's really using disk space across the entire system
"""

import os
import subprocess
import sys
from pathlib import Path


def get_disk_usage():
    """Get overall disk usage"""
    try:
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            if len(parts) >= 5:
                return {
                    'total': parts[1],
                    'used': parts[2],
                    'available': parts[3],
                    'percent': parts[4]
                }
    except Exception as e:
        print(f"Error getting disk usage: {e}")
    return None


def get_directory_size(path):
    """Get size of directory in bytes using du"""
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
    except Exception:
        pass
    return 0


def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:8.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:8.2f} PB"


def scan_system_directories():
    """Scan critical system directories"""
    print("\n" + "="*60)
    print("📊 SYSTEM DISK ANALYSIS")
    print("="*60)
    
    # Get overall disk usage
    disk_info = get_disk_usage()
    if disk_info:
        print(f"\n💾 Overall Disk Usage:")
        print(f"   Total: {disk_info['total']}")
        print(f"   Used: {disk_info['used']} ({disk_info['percent']})")
        print(f"   Available: {disk_info['available']}")
    
    print("\n" + "─"*60)
    print("📁 TOP SYSTEM DIRECTORIES:")
    print("─"*60)
    
    # Directories to check
    check_dirs = [
        '/var/log',
        '/var/lib/docker',
        '/var/cache',
        '/var/tmp',
        '/tmp',
        '/root',
        '/opt',
        '/home',
        '/usr/share',
        '/var/lib/apt',
    ]
    
    dir_sizes = []
    
    print("\n🔍 Scanning system directories...")
    for directory in check_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            try:
                print(f"   Checking {directory}...", end='', flush=True)
                size = get_directory_size(directory)
                if size > 0:
                    dir_sizes.append((directory, size))
                    print(f" {format_size(size)}")
                else:
                    print(" (empty or no access)")
            except Exception as e:
                print(f" (error: {e})")
    
    # Sort by size
    dir_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📊 Results (sorted by size):")
    print()
    for i, (directory, size) in enumerate(dir_sizes[:15], 1):
        print(f"{i:2d}. {format_size(size)} - {directory}")
    
    # Recommendations
    print("\n" + "─"*60)
    print("💡 RECOMMENDATIONS:")
    print("─"*60)
    print()
    
    recommendations = []
    
    for directory, size in dir_sizes:
        size_gb = size / (1024**3)
        
        if '/var/log' in directory and size_gb > 1:
            recommendations.append({
                'priority': 1,
                'dir': directory,
                'size': size,
                'msg': f"🔴 {directory} is using {format_size(size)}",
                'cmd': [
                    "Clean system logs:",
                    "  sudo journalctl --vacuum-size=100M",
                    "  sudo find /var/log -name '*.gz' -delete",
                    "  sudo find /var/log -name '*.old' -delete"
                ]
            })
        
        if '/var/lib/docker' in directory and size_gb > 1:
            recommendations.append({
                'priority': 1,
                'dir': directory,
                'size': size,
                'msg': f"🔴 {directory} is using {format_size(size)}",
                'cmd': [
                    "Clean Docker:",
                    "  docker system prune -af --volumes"
                ]
            })
        
        if '/var/cache' in directory and size_gb > 0.5:
            recommendations.append({
                'priority': 2,
                'dir': directory,
                'size': size,
                'msg': f"🟡 {directory} is using {format_size(size)}",
                'cmd': [
                    "Clean APT cache:",
                    "  sudo apt clean",
                    "  sudo apt autoremove -y"
                ]
            })
        
        if directory in ['/tmp', '/var/tmp'] and size_gb > 0.5:
            recommendations.append({
                'priority': 2,
                'dir': directory,
                'size': size,
                'msg': f"🟡 {directory} is using {format_size(size)}",
                'cmd': [
                    f"Clean temporary files:",
                    f"  sudo rm -rf {directory}/*"
                ]
            })
    
    # Sort by priority
    recommendations.sort(key=lambda x: (x['priority'], -x['size']))
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['msg']}")
            for cmd in rec['cmd']:
                print(f"   {cmd}")
            print()
        
        total_potential = sum(r['size'] for r in recommendations)
        print(f"💾 Potential space to free: {format_size(total_potential)}")
    else:
        print("✓ No major issues found")
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)
    print()


def main():
    """Main function"""
    if os.geteuid() != 0:
        print("⚠️  Warning: Running without root privileges.")
        print("   Some directories may not be accessible.")
        print("   Run with 'sudo' for complete analysis.")
        print()
    
    try:
        scan_system_directories()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
