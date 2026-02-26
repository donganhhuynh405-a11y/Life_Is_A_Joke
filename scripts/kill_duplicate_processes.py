#!/usr/bin/env python3
"""
Kill Duplicate Bot Processes
Останавливает дублирующиеся процессы бота
"""

import os
import sys
import psutil
import signal
import time
import argparse
from typing import List

def get_bot_processes() -> List[psutil.Process]:
    """Найти все процессы бота"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if 'python' in proc.info['name'].lower() and ('main.py' in cmdline or 'bot.py' in cmdline or 'trading-bot' in cmdline):
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    # Сортировать по времени создания (старые первыми)
    processes.sort(key=lambda p: p.create_time())
    return processes

def kill_process(proc: psutil.Process, force: bool = False) -> bool:
    """Остановить процесс"""
    try:
        if force:
            proc.kill()  # SIGKILL
        else:
            proc.terminate()  # SIGTERM
        
        # Ждать завершения
        proc.wait(timeout=10)
        return True
    except psutil.TimeoutExpired:
        return False
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True

def main():
    parser = argparse.ArgumentParser(description='Остановить дублирующиеся процессы бота')
    parser.add_argument('--dry-run', action='store_true', help='Показать что будет сделано без выполнения')
    parser.add_argument('--auto', action='store_true', help='Автоматический режим без подтверждения')
    parser.add_argument('--force', action='store_true', help='Принудительная остановка (SIGKILL)')
    parser.add_argument('--keep', type=int, default=0, help='Оставить N новейших процессов')
    args = parser.parse_args()
    
    print("🔍 Поиск процессов бота...")
    processes = get_bot_processes()
    
    if not processes:
        print("✓ Процессы бота не найдены")
        return 0
    
    print(f"✓ Найдено процессов: {len(processes)}")
    print()
    
    # Показать процессы
    print("Процессы:")
    for i, proc in enumerate(processes):
        try:
            mem_mb = proc.memory_info().rss / 1024 / 1024
            cpu_percent = proc.cpu_percent(interval=0.1)
            age_hours = (time.time() - proc.create_time()) / 3600
            cmdline = ' '.join(proc.cmdline()[:3]) if proc.cmdline() else 'N/A'
            print(f"  {i+1}. PID {proc.pid}: {mem_mb:.1f}MB, CPU {cpu_percent:.1f}%, Age {age_hours:.1f}h")
            print(f"      {cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print(f"  {i+1}. PID {proc.pid}: (недоступен)")
    print()
    
    # Определить какие останавливать
    if args.keep > 0:
        to_kill = processes[:-args.keep] if len(processes) > args.keep else []
        to_keep = processes[-args.keep:] if len(processes) > args.keep else processes
        print(f"Оставить новейших: {len(to_keep)}")
        print(f"Остановить старых: {len(to_kill)}")
    else:
        to_kill = processes
        to_keep = []
        print(f"Остановить всех: {len(to_kill)}")
    
    if not to_kill:
        print("✓ Нечего останавливать")
        return 0
    
    if args.dry_run:
        print("\n[DRY RUN] Будет остановлено:")
        for proc in to_kill:
            print(f"  - PID {proc.pid}")
        print("\nЗапустите без --dry-run для выполнения")
        return 0
    
    # Подтверждение
    if not args.auto:
        response = input(f"\nОстановить {len(to_kill)} процесс(ов)? [y/N]: ")
        if response.lower() != 'y':
            print("Отменено")
            return 1
    
    # Остановка
    print("\n🛑 Остановка процессов...")
    killed = 0
    failed = []
    
    for proc in to_kill:
        try:
            print(f"  Остановка PID {proc.pid}...", end=' ', flush=True)
            if kill_process(proc, force=args.force):
                print("✓")
                killed += 1
            else:
                print("✗ (timeout)")
                failed.append(proc.pid)
        except Exception as e:
            print(f"✗ ({e})")
            failed.append(proc.pid)
    
    print()
    print(f"✓ Остановлено: {killed}/{len(to_kill)}")
    
    if failed:
        print(f"⚠️  Не удалось остановить: {len(failed)}")
        for pid in failed:
            print(f"    PID {pid}")
        print("\n💡 Попробуйте с флагом --force")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
