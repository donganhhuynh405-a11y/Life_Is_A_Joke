#!/usr/bin/env python3
"""
Elite AI Diagnostic Tool
Checks why Elite AI might not be working
"""

import os
import sys


def check_env_vars():
    """Check if Elite AI environment variables are set"""
    print("=" * 60)
    print("1. Checking .env Elite AI Configuration:")
    print("=" * 60)

    flags = [
        'ENABLE_ELITE_RISK_MANAGEMENT',
        'ENABLE_REGIME_DETECTION',
        'ENABLE_MTF_ANALYSIS',
        'ENABLE_ELITE_POSITION_MGMT'
    ]

    all_enabled = True
    for flag in flags:
        value = os.getenv(flag, 'NOT SET')
        status = "✓" if value.lower() == 'true' else "✗"
        print(f"  {status} {flag} = {value}")
        if value.lower() != 'true':
            all_enabled = False

    if all_enabled:
        print("\n✓ All Elite AI flags are ENABLED")
    else:
        print("\n✗ Some Elite AI flags are NOT ENABLED")
        print("   Fix: Set all flags to 'true' in .env file")

    return all_enabled


def check_files():
    """Check if Elite AI files exist"""
    print("\n" + "=" * 60)
    print("2. Checking Elite AI Files:")
    print("=" * 60)

    base_path = "/opt/trading-bot"
    files = [
        "src/core/elite_bot_integrator.py",
        "src/ml/advanced_risk_manager.py",
        "src/ml/market_regime_detector.py",
        "src/ml/multi_timeframe_analyzer.py",
        "src/ml/elite_position_manager.py"
    ]

    all_exist = True
    for file_path in files:
        full_path = os.path.join(base_path, file_path)
        exists = os.path.exists(full_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n✓ All Elite AI files are present")
    else:
        print("\n✗ Some Elite AI files are MISSING")
        print("   Fix: Run deployment script to copy files")

    return all_exist


def check_bot_integration():
    """Check if bot.py has Elite AI integration code"""
    print("\n" + "=" * 60)
    print("3. Checking bot.py Integration:")
    print("=" * 60)

    bot_file = "/opt/trading-bot/src/core/bot.py"

    if not os.path.exists(bot_file):
        print("  ✗ bot.py not found at", bot_file)
        return False

    with open(bot_file, 'r') as f:
        content = f.read()

    checks = [
        ('elite_integrator', 'EliteBotIntegrator import'),
        ('EliteBotIntegrator(', 'EliteBotIntegrator initialization'),
        ('detect_market_regime', 'Regime detection integration'),
        ('validate_with_mtf', 'MTF analysis integration'),
    ]

    all_present = True
    for search_str, description in checks:
        present = search_str in content
        status = "✓" if present else "✗"
        print(f"  {status} {description}")
        if not present:
            all_present = False

    if all_present:
        print("\n✓ bot.py has Elite AI integration code")
    else:
        print("\n✗ bot.py is MISSING Elite AI integration")
        print("   Fix: Deploy updated bot.py file")

    return all_present


def check_imports():
    """Try to import Elite AI modules"""
    print("\n" + "=" * 60)
    print("4. Testing Module Imports:")
    print("=" * 60)

    sys.path.insert(0, '/opt/trading-bot')

    modules = [
        ('src.core.elite_bot_integrator', 'EliteBotIntegrator'),
        ('src.ml.advanced_risk_manager', 'AdvancedRiskManager'),
        ('src.ml.market_regime_detector', 'MarketRegimeDetector'),
        ('src.ml.multi_timeframe_analyzer', 'MultiTimeframeAnalyzer'),
        ('src.ml.elite_position_manager', 'ElitePositionManager'),
    ]

    all_importable = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {class_name} from {module_name}")
        except Exception as e:
            print(f"  ✗ {class_name} from {module_name}: {e}")
            all_importable = False

    if all_importable:
        print("\n✓ All Elite AI modules can be imported")
    else:
        print("\n✗ Some Elite AI modules CANNOT be imported")
        print("   Fix: Check for syntax errors or missing dependencies")

    return all_importable


def main():
    print("\n🔍 ELITE AI DIAGNOSTIC TOOL")
    print("=" * 60)

    results = {
        'env_vars': check_env_vars(),
        'files': check_files(),
        'integration': check_bot_integration(),
        'imports': check_imports()
    }

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY:")
    print("=" * 60)

    for key, value in results.items():
        status = "✓ PASS" if value else "✗ FAIL"
        print(f"  {status}: {key}")

    if all(results.values()):
        print("\n✅ All checks PASSED - Elite AI should be working!")
        print("   If it's still not working, check bot logs for errors:")
        print("   sudo journalctl -u trading-bot -n 100 | grep -i elite")
    else:
        print("\n❌ Some checks FAILED - follow the fixes above")
        print("   Then run the deployment script to update files")

    print("=" * 60)


if __name__ == "__main__":
    main()
