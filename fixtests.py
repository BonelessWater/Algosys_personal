#!/usr/bin/env python3
"""
Quick fix script for AlgoSystem tests
"""

import os
import shutil
import sys
from pathlib import Path

def fix_test_structure():
    """Fix the test directory structure and files."""
    
    print("🔧 Fixing AlgoSystem test structure...")
    
    # 1. Create tests directory at root if it doesn't exist
    root_tests_dir = Path("tests")
    if not root_tests_dir.exists():
        root_tests_dir.mkdir()
        print("✅ Created tests/ directory at root level")
    
    # 2. Move tests from algosystem/tests/ to tests/
    algosystem_tests_dir = Path("algosystem/tests")
    if algosystem_tests_dir.exists():
        print("📁 Moving test files from algosystem/tests/ to tests/...")
        
        for test_file in algosystem_tests_dir.glob("*.py"):
            dest_file = root_tests_dir / test_file.name
            shutil.copy2(test_file, dest_file)
            print(f"   Moved {test_file.name}")
        
        # Remove old tests directory
        shutil.rmtree(algosystem_tests_dir)
        print("🗑️  Removed algosystem/tests/ directory")
    
    # 3. Create conftest.py if it doesn't exist
    conftest_file = root_tests_dir / "conftest.py"
    if not conftest_file.exists():
        print("⚠️  conftest.py is missing - you need to create it with the fixtures")
        print("   See the conftest.py content I provided above")
    
    # 4. Update pytest.ini in root
    pytest_ini_content = '''[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --disable-warnings"
testpaths = [
    "tests",
]
python_files = [
    "test_*.py",
    "*_test.py",
]
python_classes = [
    "Test*",
]
python_functions = [
    "test_*",
]

# Coverage settings
[tool.coverage.run]
source = ["algosystem"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*",
]
'''
    
    pytest_ini_file = Path("pytest.ini")
    with open(pytest_ini_file, "w") as f:
        f.write(pytest_ini_content)
    print("✅ Updated pytest.ini")
    
    print("\n🎉 Test structure fixed!")
    print("\nNext steps:")
    print("1. Create the conftest.py file with the fixtures (content provided)")
    print("2. Fix the bugs in your Engine and metrics code")
    print("3. Run tests with: pytest tests/test_engine.py")

def create_fixed_metrics():
    """Create a patch for the metrics.py file."""
    patch_content = '''
# Add this fix to algosystem/backtesting/metrics.py around line 184:

# OLD CODE:
# metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(strategy_returns)) - 1

# NEW CODE:
if len(strategy_returns) > 0:
    metrics['annualized_return'] = (1 + metrics['total_return']) ** (252 / len(strategy_returns)) - 1
else:
    metrics['annualized_return'] = 0.0
'''
    
    with open("metrics_fix.patch", "w") as f:
        f.write(patch_content)
    print("📝 Created metrics_fix.patch with the fix for division by zero")

def create_fixed_engine():
    """Create a patch for the engine.py file."""
    patch_content = '''
# Add this fix to algosystem/backtesting/engine.py around line 84:

# OLD CODE:
# logger.info(f"Initialized backtest from {self.start_date.date()} to {self.end_date.date()}")

# NEW CODE:
if hasattr(self.start_date, 'date') and hasattr(self.end_date, 'date'):
    logger.info(f"Initialized backtest from {self.start_date.date()} to {self.end_date.date()}")
else:
    logger.info(f"Initialized backtest from {self.start_date} to {self.end_date}")
'''
    
    with open("engine_fix.patch", "w") as f:
        f.write(patch_content)
    print("📝 Created engine_fix.patch with the fix for non-datetime index")

if __name__ == "__main__":
    fix_test_structure()
    create_fixed_metrics()
    create_fixed_engine()
    
    print("\n📋 Summary of fixes needed:")
    print("1. ✅ Fixed test directory structure")
    print("2. ⚠️  Need to create conftest.py with fixtures")
    print("3. ⚠️  Need to apply metrics_fix.patch")
    print("4. ⚠️  Need to apply engine_fix.patch")
    
    print("\nAfter applying all fixes, run: pytest tests/test_engine.py -v")