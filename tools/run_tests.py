"""Запустити перевірки з тимчасовими файлами всередині робочого проєкту.

Це обходить обмеження доступу до системної тимчасової папки Windows,
не змінюючи систему та не потребуючи підвищених прав.
"""
from pathlib import Path
import os
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
# Імпорти src мають працювати також при запуску python tools/run_tests.py.
import sys
sys.path.insert(0, str(ROOT))
temp_path = ROOT/'outputs/local/test-temp'
temp_path.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temp_path)
suite = unittest.defaultTestLoader.discover('tests')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
