#!/usr/bin/env python3
"""Demo script showing GPU setup with meta_trainer.py (first 30 lines of output)"""

import subprocess
import sys

print("=" * 70)
print("DEMO: Running meta_trainer.py with GPU logging")
print("This will show the first 30 lines of output including GPU setup")
print("=" * 70)
print()

# Run meta_trainer.py and capture first 30 lines
cmd = [sys.executable, "meta_trainer.py"]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

line_count = 0
for line in process.stdout:
    print(line, end='')
    line_count += 1
    if line_count >= 30:
        process.terminate()
        break

print("\n" + "=" * 70)
print("Demo completed - showing first 30 lines of meta_trainer.py execution")
print("=" * 70)