#!/usr/bin/env python3
import re
from pathlib import Path

START_MARKER = r'# ---start---'
END_MARKER   = r'# ---end---'

pattern = re.compile(
    rf'(^[ \t]*{re.escape(START_MARKER)}[ \t]*\n)'  # group 1: the line with "# ---start---"
    rf'(.*?\n)'                                    # group 2: all code up until...
    rf'(^[ \t]*{re.escape(END_MARKER)}[ \t]*$)',    # group 3: the line with "# ---end---"
    re.DOTALL | re.MULTILINE
)

def process_file(path: Path):
    text = path.read_text(encoding='utf-8')
    def replacer(m):
        return f"{m.group(1)}# Fill your code\n{m.group(3)}"
    new_text, count = pattern.subn(replacer, text)
    if count:
        path.write_text(new_text, encoding='utf-8')
        print(f"  → Updated {path.relative_to(Path.cwd())}")

EXCLUDE_DIRS = {'venv', '.venv', '__pycache__'}

def main():
    print("Scanning for .py files…")
    for py in Path('.').rglob('*.py'):

        if any(part in EXCLUDE_DIRS for part in py.parts):
            continue

        process_file(py)
    print("Done.")

if __name__ == '__main__':
    main()
