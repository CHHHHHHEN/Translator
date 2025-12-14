"""Build helpers for packaging the Translator app."""

import subprocess
import sys


def main() -> int:
    print("Building translator distribution via PyInstaller...")
    return subprocess.call([
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "translator",
        "-y",
        "-F",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
