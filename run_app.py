import subprocess
import sys
from pathlib import Path

from config import APP_HEADLESS, APP_HOST, APP_PORT

APP_FILE = Path(__file__).resolve().parent / "app.py"


def main():
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.address",
        APP_HOST,
        "--server.port",
        str(APP_PORT),
        "--server.headless",
        str(APP_HEADLESS).lower(),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
