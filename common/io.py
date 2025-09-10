import json
from pathlib import Path
from datetime import datetime
import platform
import sys

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(data, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def now_iso_pst():
    # store local ISO; your OS tz is fine
    return datetime.now().astimezone().isoformat(timespec="seconds")

def runtime_versions():
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})"
    }
