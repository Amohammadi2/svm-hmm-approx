import json
from dataclasses import asdict
from pathlib import Path

from utils.datastructs import SVMParameters


_CACHE_FILE = Path(__file__).resolve().parent / "cache" / "svm_params.json"


def load_cached_params() -> SVMParameters:
    try:
        with _CACHE_FILE.open("r", encoding="utf-8") as f:
            return SVMParameters(**json.load(f))
    except:
        return None

def save_params_to_cache(params: SVMParameters):
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=4)