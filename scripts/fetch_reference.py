"""Cache season and player tables, conservatively spaced; completed seasons are cached indefinitely."""
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://www.basketball-reference.com'
_last = 0.0

def fetch(path, force=False):
    global _last
    target = ROOT / 'data/raw' / path.lstrip('/')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target.read_text()
    time.sleep(max(0, 7.6 - (time.monotonic() - _last)))
    response = requests.get(BASE + path, timeout=45)
    _last = time.monotonic()
    response.raise_for_status()
    response.encoding = "utf-8"
    if '<table' not in response.text:
        raise ValueError(f'No tables: {path}')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(response.text)
    temporary.replace(target)
    print(path, flush=True)
    return response.text

if __name__ == '__main__':
    for year in range(2002, 2027):
        for suffix in ['', '_advanced']:
            fetch(f'/leagues/NBA_{year}{suffix}.html')
