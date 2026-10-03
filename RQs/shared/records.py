"""Read the merged evaluation records without replaying corrections."""
import csv
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data/evaluation"
IO = {"io_correct": "correct_io_pass_rate", "io_wrong": "wrong_io_reject_rate", "io_invalid": "invalid_test_filtering_rate"}

def read_csv(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def read_json(path, expected=None):
    raw = path.read_bytes()
    if expected is not None and sha(raw) != expected:
        raise ValueError(f'Source hash mismatch: {path}')
    return json.loads(raw)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=1) + chr(10)).encode()
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(raw)
    temporary.replace(path)
    return sha(raw)

def generated(payload, name):
    metric = payload['metrics'][name]
    return metric.get('generated', metric)
