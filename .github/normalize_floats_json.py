import json
import sys
from pathlib import Path

SIG_DIGITS = 12


def normalize(obj):
    if isinstance(obj, float):
        # Use 8 significant digits (auto scientific notation if needed)
        return float(format(obj, f".{SIG_DIGITS}g"))

    elif isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [normalize(v) for v in obj]

    return obj


def process_file(path):
    path = Path(path)

    try:
        with path.open("r") as f:
            original = json.load(f)
    except Exception:
        return  # skip invalid JSON

    normalized = normalize(original)

    # Only write if something actually changed
    if normalized != original:
        with path.open("w") as f:
            json.dump(normalized, f, indent=2, sort_keys=False)
            f.write("\n")


if __name__ == "__main__":
    for file in sys.argv[1:]:
        process_file(file)
