"""
Compare the files a run just produced with the versions committed in git.
Numbers may differ only by tiny floating-point amounts (different computers and
library versions can change the last digits). Anything larger fails.

Usage, from the repository root:  python scripts/check_reproducible.py
"""

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

FILES = sorted(Path("data/processed").glob("*.csv")) + sorted(Path("results").glob("*.*"))
TOLERANCE = 1e-6


def committed(path):
    """The version of a file in the last commit, or None if the file is new."""
    result = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return None
    return result.stdout


def same_json(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_json(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE) or (a != a and b != b)
    return a == b


failures = []
for path in FILES:
    old = committed(path)
    if old is None:
        failures.append(f"{path}: new file, not in the last commit yet")
        continue
    if path.suffix == ".csv":
        try:
            pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(io.StringIO(old)),
                                          check_dtype=False, rtol=TOLERANCE, atol=TOLERANCE)
        except AssertionError as e:
            failures.append(f"{path}: {str(e).splitlines()[0]}")
    elif path.suffix == ".json":
        # NaN is written by Python's json module; allow it when parsing
        if not same_json(json.loads(path.read_text()), json.loads(old)):
            failures.append(f"{path}: values differ")
    print(f"  checked {path}")

if failures:
    print("\nOutputs differ from the committed versions:")
    print("\n".join(f"  {f}" for f in failures))
    sys.exit(1)
print(f"\nAll {len(FILES)} files match the committed versions.")
