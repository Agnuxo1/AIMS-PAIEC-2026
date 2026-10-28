"""Validate the submission archives without the competition data or network.

Checks, for every ZIP in submissions/:
  * it contains model.py at the archive root;
  * model.py imports nothing outside the standard library, opens no file and
    makes no network call (AST-level, so an obfuscated import still shows up as
    a call we do not allow);
  * predict() returns a finite float in [0, 1] on a battery of edge cases,
    including malformed evidence;
  * the shrinkage identity holds exactly on random (n, k) draws;
  * the contextual and item-peer archives return the original probabilities
    when no peer labels are supplied.

Exit code 0 means every archive is self-contained and well behaved.
"""
from __future__ import annotations

import ast
import importlib.util
import io
import math
import os
import random
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBS = os.path.join(ROOT, "submissions")

ALLOWED_IMPORTS = {
    "math", "random", "json", "os", "sys", "typing", "collections",
    "itertools", "functools", "statistics", "dataclasses", "re",
}
FORBIDDEN_CALLS = {"open", "eval", "exec", "compile", "__import__", "input"}

SUBJECT = {
    "normalized_name": "Example LM", "provider": "Example",
    "release_date": "2025-01-01", "access_date": "2025-06-01",
    "harness": "default", "harness_version": "1", "reasoning_effort": "",
    "subject_features_extra": "",
}
ITEM = {"item_content": "q", "item_features": "", "interactors": "",
        "benchmark_id": "bench_a"}
OTHER_ITEM = dict(ITEM, benchmark_id="bench_b")
OTHER_SUBJECT = dict(SUBJECT, normalized_name="Peer LM")

fails: list[str] = []


def check(cond, msg):
    print(("  [OK ] " if cond else "  [FAIL] ") + msg)
    if not cond:
        fails.append(msg)


def static_scan(source: str, name: str):
    tree = ast.parse(source)
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    extra = imported - ALLOWED_IMPORTS
    bad = called & FORBIDDEN_CALLS
    check(not extra, f"{name}: no imports outside the standard library allowlist {sorted(extra)}")
    check(not bad, f"{name}: no file or dynamic-execution calls {sorted(bad)}")


def load(source: str, name: str):
    path = os.path.join(tempfile.mkdtemp(), "m.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(source)
    spec = importlib.util.spec_from_file_location(f"sub_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edge_cases(mod, name):
    cases = [
        ("no evidence", []),
        ("evidence is None", None),
        ("one success", [[[SUBJECT, ITEM], 1]]),
        ("one failure", [[[SUBJECT, ITEM], 0]]),
        ("31 successes", [[[SUBJECT, ITEM], 1]] * 31),
        ("other benchmark only", [[[SUBJECT, OTHER_ITEM], 1]] * 10),
        ("other subject only", [[[OTHER_SUBJECT, ITEM], 1]] * 10),
        ("malformed entries", [1, "x", None, [[SUBJECT], 1], [[SUBJECT, ITEM]]]),
        ("non-binary labels", [[[SUBJECT, ITEM], "yes"], [[SUBJECT, ITEM], None],
                               [[SUBJECT, ITEM], 1]]),
        ("missing benchmark_id", [[[SUBJECT, {"item_content": "q"}], 1]] * 5),
    ]
    ok = True
    for label, evidence in cases:
        try:
            p = mod.predict([SUBJECT, ITEM], evidence)
        except Exception as exc:                      # noqa: BLE001
            print(f"      {label}: raised {type(exc).__name__}")
            ok = False
            continue
        good = isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0
        if not good:
            print(f"      {label}: returned {p!r}")
        ok = ok and good
    check(ok, f"{name}: every edge case returns a finite float in [0, 1]")


def identity(mod, name, p0=0.5, strength=3.0):
    rng = random.Random(20261024)
    worst = 0.0
    for _ in range(4000):
        n = rng.randint(0, 31)
        k = rng.randint(0, n)
        evidence = [[[SUBJECT, ITEM], 1]] * k + [[[SUBJECT, ITEM], 0]] * (n - k)
        rng.shuffle(evidence)
        expected = (p0 * strength + k) / (strength + n)
        worst = max(worst, abs(mod.predict([SUBJECT, ITEM], evidence) - expected))
    check(worst < 1e-9, f"{name}: shrinkage identity holds on 4000 draws (max error {worst:.2e})")


def fallback(mod, original, name):
    """The contextual archive must equal the original with no peer evidence."""
    rng = random.Random(7)
    worst = 0.0
    for _ in range(2000):
        n = rng.randint(0, 31)
        k = rng.randint(0, n)
        evidence = [[[SUBJECT, ITEM], 1]] * k + [[[SUBJECT, ITEM], 0]] * (n - k)
        worst = max(worst, abs(mod.predict([SUBJECT, ITEM], evidence)
                               - original.predict([SUBJECT, ITEM], evidence)))
    check(worst == 0.0, f"{name}: identical to the original with no peer labels "
                        f"(max difference {worst:.2e})")


def main():
    archives = sorted(f for f in os.listdir(SUBS) if f.endswith(".zip"))
    check(bool(archives), f"submissions/ contains at least one archive ({archives})")
    loaded = {}
    for fname in archives:
        print(f"\n{fname}")
        with zipfile.ZipFile(os.path.join(SUBS, fname)) as z:
            names = z.namelist()
            check("model.py" in names, f"{fname}: model.py sits at the archive root")
            bad = [n for n in names if n.startswith("/") or ".." in n]
            check(not bad, f"{fname}: no absolute or traversing paths {bad}")
            source = z.read("model.py").decode("utf-8")
        static_scan(source, fname)
        mod = load(source, fname.replace(".", "_"))
        check(hasattr(mod, "predict"), f"{fname}: exposes predict()")
        edge_cases(mod, fname)
        loaded[fname] = mod
        if "original" in fname:
            identity(mod, fname)

    orig = next((m for f, m in loaded.items() if "original" in f), None)
    for fname, mod in loaded.items():
        if orig is not None and ("context" in fname or "item_peer" in fname):
            print(f"\n{fname} vs the original")
            fallback(mod, orig, fname)

    print("\n" + "=" * 62)
    if fails:
        print(f"FAILED: {len(fails)} check(s)")
        for f in fails:
            print("  -", f)
        return 1
    print("All submission checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
