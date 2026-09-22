"""Verify the golden fixture still matches oracle.json.

Gate 0.5's success criteria are claims about the fixture ("baseline fails for
the intended reason", "known solution passes", "touches at least two files").
This makes them checkable instead of trusted, and catches the failure mode the
prompt warns about: committing a repo/ that someone already solved, which would
silently void the baseline signature every arm is measured against.

Also checks the oracle's P1-P4 label data against the actual tree, so the Exp 3a
ground truth can't drift away from the code it describes.

stdlib only. Run from this directory:

    python3 verify_fixture.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import shutil
import subprocess
import tempfile

HERE = pathlib.Path(__file__).parent
REPO = HERE / "repo"
ORACLE = json.loads((HERE / "oracle.json").read_text())

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def run_tests(cwd: pathlib.Path) -> tuple[int, int, int]:
    """Returns (total, failures, errors) parsed from unittest's summary."""
    proc = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    out = proc.stderr + proc.stdout
    total = int(next(l for l in out.splitlines() if l.startswith("Ran ")).split()[1])
    fails = errs = 0
    for line in out.splitlines():
        if line.startswith("FAILED"):
            for part in line[line.find("(") + 1 : line.rfind(")")].split(","):
                key, _, value = part.strip().partition("=")
                if key == "failures":
                    fails = int(value)
                elif key == "errors":
                    errs = int(value)
    return total, fails, errs


def imports_of(path: pathlib.Path) -> set[str]:
    """Module-level imports as dotted names.

    Both halves of an `ImportFrom` matter here: `from srcpkg import units`
    carries the target module in the ALIAS list, not in `node.module`, so
    collecting only `node.module` misses every test-to-module edge.
    """
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            for alias in node.names:
                names.add(f"{node.module}.{alias.name}" if node.module else alias.name)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


def module_symbols(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return out


# --- baseline state -------------------------------------------------------
total, fails, errs = run_tests(REPO)
b = ORACLE["baseline"]
check(total == b["total_tests"], f"baseline ran {total} tests, oracle says {b['total_tests']}")
check(
    fails + errs == b["expected_failures"],
    f"baseline had {fails + errs} failing, oracle says {b['expected_failures']} "
    "(a passing baseline means repo/ was committed already solved)",
)

# --- solution patch round-trip -------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    work = pathlib.Path(tmp) / "repo"
    shutil.copytree(REPO, work)
    patch = (HERE / ORACLE["solution"]["patch"].removeprefix("../")).read_text()
    applied = subprocess.run(
        ["patch", "-p1", "-s"], cwd=work, input=patch, capture_output=True, text=True
    )
    check(applied.returncode == 0, f"solution.patch did not apply: {applied.stderr.strip()[:200]}")
    if applied.returncode == 0:
        t2, f2, e2 = run_tests(work)
        check(f2 + e2 == 0, f"solution.patch applied but {f2 + e2} tests still fail")
        check(t2 == total, "solution changed the test count")

check(
    len(ORACLE["expected_changed_files"]) >= 2,
    "Gate 0.5 requires the task to touch at least two files",
)
patch_text = (HERE / "solution.patch").read_text()
for path in ORACLE["expected_changed_files"]:
    check(f"a/{path}" in patch_text, f"{path} is listed as changed but absent from solution.patch")

# --- P1: import graph ----------------------------------------------------
for src, dst in ORACLE["p1_file_graph"]["import_edges"]:
    src_path = REPO / src
    check(src_path.exists(), f"P1 node missing: {src}")
    if not src_path.exists():
        continue
    dst_mod = pathlib.Path(dst).stem
    found = any(dst_mod == i.split(".")[-1] for i in imports_of(src_path))
    check(found, f"P1 edge {src} -> {dst} not present in the actual imports")

for node in ORACLE["p1_file_graph"]["nodes"]:
    check((REPO / node).exists(), f"P1 node listed but missing on disk: {node}")

# --- P2: symbol surface --------------------------------------------------
for sym in ORACLE["p2_symbol_surface"]:
    path = REPO / sym["file"]
    check(path.exists(), f"P2 file missing: {sym['file']}")
    if not path.exists():
        continue
    present = sym["name"] in module_symbols(path)
    if sym["status"] == "to_add":
        check(not present, f"P2 {sym['name']} is marked to_add but already exists in the baseline")
    else:
        check(present, f"P2 {sym['name']} marked {sym['status']} but is absent from {sym['file']}")

# --- P3: test dependencies ----------------------------------------------
for test_file, deps in ORACLE["p3_test_dependencies"].items():
    path = REPO / test_file
    check(path.exists(), f"P3 test file missing: {test_file}")
    if not path.exists():
        continue
    direct = {i.split(".")[-1] for i in imports_of(path)}
    first = pathlib.Path(deps[0]).stem
    check(first in direct, f"P3 {test_file} should import {deps[0]} directly")

# --- P4: ambiguity arithmetic -------------------------------------------
case = ORACLE["p4_ambiguity"]["discriminating_case"]
mib = 1024 * 1024
total_b = sum(case["samples"])
check(
    abs((total_b / mib) / case["seconds"] - case["mebibytes_per_second"]) < 1e-9,
    "P4 mebibytes_per_second value does not match the samples/seconds given",
)
check(
    abs((total_b * 8 / mib) / case["seconds"] - case["mebibits_per_second"]) < 1e-9,
    "P4 mebibits_per_second value does not match the samples/seconds given",
)
check(
    case["mebibytes_per_second"] != case["mebibits_per_second"],
    "P4 candidates are not discriminable on this case",
)
check(
    ORACLE["p4_ambiguity"]["committed"] in ORACLE["p4_ambiguity"]["candidates"],
    "P4 committed answer is not one of the listed candidates",
)
distractor = ORACLE["p4_ambiguity"]["distractor_symbol"].split(":")[-1]
check(
    distractor in module_symbols(REPO / "srcpkg" / "units.py"),
    f"P4 distractor {distractor} is missing, so the ambiguity is not real",
)

# --- report --------------------------------------------------------------
print(f"ran {checks} checks")
if failures:
    print(f"\nFAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print(f"\nPASSED: fixture matches oracle.json")
print(f"  baseline {total} tests, {fails + errs} failing ({b['root_cause']})")
print(f"  solution.patch applies, all {total} pass, touches {len(ORACLE['expected_changed_files'])} files")
