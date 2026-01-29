import re
from typing import Any


def _normalize_module_path(file_path: str) -> str:
    # Convert e.g. "sympy/matrices/tests/test_x.py" -> "sympy.matrices.tests.test_x"
    if file_path.endswith(".py"):
        file_path = file_path[:-3]
    return file_path.replace("/", ".")


def parse_sympy_tests_output(output: str) -> dict[str, Any]:
    """
    Parse SymPy's test runner output from `python bin/test`.

    Expected structure:
    - Per-file progress lines like: `sympy/.../test_foo.py[10] .... [OK]` (not used directly).
    - For failures/errors, detailed blocks appear as:

        ________________________________________________________________
         sympy/path/to/test_file.py:test_name
           <stack trace lines>
        ________________________________________________________________

    - Final summary line (can wrap across lines), e.g.:
        tests finished: 2852 passed, 1 failed, 79 skipped, 51 expected to fail,
        6 exceptions, in 22.06 seconds

    Returns dict with keys:
      - tests: list of {test_full_path, status, test_method, module_path, stacktrace}
      - summary: {total_tests, status, details}
          where details has at least {failures, errors, skipped, expected_failures, passed}
    """
    results: list[dict[str, Any]] = []

    # 1) Identify files that ended with [FAIL] in the progress section.
    # Handle both same-line and next-line placements of [FAIL].
    fail_progress_pattern = re.compile(r"(?m)^(sympy/[^\s\[]+\.py)\[\d+\][^\n]*?(?:\n\s*)?\[FAIL\]")
    failed_files = {m.group(1) for m in fail_progress_pattern.finditer(output)}

    # 2) Build a map from file path -> list of (test_method, stacktrace) from detailed sections.
    header_pattern = re.compile(r"(?m)^\s*(sympy/[^\s:]+\.py):([A-Za-z0-9_]+)\s*$")
    header_matches = list(header_pattern.finditer(output))

    def _next_boundary(start_idx: int) -> int:
        for m in header_matches:
            if m.start() > start_idx:
                return m.start()
        return len(output)

    file_to_stacks: dict[str, list[tuple[str, str]]] = {}
    for m in header_matches:
        file_path = m.group(1)
        test_method = m.group(2)
        start_idx = m.end()
        end_idx = _next_boundary(start_idx)
        stacktrace = output[start_idx:end_idx].strip()
        stacktrace = re.sub(r"(?m)^\s*_+\s*$\n?", "", stacktrace).strip()
        file_to_stacks.setdefault(file_path, []).append((test_method, stacktrace))

    # 3) For each failed file, pick some stack trace (first available) and record a FAIL entry.
    for file_path in failed_files:
        module_path = _normalize_module_path(file_path)
        test_method = "<unknown>"
        stacktrace = ""
        if file_path in file_to_stacks and file_to_stacks[file_path]:
            test_method, stacktrace = file_to_stacks[file_path][0]

        results.append(
            {
                "test_full_path": f"{module_path}.{test_method}",
                "status": "FAIL",
                "test_method": test_method,
                "module_path": module_path,
                "stacktrace": stacktrace,
            }
        )

    # 2) Parse the final summary block
    # It can span multiple lines before the "in XX.XX seconds" part
    summary_pattern = re.compile(
        r"tests finished:\s*(.*?)\s*in\s*[\d.]+\s*seconds",
        re.IGNORECASE | re.DOTALL,
    )
    summary_match = summary_pattern.search(output)

    if not summary_match:
        return {"tests": results, "summary": None}

    summary_body = summary_match.group(1)

    # Extract pairs like "2852 passed", "1 failed", possibly separated by commas/newlines
    item_pattern = re.compile(r"(\d+)\s+([A-Za-z ]+?)(?:,|$)")
    counts: dict[str, int] = {}
    for count_str, label in item_pattern.findall(summary_body):
        n = int(count_str)
        key = label.strip().lower()
        if key.startswith("passed"):
            counts["passed"] = n
        elif key.startswith("failed"):
            counts["failed"] = n
        elif key.startswith("skipped"):
            counts["skipped"] = n
        elif key.startswith("expected to fail"):
            counts["expected_failures"] = n
        elif key.startswith("exceptions"):
            counts["errors"] = n

    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    skipped = counts.get("skipped", 0)
    expected_failures = counts.get("expected_failures", 0)
    errors = counts.get("errors", 0)

    total_tests = passed + failed + skipped + expected_failures + errors
    overall_status = "OK" if (failed == 0 and errors == 0) else "FAILED"

    summary_info = {
        "total_tests": total_tests,
        "status": overall_status,
        "details": {
            "failures": failed,
            "errors": errors,
            "skipped": skipped,
            "expected_failures": expected_failures,
            "passed": passed,
        },
    }

    return {"tests": results, "summary": summary_info}
