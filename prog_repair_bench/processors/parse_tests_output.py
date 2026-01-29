import re
from typing import Any

from prog_repair_bench.processors.parse_tests_output_sympy import parse_sympy_tests_output


def parse_tests_output(output: str, repository: str = "django") -> dict[str, Any]:
    if repository == "sympy":
        return parse_sympy_tests_output(output)
    elif repository == "django":
        return parse_django_tests_output(output)
    else:
        raise ValueError(f"Invalid repository: {repository}")


def parse_django_tests_output(output: str) -> dict[str, Any]:
    test_pattern = re.compile(
        r"^(={70})\n"
        r"([A-Z]+):\s"  # ===== header  # Status, eg: ERROR:
        # r"([^\s]+) [^\n]*\n"  # Test name, eg: forms_tests.tests.test_deprecation_forms
        r"([^\s]+)\s+\(([^)]+)\)\n"  # Test method and module path in parentheses
        r"(?:.*?)"
        r"[-]{70}\n"  # ------
        r"(.*?)(?=\n[=-]{70}|\Z)",  # Stacktrace (non-greedy, up to next ===== or end of file
        re.DOTALL | re.MULTILINE,
    )
    # Pattern for summary statistics at end of file
    summary_pattern = re.compile(
        r"Ran (\d+) tests? in [\d.]+s\s*\n\s*"  # Ran X tests in Y.ZZZs
        r"([A-Z]+)(?: \(([^)]+)\))?",  # FAILED or OK, optionally with details in parentheses
        re.MULTILINE,
    )
    results = []
    for match in test_pattern.finditer(output):
        status = match.group(2)
        test_method = match.group(3)  # Just the method name (test_i18n_app_dirs)
        module_path = match.group(4)  # Module path (i18n.tests.WatchForTranslationChangesTests)
        full_test_path = f"{module_path}.{test_method}"  # Complete test path
        stacktrace = match.group(5)
        results.append(
            {
                "test_full_path": full_test_path,
                "status": status,
                "test_method": test_method,
                "module_path": module_path,
                "stacktrace": stacktrace.strip(),
            }
        )

    # Extract summary information
    summary_match = summary_pattern.search(output)
    if summary_match:
        total_tests = int(summary_match.group(1))
        overall_status = summary_match.group(2)  # "FAILED" or "OK"
        details_str = summary_match.group(3)
        # e.g. "failures=1, errors=14, skipped=998, expected failures=4"

        # Parse the details into a dictionary
        details = {}
        if details_str:
            for item in details_str.split(", "):
                key, value = item.split("=")
                details[key] = int(value)

        summary_info = {
            "total_tests": total_tests,
            "status": overall_status,
            "details": details,
        }
    else:
        summary_info = None

    return {"tests": results, "summary": summary_info}


def get_percentage_passed(test_results: dict[str, Any]) -> float:

    summary = test_results.get("summary", {})
    if summary is None:
        return 0.0
    total_tests = summary.get("total_tests", 0)
    details = summary.get("details", {})
    if (details is None) or (total_tests is None) or (total_tests == 0):
        return 0.0

    total_tests = max((details.get("failures", 0) + details.get("errors", 0)), total_tests)
    passed_tests = total_tests - (details.get("failures", 0) + details.get("errors", 0))
    percentage_passed = (passed_tests / total_tests) if total_tests > 0 else 0.0

    return percentage_passed
