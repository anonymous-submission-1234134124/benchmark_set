"""
Trim a Python file down to a maximum number of characters while preserving a target
method inside a target class.

Rules enforced:
1) Cut from the end of the file.
   - If the target class is the last class in the file, trim other classes first
     (from the end moving toward the beginning of the file).
2) Cut method-wise (never cut inside a method).
   - If the target method would be the next to cut, skip it and cut other methods
     first (again, from the end of the file).
3) If a class has no methods left, remove the entire class block.

Notes:
- Trailing non-class code at the very end of the file (e.g., a __main__ block) is
  considered a "tail" and is cut first since trimming proceeds from the end.
- Only class methods are removed method-wise. Top-level functions are part of the tail.
- The script never removes the target method, and it will not remove the target class
  if it still contains the target method.

"""

import ast
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable


# ------------------------------- Data structures ------------------------------


@dataclass
class MethodInfo:
    name: str
    start: int  # 1-based line index (includes decorators)
    end: int    # inclusive 1-based line index
    is_async: bool = False


@dataclass
class ClassInfo:
    name: str
    start: int
    end: int
    methods: list[MethodInfo] = field(default_factory=list)

    def has_non_target_methods(self, target_class: str, target_method: str) -> bool:
        if self.name != target_class:
            return bool(self.methods)
        return any(m.name != target_method for m in self.methods)

# ------------------------------- Regexp fallbacks ------------------------------

CLASS_RE = re.compile(r'^\s*class\s+([A-Za-z_]\w*)\s*(\(|:)', re.ASCII)
DEF_RE = re.compile(r'^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(')


def _indent_width(line: str) -> int:
    """Count leading spaces/tabs as indentation width (tabs treated as 4)."""
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            count += 4
        else:
            break
    return count


def _collect_classes_and_methods_fallback(
    file_content: str,
) -> tuple[list[ClassInfo], list[str]]:
    """
    Fallback: heuristically collect classes and methods using regex + indentation,
    without parsing the file as AST. Works even if the file has syntax errors.
    """
    lines = file_content.splitlines(keepends=True)
    n = len(lines)
    classes: list[ClassInfo] = []

    i = 0
    while i < n:
        line = lines[i]
        m_cls = CLASS_RE.match(line)
        if not m_cls:
            i += 1
            continue

        class_name = m_cls.group(1)
        class_indent = _indent_width(line)
        class_start = i + 1  # 1-based

        # Find end of class block by indentation
        j = i + 1
        while j < n:
            l = lines[j]
            stripped = l.lstrip()
            if stripped == "" or stripped.startswith("#"):
                j += 1
                continue
            ind = _indent_width(l)
            if ind <= class_indent:
                break
            j += 1
        class_end = j  # 1-based inclusive

        # Scan for methods inside [i+1, j)
        methods: list[MethodInfo] = []
        body_start_idx = i + 1
        body_end_idx = j

        k = body_start_idx
        while k < body_end_idx:
            l = lines[k]
            stripped = l.lstrip()
            if stripped == "" or stripped.startswith("#"):
                k += 1
                continue

            ind = _indent_width(l)
            if ind <= class_indent:
                k += 1
                continue

            m_def = DEF_RE.match(l)
            if not m_def:
                k += 1
                continue

            method_name = m_def.group(1)
            method_indent = ind
            m_start = k + 1  # 1-based

            # Find method end: until something dedents to class level or another method/class
            t = k + 1
            while t < body_end_idx:
                lt = lines[t]
                stripped_t = lt.lstrip()
                if stripped_t == "" or stripped_t.startswith("#"):
                    t += 1
                    continue

                ind_t = _indent_width(lt)
                # End if dedent to class level or a new def at same/higher level
                if ind_t <= class_indent:
                    break
                if ind_t <= method_indent and DEF_RE.match(lt):
                    break

                t += 1

            m_end = t  # 1-based inclusive (t is first line after the method body)
            is_async = stripped.startswith("async")
            methods.append(
                MethodInfo(
                    name=method_name,
                    start=m_start,
                    end=m_end,
                    is_async=is_async,
                )
            )
            k = t

        methods.sort(key=lambda m: m.start)
        classes.append(
            ClassInfo(
                name=class_name,
                start=class_start,
                end=class_end,
                methods=methods,
            )
        )

        i = j  # continue scanning after this class

    classes.sort(key=lambda c: c.start)
    return classes, lines

# ------------------------------- AST utilities --------------------------------

def trim_extra_empty_lines(text: str, max_empty: int = 2) -> str:
    """
    Collapse runs of empty lines (strip() == '') so that no more than
    `max_empty` consecutive empty lines remain.
    """
    out_lines = []
    empty_streak = 0

    for line in text.splitlines(keepends=True):
        if line.strip() == "":
            empty_streak += 1
            if empty_streak <= max_empty:
                out_lines.append(line)
        else:
            empty_streak = 0
            out_lines.append(line)

    return "".join(out_lines)


def _earliest_decorator_lineno(node: ast.AST) -> int:
    """Return the earliest line among the node and its decorators (if any)."""
    linenos = [getattr(node, "lineno", None)]
    for deco in getattr(node, "decorator_list", []) or []:
        ln = getattr(deco, "lineno", None)
        if ln:
            linenos.append(ln)
    # Filter None, default to node.lineno
    linenos = [ln for ln in linenos if ln is not None]
    return min(linenos) if linenos else getattr(node, "lineno", 1)


def _compute_end_from_siblings(
    start_idx: int,
    siblings: list[ast.AST],
    fallback_end: int,
) -> int:
    """
    Conservative fallback when node.end_lineno is unavailable:
    end line = (next sibling's lineno - 1) or fallback_end.
    """
    for sib in siblings[start_idx + 1:]:
        # earliest line of sibling, including decorators
        first_line = _earliest_decorator_lineno(sib)
        if first_line is not None:
            return first_line - 1
    return fallback_end


def _collect_classes_and_methods(file_content: str) -> tuple[list[ClassInfo], list[str]]:
    """
    Parse the module and collect top-level classes with their direct methods.
    Returns (classes, lines) where lines are the original lines including newlines.
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        return _collect_classes_and_methods_fallback(file_content)

    lines = file_content.splitlines(keepends=True)

    classes: list[ClassInfo] = []

    # Prepare a quick mapping of top-level nodes to enable fallback end positions
    top_level = list(tree.body)

    for idx, node in enumerate(top_level):
        if not isinstance(node, ast.ClassDef):
            continue

        class_start = _earliest_decorator_lineno(node)
        class_end = getattr(node, "end_lineno", None)
        if class_end is None:
            # Next top-level sibling - 1, or end of file
            class_end = _compute_end_from_siblings(idx, top_level, len(lines))

        # Collect direct methods (ignore nested classes/functions deeper inside)
        methods: list[MethodInfo] = []
        body_siblings = list(node.body)
        # Fallback for last method's end uses class_end
        for j, child in enumerate(body_siblings):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m_start = _earliest_decorator_lineno(child)
                m_end = getattr(child, "end_lineno", None)
                if m_end is None:
                    m_end = _compute_end_from_siblings(j, body_siblings, class_end)
                methods.append(
                    MethodInfo(
                        name=child.name,
                        start=m_start,
                        end=m_end,
                        is_async=isinstance(child, ast.AsyncFunctionDef),
                    )
                )

        methods.sort(key=lambda m: m.start)  # textual order within the class
        classes.append(ClassInfo(name=node.name, start=class_start, end=class_end, methods=methods))

    classes.sort(key=lambda c: c.start)  # textual order of classes
    return classes, lines


# ---------------------------- Trimming plan builder ----------------------------


@dataclass
class Removal:
    start: int
    end: int
    kind: str  # "tail" | "method" | "class"
    class_name: str | None = None
    method_name: str | None = None


def _generate_removal_plan(
    classes: list[ClassInfo],
    total_lines: int,
    target_class: str,
    target_method: str,
) -> Iterable[Removal]:
    """
    Yields removal actions in the order they should be applied to trim from the end
    while protecting the target class/method per the rules.
    """
    # 1) Remove trailing non-class "tail" first (everything after the last class).
    last_class_end = max((c.end for c in classes), default=0)
    if last_class_end < total_lines:
        yield Removal(
            start=last_class_end + 1,
            end=total_lines,
            kind="tail",
        )

    # 2) Determine class processing order:
    #    Process every class from the end toward the beginning, but PUT the target class LAST.
    indices = list(range(len(classes)))
    # Move the last occurrence of the target class to the end of the order
    target_indices = [i for i, c in enumerate(classes) if c.name == target_class]
    if not target_indices:
        raise SystemExit(f"Target class '{target_class}' not found.")
    t_idx = target_indices[-1]
    ordered = [i for i in reversed(indices) if i != t_idx] + [t_idx]

    # 3) For each class in that order:
    #    - remove its methods from the end (skipping the target method in the target class),
    #    - if a non-target class ends up with no methods left, remove the whole class.
    for i in ordered:
        cls = classes[i]
        # methods we are allowed to remove for this class
        if cls.name == target_class:
            candidate_methods = [m for m in cls.methods if m.name != target_method]
        else:
            candidate_methods = list(cls.methods)

        # Remove methods from the end of the class
        for m in reversed(candidate_methods):
            yield Removal(
                start=m.start,
                end=m.end,
                kind="method",
                class_name=cls.name,
                method_name=m.name,
            )

        # If it's NOT the target class, after all its methods are gone we can remove the class shell.
        if cls.name != target_class:
            yield Removal(
                start=cls.start,
                end=cls.end,
                kind="class",
                class_name=cls.name,
            )

def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/remotely adjacent line intervals (inclusive, 1-based)."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _apply_removals_until_limit(
    lines: list[str],
    plan: Iterable[Removal],
    limit: int,
) -> tuple[str, list[Removal]]:
    """Apply removals in order until the output's character count <= limit."""
    applied: list[Removal] = []
    intervals: list[tuple[int, int]] = []

    def build_output() -> str:
        merged = _merge_intervals(intervals)
        if not merged:
            return "".join(lines)
        out_parts: list[str] = []
        cur_line = 1
        for s, e in merged:
            # keep [cur_line, s-1]
            if cur_line <= s - 1:
                out_parts.append("".join(lines[cur_line - 1 : s - 1]))
            cur_line = e + 1
        if cur_line <= len(lines):
            out_parts.append("".join(lines[cur_line - 1 :]))
        return "".join(out_parts)

    current = build_output()
    if len(current) <= limit:
        return current, applied

    for step in plan:
        intervals.append((step.start, step.end))
        applied.append(step)
        current = build_output()
        if len(current) <= limit:
            return current, applied

    # If we exhausted the plan and still exceed limit (shouldn't happen if the
    # target method is smaller than limit), return the best effort.
    return current, applied


def _verify_target_present(text: str, target_class: str, target_method: str) -> None:
    """
    Ensure the resulting text still defines target class + method.
    If parsing fails or not found, we print a warning (and leave the output as-is).
    """
    try:
        t = ast.parse(text)
    except SyntaxError:
        # sys.stderr.write(
        #     "[WARN] Output is not syntactically valid after trimming; "
        #     "could not verify target presence.\n"
        # )
        return

    found_class = None
    for node in t.body:
        if isinstance(node, ast.ClassDef) and node.name == target_class:
            found_class = node
            break
    if not found_class:
        sys.stderr.write(
            f"[WARN] Target class '{target_class}' not found in trimmed output.\n"
        )
        return

    for child in found_class.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == target_method:
            return

    # sys.stderr.write(
    #     f"[WARN] Target method '{target_method}' not found in class '{target_class}' "
    #     "after trimming.\n"
    # )


def trim_file(
    file_content: str,
    target_class: str,
    target_method: str,
    limit: int,
) -> str:

    classes, lines = _collect_classes_and_methods(file_content)

    if not classes:
        # No classes at all; just cut from the end as a tail, because we must keep
        # a target class/method that doesn't exist—signal clearly.
        # Tail-only trimming
        trimmed = file_content[:limit]
        return trimmed

    # Confirm the target class/method exist (best effort: use last occurrence of class).
    target_class_indices = [i for i, c in enumerate(classes) if c.name == target_class]
    if not target_class_indices:
        sys.stderr.write(
            f"[WARN] Target class '{target_class}' not found; trimming will proceed "
            "without preserving it.\n"
        )
    else:
        t_idx = target_class_indices[-1]
        cls = classes[t_idx]
        if not any(m.name == target_method for m in cls.methods):
            pass
            # sys.stderr.write(
            #     f"[WARN] Target method '{target_method}' not found in class '{target_class}'; "
            #     "trimming will proceed without preserving it.\n"
            # )

    plan = _generate_removal_plan(
        classes=classes,
        total_lines=len(lines),
        target_class=target_class,
        target_method=target_method,
    )
    output, _applied = _apply_removals_until_limit(lines, plan, limit)
    _verify_target_present(output, target_class, target_method)

    output = trim_extra_empty_lines(output)

    return output

