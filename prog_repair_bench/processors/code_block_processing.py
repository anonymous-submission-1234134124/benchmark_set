import re


def parse_code_block(text: str) -> str | None:
    """
    Extract first python code-block from the model output.
    """
    match = re.search(r"```python(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    return None

def parse_last_code_block(text: str) -> str | None:
    """
    Extract last python code-block from the model output.
    """
    matches = re.findall(r"```python(.*?)```", text, flags=re.DOTALL)
    if matches:
        return matches[-1]
    return None


def get_code_block(text: str) -> str:
    code_block = parse_last_code_block(text)
    if code_block is None:
        code_block = ""
    code_block = code_block.replace("\r\n", "\n")
    code_block = code_block.replace("\r", "\n")
    code_block = code_block.strip("\n")

    return code_block


def normalize_indent(code: str) -> str:
    # assumes the first line is the function definition
    code = code.strip("\n")
    lines = code.splitlines()
    line = lines[0]
    indent_len = len(line) - len(line.lstrip())
    # Remove up to indent_len spaces from the start of every line
    normalized = [line[indent_len:] if len(line) >= indent_len else "" for line in lines]
    return "\n".join(normalized)

def strip_decorators(code: str) -> str:
    """
    Remove leading decorators that directly precede a function definition.
    Keeps the rest of the code unchanged.
    """
    if not code:
        return code

    lines = code.splitlines()
    lines = [line for line in lines if line.strip()]
    lines_without_decorator = []
    is_method_body = False
    for line in lines:
        if line.strip().startswith("@") and not is_method_body:
            continue
        else:
            is_method_body = True
        lines_without_decorator.append(line)
    return "\n".join(lines_without_decorator)