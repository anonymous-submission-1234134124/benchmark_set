
import math
from datasets import Dataset as HFDataset

def split_method_for_completion(method_code: str, method_declaration: str) -> tuple[str, str]:
    """
    Split a method into two parts for code completion task.
    
    Rules:
    - Split follows line breaks (no mid-line splits)
    - Empty lines are NOT counted when calculating the split point
    - For methods with 0-1 non-empty body lines: prefix = declaration, target = full body
    - For methods with 2+ non-empty body lines: split roughly in half
    - First part includes method declaration + first half of body
    - Second part is what needs to be completed
    
    Args:
        method_declaration: The method declaration line
        method_body: The method body
        
    Returns:
        Tuple of (prefix_code, completion_target)
        - prefix_code: The first half to provide as context
        - completion_target: The second half that should be generated
    """
    declaration_lines = (method_declaration+'\n').splitlines(keepends=True)
    body_lines = method_code.splitlines(keepends=True)
    
    non_empty_body_lines = [line for line in body_lines if line.strip()]
    num_non_empty_lines = len(non_empty_body_lines)
    
    # Special case: 0-1 non-empty lines in body
    # Generate the entire body from signature
    if num_non_empty_lines <= 1:
        return "".join(declaration_lines), "".join(body_lines)
    
    # Number of non-empty lines to generate (round up half)
    non_empty_to_generate = math.ceil(num_non_empty_lines / 2)
    
    # Find the split point by counting non-empty lines
    non_empty_count = 0
    split_idx = 0
    
    for i, line in enumerate(body_lines):
        if line.strip():  # Non-empty line
            non_empty_count += 1
            if non_empty_count >= (num_non_empty_lines - non_empty_to_generate):
                split_idx = i + 1
                break
    
    lines_to_keep = split_idx
    
    # Check if we're splitting in the middle of a docstring
    # If so, include the complete docstring in the prefix
    if lines_to_keep < len(body_lines):
        lines_to_keep = _adjust_for_docstring(body_lines, lines_to_keep)
    
    # Split the method
    prefix_lines = declaration_lines + body_lines[:lines_to_keep]
    completion_lines = body_lines[lines_to_keep:]
    
    prefix_code = "".join(prefix_lines)
    completion_target = "".join(completion_lines)
    
    return prefix_code, completion_target


def _adjust_for_docstring(body_lines: list[str], lines_to_keep: int) -> int:
    """
    Adjust split point to avoid breaking in the middle of a docstring.
    
    Args:
        body_lines: Lines of the method body
        lines_to_keep: Current split point
        
    Returns:
        Adjusted split point that doesn't break docstrings
    """
    in_docstring = False
    docstring_quote = None
    
    for i in range(lines_to_keep):
        line = body_lines[i].strip()
        # Check for docstring start
        if line.startswith('"""') or line.startswith("'''"):
            quote = '"""' if line.startswith('"""') else "'''"
            # Check if it's a single-line docstring
            if line.count(quote) >= 2:
                continue  # Complete docstring on one line
            else:
                in_docstring = True
                docstring_quote = quote
        # Check for docstring end
        elif in_docstring and docstring_quote in line:
            in_docstring = False
            docstring_quote = None
    
    return lines_to_keep


def augment_dataset_with_completion(dataset: HFDataset) -> HFDataset:
    """
    Augments the dataset with 'completion' field if not present.
    The 'completion' field contains:
    - prompt_file_context: The file content with the target method cut in half.
    - half_method: The prefix of the method (what remains).
    - continuation: The suffix of the method (what needs to be generated).
    - full_method: The full method code.
    """

    def process_item(item):
        '''
        if 'completion' in item and item['completion']:
            return item
        '''

        method_info = item['method']
        method_declaration = method_info['declaration']
        method_body = method_info['body']

        try:
            prefix, target = split_method_for_completion(method_body, method_declaration)
            
            # Fallback if target empty (edge case)
            if not target or not target.strip():
                if method_body and method_body.strip():
                    target = method_body
                    prefix = method_declaration
                    if not prefix.endswith('\n'):
                        prefix += '\n'
                else:
                    return item
            
            # Prepare file context (file content up to split + prefix)
            raw_file = item['raw_file_content']
            file_lines = raw_file.splitlines(keepends=True)
            start_idx = method_info['global_method_declaration_index'][0]
            
            # Before method
            before_method = "".join(file_lines[:start_idx])
            prompt_file_context = before_method + prefix
            full_method = prefix + target

            item['completion'] = {
                'prompt_file_context': prompt_file_context,
                'half_method': prefix,
                'continuation': target,
                'full_method': full_method
            }
        except Exception as e:
            raise ValueError(f"Error processing completion for item {item.get('idx')}: {e}")
            
        return item

    return dataset.map(process_item)
