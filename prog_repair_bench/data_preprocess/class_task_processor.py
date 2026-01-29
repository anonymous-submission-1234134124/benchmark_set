from collections import defaultdict
from copy import deepcopy
from datasets import Dataset


def is_list_of_float(obj) -> bool:
    return isinstance(obj, list) and all(isinstance(x, float) for x in obj)


# Python
def is_list_of_str(obj) -> bool:
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)


def combine_lists(test_lists: list, max_tests: int = 20):
    test_lists_copy = deepcopy(test_lists)
    tests_combined = set()
    added_test = True
    while (len(tests_combined) <= max_tests) and added_test:
        added_test = False
        for test_list in test_lists_copy:
            if test_list:
                test = test_list.pop(0)
                tests_combined.add(test)
                added_test = True
    return list(tests_combined)


def process_tests(tests) -> dict:
    tests_processed = defaultdict(lambda: defaultdict(list))

    for key, value in tests.items():
        for d_item in value:
            for k, v in d_item.items():
                tests_processed[key][k].append(v)
        tests_processed[key] = dict(tests_processed[key])
    tests_processed = dict(tests_processed)

    for key, value in tests_processed.items():
        for k, v in value.items():
            if is_list_of_float(v):
                value[k] = min(v)
            elif is_list_of_str(v):
                value[k] = v[0]
            else:
                value[k] = combine_lists(v)

    return tests_processed


def process_class_dataset(dataset):
    method_start_lines = defaultdict(list)
    tests = defaultdict(list)

    for item in dataset:
        method_start_line_num = item["method"]["global_method_declaration_index"]
        class_id = item["file_path"] + item["class_name"]
        method_start_lines[class_id].append(method_start_line_num[0])
        tests[class_id].append(item["tests"])

    tests = process_tests(tests)
    classes_done = set()
    items = []

    for item in dataset:
        class_id = item["file_path"] + item["class_name"]
        if class_id in classes_done:
            continue
        file_content = item["raw_file_content"]
        class_start_line_num = item["class_position_in_file"]
        methods_start_line_num = min(method_start_lines[class_id])
        class_lines = item["class_code"].splitlines()
        class_num_lines = len(class_lines)
        item["global_class_declaration_index"] = [class_start_line_num, class_start_line_num]
        item["global_class_vars_index"] = [class_start_line_num + 1, methods_start_line_num - 1]
        item["global_class_body_index"] = [
            methods_start_line_num,
            class_start_line_num + class_num_lines - 1,
        ]
        file_lines = file_content.splitlines()
        before_class = file_lines[:class_start_line_num]
        after_class = file_lines[item["global_class_body_index"][1] + 1:]
        item["file_without_class"] = "\n".join(before_class + after_class)
        item["class_declaration"] = file_lines[class_start_line_num]
        class_vars = "\n".join(
            file_lines[item["global_class_vars_index"][0]: item["global_class_vars_index"][1]]
        )
        item["class_vars"] = class_vars if class_vars.strip() else '"""Empty docstring"""'
        item["tests_class"] = tests[class_id]
        items.append(item)
        classes_done.add(class_id)

    return Dataset.from_list(items)