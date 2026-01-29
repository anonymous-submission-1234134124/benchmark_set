import json
from pathlib import Path


def read_jsonl(filename: str | Path) -> list[dict]:

    results = []

    with open(filename, encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # Parse JSON line
                data = json.loads(line.strip())
                results.append(data)

    return results


def save_jsonl(filename: str | Path, data: list[dict], mode="a") -> None:
    with open(filename, mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def get_simple_results(results: list[dict]) -> list[dict]:

    result_simple = []
    for item in results:
        line_nums = item["dp_item"]["method"]["global_method_body_index"]
        simp_dict = {
            "split": item["dp_item"]["split"],
            "idx": item["dp_item"]["idx"],
            "dp_id": item["dp_item"]["dp_id"],
            "turn": item["turn"],
            "stop_reason": item["stop_reason"],
            "percentage_passed": item["percentage_passed"],
            "num_lines": line_nums[1] - line_nums[0] + 1,
            "timestamp": item["timestamp_start"],
            "test_status": item["test_status"],
            "duration": item["duration"],
        }
        
        if item.get("test_results"):
            last_res = item["test_results"][-1]
            for metric in ["chrf", "common_prefix_ratio", "match_method_name"]:
                if metric in last_res:
                    simp_dict[metric] = last_res[metric]

        result_simple.append(simp_dict)

    return result_simple


def covert_to_simple_res(outputs_dir: str | Path) -> None:

    outputs_dir = Path(outputs_dir)

    for folder in outputs_dir.iterdir():
        if not folder.is_dir():
            continue
        results_path = folder / "results.jsonl"
        results = read_jsonl(results_path)
        results_simp = get_simple_results(results)
        simp_res_path = outputs_dir / f"{folder.name}.jsonl"
        save_jsonl(simp_res_path, results_simp, mode="w")
        print(f"{folder.name} converted to simple results and saved to {simp_res_path}")
