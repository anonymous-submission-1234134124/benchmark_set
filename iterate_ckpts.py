"""
Script to run multiturn inference with different model checkpoints.
Automatically modifies the config file for each model and runs the inference.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml


def get_checkpoint_paths(base_path: str) -> list[str]:
    """
    Parse the specified folder for directories starting with 'checkpoint-'
    and return a list of their full paths.

    Args:
        base_path: Path to the directory to search in

    Returns:
        List of full paths to checkpoint directories
    """
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Warning: Directory {base_path} does not exist")
        return []

    if not base_dir.is_dir():
        print(f"Warning: {base_path} is not a directory")
        return []

    # Find all directories starting with "checkpoint-"
    checkpoint_dirs = [
        str(path.absolute())
        for path in base_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    ]

    # Sort the paths for consistent ordering
    checkpoint_dirs.sort()

    return checkpoint_dirs


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def save_config(config: dict, config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def run_multiturn_command(config_path: str, gpu_ids: str, additional_args: list[str] = None):
    """Run the multiturn command with specified parameters."""
    cmd = ["run-multiturn", f"--config={config_path}", "--autorun", f"--gpu_ids={gpu_ids}"]

    if additional_args:
        cmd.extend(additional_args)

    print(f"Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, text=True)
        print("Command completed successfully")
        # print("STDOUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run multiturn inference with different models")
    parser.add_argument("--config", default="resources/config.yaml", help="Path to config file")
    parser.add_argument("--gpu_ids", default="0,1", help="GPU IDs to use")
    parser.add_argument("--ckpts_path", required=True, help="Path to folder with checkpoints")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with next model if current one fails",
    )

    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist")
        sys.exit(1)

    # Load original configuration
    original_config = load_config(config_path)
    res_folder = Path(original_config["paths"]["output_folder"])
    out_res_folders = [str(item) for item in os.listdir(res_folder)]
    ckpt_nums = {int(re.search(r"checkpoint-(\d+)", dir).group(1)) for dir in out_res_folders}

    models = get_checkpoint_paths(args.ckpts_path)

    # return None

    successful_runs = []
    failed_runs = []

    for model_name in models:
        checkpoint_number = int(re.search(r"checkpoint-(\d+)", model_name).group(1))
        if checkpoint_number in ckpt_nums:
            continue

        print(f"\n{'=' * 60}")
        print(f"Running inference with model: {Path(model_name).name}")
        print(f"{'=' * 60}")

        # Create modified config
        config = original_config.copy()
        config["inference"]["model_name"] = model_name

        # Save temporary config
        temp_config_path = (
            config_path.parent
            / f"temp_config_{model_name.replace('/', '_').replace('.', '_')}.yaml"
        )
        save_config(config, str(temp_config_path))

        try:
            # Run the command
            success = run_multiturn_command(str(temp_config_path), args.gpu_ids)

            if success:
                successful_runs.append(model_name)
                print(f"✓ Successfully completed inference for {model_name}")
            else:
                failed_runs.append(model_name)
                print(f"✗ Failed inference for {model_name}")
            #     if not args.continue_on_error:
            #         print("Stopping execution due to error. Use --continue-on-error to continue with next model.")
            #         break

        except Exception as e:
            failed_runs.append(model_name)
            print(f"✗ Exception during inference for {model_name}: {e}")

            if not args.continue_on_error:
                break

        finally:
            # Clean up temporary config
            if temp_config_path.exists():
                temp_config_path.unlink()
        print("Waiting 15 seconds before running next model...")
        time.sleep(15)

    # Summary
    print(f"\n{'=' * 60}")
    print("EXECUTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successful runs ({len(successful_runs)}):")
    for model in successful_runs:
        print(f"  ✓ {model}")

    if failed_runs:
        print(f"\nFailed runs ({len(failed_runs)}):")
        for model in failed_runs:
            print(f"  ✗ {model}")

    print(f"\nTotal: {len(successful_runs)} successful, {len(failed_runs)} failed")


if __name__ == "__main__":
    main()
