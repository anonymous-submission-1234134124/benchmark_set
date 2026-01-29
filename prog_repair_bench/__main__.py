"""
Entry point for the package.
This allows running the package with `python -m prog_repair_bench`.
"""

import asyncio
from pathlib import Path

from fire import Fire
from omegaconf import OmegaConf

from prog_repair_bench.run_multiturn_inference import main
from prog_repair_bench.serve_vllm import LOG_FOLDER_NAME, kill_vllm_cli, serve_vllm_cli
from prog_repair_bench.processors.config_merger import load_config_with_default


def serve_vllm():
    """Entry point for serve-vllm command"""
    Fire(serve_vllm_cli)


def kill_vllm():
    """Entry point for kill-vllm command"""
    Fire(kill_vllm_cli)


def run_with_autorun(**kwargs):
    autorun = kwargs.pop("autorun", False)
    if autorun:
        print("Running in autorun mode!")
        config = load_config_with_default(kwargs["config"])
        print("Running config:")
        print(70 * "-")
        print(OmegaConf.to_yaml(config))
        print(70 * "-")
        vllm_log_dir = Path(config.paths.local_dir) / LOG_FOLDER_NAME
        vllm_keys = {"gpu_ids", "wait_time", "base_port", "verbose_vllm"}
        vllm_args = {key: kwargs.pop(key) for key in vllm_keys if key in kwargs}
        vllm_args["config"] = kwargs["config"]
        vllm_args["vllm_log_dir"] = vllm_log_dir
        ports = None
        try:
            if config.inference.provider == "vllm" and not config.inference.get("check_ground_truth", False):
                _, ports, _ = serve_vllm_cli(**vllm_args)
            kwargs["ports"] = ports
            asyncio.run(main(**kwargs))
        except Exception as e:
            raise e
        finally:
            if config.inference.provider == "vllm" and not config.inference.get("check_ground_truth", False):
                print("Finished running! Killing the vLLM server.")
                kill_vllm_cli(ports=ports, vllm_log_dir=vllm_log_dir)
    else:
        # Normal execution
        asyncio.run(main(**kwargs))


def run_main():
    """
    Run the main function with Fire CLI.
    """

    Fire(run_with_autorun)


if __name__ == "__main__":
    run_main()
