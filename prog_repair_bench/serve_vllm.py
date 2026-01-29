import os
import re
import signal
import subprocess
import time
from pathlib import Path

import requests
from omegaconf import OmegaConf

from prog_repair_bench.data_preprocess.s3_folders_manager import get_local_object
from prog_repair_bench.processors.config_merger import load_config_with_default

BASE_PORT = 3117
LOG_FOLDER_NAME = "logs"
VLLM_PORTS_FILENAME = "vll_ports.txt"
PID_FILENAME = "vllm_pids.txt"


def natural_sort_key(path: Path):
    s = path.name
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def get_ckpts(
    ckpts_dir: str | Path | None, ckpts_list: list[str] | None = None
) -> list[Path | None]:
    if ckpts_dir is None:
        return [None]
    ckpts_dir = Path(ckpts_dir)
    if not ckpts_dir.exists():
        raise ValueError(f"LoRA checkpoint directory {ckpts_dir} does not exist")
    if not ckpts_dir.is_dir():
        raise ValueError(f"LoRA checkpoint directory {ckpts_dir} is not a directory")

    ckpts = [ckpt for ckpt in ckpts_dir.iterdir() if ckpt.is_dir()]
    if ckpts_list:
        ckpts = [ckpt for ckpt in ckpts if ckpt.name in ckpts_list]
    ckpts = sorted(ckpts, key=natural_sort_key)

    return ckpts


def get_ckpt_names(ckpts):
    return [ckpt.name if ckpt else None for ckpt in ckpts]


def get_lora_rank(ckpts) -> int:
    ranks = []
    for ckpt in ckpts:
        if not (ckpt / "adapter_config.json").exists():
            raise ValueError(
                f"LoRA checkpoint directory {ckpt} does not contain adapter_config.json"
            )
        adapter_config = OmegaConf.load(ckpt / "adapter_config.json")
        ranks.append(adapter_config.r)
    return max(ranks)


def get_vllm_serve_commands(model_name, port, ckpts: None | list[Path] = None) -> list[str]:

    cmd = [
        "vllm",
        "serve",
        model_name,
        "--dtype",
        "auto",
        "--port",
        str(port),
        "--max-model-len",
        "32k",
    ]
    # Somehow this is needed at least for Mistral-Small-3.2-24B but not for mistralai/Ministral-8B-Instruct-2410
    if "Mistral-Small-3.2-24B".lower() in model_name.lower():
        cmd.extend(
            [
                "--tokenizer-mode",
                "mistral",
                "--config-format",
                "mistral",
                "--load-format",
                "mistral",
            ]
        )

    if ckpts == [None]:
        return cmd
    else:
        ckpts_names = get_ckpt_names(ckpts)
        rank = get_lora_rank(ckpts)
        cmd_lora = ["--enable-lora", "--max-lora-rank", str(rank), "--lora-modules"]
        print("Serving LoRA adapters:")
        for ckpt, name in zip(ckpts, ckpts_names):
            lora_ckpt = f"{name}={ckpt.absolute()}"
            print(lora_ckpt)
            cmd_lora.append(lora_ckpt)
        cmd.extend(cmd_lora)

        return cmd


def remove_port_from_file(ports: int | list[int], ports_file_path: str | Path) -> None:
    if isinstance(ports, int):
        ports = [ports]
    ports_file = Path(ports_file_path)
    if not ports_file.exists():
        print(f"{ports_file_path} does not exist.")
        return

    with open(ports_file, "r") as f:
        ports_in_file = [int(line.strip()) for line in f if line.strip()]
    ports_to_keep = [p for p in ports_in_file if p not in ports]

    with open(ports_file, "w") as f:
        for p in ports_to_keep:
            f.write(f"{p}\n")


def check_port_available(port: int) -> bool:
    """
    Check if a port is available (not in use by any process).

    Args:
        port: Port number to check

    Returns:
        bool: True if port is available, False if in use
    """
    # Find process using the port
    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)

    # If lsof returns 0 and has output, port is in use
    if result.returncode == 0 and result.stdout.strip():
        return False

    try:
        ss_result = subprocess.run(["ss", "-tuln"], capture_output=True, text=True)

        if ss_result.returncode == 0:
            # Check if port appears in ss output
            for line in ss_result.stdout.splitlines():
                if f":{port}" in line:
                    return False
    except Exception:
        pass

    return True


def get_reserved_ports(vllm_ports_file: Path) -> set[int]:
    if not vllm_ports_file.exists():
        return set()
    with open(vllm_ports_file, "r") as f:
        reserved_ports = {int(line.strip()) for line in f if line.strip()}
    return reserved_ports


def find_available_port(base_port: int, vllm_ports_file: Path, max_attempts: int = 100) -> int:
    """
    Find the next available port starting from base_port.

    Args:
        base_port: Starting port number
        max_attempts: Maximum number of ports to check

    Returns:
        int: First available port number

    Raises:
        RuntimeError: If no available port found within max_attempts
    """

    reserved_ports = get_reserved_ports(vllm_ports_file)
    for i in range(max_attempts):
        port = base_port + i
        if (port not in reserved_ports) and check_port_available(port):
            return port
        else:
            print(f"Port {port} is in use, trying next port...")

    raise RuntimeError(
        f"No available port found starting from {base_port} within {max_attempts} attempts"
    )


def check_vllm_health_all_ports(gpu_ids: list[int], ports: list[int], timeout: int = 300) -> dict:
    """
    Check if all vLLM servers are ready to serve requests

    Args:
        gpu_ids: List of GPU IDs that have vLLM servers
        base_port: Base port number (default from your config/script)
        timeout: Timeout in seconds for each server

    Returns:
        Dictionary with port as key and health status as value
    """
    health_status = {}

    for gpu_id, port in zip(gpu_ids, ports):

        url = f"http://localhost:{port}/health"

        print(f"Checking vLLM health on GPU {gpu_id} at port {port}...")

        start_time = time.time()
        is_healthy = False

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"✓ vLLM on GPU {gpu_id} (port {port}) is healthy")
                    is_healthy = True
                    break
            except requests.RequestException:
                pass  # Continue trying
            time.sleep(1)

        if not is_healthy:
            print(f"✗ vLLM on GPU {gpu_id} (port {port}) is not responding")

        health_status[port] = is_healthy

    return health_status


def serve_vllm_cli(
    config: str,
    gpu_ids: list[int] | int,
    vllm_log_dir: str | Path,
    wait_time: float = 30,
    base_port: int = BASE_PORT,
    verbose_vllm: bool = False,
) -> tuple[list[int], list[int], list[str]]:
    """
    Start vLLM servers on specified GPUs based on configuration.

    Args:
        config_path: Path to the configuration YAML file
        gpu_ids: List of GPU IDs to use for serving

    Returns:
        Tuple of (process_ids, log_file_paths)
    """
    # Load configuration
    vllm_log_dir = Path(vllm_log_dir)
    vllm_log_dir.mkdir(parents=True, exist_ok=True)
    vllm_ports_file = vllm_log_dir / VLLM_PORTS_FILENAME
    pid_file = vllm_log_dir / PID_FILENAME

    config_path = config
    config = load_config_with_default(config_path)

    local_dir = config.paths.local_dir
    model_name_or_path = get_local_object(config.inference.model_name, Path(local_dir) / "model")
    lora_ckpts_dir = get_local_object(
        config.inference.lora_ckpts_dir, Path(local_dir) / "lora_checkpoints"
    )
    ckpt_list = config.inference.lora_ckpts_list
    if ckpt_list is not None:
        ckpt_list = [ckpt for ckpt in ckpt_list if ckpt]

    ckpts = get_ckpts(lora_ckpts_dir, ckpt_list)

    # Check if provider is vllm
    if config.inference.provider != "vllm":
        raise ValueError("Provider must be 'vllm' in config")

    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    else:
        gpu_ids = list(gpu_ids)
    process_ids = []
    log_file_paths = []
    ports = []
    gpu_ids.sort()
    initial_port = base_port + min(gpu_ids)

    # Clear PID file
    with open(pid_file, "w") as f:
        f.write("")

    print(f"Starting vLLM servers for the model {model_name_or_path}")

    for i, gpu_id in enumerate(gpu_ids):

        port = find_available_port(initial_port, vllm_ports_file)
        initial_port = port + 1
        ports.append(port)

        log_file = vllm_log_dir / f"vllm_gpu{gpu_id}.log"
        log_file_paths.append(str(log_file.resolve()))

        # Prepare environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # For deterministic behaviour https://docs.nvidia.com/cuda/archive/12.6.2/cublas
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Start vLLM server
        cmd = get_vllm_serve_commands(model_name_or_path, port, ckpts)
        print(f"Starting vLLM with command:\n{' '.join(cmd)}")

        with open(vllm_ports_file, "a") as f:
            f.write(f"{port}\n")

        if verbose_vllm:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=None,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # Create new process group
            )
        else:
            with open(log_file, "w") as log_f:
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # Create new process group
                )

        process_ids.append(process.pid)
        print(f"Started vLLM on GPU {gpu_id} at port {port}, PID = {process.pid}")

        # Store PID
        with open(pid_file, "a") as f:
            f.write(f"{process.pid}\n")

        # Wait a bit before starting next server
        if len(gpu_ids) > 1 and i < len(gpu_ids) - 1:
            print(f"Waiting {wait_time} seconds before starting next server...")
            time.sleep(wait_time)

    print("All vLLM instances launched. Waiting for servers to start...")
    print(f"PIDs stored in {pid_file}")
    print(f"Log files: {log_file_paths}")

    check_vllm_health_all_ports(gpu_ids, ports, timeout=600)

    return process_ids, ports, log_file_paths


def kill_vllm_processes(pid_file: str) -> None:
    """
    Kill vLLM processes based on PIDs stored in file.

    Args:
        pid_file: Path to file containing process IDs
    """
    if not os.path.exists(pid_file):
        print(f"PID file {pid_file} not found")
        return

    killed_pids = []
    failed_pids = []

    with open(pid_file, "r") as f:
        pids = [line.strip() for line in f if line.strip()]

    for pid_str in pids:
        try:
            pid = int(pid_str)
            # Try to kill the process group first (kills child processes too)
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                print(f"Killed process group for PID {pid}")
            except ProcessLookupError:
                # Process group doesn't exist, try killing individual process
                os.kill(pid, signal.SIGTERM)
                print(f"Killed process PID {pid}")

            killed_pids.append(pid)

        except (ValueError, ProcessLookupError, PermissionError) as e:
            print(f"Failed to kill process {pid_str}: {e}")
            failed_pids.append(pid_str)

    # Wait a bit for graceful shutdown
    time.sleep(5)

    # Force kill any remaining processes
    for pid_str in [str(pid) for pid in killed_pids]:
        pid = int(pid_str)
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Force killed process PID {pid}")
        except ProcessLookupError:
            # Process already terminated
            pass
        except Exception as e:
            print(f"Error force killing {pid}: {e}")

    # Clean up PID file
    try:
        os.remove(pid_file)
        print(f"Removed PID file {pid_file}")
    except OSError as e:
        print(f"Error removing PID file: {e}")

    if killed_pids:
        print(f"Successfully killed processes: {killed_pids}")
    if failed_pids:
        print(f"Failed to kill processes: {failed_pids}")


def kill_vllm_by_port(
    vllm_ports_file: Path, ports: list[int] | None = None, base_port: int = BASE_PORT
) -> None:
    """
    Kill vLLM processes by finding processes listening on specified ports.

    Args:
        ports: List of ports to check for vLLM processes
    """
    if ports is None:
        if vllm_ports_file.exists():
            with open(vllm_ports_file, "r") as f:
                ports = [int(line.strip()) for line in f if line.strip()]
        else:
            ports = list(range(base_port, base_port + 8))
    print(f"Killing processes on ports {ports}...")
    killed_pids = []

    for port in ports:
        try:
            # Find process using the port
            result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid_str in pids:
                    try:
                        pid = int(pid_str)
                        os.kill(pid, signal.SIGTERM)
                        killed_pids.append(pid)
                        print(f"Killed process PID {pid} on port {port}")
                    except (ValueError, ProcessLookupError) as e:
                        print(f"Failed to kill process {pid_str} on port {port}: {e}")
            else:
                print(f"No process found on port {port}")

        except FileNotFoundError:
            print("lsof command not found. Please install it or use kill_vllm_processes() instead.")
        except Exception as e:
            print(f"Error checking port {port}: {e}")

    if killed_pids:
        print(f"Successfully killed processes: {killed_pids}")

    remove_port_from_file(ports, vllm_ports_file)


def kill_vllm_cli(
    vllm_log_dir: str | Path, pid_file: str | None = None, ports: list[int] | None = None
) -> None:
    """
    CLI wrapper for kill_vllm_processes function.

    Args:
        pid_file: Path to file containing process IDs
    """
    vllm_log_dir = Path(vllm_log_dir)
    if pid_file is None:
        pid_file = vllm_log_dir / PID_FILENAME
    vllm_ports_file = vllm_log_dir / VLLM_PORTS_FILENAME
    kill_vllm_processes(pid_file)
    kill_vllm_by_port(ports=ports, vllm_ports_file=vllm_ports_file)
