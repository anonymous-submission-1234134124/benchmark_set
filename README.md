# Repository-based multi-task benchmark suite

This repository accompanies the "Position: Current Coding Benchmarks Measure Task Performance, not Coding Capabilities" ICML submission. In it, we provide the code for the benchmark suite described in the paper. The accompanying test dataset can be found at [anonauthor/django_multi_task](https://huggingface.co/datasets/anonauthor/django_multi_task).

We provide three benchmarks for evaluating LLMs on code-related tasks:

- method generation
- completion
- program repair

Our benchmark uses Sandbox testing infrastructure to run generated code in a secure environment and provide feedback.
Here **_Sandbox_** stands for a sandboxed Python environment for testing generated code. It is proprietary for now, but hopefully would be released soon.
For now we suggest implementing the REST API service for running the tests in a sandboxed environment. Example could be seen in `prog_repair_bench/sandbox_api/api.py`. Thus, before the release of Sandbox our benchmark sadly cannot be run out of the box.

Method generation task would be supporting two repositories - [django](https://github.com/django/django) and [sympy](https://github.com/sympy/sympy).
For now we only support django.

For the method generation task, the benchmark supports supplementary feature - multi-turn method generation operating in the feedback loop:

1. An LLM generates method implementations or fixes according to a description/context.
2. The generated code is tested against real test suites using a Python test execution environment (Docker-based sandbox).
3. Test results and feedback are provided back to the model.
4. The process repeats until tests pass or maximum iterations are reached.

## Installation

### Local Runs
For local runs, install the package with local dependencies:
`poetry install --extras local`

### Kubernetes (k8s) Runs
For Kubernetes runs, install the bare package. It is lightweight and contains only the `omegaconf` package and formatters:
`poetry install --extras local`

## Configuration

The configuration is managed via `config.yaml`. It is recommended to review these before running the benchmark.

### Config Keys (resources/config.yaml)
- `repository`: The project to benchmark (e.g., "django", "sympy"), for now only "django" supported.
- `benchmark`: The task to run. Options:
    - `method_gen_multiturn`: Generates method implementations. To run method generation benchmark in single-turn setup (as done in the paper), set `max_iter` key described below to `1`.
    - `completion`: Completion task.
    - `program_repair`: Fixing bugs in existing code.
- `paths`: S3 or local paths for data, outputs, and checkpoints.
- `inference`: LLM parameters (model, lora checkpoionts, temperature, etc.) and benchmark-specific settings (max iterations, feedback limits).

More fine-grained keys can be found in the comments of `resources/config_default.yaml`.

## Prerequisites

### Kubernetes
If you're planning to run benchmark on Kubernetes,
Install [`kubectl`](https://kubernetes.io/docs/tasks/tools/).

### AWS Credentials (for local running)
Data is typically downloaded from S3. You need to provide AWS credentials:
```bash
pip install awscli
aws configure
```

### Environmental Variables
Configure your API keys:
```bash
export WANDB_API_KEY="..."
export WANDB_EMAIL=name.surname@domain.com
export LITELLM_KEY="..." # If using LiteLLM
```
You can also use a `.env` file.

## Usage

### 1. Generate Configuration
First, generate the config and edit if needed:
```bash
# Copy default resources to "resources" directory. Includes configs and prompts.
get-resources resources
```
You MUST provide a config file to the main script.

### 2. Switching Between Benchmarks
To switch between tasks, edit the `benchmark` key in `resources/config.yaml`.
Examples:
- `benchmark: "method_gen_multiturn"`
- `benchmark: "completion"`
- `benchmark: "program_repair"`

### 3. Running Locally
You can run the benchmark with auto-serving vLLM:
```bash
run-multiturn --config=resources/config.yaml --autorun --gpu_ids=0,1 --max_items=10
```
- `--repo=django`: Specifies the repository (overrides config).
- `--max_items`: Useful for debugging (runs only N items).

### 4. Running on Kubernetes
```bash
run-multiturn-k8s --config=resources/config.yaml --k8s_config_file=resources/config_k8s.yaml
```
Ensure S3 paths for output and checkpoints are provided in the config when running on k8s.

Most important keys in `config_k8s.yaml` correspond to the k8s resources:

```yaml
gpu_requests: 1
gpu_limits: 1
cpu_requests: 16
cpu_limits: 16
 # RAM size
memory_requests: "50Gi"
memory_limits: "50Gi"
 # Disk size, i.e. for model and results storage
tmp_disk_size: "50Gi"
```

Others keys provide some parameters of the k8s cluster and env variables.

### 5. Sandbox Options
Controlled by `inference.sandbox_provider` in `config.yaml`:
- `local`: Runs tests in local Docker containers.
- `restapi`: Uses a REST API service (default for k8s runs).

## Managing vLLM Servers
If not using `--autorun`, you can manage servers manually:
```bash
serve-vllm --config=resources/config.yaml --gpu_ids=0,1 --base_port=3117
kill-vllm
```
