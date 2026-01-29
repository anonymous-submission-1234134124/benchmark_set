import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from fire import Fire
from omegaconf import OmegaConf


def run_command(command, cwd=None):
    print(
        f"Executing command: {' '.join(command) if isinstance(command, list) else command}"
    )  # Optional: keep for debugging
    subprocess.run(
        command,
        check=True,  # Raise CalledProcessError if non-zero exit code
        text=True,  # Decode stdout/stderr as text
        cwd=cwd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _inject_env(doc: dict, repo: str, num_servers: int):
    # Works for Deployment/Job with template.spec.containers[0]
    containers = doc["spec"]["template"]["spec"]["containers"]
    if not containers:
        return
    container = containers[0]
    env = container.get("env")
    # Avoid duplicates
    env[:] = [
        e for e in env if not (isinstance(e, dict) and e.get("name") in {"REPO", "NUM_SERVERS"})
    ]
    env.append({"name": "REPO", "value": repo})
    env.append({"name": "NUM_SERVERS", "value": str(num_servers)})


def create_job_yaml(job_template_path: str | Path, repo: str = "sympy", num_servers: int = 4, namespace: str = "namespace"):
    with open(job_template_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    if not docs:
        raise ValueError("YAML template is empty.")
    # Inject into first workload-like doc (your Deployment is second doc)
    res = []
    for doc in docs:
        doc = OmegaConf.create(doc)
        template_conf = {"namespace": namespace}
        doc = OmegaConf.merge(doc, template_conf)
        OmegaConf.resolve(doc)

        for k in template_conf.keys():
            doc.pop(k)

        if isinstance(doc, dict) and doc.get("kind") in {
            "Deployment",
            "Job",
            "StatefulSet",
            "DaemonSet",
        }:
            _inject_env(doc, repo, num_servers)
            break
        res.append(doc)

    return res


def submit_sandbox_service(
    repo: str = "sympy", num_servers: int = 4, job_template_path: str | None = None,
    namespace: str = "namespace",
) -> None:

    if job_template_path is None:
        job_template_path = Path(__file__).parent / "sandbox-api.yaml"

    docs = create_job_yaml(job_template_path, repo, num_servers, namespace)

    # Print final YAML
    yaml_data = "---\n".join(OmegaConf.to_yaml(doc) for doc in docs)
    print(yaml_data)

    # Write and apply
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".yaml", encoding="utf-8"
    ) as f:
        tmp_path = f.name
        f.write(yaml_data)
    try:
        run_command(["kubectl", "apply", "-f", tmp_path])
    finally:
        os.remove(tmp_path)


def kill_sandbox_service() -> None:
    try:
        run_command(["kubectl", "delete", "deployment", "sandbox-sympy-api"])
    except Exception:
        pass
    try:
        run_command(["kubectl", "delete", "service", "sandbox-sympy-api"])
    except Exception:
        pass


def submit_sandbox_service_on_k8s():
    """
    Run the main function with Fire CLI.
    """
    Fire(submit_sandbox_service)


def kill_sandbox_service_on_k8s():
    kill_sandbox_service()
