"""Build docker image and submit Job to k8s.

Usage:
    python k8s/build_bench_image.py --config k8s/config.yaml
    python k8s/build_bench_image.py --config k8s/config.yaml --just_build_image
    python k8s/build_bench_image.py --config k8s/config.yaml --image_tag 7053ff3
"""

import os
import random
import re
import string
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
import base64
from dotenv import load_dotenv
import json
import urllib.parse

from fire import Fire
from omegaconf import OmegaConf, DictConfig
from prog_repair_bench.processors.config_merger import load_config_with_default

MAX_CHARACTERS = 40

# TODO Restore
LOKI_BASE_URL = "https://loki.com"
LOKI_UID = "af3p6xjljiqyoa"
LOKI_ORG = "37"
LOKI_FROM = "now-24h"
LOKI_TO = "now"


def get_pod_name(job_name, namespace=None):
    """Return the first pod name belonging to a given Kubernetes Job."""
    cmd = ["kubectl", "get", "pods", f"--selector=job-name={job_name}"]
    if namespace:
        cmd.extend(["-n", namespace])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        print(f"⚠️ No pods found for job '{job_name}'")
        return None

    pod_name = lines[1].split()[0]
    return pod_name

def get_log_link(pod: str, ns: str | None = None):

    expr = f'{{pod="{pod}"}} |= ``' if not ns else f'{{pod="{pod}", namespace="{ns}"}} |= ``'

    panes = {
        "vkw": {
            "datasource": LOKI_UID,
            "queries": [{
                "refId": "A",
                "expr": expr,
                "queryType": "range",
                "datasource": {"type": "loki", "uid": LOKI_UID},
                "editorMode": "builder",
                # "direction": "forward"
            }],
            "range": {"from": LOKI_FROM, "to": LOKI_TO}
        }
    }

    qs = {
        "schemaVersion": "1",
        "panes": json.dumps(panes, separators=(",", ":")),
        "orgId": LOKI_ORG
    }
    log_link = f"{LOKI_BASE_URL}?{urllib.parse.urlencode(qs)}"

    return log_link

def get_rnd_string(length):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def get_git_hash(short=True):
    try:
        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        result = subprocess.check_output(cmd).decode("utf-8").strip()
        return result
    except subprocess.CalledProcessError:
        return None


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


def verify_prog_repair_config(config: DictConfig):

    output_folder = config.paths.output_folder
    # data_folder = config.data.path
    sandbox_provider = config.inference.sandbox_provider
    lora_ckpts_dir = config.inference.lora_ckpts_dir

    assert output_folder.startswith(
        "s3://"
    ), "Output folder must be an S3 URI to run benchmark on k8s"
    # assert data_folder.startswith("s3://"), "Data folder must be an S3 URI to run benchmark on k8s"
    assert sandbox_provider in {
        "restapi",
    }, "Sandbox provider must be 'restapi' to run benchmark on k8s"
    assert lora_ckpts_dir is None or lora_ckpts_dir.startswith(
        "s3://"
    ), "Lora ckpts dir must be an S3 URI to run benchmark on k8s"


def validate_k8s_name(name):
    """Check if a string is a valid k8s resource name."""
    print(f"Validating k8s name: {name}")
    if len(name) > MAX_CHARACTERS:
        raise ValueError(f"job name is too long: {name}. Max length is 63 characters. ")
    pattern = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
    if not bool(re.match(pattern, name)):
        raise ValueError(
            f"job name is not a valid k8s resource name: {name}. Must match the regex: {pattern}. "
        )


def sanitize_username(username):
    """Sanitize the username to be a valid k8s resource name."""
    username = username.lower().replace(".", "-")
    return username


def setup_secrets(k8s_conf):
    """Create or update the wandb secret in k8s."""
    secrets = k8s_conf.get("secrets", {})
    if not secrets:
        return None

    try:
        run_command(
            [
                "kubectl",
                "-n",
                k8s_conf.namespace,
                "delete",
                "secret",
                "custom-secrets",
                "--ignore-not-found=true",
            ]
        )
    except subprocess.CalledProcessError:
        pass

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for k, v in secrets.items():
            f.write(f"{k}={v}\n")
        env_file = f.name

    try:
        run_command(
            [
                "kubectl",
                "-n",
                k8s_conf.namespace,
                "create",
                "secret",
                "generic",
                "custom-secrets",
                "--from-env-file",
                env_file,
            ]
        )
        print("Created/updated custom-secrets in k8s")
    finally:
        try:
            os.remove(env_file)
        except FileNotFoundError:
            pass


def create_config_map(run_conf_path: str, job_name: str, namespace: str, file_key: str):
    cfg_id = get_rnd_string(6)
    conf_name = job_name + f"-config-{cfg_id}"
    if len(conf_name) > MAX_CHARACTERS:
        conf_name = conf_name[-MAX_CHARACTERS:]
    validate_k8s_name(conf_name)
    cmd = [
        "kubectl",
        "create",
        "-n",
        namespace,
        "configmap",
        conf_name,
        f"--from-file={file_key}={run_conf_path}",
    ]
    run_command(cmd)
    return conf_name

def create_job_yaml(k8s_conf, job_template, run_conf_path, git_hash, prompts_path: str | None = None):

    username = sanitize_username(os.environ.get("USER"))
    job_name_suffix = k8s_conf.get("job_name_suffix", uuid.uuid4().hex)[:6]
    job_name = f"prog-repair-{username[:6]}-{git_hash}-{job_name_suffix}"
    validate_k8s_name(job_name)

    conf_name = create_config_map(run_conf_path, job_name, k8s_conf.namespace, file_key="config.yaml")

    if prompts_path:
        if not Path(prompts_path).exists():
            raise FileNotFoundError(f"prompts file not found: {prompts_path}")
        conf_name_prompts = create_config_map(prompts_path, job_name, k8s_conf.namespace, file_key="prompts.yaml")
    else:
        conf_name_prompts = "prompts-are-default"

    template_vals = OmegaConf.merge(k8s_conf, {
        "username": username,
        # TODO Restore
        "image_name": f"registry/image/path:{git_hash}",
        "job_name": job_name,
        "command": ["/bin/bash", "-c", k8s_conf["command"]],
        # We make sure inside main() that these variable were set.
        "wandb_email": os.environ["WANDB_EMAIL"],
        # "wandb_project": os.environ["WANDB_PROJECT"],
        # "app_config_yaml_base64": base64.b64encode(OmegaConf.to_yaml(train_conf).encode("utf-8")).decode("utf-8"),
        "configMap": conf_name,
        "configMapPrompts": conf_name_prompts,
    })

    job_conf = OmegaConf.merge(job_template, template_vals)

    OmegaConf.resolve(job_conf)

    for key in template_vals.keys():
        job_conf.pop(key)

    container = job_conf["spec"]["template"]["spec"]["containers"][0]
    additional_envs = k8s_conf.get("env", {})
    container["env"].extend([{"name": key, "value": value} for key, value in additional_envs.items()])

    return job_conf, job_name, conf_name


def get_job_uid(job_name: str, namespace: str, dry_run: bool = False) -> str:
    cmd = [
        "kubectl",
        "-n",
        namespace,
        "get",
        "job",
        job_name,
        "-o",
        "jsonpath={.metadata.uid}",
    ]
    print(f"Executing command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    if dry_run:
        return "DRY_RUN_JOB_UID"

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def set_owner_reference(
        job_name: str,
        job_uid: str,
        config_map_name: str,
        namespace: str,
) -> None:
    patch_body = {
        "metadata": {
            "ownerReferences": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "name": job_name,
                    "uid": job_uid,
                }
            ]
        }
    }
    cmd = [
        "kubectl",
        "-n",
        namespace,
        "patch",
        "configmap",
        config_map_name,
        "--type",
        "merge",
        "--patch",
        json.dumps(patch_body),
    ]
    run_command(cmd)


def main(config, k8s_config_file, image_tag=None, prompts_yaml: str | None = None) -> None:

    load_dotenv()
    run_conf_path = config
    train_config = load_config_with_default(run_conf_path)

    if image_tag is None:
        image_tag_file = Path(__file__).parent / "resources" / "image_tag.txt"
        with open(image_tag_file) as f:
            image_tag = f.read().strip()

    k8s_conf = OmegaConf.load(k8s_config_file)
    verify_prog_repair_config(train_config)
    if train_config.inference.provider == "litellm":
        k8s_conf.gpu_requests = 0
        k8s_conf.gpu_limits = 0
        k8s_conf.cpu_requests = 6
        k8s_conf.cpu_limits = 6

    job_template = OmegaConf.load(k8s_conf.get("job_template", "resources/job_template.yaml"))

    git_hash = image_tag

    setup_secrets(k8s_conf)

    job_yaml, job_name, conf_name = create_job_yaml(k8s_conf, job_template, run_conf_path, git_hash, prompts_yaml)

    print(OmegaConf.to_yaml(job_yaml))
    # delete = True for temp file does not work properly on Windows.
    # On Windows an opened NamedTemporaryFile is kept locked for exclusive access;
    # while the file handle is still alive another process (kubectl in this case) cannot reopen it.
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as f:
        tmp_path = f.name
    OmegaConf.save(job_yaml, tmp_path)
    try:
        cmd = ["kubectl", "-n", k8s_conf.namespace, "apply", "-f", tmp_path]
        run_command(cmd)
    finally:
        os.remove(tmp_path)

    # Associate the config map with the job, so that it is deleted after
    # the job is finished.
    job_uid = get_job_uid(job_name, k8s_conf.namespace)
    set_owner_reference(job_name, job_uid, conf_name, k8s_conf.namespace)

    time.sleep(5)
    pod_name = get_pod_name(job_name, k8s_conf.namespace)
    if pod_name:
        print(f"Pod name: {pod_name}")
        log_link = get_log_link(pod_name, k8s_conf.namespace)
        print(f"Link to logs:\n\n{log_link}")
        print(f"\n\nTo get logs in the console, run the following command:\nkubectl logs {pod_name}")
        print(f"\n\nTo get into pod console, run the following command:\nkubectl exec -it {pod_name} -n {k8s_conf.namespace} -- /bin/bash")
    else:
        print("Could not generate the link to the logs.")


def run_on_k8s():
    """
    Run the main function with Fire CLI.
    """

    Fire(main)


if __name__ == "__main__":
    Fire(main)