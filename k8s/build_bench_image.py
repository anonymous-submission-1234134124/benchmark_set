"""Build docker image and submit Job to k8s.

Usage:
    python k8s/build_bench_image.py
"""

import subprocess
import sys


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


def build_and_push_docker_image():
    tag = get_git_hash()

    # TODO Restore
    docker_image_tag = f"base_image:{tag}"

    print(f"--- Building Docker Image: {docker_image_tag} ---")
    docker_build_cmd = [
        "docker",
        "build",
        "-f",
        "./k8s/Dockerfile",
        "-t",
        docker_image_tag,
        "--platform",
        "linux/amd64",
        ".",  # The build context
    ]

    run_command(docker_build_cmd)

    print(f"\n--- Pushing Docker Image: {docker_image_tag} ---")
    docker_push_cmd = ["docker", "push", docker_image_tag]
    run_command(docker_push_cmd)
    print(f"Docker image pushed: {docker_image_tag}")

    with open("prog_repair_bench/resources/image_tag.txt", "w") as f:
        f.write(tag)

    return tag  # Return the tag for use in job creation


if __name__ == "__main__":
    build_and_push_docker_image()
