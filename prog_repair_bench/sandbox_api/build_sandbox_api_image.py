import subprocess
import sys
from pathlib import Path


def get_git_hash(short=True):
    try:
        cmd = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
        result = subprocess.check_output(cmd).decode("utf-8").strip()
        return result
    except subprocess.CalledProcessError:
        return None


def _rewrite_dockerfile_cmd_to_temp(dockerfile_path: Path, repo: str, num_servers) -> Path:
    """
    Read the given dockerfile, strip lines, replace the last line (expected to start with CMD)
    with the required JSON-array form, write to a temp file, and return its path.
    """
    docker_commands = (dockerfile_path.read_text(encoding="utf-8")).strip()
    content = docker_commands.splitlines()
    if not content:
        raise ValueError("Dockerfile is empty")

    # Ensure it starts with CMD and replace it
    last_line = content[-1]
    if not last_line.startswith("CMD"):
        raise ValueError("Last non-empty line of Dockerfile does not start with CMD")

    # TODO Restore
    new_cmd = (
        'CMD ["uvicorn", "sandbox_django.api:app", "--host", "0.0.0.0", "--port", "8000"]'
    )
    new_cmd = 'CMD ["python", "prog_repair_bench/sandbox_api/run_api.py", "--repo , "sympy", "--num_servers", "4"]'
    content[-1] = new_cmd
    new_commands = "\n".join(content) + "\n"

    # Write to a temp file placed next to the original Dockerfile for correct build context
    tmp_dockerfile = Path("Dockerfile_tmp")
    tmp_dockerfile.write_text(new_commands, encoding="utf-8")
    return tmp_dockerfile


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

    # tag = get_git_hash()
    # TODO Restore
    docker_image_tag = (
        "registry/image/path"
    )
    here = Path(__file__).parent
    dockerfile_path = here / "Dockerfile"

    print(f"--- Building Docker Image: {docker_image_tag} ---")
    # tmp_dockerfile = _rewrite_dockerfile_cmd_to_temp(dockerfile_path)

    docker_build_cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile_path),
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


if __name__ == "__main__":
    build_and_push_docker_image()
