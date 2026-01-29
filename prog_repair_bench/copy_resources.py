import shutil
from pathlib import Path

from fire import Fire


def _overwrite(path: Path, item_type: str) -> bool:
    """
    Prompt user for overwrite confirmation.

    Args:
        path: Path that would be overwritten
        item_type: Type of item (file/directory) for user message

    Returns:
        True if user wants to overwrite, False otherwise
    """
    while True:
        response = input(f"{item_type} '{path}' already exists. Overwrite? (y/n): ").lower().strip()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def copy_config(destination: str | Path = "config.yaml") -> None:
    """
    Copy the default configuration file to a specified location.

    Args:
        destination: Path where the config file should be copied. Defaults to "config.yaml".
    """
    config_path = Path(__file__).parent / "resources" / "config.yaml"
    dest_path = Path(destination)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    # Check if destination exists and ask for overwrite
    if dest_path.exists():
        if not _overwrite(dest_path, "File"):
            print("Operation cancelled.")
            return

    # Create parent directory if it doesn't exist
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, dest_path)
    print(f"Config file copied to: {dest_path.absolute()}")


def copy_prompts(destination: str | Path = "prompts.yaml") -> None:

    for prompt_type in ["method", "class"]:
        copy_prompts_single(destination, prompt_type)


def copy_prompts_single(
    destination: str | Path = "prompts.yaml", prompt_type: str = "method"
) -> None:
    """
    Copy the default prompt to a specified location.

    Args:
        destination: Path where the prompts should be copied. Defaults to "prompts.yaml".
    """
    prompts_path = Path(__file__).parent / "resources" / f"prompts_{prompt_type}.yaml"
    dest_path = Path(destination)
    dest_path = dest_path.with_stem(f"{dest_path.stem}_{prompt_type}")

    if dest_path.exists():
        if not _overwrite(dest_path, "File"):
            print("Operation cancelled.")
            return
        dest_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(prompts_path, dest_path)
    print(f"Prompts copied to: {dest_path.absolute()}")


def copy_dockerfile(destination: str | Path = "dockerfile_commands_django") -> None:
    """
    Copy the default docker file commands to a specified location.

    Args:
        destination: Path where the docker file commands should be copied. Defaults to "dockerfile_commands".
    """
    prompts_path = Path(__file__).parent / "resources" / "dockerfile_commands_django"
    dest_path = Path(destination)

    if dest_path.exists():
        if not _overwrite(dest_path, "File"):
            print("Operation cancelled.")
            return
        dest_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(prompts_path, dest_path)
    print(f"Dockerfile copied to: {dest_path.absolute()}")


def copy_dockerfile_sympy(destination: str | Path = "dockerfile_commands_sympy") -> None:
    """
    Copy the default docker file commands to a specified location.

    Args:
        destination: Path where the docker file commands should be copied. Defaults to "dockerfile_commands_sympy".
    """
    prompts_path = Path(__file__).parent / "resources" / "dockerfile_commands_sympy"
    dest_path = Path(destination)

    if dest_path.exists():
        if not _overwrite(dest_path, "File"):
            print("Operation cancelled.")
            return
        dest_path.unlink()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(prompts_path, dest_path)
    print(f"Dockerfile copied to: {dest_path.absolute()}")


def copy_resources(destination: str | Path = "resources") -> None:
    """
    Copy all files from the target resources folder to a specified destination,
    excluding 'image_tag.txt'.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    # Source folder
    source_path = Path(__file__).parent / "resources"

    # Iterate over files in the source directory
    for item in source_path.iterdir():
        if item.name == "image_tag.txt":
            continue

        dest_item = destination / item.name

        if item.is_dir():
            shutil.copytree(item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest_item)


def get_config():
    """Entry point for serve-vllm command"""
    Fire(copy_config)


def get_prompts():
    """Entry point for serve-vllm command"""
    Fire(copy_prompts)


def get_dockerfile():
    """Entry point for serve-vllm command"""
    Fire(copy_dockerfile)


def get_dockerfile_sympy():
    """Entry point for serve-vllm command"""
    Fire(copy_dockerfile_sympy)


def get_resources():
    """Entry point for serve-vllm command"""
    Fire(copy_resources)
