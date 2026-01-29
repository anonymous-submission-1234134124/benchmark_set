from abc import ABC, abstractmethod
from pathlib import Path

import yaml


class PromptBuilderBase(ABC):
    """Base class for building prompts from YAML templates."""

    def __init__(self, prompts_file: str | Path):
        with open(prompts_file) as f:
            prompts = yaml.safe_load(f)

        # Load all YAML keys as attributes
        for key in self.get_yaml_keys():
            setattr(self, key, prompts[key])

        # Build composite prompt for the first turn
        self.user_prompt = self._compose_user_prompt()

    def build_user_prompt(self, item: dict, file_content_cut: str) -> str:
        """Builds prompt for the first turn with full context."""
        format_keys = self.get_format_keys(item, file_content_cut)
        return self.user_prompt.format(**format_keys)

    @abstractmethod
    def get_yaml_keys(self) -> list[str]:
        """Returns list of YAML keys to load from prompts file."""
        pass

    @abstractmethod
    def _compose_user_prompt(self) -> str:
        """Compose the prompt used for the first turn from loaded YAML attributes."""
        pass

    @abstractmethod
    def get_format_keys(self, item: dict, file_content_cut: str) -> dict:
        """
        Extract formatting keys from item and file_content_cut.
        Returns dict that will be used for .format(**keys).
        """
        pass


class PromptBuilderMultiturnGen(PromptBuilderBase):

    def get_yaml_keys(self) -> list[str]:
        return ["system_prompt", "base_prompt", "dp_description", "iter_prompt_template"]

    def build_iter_prompt(self, item: dict, test_output: str) -> str:
        """Builds prompt for feedback iteration with test results."""
        feedback = self.iter_prompt_template.format(test_output=test_output)
        base_task = self.base_prompt.format(**self.get_format_keys(item, ""))
        return feedback + "\n" + base_task

    def _compose_user_prompt(self) -> str:
        return "\n".join([self.base_prompt, self.dp_description])

    def get_format_keys(self, item: dict, file_content_cut: str) -> dict:
        return {
            "method_name": item["method"]["name"],
            "class_name": item["class_name"],
            "method_declaration": item["method"]["declaration"],
            "method_description": item["method"]["description"],
            "file_content": file_content_cut,
            "class_description": item["class_doc"],
            "class_declaration": item["class_declaration"] if "class_declaration" in item else "",
            "class_vars": item["class_vars"] if "class_vars" in item else "",
        }


class PromptBuilderCompletion(PromptBuilderBase):
    """Builds prompts for code completion tasks."""

    def get_yaml_keys(self) -> list[str]:
        return ["system_prompt", "base_prompt"]

    def _compose_user_prompt(self) -> str:
        return self.base_prompt

    def get_format_keys(self, item: dict, file_content_cut: str) -> dict:
        return {"file_content": file_content_cut}


class PromptBuilderProgramRepair(PromptBuilderBase):
    """Builds prompts for program repair tasks."""

    def get_yaml_keys(self) -> list[str]:
        return ["system_prompt", "user_prompt_template"]

    def _compose_user_prompt(self) -> str:
        return self.user_prompt_template

    def get_format_keys(self, item: dict, file_content_cut: str = None) -> dict:
        return {
            "method_description": item["method"]["description"],
            "file_content": file_content_cut,
            "test_output": item["test_error_message"][-20_000:]
        }
