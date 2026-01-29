from pathlib import Path
from langchain_openai import ChatOpenAI
import ast
from textwrap import dedent

from prog_repair_bench.test_runner import TestPipeline
from prog_repair_bench.bench_runners.base_langgraph import BaseBenchRunner
from prog_repair_bench.processors.prompt_builder import PromptBuilderProgramRepair
from prog_repair_bench.bench_runners.method_gen_multiturn_langgraph import (
    is_good_start,
    fit_indent_to_declaration,
)
from prog_repair_bench.processors import normalize_indent, strip_decorators
from prog_repair_bench.processors.cut_file import trim_file

def get_method_name(code: str) -> str | None:
    """
    Parse Python code containing a class method and return its method name.
    Returns None if no suitable method is found.
    """
    try:
        # Remove leading indentation that often appears in snippets
        tree = ast.parse(dedent(code))
    except SyntaxError:
        return None

    # Walk the tree and look for FunctionDef nodes that look like methods
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name:  # non-empty arg list
                return node.name

    return None


class ProgramRepairRunner(BaseBenchRunner):
    """
    Program Repair Benchmark Runner.
    
    The model is given:
    - A complete file with one incorrect method
    - Method documentation
    - Test failure output
    
    The model must:
    - Identify the incorrect method
    - Output a fixed implementation with exact signature
    """

    def __init__(
        self,
        max_iterations: int,
        test_runner: TestPipeline,
        gen_engine: list[ChatOpenAI],
        prompts_file: str | Path,
        think: bool = True,
        max_context_symb: int = 60_000,
        max_num_tests: int = 100,
        debug=False,
        repository: str = "django",
        check_ground_truth: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            max_iterations=max_iterations,
            test_runner=test_runner,
            gen_engine=gen_engine,
            prompts_file=prompts_file,
            think=think,
            max_context_symb=max_context_symb,
            context_source="file",
            max_num_tests=max_num_tests,
            debug=debug,
            repository=repository,
            prompt_builder_cls=PromptBuilderProgramRepair,
            check_ground_truth=check_ground_truth,
        )

    def build_prompt(self, item: dict) -> str:
        """
        Build the program repair prompt.

        PromptBuilder will handle truncation using max_context_symb.
        """
        context = item["file_with_buggy_method"]
        if len(context) > self.max_context_symb:
            context = trim_file(file_content=context,
                                target_class=item["class_name"],
                                target_method=item["method"]["name"],
                                limit=self.max_context_symb)
        user_prompt = self.prompt_builder.build_user_prompt(
            item=item, file_content_cut=context
        )

        return user_prompt

    def is_good_format(self, code: str, item_dict: dict) -> tuple[bool, str]:
        """
        Check if the generated code starts with method declaration or decorator.
        """
        if not code.strip():
            return False, "No code block provided."
        
        method_name = item_dict["method"]["name"]
        good_format = is_good_start(method_name=method_name, code=code)
        
        if not good_format:
            error_message = (
                f"Code does not start with 'def {method_name}' or decorator. "
                "Please provide only the repaired method."
            )
            return False, error_message
        
        return True, ""

    def process_code_block(self, code: str, item: dict) -> str:
        """
        Process the generated code block:
        1. Normalize indentation
        2. Strip decorators (as per benchmark rules)
        3. Fit indentation to match the original declaration
        """
        if not code:
            return code
        
        code = normalize_indent(code)
        code = strip_decorators(code)
        
        instance_dec = item["method"]["declaration"]
        code = fit_indent_to_declaration(code, instance_dec=instance_dec)
        
        return code

    def build_item(self, item: dict, code: str = ""):
        """
        Build ItemToRun for testing.
        
        Replace the buggy method with the generated fix.
        """
        startline = item["method"]["global_method_declaration_index"][0] + 1
        endline = item["method"]["global_method_body_index"][1] + 1
        instance_name = item["method"]["name"]
        
        return self.build_item_basic(item, startline, endline, code, instance_name)

    def _compute_similarity_metrics(self, prediction: str, reference: str) -> dict:

        metrics = super()._compute_similarity_metrics(prediction, reference)
        method_name_ref = get_method_name(reference)
        method_name_pred = get_method_name(prediction)

        metrics["match_method_name"] = method_name_ref == method_name_pred

        return metrics

