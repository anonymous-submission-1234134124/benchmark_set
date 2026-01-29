from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage

from prog_repair_bench.test_runner import TestPipeline
from prog_repair_bench.bench_runners.base_langgraph import BaseBenchRunner, TestResultState
from prog_repair_bench.processors.prompt_builder import PromptBuilderCompletion
from prog_repair_bench.processors.cut_file import trim_file

class CompletionRunner(BaseBenchRunner):
    """
    A thin orchestration layer that:

    1. Generates a prompt for the model
    2. Repeatedly asks the model for a new implementation of the method
    3. Runs the produced implementation against IDE-Gym tests
    4. Provides the test feedback back to the model
    5. Stops either when the tests pass or `max_iterations` is reached
    """

    def __init__(
        self,
        max_iterations: int,
        test_runner: TestPipeline,
        gen_engine: list[ChatOpenAI],  # LangChain chat-model (e.g. ChatOpenAI)
        prompts_file: str | Path,
        think: bool = True,
        max_context_symb: int = 60_000,
        context_source: str = "file",
        max_num_tests: int = 100,
        debug=False,  # outputs each item in the graph to the console
        repository: str = "django",
        check_ground_truth: bool = False,
        # To avoid init errors
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
            context_source=context_source,
            max_num_tests=max_num_tests,
            debug=debug,
            repository=repository,
            prompt_builder_cls=PromptBuilderCompletion,
            check_ground_truth=check_ground_truth,
        )

    def get_ground_truth(self, item: dict) -> str | None:
        """
        Extract ground truth code for completion benchmark.
        
        Returns:
            The continuation part that should be generated
        """
        completion_data = item.get("completion")
        return completion_data["continuation"]

    def build_prompt(self, item: dict) -> str:
        """
        Build prompt using the completion prompt builder.
        We expect the dataset item to have 'completion' field populated.
        """
        completion_data = item.get("completion")
        # 'prompt_file_context' from new dataset augmentation logic
        file_content = completion_data["prompt_file_context"]
        if len(file_content) > self.max_context_symb:
            file_content = trim_file(file_content = file_content,
                                target_class = item["class_name"],
                                target_method = item["method"]["name"],
                                limit=self.max_context_symb)
            
        return self.prompt_builder.build_user_prompt(item, file_content)

    def is_good_format(self, code: str, item_dict: dict) -> tuple[bool, str]:
        """
        Check if code block is present.
        """
        # code is already the extracted content from BaseBenchRunner
        is_good = code.strip() != ""
        error_message = "" if is_good else "Code block is empty"
        return is_good, error_message

    def process_code_block(self, code: str, item_dict: dict) -> str:
        """
        Extract code block.
        """
        # code is already extracted
        return code

    def build_item(
        self,
        item_dict: dict,
        code: str = "",
        tests: list[str] | None = None,
    ):
        """
        Construct ItemToRun for testing.
        We need to combine the half_method (from completion data) with the generated code (continuation).
        Then replace the original method in the file.
        """
        completion_data = item_dict.get("completion")
        if not completion_data:
             raise ValueError(f"Item {item_dict.get('idx')} missing 'completion' field.")
        
        half_method = completion_data["half_method"]

        # Note: generated 'code' is appended to 'half_method' (prefix) to form full method
        full_method_candidate = half_method + "\n" + code
        
        # We replace the original method using original indices.
        method_info = item_dict["method"]
        start_line = method_info["global_method_declaration_index"][0] # 0-based
        end_line = method_info["global_method_body_index"][1]          # 0-based
        
        start_line_1based = start_line + 1
        end_line_1based = end_line + 1
        
        return self.build_item_basic(
            item_dict, 
            start_line_1based, 
            end_line_1based, 
            full_method_candidate
        )

    async def _call_model(self, state: TestResultState) -> TestResultState:
        """
        Override to support check_ground_truth mode.
        """
        # TODO We can generalize it and move to the basic class later
        if self.check_ground_truth:
             # Simulate model output with ground truth
             item = state["dp_item"]
             completion_data = item.get("completion")
             if completion_data:
                 continuation = completion_data["continuation"]
                 # Wrap in python block
                 content = f"```python\n{continuation}\n```"
                 
                 # Update state as if model returned this
                 state["turn"] = state["turn"] + 1
                 new_message = AIMessage(content=content)
                 messages = list(state["messages"])
                 messages.append(new_message)
                 state["messages"] = messages
                 
                 responses_raw = list(state["responses_raw"])
                 responses_raw.append(new_message)
                 state["responses_raw"] = responses_raw
                 
                 state["should_continue"] = True
                 return state
        
        return await super()._call_model(state)


