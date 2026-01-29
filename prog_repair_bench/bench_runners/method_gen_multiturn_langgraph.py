from pathlib import Path
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from prog_repair_bench.test_runner import TestPipeline
from prog_repair_bench.processors import (
    PromptBuilderMultiturnGen,
    normalize_indent,
    strip_decorators
)
from prog_repair_bench.processors.cut_file import trim_file
from prog_repair_bench.bench_runners.base_langgraph import BaseBenchRunner, TestResultState

def is_good_start(code: str, method_name: str) -> bool:
    method_dec = f"def {method_name}"
    method_dec_async = f"async def {method_name}"
    # allow starting with decorators immediately preceding the function
    # e.g., @decorator(args)
    stripped = code.strip()
    good_start = stripped.startswith((method_dec, method_dec_async)) or stripped.startswith("@")
    return good_start

def fit_indent_to_declaration(code: str, instance_dec: str) -> str:

    declaration_indent = (len(instance_dec) - len(instance_dec.lstrip(" "))) * " "
    # Generated code will start with method declaration
    code_lines = code.splitlines()
    instance_dec = code_lines[0]
    if (len(instance_dec) - len(instance_dec.lstrip(" "))) == 0:
        code = "\n".join(declaration_indent + line for line in code_lines)

    return code


class MultiTurnRunner(BaseBenchRunner):
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
        gen_target: str,
        think: bool = True,
        max_test_feedback_symb: int = 20_000,
        max_context_symb: int = 60_000,
        context_source: str = "file",
        max_num_tests: int = 100,
        debug=False,  # outputs each item in the graph to the console
        # Arguments for control experiments
        no_feedback: bool = False,  # do not add tests feedback to the prompt
        no_multiturn: bool = False,  # do not add history, pure sampling
        no_history: bool = False,  # keep only the last prompt and last AI response
        repair_sampling: bool = False,  # Pass @k of repair of same AI solution
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
            prompt_builder_cls = PromptBuilderMultiturnGen,
        )

        assert isinstance(no_feedback, bool), "no_feedback must be a bool"
        assert isinstance(no_multiturn, bool), "no_multiturn must be a bool"
        assert isinstance(no_history, bool), "no_history must be a bool"
        assert isinstance(repair_sampling, bool), "repair_sampling must be a bool"

        self.max_test_feedback_symb = max_test_feedback_symb

        self.no_feedback = no_feedback
        self.no_multiturn = no_multiturn
        self.no_history = no_history
        self.repair_sampling = repair_sampling

        self.gen_target = gen_target
        self.check_ground_truth = check_ground_truth

    def build_prompt(self, item: dict) -> str:
        if self.gen_target == "method":
            if self.context_source == "file":
                context = item["file_without_method"]
            elif self.context_source == "class":
                context = item["class_without_method"]
            else:
                raise ValueError(f"Unknown context source: {self.context_source}")
        elif self.gen_target == "class":
            context = item["file_without_class"]
        else:
            raise ValueError(f"Unknown generation targets: {self.gen_target}")
        if len(context) > self.max_context_symb:
            context = trim_file(file_content = context,
                                target_class = item["class_name"],
                                target_method = item["method"]["name"],
                                limit=self.max_context_symb)
        user_prompt = self.prompt_builder.build_user_prompt(item=item, file_content_cut=context)

        return user_prompt

    def is_good_format(self, code: str, item_dict: dict) -> tuple[bool, str | None]:
        '''
        check if the output code is properly formatted
        '''
        error_message = ""
        if self.gen_target == "method":
            method_name = item_dict["method"]["name"]
            good_format = is_good_start(method_name=method_name, code=code)
            if not good_format:
                error_message = f"Code does not start with 'def {method_name}' or decorator."
        elif self.gen_target == "class":
            class_dec = item_dict["class_declaration"]
            good_format = (code.strip()).startswith(class_dec)
            if not good_format:
                error_message = f"Code does not start with '{class_dec}'."
        else:
            raise ValueError(f"Unknown generation target: {self.gen_target}")

        return good_format, error_message

    def get_ground_truth(self, item: dict) -> str | None:

        """
        Extract ground truth method.
        Override in subclasses for different benchmark types.
        """
        if self.gen_target == "method":
            method_dec = item["method"]["declaration"]
            method_body = item["method"]["body"]
            gt = method_dec + "\n" + method_body
        elif self.gen_target == "class":
            gt = item["class_code"]
        else:
            raise ValueError(f"Unknown genenration target: {self.gen_target}")

        return gt.rstrip()

    def process_code_block(self, code: str, item: dict) -> str:
        if not code:
            return code

        code = normalize_indent(code)
        code = strip_decorators(code)

        if self.gen_target == "method":
            instance_dec = item["method"]["declaration"]
        elif self.gen_target == "class":
            instance_dec = item["class_declaration"]
        else:
            raise ValueError(f"Unknown gen_target: {self.gen_target}")

        code = fit_indent_to_declaration(code, instance_dec=instance_dec)

        return code

    def build_item(
        self,
        item: dict,
        code: str = "",
    ):

        if self.gen_target == "method":
            startline = item["method"]["global_method_declaration_index"][0] + 1
            endline = item["method"]["global_method_body_index"][1] + 1
            instance_name = item["method"]["name"]
        elif self.gen_target == "class":
            startline = item["global_class_declaration_index"][0] + 1
            endline = item["global_class_body_index"][1] + 1
            instance_name = item["class_name"]
        else:
            raise ValueError(f"Unknown geneneration target: {self.gen_target}")

        return self.build_item_basic(item, startline, endline, code, instance_name)


    async def _process_test_results(self, state: TestResultState) -> TestResultState:

        test_results = state["test_results"][-1]
        test_passed = state["test_status"] == "OK"
        test_crushed = state["test_status"] == "CRUSHED"

        # Calculate metrics if ground truth is available
        raw_code = state["generate_codes"][-1] if state["generate_codes"] else ""
        item_dict = state["dp_item"]

        test_results["chrf"] = 0.0
        test_results["common_prefix_ratio"] = 0.0

        if raw_code:
            is_good, _ = self.is_good_format(raw_code, item_dict)
            if is_good:
                prediction = self.process_code_block(raw_code, item_dict)
                reference = self.get_ground_truth(item_dict)
                metrics = self._compute_similarity_metrics(prediction, reference)
                test_results.update(metrics)

        test_output = test_results["test_output"]
        test_output = test_output[-self.max_test_feedback_symb :]
        if not self.no_feedback:
            prompt = self.prompt_builder.build_iter_prompt(
                item=state["dp_item"], test_output=test_output
            )
        else:
            # base prompt is just
            # <<Generate code for the method `{method_name}` in the class `{class_name}`.
            # Do not generate class, only target method. Start with method declaration.>>
            prompt = self.prompt_builder.build_base_prompt(item=state["dp_item"])
            prompt = "Something wrong. Correct generated code. Reminding the task: " + prompt

        previous_messages = list(state["messages"])
        if self.no_history:
            # Remove old history, keeping only original prompt and last AI response
            # ['Sy', 'Hu1', 'Ai1', 'Hu2', 'Ai2'] -> ['Sy', 'Hu1', 'Ai2']
            previous_messages = previous_messages[:2] + [previous_messages[-1]]

        previous_messages.append(HumanMessage(content=prompt))

        if self.repair_sampling:
            # Keep only the task and the first request of the refinement
            # ['Sy', 'Hu1', 'Ai1', 'Hu2', 'Ai2'] -> ['Sy', 'Hu1', 'Ai1', 'Hu2']
            previous_messages = previous_messages[:4]
        if self.no_multiturn:
            previous_messages = previous_messages[:2]

        state["messages"] = previous_messages
        if test_passed:
            state["should_continue"] = False
            state["stop_reason"] = "passed_tests"
        elif test_crushed:
            state["should_continue"] = False
            state["stop_reason"] = "crushed_tests"
        elif state["turn"] >= state["max_turns"]:
            state["should_continue"] = False
            state["stop_reason"] = "max_turns_reached"
        else:
            state["should_continue"] = True

        return state

    def _build_graph(self):
        """
        Constructs and compiles lang-graph

        Description.

        1. Run the model on input.
        2. Run the tests, gather feedback:
            - if tests passed, stop.
            - if the number of turns exceeds the limit, stop.
        3. GOTO 1.

        """
        workflow = StateGraph(TestResultState)

        # Register nodes
        # workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("model", self._call_model)
        workflow.add_node("process_output", self._process_model_output)
        workflow.add_node("run_tests", self._run_tests_async)
        workflow.add_node("process_tests", self._process_test_results)
        workflow.add_node("finalize", self._finalize)

        # Wire the graph
        # workflow.add_edge("initialize", "model")
        # This condition in case of failed generation
        workflow.add_conditional_edges(
            "model",
            lambda s: s["should_continue"],
            {True: "process_output", False: "finalize"},
        )
        workflow.add_edge("process_output", "run_tests")
        workflow.add_edge("run_tests", "process_tests")
        workflow.add_conditional_edges(
            "process_tests",
            lambda s: s["should_continue"],
            {True: "model", False: "finalize"},
        )
        workflow.add_edge("finalize", END)

        workflow.set_entry_point("model")

        return workflow.compile(debug=self.debug)
