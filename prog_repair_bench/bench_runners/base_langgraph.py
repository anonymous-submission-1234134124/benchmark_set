import logging
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Annotated, Sequence, TypedDict

import wandb
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import convert_to_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import BadRequestError
from sacrebleu.metrics import CHRF

from prog_repair_bench.test_runner import TestPipeline
from prog_repair_bench.processors import (
    PromptBuilderMultiturnGen,
    get_code_block,
    parse_tests_output,
)
from prog_repair_bench.processors.parse_tests_output import get_percentage_passed
from prog_repair_bench.run_item import ItemToRun


class TestResultState(TypedDict):
    """
    A minimal state that is passed between the nodes of the lang-graph.
    """

    messages: Annotated[
        Sequence[BaseMessage],
        "The conversation messages. Responses from LLM are filtered from think",
    ]
    responses_raw: Annotated[Sequence[BaseMessage | str], "The raw responses from the LLM"]
    test_results: list[dict]
    test_status: str | None  # Last test status
    percentage_passed: list[float]
    generate_codes: list[str]
    turn: Annotated[int, "Current turn number"]
    max_turns: Annotated[int, "Maximum number of turns allowed"]
    should_continue: bool | None
    stop_reason: str | None
    dp_item: Annotated[dict | None, "Datapoint to run tests on"]
    edited_file: list[str]
    timestamp_start: str | None
    timestamp_end: str | None
    duration: float | None
    tests_duration: float | None


class BaseBenchRunner:
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
        test_runner: TestPipeline,
        gen_engine: list[ChatOpenAI],  # LangChain chat-model (e.g. ChatOpenAI)
        prompts_file: str | Path,
        max_iterations: int = 1,
        think: bool = True,
        max_context_symb: int = 60_000,
        context_source: str = "file",
        max_num_tests: int = 100,
        debug=False,  # outputs each item in the graph to the console
        repository: str = "django",
        prompt_builder_cls: type = PromptBuilderMultiturnGen,
        check_ground_truth: bool = False,
    ):

        self.prompt_builder = prompt_builder_cls(prompts_file)

        assert max_iterations > 0, "max_iterations must be > 0"

        self.max_iterations = max_iterations
        self.context_source = context_source
        self.max_context_symb = max_context_symb
        self.think = think
        self.max_num_tests = max_num_tests

        self.gen_target = "method"
        self.repository = repository
        self.check_ground_truth = check_ground_truth 
        self.test_runner = test_runner
        if not isinstance(gen_engine, list):
            gen_engine = [gen_engine]
        self.gen_engine = gen_engine
        self.debug = debug
        self.system_prompt = self.prompt_builder.system_prompt
        self.chrf = CHRF()
        self.graph = self._build_graph()
        self.too_long_ai_message = "The thinking was too long. I should be more concise next time."
        self.test_result_keys = [
            "test_output",
            "time_test",
            "time_reset",
            "time_cat",
            "time_edit",
            "time_total",
            "summary",
            "details",
            "status",
            "percentage_passed",
            "chrf",
            "common_prefix_ratio",
        ]

    def build_prompt(self, item_dict: dict) -> str:
        '''
        Build task for the model
        '''
        raise NotImplementedError

    def is_good_format(self, code: str, item_dict: dict) -> tuple[bool, str]:
        '''
        Check, the code is correct.
        I.e. for program repair it should start from def or decorator
        Returns (is_good, error_message)
        '''
        raise NotImplementedError


    def process_code_block(self, code: str, item_dict: dict) -> str:
        '''
        Process code bock.
        I.e. for program repair strip decorators, normalize indentations
        '''

        raise NotImplementedError

    def build_item_basic(
        self,
        item: dict,
        startline: int,
        endline: int,
        code: str,
        instance_name: str | None = None,
    ):
        """
        Factory method to create ItemToRun from an item dict.
        """

        if instance_name is None:
            instance_name = item["method"]["name"]
        tests = item["tests"]["full_paths"]

        if tests and self.max_num_tests > 0:
            tests = tests[:self.max_num_tests]

        return ItemToRun(
            idx=item["idx"],
            dp_id=item["dp_id"],
            file_path=item["file_path"],
            replace_content=code,
            method_name=instance_name,
            start_line=startline,
            end_line=endline,
            tests=tests,
        )

    def build_item(
        self,
        item: dict,
        code: str = "",
    ):

        ''''
        Example:
        startline = item["method"]["global_method_declaration_index"][0] + 1
        endline = item["method"]["global_method_body_index"][1] + 1

        return self.build_item_basic(item, startline, endline, code)
        '''

        raise NotImplementedError

    def build_init_state(self, item: dict) -> TestResultState:

        user_prompt = self.build_prompt(item)
        logging.info(f"Started working on item {item["idx"]}.")
        initial_state: TestResultState = {
            "messages": convert_to_messages(
                [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            ),
            "responses_raw": [],
            "test_results": [],
            "test_status": None,
            "percentage_passed": [],
            "generate_codes": [],
            "turn": 0,
            "max_turns": self.max_iterations,
            "should_continue": None,
            "stop_reason": None,
            "dp_item": item,
            "edited_file": [],
            "timestamp_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_end": None,
            "duration": None,
            "tests_duration": None,
        }

        return initial_state

    def build_init_state_batch(self, item_list: list[dict]) -> list[TestResultState]:
        return [self.build_init_state(item) for item in item_list]

    def get_ground_truth(self, item: dict) -> str | None:

        """
        Extract ground truth method.
        Override in subclasses for different benchmark types.
        """

        method_dec = item["method"]["declaration"]
        method_body = item["method"]["body"]
        gt = method_dec + "\n" + method_body

        return gt.rstrip()

    def _compute_similarity_metrics(self, prediction: str, reference: str) -> dict:
        """
        Calculate code similarity metrics between prediction and reference.
        
        Args:
            prediction: Generated code
            reference: Ground truth code
            
        Returns:
            Dictionary with metrics: common_prefix_ratio and chrf
        """
        metrics = {}

        common_len = 0
        min_len = min(len(prediction), len(reference))
        for i in range(min_len):
            if prediction[i] == reference[i]:
                common_len += 1
            else:
                break
        
        if len(reference) > 0:
            ratio = common_len / len(reference)
        else:
            ratio = 1.0 if len(prediction) == 0 else 0.0
        metrics["common_prefix_ratio"] = ratio

        score = self.chrf.corpus_score([prediction], [[reference]])
        metrics["chrf"] = score.score
            
        return metrics

    @staticmethod
    def trim_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        """
        Trim the messages list to fit max_messages, keeping the first two always.

        :param messages: List of messages (System, Human, AI, etc)
        :return: Trimmed messages list
        """

        # Keep first two and as many of the latest messages as will fit
        start = messages[:2]
        end = messages[2:]
        # remove first AIresponse and feedback
        end = end[2:]
        return start + end

    async def _run_model(
        self, messages: Sequence[BaseMessage], idx: int
    ) -> tuple[AIMessage | None, str, str | Exception | BadRequestError]:
        # Process each item in GPU (same item goes to same GPU to keep caching)
        caught_exception = ""
        gen_engine = self.gen_engine[idx % len(self.gen_engine)]
        try:
            start_time = time.perf_counter()
            new_message = await gen_engine.ainvoke(messages)
            time_used = time.perf_counter() - start_time
            new_message.usage_metadata["time_used"] = time_used
            status = "success"
        except BadRequestError as e:
            caught_exception = e
            new_message = None
            if "maximum context length" in str(e):
                logging.warning(f"Context too long: {e}")
                status = "context_too_long"
            else:
                logging.warning(f"Generation error occurred: {e}")
                status = "error"
        except Exception as e:
            caught_exception = e
            logging.warning(f"Generation error occurred: {e}")
            status = "error"
            new_message = None

        return new_message, status, caught_exception

    async def _run_model_with_context_trim(
        self, messages: Sequence[BaseMessage], idx: int, turn: int
    ) -> tuple[AIMessage | None, str | Exception | BadRequestError]:
        # If we get context too long, trim it and try again.
        while True:
            logging.info(
                f"Calling model for item {idx}, at llm {idx % len(self.gen_engine)}, turn = {turn}."
            )
            new_message, status, caught_exception = await self._run_model(messages, idx)
            if status == "context_too_long" and len(messages) > 2:
                logging.warning(
                    f"Trimming context to fit context for item idx = {idx}, turn = {turn}, (current length: {len(messages)})."
                )
                messages = self.trim_messages(messages)
            else:  # if status is "error" or "success"
                break

        return new_message, caught_exception

    async def _call_model(self, state: TestResultState) -> TestResultState:
        """
        Node #1. Sends current conversation to the LLM and appends the answer.
        """
        if self.check_ground_truth:
            ground_truth_code = self.get_ground_truth(state["dp_item"])
            content = f"```python\n{ground_truth_code}\n```"
            
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

        state["turn"] = state["turn"] + 1
        messages = list(state["messages"])
        item_idx = state["dp_item"]["idx"]
        if not self.think:
            last_message = messages[-1]
            assert isinstance(last_message, HumanMessage)
            last_message.content = last_message.content + " /no_think"
            messages[-1] = last_message

        new_message, caught_exception = await self._run_model_with_context_trim(
            messages, idx=item_idx, turn=state["turn"]
        )

        # In case the generation failed.
        if new_message is None:
            state["should_continue"] = False
            state["stop_reason"] = "generation_error"
            print(f"Generation error! Exception: {caught_exception}")
            state["test_status"] = None
            state["percentage_passed"].append(0)
            responses_raw = list(state["responses_raw"])
            responses_raw.append(str(caught_exception))
            state["responses_raw"] = responses_raw
            return state

        messages = list(state["messages"])
        responses_raw = list(state["responses_raw"])
        responses_raw.append(deepcopy(new_message))
        state["responses_raw"] = responses_raw

        # Remove possible chain-of-thought that the model outputs between
        # <think> … </think>
        content_filtered, num_subs = re.subn(
            r"^.*?</think>", "", str(new_message.content), flags=re.DOTALL
        )
        # If there is no closing </think> tag, the thinking was too long and there was no answer.
        if num_subs > 0:
            new_message.content = content_filtered.strip("\n")
        # TODO Should we look not only at the start?
        elif new_message.content.strip().startswith("<think>"):
            new_message.content = self.too_long_ai_message
        messages.append(new_message)

        state["messages"] = messages
        state["should_continue"] = True
        return state

    def _process_model_output(self, state: TestResultState) -> TestResultState:
        """
        Node #2.1. Decides whether we have something to test and whether another
        iteration is required.
        """
        # Ran out of allowed turns – finish

        last_message = state["messages"][-1]
        assert isinstance(last_message, AIMessage)

        code_block = get_code_block(str(last_message.content))
        state["generate_codes"].append(code_block)

        return state

    def _record_test_results(self, test_result: dict | None, good_format: bool, format_error: str = "") -> tuple[dict, str]:
        if not good_format or not test_result:
            test_result = dict.fromkeys(self.test_result_keys)
            if not good_format:
                test_output = format_error or "Code does not start with def or no valid codeblock provided. Revise the answer."
                test_result["status"] = "NOT_RUN"
            else:
                test_output = "Tests crashed. Could not run the tests."
                test_result["status"] = "CRUSHED"
            edited_file = ""
            test_result["test_output"] = test_output
        else:
            test_output = test_result.get("test_output", "")
            edited_file = test_result.pop("edited_file", "")
            test_result.pop("datapoint", None)
            test_result_dict = parse_tests_output(test_output, repository=self.repository)
            test_result["summary"] = test_result_dict.pop("summary")
            test_result["details"] = test_result_dict["tests"]

            if test_result["summary"] is not None:
                test_result["status"] = test_result["summary"]["status"]
            else:
                test_result["status"] = "FAILED"

        test_result["percentage_passed"] = get_percentage_passed(test_result)

        return test_result, edited_file

    async def _run_tests_async(self, state: TestResultState) -> TestResultState:
        """
        Node 3. Executes IDE-Gym tests for the extracted implementation
        and appends the feedback for the next turn.
        """

        code = state["generate_codes"][-1]
        last_message = state["messages"][-1].content
        item_dict = state["dp_item"]

        good_format, format_error = self.is_good_format(code, item_dict)

        test_result = None
        if good_format:
            code = self.process_code_block(code, item_dict)
            # Prepare item for IDE-Gym runner
            item_to_run = self.build_item(item_dict, code=code)
            # The test runner will handle session allocation and locking internally
            test_result = await self.test_runner.run_single_task(item_to_run)

        test_result_processed, edited_file = self._record_test_results(test_result, good_format, format_error)

        if last_message == self.too_long_ai_message:
            test_result_processed["test_output"] = (
                "The thinking was too long. Be more concise next time."
            )

        test_result_processed["chrf"] = 0.0
        test_result_processed["common_prefix_ratio"] = 0.0
        if good_format:
            reference = self.get_ground_truth(item_dict)
            if reference:
                metrics = self._compute_similarity_metrics(code, reference)
                test_result_processed.update(metrics)

        state["test_status"] = test_result_processed["status"]
        state["test_results"].append(test_result_processed)
        state["percentage_passed"].append(test_result_processed["percentage_passed"])
        state["edited_file"].append(edited_file)

        return state

    async def _process_test_results(self, state: TestResultState) -> TestResultState:

        test_passed = state["test_status"] == "OK"
        test_crushed = state["test_status"] == "CRUSHED"

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

    def _finalize(self, state: TestResultState) -> TestResultState:

        '''
        Finalize state - log metrics and results
        '''

        assert len(state["percentage_passed"]) == state["turn"]
        state["timestamp_end"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        start_time = datetime.strptime(state["timestamp_start"], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(state["timestamp_end"], "%Y-%m-%d %H:%M:%S")
        time_diff = end_time - start_time
        seconds_diff = time_diff.total_seconds()

        test_time = 0
        for test in state["test_results"]:
            test_time += test["time_total"] or 0

        state["duration"] = seconds_diff
        state["tests_duration"] = test_time

        gen_time = state["duration"] - test_time
        wandb.log(
            {
                "duration": state["duration"],
                "gen_time": gen_time,
                "time_for_tests": state["tests_duration"],
                "status": state["stop_reason"],
            }
        )

        return state

    def _build_graph(self):
        """
        Constructs and compiles lang-graph

        Description.

        1. Run the model on input.
        2. Run the tests, gather feedback:
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
        workflow.add_edge("process_tests", "finalize")
        workflow.add_edge("finalize", END)

        workflow.set_entry_point("model")

        return workflow.compile(debug=self.debug)

    async def run_single(self, item: dict) -> TestResultState:
        initial_state = self.build_init_state(item)

        final_state: TestResultState = await self.graph.ainvoke(initial_state)

        return final_state

    async def run_batch(self, item_list: list[dict]) -> list[TestResultState]:
        states_batch = self.build_init_state_batch(item_list)

        final_states = []
        async for idx, result in self.graph.abatch_as_completed(states_batch):
            final_states.append(result)

        return final_states
