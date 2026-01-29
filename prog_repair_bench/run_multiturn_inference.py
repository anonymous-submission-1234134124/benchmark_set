import logging
import os
import shutil
import json
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from importlib.metadata import version

import requests
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from fire import Fire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from prog_repair_bench.data_preprocess.hf_dataset_loader import hf_dataset_loader
from prog_repair_bench.data_preprocess.completion_dataset import augment_dataset_with_completion
from prog_repair_bench.data_preprocess.s3_folders_manager import (
    get_local_object,
    upload_s3_folder,
)
from prog_repair_bench.data_preprocess.class_task_processor import process_class_dataset

from prog_repair_bench.bench_runners import MultiTurnRunner, CompletionRunner,  ProgramRepairRunner
from prog_repair_bench.test_runner import TestPipeline
from prog_repair_bench.test_runner_api import TestPipelineRestApi
from prog_repair_bench.processors.output_reader import (
    load_results_from_jsonl,
)
from prog_repair_bench.processors.process_results import get_simple_results, save_jsonl
from prog_repair_bench.serve_vllm import (
    LOG_FOLDER_NAME,
    VLLM_PORTS_FILENAME,
    get_ckpt_names,
    get_ckpts,
)
from prog_repair_bench.wandb_plots import log_results_and_graphs
from prog_repair_bench.processors.config_merger import load_config_with_default, paths_to_abs_str

load_dotenv()

# TODO Restore
DJANGO_DATA_PATH = "anonauthor/django_multi_task"
# TODO Restore
SYMPY_DATA_PATH = "s3://sympy-data/"

BENCH_DICT = {
    "method_gen_multiturn": MultiTurnRunner,
    "completion": CompletionRunner,
    "program_repair": ProgramRepairRunner,
}

def dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances inside containers to dicts."""
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, (AIMessage, BaseMessage, HumanMessage, SystemMessage)):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(i) for i in obj]
    if isinstance(obj, Exception):
        return str(obj)
    else:
        return obj


def print_and_log_summary(results: list[dict], max_iter: int, wandb_run, results_filename: Path):
    total = len(results)
    pass_rates = defaultdict(float)
    percent_pass = defaultdict(float)
    summary_add_metrics = {"chrf": 0, "common_prefix_ratio": 0, "match_method_name": 0}
    for res in results:
        turn = res["turn"]
        if res["test_status"] == "OK":
            pass_rates[turn] += 1
        for key, value in summary_add_metrics.items():
            if res["test_results"]:
                summary_add_metrics[key] += res["test_results"][-1].get(key, 0)
    for key, value in summary_add_metrics.items():
        summary_add_metrics[key] /= total
    for turn in range(1, max_iter + 1):
        for res in results:
            if turn <= res["turn"]:
                percent_pass[turn] += res["percentage_passed"][turn - 1]
            else:
                percent_pass[turn] += res["percentage_passed"][-1]
        percent_pass[turn] = percent_pass[turn] / total * 100

    passed_upto = 0
    pass_rates_dict = dict()
    print("Pass rate by turn (cumulative):")
    for i in range(1, max_iter + 1):
        passed_upto += pass_rates.get(i, 0)
        rate = passed_upto / total * 100
        pass_rates_dict[i] = rate
        print(f"  <= turn {i}: {rate:.2f}%")
        wandb_run.summary[f"pass_rate_by_turn/{i}"] = rate

    print("")
    print("Percentage passed by turn (cumulative):")
    for i in range(1, max_iter + 1):
        print(f"  <= turn {i}: {percent_pass[i]:.2f}%")
        wandb_run.summary[f"percentage_pass_by_turn/{i}"] = percent_pass[i]

    for key, value in summary_add_metrics.items():
        print(f"{key}: {value:.2f}")
        wandb_run.summary[key] = value

    pass_summary = {
                "num_datapoints": len(results),
                "pass_rates": pass_rates_dict,
                "percent_pass": percent_pass,
            }
    pass_summary.update(summary_add_metrics)

    filename_summary = results_filename.with_name(
        f"{results_filename.stem}_summary{results_filename.suffix}"
    )
    with open(filename_summary, "w") as f:
        json.dump(pass_summary, f,)


def _add_name_suffix(folder_name: str, config: DictConfig) -> str:
    # add suffix to output folder
    inference = config.inference
    flags = ["repair_sampling", "no_feedback", "no_multiturn", "no_history"]
    for flag in flags:
        if getattr(inference, flag):
            folder_name += "_" + flag
    if not inference.think:
        folder_name += "-no_think"

    return folder_name


def format_paths(config: DictConfig):
    model_name = config.inference.model_name
    lora_ckpts_dir = config.inference.get("lora_ckpts_dir")

    if lora_ckpts_dir:
        parts = Path(lora_ckpts_dir).parts
        folder_name = parts[-1]
        name_parts = folder_name.split('-')
        wandb_run_name = '-'.join(name_parts[:4]) if len(name_parts) >= 4 else folder_name
        res_folder = folder_name
    elif config.inference.provider == "vllm" and (
            Path(model_name).exists() or model_name.startswith("s3://")
    ):
        parts = Path(model_name).parts
        res_folder = str(Path(*parts[-2:]))
        wandb_run_name = res_folder.replace("/", "-")
    else:
        wandb_run_name = model_name.replace("/", "__")
        res_folder = wandb_run_name

    if config.paths.output_folder.startswith("s3://"):
        config.paths.output_folder = f"{config.paths.output_folder.rstrip('/')}/{res_folder}"

    wandb_run_name_with_info = _add_name_suffix(wandb_run_name, config)

    replacements = {
        "_no_think": "-no_think",
        "_no_feedback": "-no_feedback",
        "_no_multiturn": "-no_multiturn",
        "_no_history": "-no_history",
        "_repair_sampling": "-repair_sampling"
    }
    for old, new in replacements.items():
        wandb_run_name_with_info = wandb_run_name_with_info.replace(old, new)

    config.wandb.run_name = wandb_run_name_with_info

    return res_folder


class MultiTurnInferenceRunner:
    def __init__(
            self,
            config: DictConfig,
            do_continue: bool | str = False,
            ports: list[int] | None = None,
            wandb_run_name: str | None = None,
    ):

        res_folder = format_paths(config)

        self.gen_target = config.benchmark_parameters.method_gen_multiturn.get("gen_target", "method")
        self.benchmark = config.benchmark
        self.repository = config.repository
        self.benchmark_pars = config.benchmark_parameters.get(self.benchmark, {})
        self.do_continue = do_continue
        self.runner = None
        self.max_iter = config.inference.max_iter
        self.batch_size = config.inference.batch_size
        package_version = version("prog-repair-bench")
        config.package_version = package_version

        print(f"Package version: {package_version}")
        print(70 * "=")
        print(f"Running inference on {self.benchmark.upper()} task")
        if self.benchmark == "method_gen_multiturn":
            print(f"{self.gen_target.upper()} generation")
        print(70 * "=")

        vllm_ports_file = Path(config.paths.local_dir) / LOG_FOLDER_NAME / VLLM_PORTS_FILENAME

        if config.paths.docker_commands is None:
            default_cmds_name = "dockerfile_commands_" + config.repository
            config.paths.docker_commands = Path(__file__).parent / "resources" / default_cmds_name

        if config.paths.prompts is None:
            custom_k8s_prompts = Path("/tmp/prompts/prompts.yaml")
            if custom_k8s_prompts.exists():
                config.paths.prompts = custom_k8s_prompts
            else:
                prompt_filename = f"prompts_{self.benchmark}"
                if self.benchmark == "method_gen_multiturn":
                    prompt_filename += "_" + self.gen_target
                elif self.benchmark == "program_repair":
                    prompt_filename += "_" + self.benchmark_pars.context
                config.paths.prompts = Path(__file__).parent / "resources" / f"{prompt_filename}.yaml"

        if config.inference.provider == "vllm":
            if ports is not None:
                self.vllm_ports = ports
            elif vllm_ports_file.exists():
                with open(vllm_ports_file, "r") as f:
                    ports = [line.strip() for line in f if line.strip()]
                self.vllm_ports = ports
            else:
                raise ValueError(
                    f"No vllm ports file found: {vllm_ports_file}. Run serve-vllm first"
                )
        else:
            self.vllm_ports = None

        self.config = config
        self.testpipeline = self.get_sandbox_runner(config)

        self.model_name = get_local_object(
            config.inference.model_name, Path(config.paths.local_dir) / "model"
        )
        self.provider = config.inference.provider

        self.llms = None
        self.output_file: Path | None = None
        ckpt_list = config.inference.lora_ckpts_list
        if ckpt_list is not None:
            ckpt_list = [ckpt for ckpt in ckpt_list if ckpt]
        lora_ckpts_dir = get_local_object(
            config.inference.lora_ckpts_dir, Path(config.paths.local_dir) / "lora_checkpoints"
        )
        self.ckpts = get_ckpts(lora_ckpts_dir, ckpt_list)
        if self.provider != "vllm":
            self.ckpts = [None]
        if self.config.paths.output_folder.startswith("s3://"):
            self.base_output_dir = Path(self.config.paths.local_dir) / "outputs" / res_folder
        else:
            self.base_output_dir = Path(self.config.paths.output_folder) / res_folder

        self.dataset = self.get_dataset()
        self.base_wandb_run_name = wandb_run_name or config.wandb.run_name or self.model_name
        self.wandb_config_dict = OmegaConf.to_container(self.config, resolve=True)

        wandb.login(
            anonymous="never",
            key=os.getenv("WANDB_API_KEY"),
            relogin=True,
            host=os.getenv("WANDB_BASE_URL"),
        )
        self.wandb_run = None
        self.current_ckpt_name = None
        self.wandb_run_ids = {}  # Store run IDs for each checkpoint

    def get_sandbox_runner(
            self, config: DictConfig
    ) -> TestPipeline | TestPipelineRestApi:
        sandbox_provider = config.inference.get("sandbox_provider", "local")

        if sandbox_provider == "local":
            return TestPipeline(
                docker_config=config.docker,
                commands_filepath=config.paths.docker_commands,
                repository=config.repository,
                **config.inference_pars,
            )
        elif sandbox_provider == "restapi":
            return TestPipelineRestApi(repository=config.repository, **config.inference_pars)
        else:
            raise ValueError(f"Unknown sandbox provider: {config.inference.sandbox_provider}")

    def set_model_and_out_name(self, model_name: str, ckpt_name: str | None = None) -> None:

        if ckpt_name is not None:
            model_name_for_dir = model_name + "_" + ckpt_name
            model_name = ckpt_name
        else:
            model_name_for_dir = model_name

        if self.config.inference.get("check_ground_truth", False):
            self.llms = [None]
        else:
            model_args = self.config.inference.get("model_args", {})
            model_args = {k: v for k, v in model_args.items() if v is not None}
            self.llms = self.get_models(
                model_name, self.provider, self.vllm_ports, model_args
            )

        self.current_ckpt_name = ckpt_name
        self.output_file = self.get_output_folder(model_name_for_dir)

        self._init_wandb_for_checkpoint(ckpt_name)

    def _init_wandb_for_checkpoint(self, ckpt_name: str | None) -> None:
        """Initialize wandb run for the current checkpoint."""

        if ckpt_name:
            wandb_run_name = f"{self.base_wandb_run_name}-{ckpt_name}"
        else:
            wandb_run_name = self.base_wandb_run_name

        self.wandb_run = wandb.init(
            project=self.config.wandb.project_name,
            config=self.wandb_config_dict,
            name=wandb_run_name,
            reinit=True
        )

        result_key = ckpt_name if ckpt_name else self.model_name
        self.wandb_run_ids[result_key] = self.wandb_run.id

    def get_output_folder(self, model_name: str) -> Path | None:

        model_prefix = model_name.replace("/", "__").replace(".", "_")
        if self.do_continue:
            if isinstance(self.do_continue, bool):
                # Find the latest result folder with the same model prefix
                result_folders = [
                    f
                    for f in self.base_output_dir.iterdir()
                    if f.is_dir() and f.name.startswith(model_prefix)
                ]
                if not result_folders:
                    return None
                result_folders.sort(key=lambda x: x.name.split("_")[-2:], reverse=True)
                output_folder = result_folders[0]
            elif isinstance(self.do_continue, str):
                output_folder = Path(self.do_continue)
            else:
                raise ValueError("do_continue should be bool or str")

            if not output_folder.exists():
                message = "Previous results has not been found. Starting from scratch."
                logging.info(message)
                print(message)
                self.do_continue = False
            else:
                message = f"Continuing from {output_folder}"
                logging.info(message)
                print(message)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.config.paths.format_run_name:
                if self.benchmark == "program_repair":
                    add_par = self.benchmark_pars.context
                elif self.benchmark == "method_gen_multiturn":
                    add_par = self.gen_target
                else:
                    add_par = ""
                result_folder_name = f"{self.benchmark}-{add_par}-{timestamp}"
            else:
                result_folder_name = model_prefix + "_" + timestamp
            result_folder_name = _add_name_suffix(result_folder_name, self.config)

            if self.current_ckpt_name:
                checkpoint_dir = f"global_step_{self.current_ckpt_name.replace('checkpoint-', '')}"
                output_folder = self.base_output_dir / checkpoint_dir / result_folder_name
            else:
                output_folder = self.base_output_dir / result_folder_name

            output_folder.mkdir(parents=True, exist_ok=True)
            cfg_to_save = paths_to_abs_str(self.config)
            with open(output_folder / "config.yaml", "w") as f:
                OmegaConf.save(config=cfg_to_save, f=f.name, resolve=True)
            shutil.copy(self.config.paths.prompts, output_folder)
            logging.info(f"Saving results from {output_folder}")
            print(f"Saving results from {output_folder}")

        output_file = output_folder / self.config.paths.output_filename

        return output_file

    async def setup_testpipeline(self):
        await self.testpipeline.setup()

    @staticmethod
    def get_bench_runner_by_name(bench_name: str) -> type[MultiTurnRunner | CompletionRunner | ProgramRepairRunner]:

        if bench_name not in BENCH_DICT:
            raise ValueError(f"Bench name {bench_name} not in BENCH_DICT")

        return BENCH_DICT[bench_name]

    def setup_multiturn_runner(self):
        runner_cls = self.get_bench_runner_by_name(self.benchmark)
        
        self.runner = runner_cls(
            max_iterations=self.max_iter,
            test_runner=self.testpipeline,
            gen_engine=self.llms,
            benchmark=self.benchmark,
            gen_target=self.gen_target,
            prompts_file=self.config.paths.prompts,
            think=self.config.inference.think,
            context_source=self.config.inference.context_source,
            max_test_feedback_symb=self.config.inference.max_test_feedback_symb,
            max_context_symb=self.config.inference.max_context_symb,
            max_num_tests=self.config.inference.max_num_tests,
            no_feedback=self.config.inference.no_feedback,
            no_multiturn=self.config.inference.no_multiturn,
            no_history=self.config.inference.no_history,
            repair_sampling=self.config.inference.repair_sampling,
            repository=self.config.repository,
            check_ground_truth=self.config.inference.get("check_ground_truth", False),
        )

    def get_models(
            self,
            model_name: str,
            provider: str,
            vllm_ports: list[int] | None,
            model_args: dict | None = None,
    ) -> list[ChatOpenAI]:
        if vllm_ports is None:
            vllm_ports = [None]
        llms = []
        for port in vllm_ports:
            model = self.get_single_model(model_name, provider, port, model_args)
            if model is not None:
                llms.append(model)
        if len(llms) == 0:
            raise ValueError("No models found")
        return llms

    @staticmethod
    def get_single_model(
            model_name: str, provider: str, vllm_port: int | None, model_args: dict | None = None
    ) -> ChatOpenAI | None:
        if model_args is None:
            model_args = {}
        if provider == "litellm":
            # TODO Restore default
            address = os.getenv("LITELLM_ENDPOINT")
            api_key = os.getenv("LITELLM_KEY")
            if api_key is None:
                raise ValueError("Provide LITELLM_KEY in environment variables")
        elif provider == "vllm":
            if vllm_port is None:
                raise ValueError("Provide vllm_port in config.yaml")
            address = f"http://localhost:{vllm_port}/v1"  # Update with your vllm server address
            api_key = "sk-empty-key"
            response = requests.get(f"{address}/models")
            if not response.ok:
                print(f"Could not find a model on port {vllm_port}")
                return None
        else:
            raise ValueError(f"Unknown model provider {provider}")

        return ChatOpenAI(base_url=address, api_key=api_key, model=model_name, **model_args)

    def get_dataset(self):
        dataset = self.get_dataset_base()
        if self.benchmark == "completion":
            dataset = augment_dataset_with_completion(dataset)
        elif self.benchmark == "method_gen_multiturn" and self.gen_target == "class":
            dataset = process_class_dataset(dataset)

        return dataset

    def get_dataset_base(self):

        data_path = OmegaConf.select(self.config, "data.path")
        if data_path is None:
            if self.repository == "django":
                data_path = DJANGO_DATA_PATH
            elif self.repository == "sympy":
                data_path = SYMPY_DATA_PATH
            else:
                raise ValueError(f"Unknown repository: {self.repository}")

        if OmegaConf.select(self.config, "data") is None:
            self.config.data = OmegaConf.create({})
        self.config.data.path = data_path

        local_cache_dir = Path(self.config.paths.local_dir) / "dataset"
        print(70 * "-")
        print(f"Dataset path: {data_path}")
        print(70 * "-")
        dataset = hf_dataset_loader(data_path, split="test", local_cache_dir=local_cache_dir)
        orig_len = len(dataset)
        # if self.config.inference.max_time_test_norm is not None:
        #     dataset = dataset.filter(lambda x: x["tests"]["time_test_norm"] < 2)
        # print(f"Dataset filtered. Keeping {len(dataset)}/{orig_len} of original items.")
        if self.do_continue and self.output_file is not None:
            previous_res = load_results_from_jsonl(self.output_file)
            indices_done = {item["dp_item"]["idx"] for item in previous_res}
            dataset = dataset.filter(lambda x: x["idx"] not in indices_done)
            message = f"Starting from index {max(indices_done) + 1}"
            logging.info(message)
            print(message)

        return dataset

    @staticmethod
    def save_results(data: list, filename: str | Path):
        filename = Path(filename)
        filename_simp = filename.with_name(f"{filename.stem}_short{filename.suffix}")

        data_serializable = dataclass_to_dict(data)
        data_simp = get_simple_results(data_serializable)

        save_jsonl(filename, data_serializable, "a")
        save_jsonl(filename_simp, data_simp, "a")

    def _maybe_upload_to_s3(self):
        if not self.config.paths.output_folder.startswith("s3://"):
            return
        # TODO refactor. Folder naming code is very ugly.
        if not self.current_ckpt_name:
            s3_path = self.config.paths.output_folder.rstrip("/") + "/" + self.output_file.parent.name + "/"
        else:
            s3_path = self.config.paths.output_folder.rstrip("/") + "/" + self.output_file.parent.parent.name + "/"  + self.output_file.parent.name + "/"

        upload_s3_folder(
            local_dir=self.output_file.parent, s3_path=s3_path
        )
        print(f"Results uploaded to {s3_path}")

    @staticmethod
    def unroll_batch(batch: dict[str, list]) -> list[dict]:
        """
        Convert a batch dictionary with list values to a list of dictionaries.

        Args:
            batch: A dictionary where each key maps to a list of values
                   (all lists should have the same length)

        Returns:
            A list of dictionaries, where each dictionary represents a single example
        """
        if not batch:
            return []

        # Get the length of the batch (length of any value list)
        batch_size = len(next(iter(batch.values())))

        # Create a list of dictionaries
        batch_unrolled = []
        for i in range(batch_size):
            example = {key: values[i] for key, values in batch.items()}
            batch_unrolled.append(example)

        return batch_unrolled

    async def run_set(self, dataset) -> tuple[list[dict], list[dict], list[dict], int, int]:

        all_results = []
        for batch_raw in tqdm(
                dataset.iter(batch_size=self.batch_size),
                total=len(dataset) // self.batch_size + 1,
        ):
            batch = self.unroll_batch(batch_raw)
            results = await self.runner.run_batch(batch)
            self.save_results(results, self.output_file)
            all_results.extend(results)

        crushed_points = 0
        gen_error = 0
        good_results = []
        bad_results = []
        for res in all_results:
            if res["test_status"] == "CRUSHED":
                crushed_points += 1
                bad_results.append(res)
            elif res["stop_reason"] == "generation_error":
                gen_error += 1
                bad_results.append(res)
            else:
                good_results.append(res)

        return all_results, good_results, bad_results, crushed_points, gen_error

    def select_dataset(self, dataset: Dataset, bad_results):

        res_ids = set()
        for res in bad_results:
            res_ids.add(res["dp_item"]["dp_id"])

        filtered_items = [item for item in dataset if item["dp_id"] in res_ids]
        dataset_bad = Dataset.from_list(filtered_items)

        return dataset_bad

    async def run_set_with_retry(self, dataset: Dataset) -> list[dict]:

        total_samples = len(dataset)
        all_results = []
        bad_results = []
        for i in range(5):
            results, good_results, bad_results, crushed_points, gen_error = await self.run_set(
                dataset
            )
            all_results.extend(good_results)
            if crushed_points + gen_error == 0:
                break
            dataset = self.select_dataset(dataset, bad_results)
            print(f"Number of crushed points: {crushed_points}")
            print(f"Number of generation errors: {gen_error}")
            print(f"Retry {i + 1} with {len(dataset)} items that has not been properly run.")
        all_results.extend(bad_results)
        assert len(all_results) == total_samples

        return all_results

    async def run_inference_single(self, dataset) -> list[dict]:

        all_results = await self.run_set_with_retry(dataset)

        crushed_points = 0
        gen_error = 0
        for res in all_results:
            crushed_points += res["test_status"] == "CRUSHED"
            gen_error += res["stop_reason"] == "generation_error"

        total = len(all_results)
        ok_count = sum(result["test_status"] == "OK" for result in all_results)
        pass_percentage = sum(result["percentage_passed"][-1] for result in all_results)
        print(60 * "-")
        print(f"Total number of items run: {total}")
        print(f"Proportion of lines with test_status='OK': {ok_count / total * 100:.2f}%")
        print(f"Average percentage of passed tests: {pass_percentage / total * 100:.2f}%")
        print(f"Number of crushed points: {crushed_points}")
        print(f"Number of generation errors: {gen_error}")
        print(60 * "-")

        # Cumulative pass rate by turn (i from 1 to max_turns):
        # numerator: count of items with test_status == 'OK' and turn <= i
        print_and_log_summary(
            all_results, self.max_iter, wandb_run=self.wandb_run, results_filename=self.output_file
        )

        return all_results

    async def _run_replicas(self, dataset, num_replicas: int = 1):
        results_all = []
        for i in range(num_replicas):
            if num_replicas > 1:
                print(60 * "-")
                print(f"Replica {i + 1} of {num_replicas}")
                print(60 * "-")
            results = await self.run_inference_single(dataset)
            results_all.append(results)
            if i + 1 < num_replicas:
                time.sleep(30)
        if len(results_all) == 1:
            results_all = results_all[0]

        return results_all

    async def run_inference(
            self, max_items: int = -1, num_replicas: int = 1
    ) -> dict[str, list[dict]]:

        if max_items > 0:
            dataset = self.dataset.select(range(max_items))
        else:
            dataset = self.dataset

        results_all = dict()
        ckpt_names = get_ckpt_names(self.ckpts)
        try:
            for ckpt_name in ckpt_names:
                try:
                    self.set_model_and_out_name(self.model_name, ckpt_name)
                    self.setup_multiturn_runner()
                    results = await self._run_replicas(dataset, num_replicas)
                    if ckpt_name:
                        results_all[ckpt_name] = results
                    else:
                        results_all[self.model_name] = results
                finally:
                    self._maybe_upload_to_s3()
        finally:
            await self.testpipeline.close()

        return results_all


async def main(
        config: str,
        max_items: int = -1,
        do_continue: bool | str = False,
        num_replicas: int = 1,
        ports: list[int] | None = None,
        wandb_run_name: str | None = None,
):
    # Configure logging to write to file
    config_path = config
    config = load_config_with_default(config_path)
    log_file = Path(config.paths.local_dir) / LOG_FOLDER_NAME / "multiturn_logs.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_file)]
    if config.inference.log_to_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    inference_runner = MultiTurnInferenceRunner(
        config, do_continue, ports=ports, wandb_run_name=wandb_run_name
    )
    await inference_runner.setup_testpipeline()
    inference_config = inference_runner.config.inference
    if not inference_config.think:
        print("---- IMPORTANT! No thinking would be used. ---------")
        logging.warning("No thinking would be used.")
    if inference_config.no_feedback:
        print("---- IMPORTANT! No feedback would be used. ---------")
        logging.warning("No feedback would be used.")
    if inference_config.no_multiturn:
        print("---- IMPORTANT! No multiturn would be used. Simple sampling. ---------")
        logging.warning("No multiturn would be used. Simple sampling.")
    if inference_config.no_history:
        print("---- IMPORTANT! No chat history would be used. ---------")
        logging.warning("No chat history would be used.")
    if inference_config.repair_sampling:
        print("---- IMPORTANT! Repair @k. Tries to repair same generation n times ---------")
        logging.warning("Repair @k. Tries to repair same generation n times")
    results = await inference_runner.run_inference(max_items=max_items, num_replicas=num_replicas)

    for result_key, result_data in results.items():
        run_id = inference_runner.wandb_run_ids.get(result_key)
        if not run_id:
            logging.warning(f"No wandb run ID found for {result_key}, skipping graph logging")
            continue

        wandb_run_name = f"{inference_runner.base_wandb_run_name}-{result_key}" if result_key != inference_runner.model_name else inference_runner.base_wandb_run_name

        inference_runner.wandb_run = wandb.init(
            project=config.wandb.project_name,
            config=inference_runner.wandb_config_dict,
            name=wandb_run_name,
            id=run_id,
            resume="must",
            reinit=True
        )

        try:
            logging.info(f"Starting W&B graph logging for {result_key}")
            log_results_and_graphs(
                wandb_config=config.wandb,
                results_by_name=results,
                run_label=result_key,
                analysis_cfg=getattr(inference_runner.config, "analysis", None),
            )
            logging.info(f"W&B logging completed for {result_key}")
        except Exception as e:
            import traceback
            error_msg = f"Reporting to wandb failed for {result_key}: {e}\nTraceback:\n{traceback.format_exc()}"
            logging.warning(error_msg)
            print(f"\n{'=' * 70}\nERROR IN W&B LOGGING:\n{error_msg}\n{'=' * 70}\n")
        finally:
            if inference_runner.wandb_run is not None:
                inference_runner.wandb_run.finish()

    print("All wandb runs completed")

    return results


if __name__ == "__main__":
    Fire(main)
