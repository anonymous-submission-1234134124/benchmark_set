import asyncio
import logging
import time
from asyncio import Lock, Queue, create_task, gather
from pathlib import Path
from typing import Any

import aiohttp
from private_sandbox.api.docker import ContainerConfig
from private_sandbox.api.git import GitRepository, GitRepositorySnapshot, GitServer
from private_sandbox.api.tools.bash import ExecuteCommandResponse as Result
from private_sandbox.client.legacy import SandboxClient
from omegaconf import DictConfig

from prog_repair_bench.run_item import ItemToRun


class TestPipeline:
    def __init__(
        self,
        docker_config: DictConfig,
        repo_snapshot: GitRepositorySnapshot | None = None,
        commands_filepath: str | Path = "dockerfile_commands_django",
        repository: str = "django",
        num_workers: int = 1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 300.0,
    ):
        if repo_snapshot is None:
            if repository == "sympy":
                repo_snapshot = GitRepositorySnapshot(
                    repository=GitRepository(
                        server=GitServer.GITHUB,
                        owner="sympy",
                        name="sympy",
                    ),
                    reference="51be47d700717e2c88c53eed81e3155651a047d2",
                )
            elif repository == "django":
                repo_snapshot = GitRepositorySnapshot(
                    repository=GitRepository(
                        server=GitServer.GITHUB,
                        owner="django",
                        name="django",
                    ),
                    reference="89807fbde8b7b17d00434bc4695535855e96fe77",
                )
            else:
                raise ValueError(f"Invalid repository: {repository}")

        self.num_workers = num_workers
        self.repo_snapshot = repo_snapshot
        commands_filepath = Path(commands_filepath)
        self.dockerfile_commands = self.load_dockerfile_commands(commands_filepath)
        self.docker_config = ContainerConfig(**docker_config)
        self.repository = repository
        self.clients: list[SandboxClient] = []
        self.sessions: list[Any] = []
        # Add session management
        self._session_locks: list[Lock] = []
        self._session_semaphore: asyncio.Semaphore
        # Retry and timeout configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        logging.info("Initializing TestPipeline")

    async def init_worker(self) -> tuple[SandboxClient, Any]:
        """
        Initialize an independent SandboxClient & session for a "worker".
        """
        client = SandboxClient()
        await client.__aenter__()
        client.init_docker()
        session = await client.start_docker_session(
            project=self.repo_snapshot,
            commands=self.dockerfile_commands,
            config=self.docker_config,
        )
        await session.__aenter__()
        await session.connected()
        return client, session

    async def setup(self) -> None:
        for _ in range(self.num_workers):
            client, session = await self.init_worker()
            await session.reset()
            self.clients.append(client)
            self.sessions.append(session)
            self._session_locks.append(Lock())

        # Create semaphore to limit concurrent access to sessions
        self._session_semaphore = asyncio.Semaphore(self.num_workers)

    async def cleanup_worker(self, client, session) -> None:
        await session.__aexit__(None, None, None)
        await client.__aexit__(None, None, None)

    async def _execute_with_timeout_and_retry(self, operation, *args, **kwargs) -> Any:
        """
        Execute an operation with timeout and retry logic.
        """
        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(operation(*args, **kwargs), timeout=self.timeout)
                return result
            except (
                aiohttp.client_exceptions.ServerDisconnectedError,
                aiohttp.client_exceptions.ClientConnectorError,
                asyncio.TimeoutError,
                ConnectionError,
            ) as e:
                logging.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {type(e).__name__}: {e}"
                )

                if attempt == self.max_retries - 1:
                    print("Could not run the tests. Exiting.")
                    logging.warning("Could not run the tests. Exiting.")
                    return None
                await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logging.error(f"Non-retryable error: {type(e).__name__}: {e}")
                raise e

    async def worker_loop(
        self, session, task_queue: Queue, results: list, do_edit: bool = True
    ) -> None:
        """
        Used only in .run_parallel() method
        For now we do not use this method, since parrallel run is handled by Langgraph
        """
        while True:
            item = await task_queue.get()
            if item is None:
                break  # Special marker for shutdown
            try:
                output = await self.run_single_task(item, session, do_edit)
                results.append(output)
            except Exception as e:
                logging.error(
                    f"Error processing item {getattr(item, 'idx', 'unknown')}: {type(e).__name__}: {e}"
                )
                # TODO this should be changed
                results.append(
                    {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "datapoint": item if hasattr(item, "__dict__") else str(item),
                    }
                )
            finally:
                task_queue.task_done()

    async def close(self) -> None:
        # Clean up all sessions/clients
        for client, session in zip(self.clients, self.sessions):
            await self.cleanup_worker(client, session)

    @staticmethod
    def load_dockerfile_commands(filepath: Path) -> list[str]:
        if not filepath.exists():
            raise FileNotFoundError(f"Dockerfile commands file not found {filepath}")
        with open(filepath) as f:
            return f.read().splitlines()

    @staticmethod
    def get_command_output(cmd_output: Result) -> str:
        return (cmd_output.stdout + "\n" + cmd_output.stderr).strip()

    async def _get_available_session(self) -> tuple[int, Lock]:
        """
        Get an available session using round-robin with locking.
        """
        async with self._session_semaphore:
            # Try to find an unlocked session
            for i, lock in enumerate(self._session_locks):
                if not lock.locked():
                    await lock.acquire()
                    return i, lock

            # If all are locked, wait for the first one
            await self._session_locks[0].acquire()
            return 0, self._session_locks[0]

    async def run_single_task(self, item: ItemToRun, session=None, do_edit: bool = True) -> dict:
        if session is not None:
            # Direct session provided - use it without locking (for worker_loop)
            # Wrap session operations with timeout and retry
            result = await self._execute_with_timeout_and_retry(
                self._execute_on_session, item=item, session=session, do_edit=do_edit
            )
            return result
        # No session provided - get one from the pool with proper locking
        session_idx, session_lock = await self._get_available_session()
        try:
            session = self.sessions[session_idx]
            # Wrap session operations with timeout and retry
            result = await self._execute_with_timeout_and_retry(
                self._execute_on_session, item=item, session=session, do_edit=do_edit
            )
            return result
        finally:
            session_lock.release()

    async def _execute_on_session(self, item: ItemToRun, session, do_edit: bool = True) -> dict:
        """
        Execute the actual task on a session. This is the core logic extracted
        to avoid duplication between locked and unlocked execution paths.
        """
        tests_str = " ".join(item.tests)
        if self.repository == "sympy":
            test_command = f"python bin/test {tests_str}"
        elif self.repository == "django":
            test_command = f"python tests/runtests.py {tests_str} --verbosity 0 --noinput"
        else:
            raise ValueError(f"Invalid repository: {self.repository}")

        time_start_global = time.perf_counter()
        if do_edit:
            await session.edit_file(
                path=item.file_path,
                start_line=item.start_line,
                end_line=item.end_line,
                new_content=item.replace_content,
            )
        time_edit = time.perf_counter() - time_start_global

        time_start = time.perf_counter()
        edited_file_res = await session.exec(command=f"cat {item.file_path}")
        time_cat = time.perf_counter() - time_start

        time_start = time.perf_counter()
        test_result = await session.exec(command=test_command)
        time_test = time.perf_counter() - time_start

        test_output = self.get_command_output(test_result)
        edited_file = self.get_command_output(edited_file_res)

        time_start = time.perf_counter()
        await session.reset()
        time_reset = time.perf_counter() - time_start
        time_total = time.perf_counter() - time_start_global
        await asyncio.sleep(1)

        return {
            "datapoint": item,
            "test_output": test_output,
            "edited_file": edited_file,
            "time_test": time_test,
            "time_reset": time_reset,
            "time_cat": time_cat,
            "time_edit": time_edit,
            "time_total": time_total,
        }

    async def run_parallel(self, items: list[ItemToRun], do_edit: bool = True) -> list[dict]:
        """
        For now we do not use this method, since parrallel run is handled by Langgraph
        """
        results: list[dict[str, Any]] = []
        task_queue: Queue[ItemToRun | None] = Queue()

        # Populate queue
        for item in items:
            await task_queue.put(item)
        # Add stop signals
        for _ in range(self.num_workers):
            await task_queue.put(None)

        workers = []
        for session in self.sessions:
            workers.append(create_task(self.worker_loop(session, task_queue, results, do_edit)))

        await gather(*workers)

        return results
