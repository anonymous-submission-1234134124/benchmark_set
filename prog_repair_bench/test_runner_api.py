import asyncio
import logging
import time
import os

import aiohttp
from tqdm import tqdm

from prog_repair_bench.run_item import ItemToRun


class TestPipelineRestApi:
    def __init__(self, repository: str, **kwargs):

        if "sandbox_server_url" in kwargs:
            self.sandbox_server_url = kwargs.get("sandbox_server_url")
        elif "SANDBOX_URL" in os.environ:
            self.sandbox_server_url = os.environ.get("SANDBOX_URL")
        elif repository == "django":
            # TODO Restore
            self.sandbox_server_url = "0.0.0.0:7778/run_item"
        elif repository == "sympy":
            # TODO Restore
            self.sandbox_server_url = "0.0.0.0:7788/run_item"
        else:
            raise ValueError(f"Unknown repo {repository}. Must be either 'django' or 'sympy'")
        # Local setting
        # self.sandbox_server_url = kwargs.get("sandbox_server_url", "http://0.0.0.0:8013/run_item")
        # Request timeout in seconds
        sandbox_api_timeout = kwargs.get("sandbox_api_timeout", 6 * 60)
        # Maximum total connections in the connection pool
        self.max_connections = kwargs.get("max_connections", 200)
        # Maximum connections per host
        self.max_connections_per_host = kwargs.get("max_connections_per_host", 200)

        # Configure connection limits and timeout
        self.session: aiohttp.ClientSession
        self.timeout_config = aiohttp.ClientTimeout(total=sandbox_api_timeout)

        self.pbar = None
        logging.info("Initializing TestPipeline")

    async def setup(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def _get_sandbox_api_response(self, item: ItemToRun, do_edit: bool = True) -> dict | None:

        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )
        try:
            async with aiohttp.ClientSession(
                connector=connector, timeout=self.timeout_config
            ) as session:
                async with session.post(
                    self.sandbox_server_url, json=item.to_dict(), params={"do_edit": str(do_edit)}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(
                            f"[ERROR] Failed to retrieve data, status code: {response.status}, item was:\n {item.to_dict()}"
                        )
                        return None
        except (asyncio.TimeoutError, aiohttp.ServerTimeoutError):
            print(
                f"[ERROR] Request timed out after {self.timeout_config.total} seconds for item: {item.to_dict()}"
            )
            return None
        except Exception as e:
            print(f"[ERROR] Request failed with exception: {e} for item: {item.to_dict()}")
            return None

    async def run_single_task(self, item: ItemToRun, do_edit: bool = True) -> dict | None:
        """
        Execute the actual task. This is the core logic extracted
        to avoid duplication between locked and unlocked execution paths.
        """

        response_start = time.perf_counter()
        response = await self._get_sandbox_api_response(item, do_edit)
        response_duration = time.perf_counter() - response_start
        if self.pbar is not None:
            self.pbar.update(1)
        if response is None:
            return None

        timing_dict = {
            "time_total": response_duration,
            # Dummy values for to fit format
            "time_test": 0,
            "time_cat": 0,
            "time_edit": 0,
            "time_reset": 0,
        }
        response.update(timing_dict)

        return response

    async def run_parallel(
        self, items: list[ItemToRun], do_edit: bool = True, use_pbar: bool = False
    ) -> list[dict]:
        """
        For now, this method simply runs each item with run_single_task sequentially (or concurrently with asyncio.gather).
        Parallel/distributed execution should be handled by Langgraph runners or by running this method on multiple processes if real parallelism is needed.
        """
        if use_pbar:
            self.pbar = tqdm(total=len(items))

        tasks = [self.run_single_task(item, do_edit=do_edit) for item in items]
        results = await asyncio.gather(*tasks)
        return results
