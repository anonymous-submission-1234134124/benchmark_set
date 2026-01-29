from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Body, FastAPI
from private_sandbox.api.auth import BasicAuth
from private_sandbox.api.exceptions import SandboxException
from private_sandbox.api.orchestrator.servers import ErrorResponse, StartServerResponse
from private_sandbox.api.tools.bash import BashCommandResponse
from private_sandbox.client.http_client import SandboxHTTPClient
from kubernetes_asyncio.client import V1ResourceRequirements
from prometheus_fastapi_instrumentator import Instrumentator

from prog_repair_bench.run_item import ItemToRun

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
REPO = os.getenv("REPO", "django")  # "django" or "sympy"
IMAGE_TAG = os.getenv("IMAGE_TAG", None)

if REPO == "django":
    if IMAGE_TAG is None:
        # TODO Restore
        IMAGE_TAG = "registry/image/path:tag"
    SERVER_NAME = "django-sandbox-api"
elif REPO == "sympy":
    if IMAGE_TAG is None:
        # TODO Restore
        IMAGE_TAG = "registry/image/path:tag"
    SERVER_NAME = "sympy-sandbox-api"
else:
    raise ValueError(f"Unknown repo {REPO}. Must be either 'django' or 'sympy'")

# Number of Sandbox servers to spin up in parallel
NUM_SERVERS = int(os.getenv("NUM_SERVERS", 1))


NAMESPACE = os.getenv("NAMESPACE", "sandbox")
RESOURCES = V1ResourceRequirements(
    requests={"cpu": "1500m", "memory": "1024Mi", "ephemeral-storage": "5Gi"},
    limits={"cpu": "3000m", "memory": "5120Mi", "ephemeral-storage": "5Gi"},
)
HEARTBEAT_INTERVAL = os.getenv("HEARTBEAT_INTERVAL", 120)  # seconds
SERVER_START_TIMEOUT = os.getenv("SERVER_START_TIMEOUT", 600)  # seconds
# ---------------------------------------------------------------------------
# Retry / queue configuration
# ---------------------------------------------------------------------------
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", 5.0))  # seconds
BASH_COMMAND_TIMEOUT = float(os.getenv("BASH_COMMAND_TIMEOUT", 60))  # seconds
# Timeout for the whole _run_single_task operation.
# It should be greater than the sum of timeouts of all operations inside.
# 10 seconds for cat + BASH_COMMAND_TIMEOUT for test_cmd + buffer for other operations.
# 10 + 60 + 10 = 80 seconds
RUN_SINGLE_TASK_TIMEOUT = float(os.getenv("RUN_SINGLE_TASK_TIMEOUT", 80.0))  # seconds

# ---------------------------------------------------------------------------
# Keep-alive configuration (prevents server idle shutdown)
# ---------------------------------------------------------------------------
KEEPALIVE_INTERVAL = float(os.getenv("KEEPALIVE_INTERVAL", 240))  # seconds


# ---------------------------------------------------------------------------
# Error handling helpers
# ---------------------------------------------------------------------------


def _format_error_response_message(response: ErrorResponse) -> str:
    """Return a concise message describing the orchestrator error."""

    body = response.body
    message = body if body else str(response)

    status_code = response.status_code
    if status_code is not None:
        return f"{message} (status_code={status_code})"
    return message


def _server_response_or_error(
    response: StartServerResponse | ErrorResponse,
    context: str,
) -> StartServerResponse:
    """Return the server info or raise a descriptive SandboxException."""
    if isinstance(response, ErrorResponse):
        raise SandboxException(f"{context}: {_format_error_response_message(response)}")

    return response


# ---------------------------------------------------------------------------
# Background keep-alive task
# ---------------------------------------------------------------------------


async def _keepalive_loop(app: FastAPI, client: SandboxHTTPClient, server_id: int):
    """
    Periodically executes a no-op command on each server to keep it alive.
    If the server is not responding, it will be replaced with a new one.
    """

    try:
        while True:
            try:
                await client.execute_bash(
                    server_id=server_id,
                    script="true",  # cheap no-op
                    command_timeout=10.0,
                )
                print(f"[KEEPALIVE] Successfully pinged server {server_id}")
            except Exception as e:
                # Log and continue; don't crash the loop
                print(f"[KEEPALIVE] Error pinging server {server_id}: {e}")

                # Remove server from list
                app.state.deleted_server_ids.add(server_id)
                # Start a new replacement server
                try:
                    replacement_response: StartServerResponse | ErrorResponse = (
                        await client.start_server(
                            image_tag=IMAGE_TAG,
                            server_name=SERVER_NAME,
                            namespace=NAMESPACE,
                            resources=RESOURCES,
                            server_start_wait_timeout_in_seconds=SERVER_START_TIMEOUT,
                        )
                    )
                    # If the client response is an ErrorResponse, it will raise an SandboxException
                    # If we got a StartServerResponse, it will return the server info
                    new_server_info = _server_response_or_error(
                        replacement_response,
                        "Failed to start replacement server",
                    )

                    server_id = new_server_info.server_id
                    # Add the new server to the queue
                    await app.state.server_ids.put(server_id)
                    print(f"[KEEPALIVE] Successfully started new server {server_id}")
                except Exception as e:
                    # We will try to start a new server in the next iteration
                    print(f"[KEEPALIVE] Failed to start replacement server: {e}")

            await asyncio.sleep(KEEPALIVE_INTERVAL)
    except asyncio.CancelledError:
        # normal during shutdown
        pass


# ---------------------------------------------------------------------------
# Background status loop (periodic status logging)
# ---------------------------------------------------------------------------


async def _status_loop(app: FastAPI):
    """
    Periodically logs the number of busy workers/servers and queue stats.
    - Busy servers: number of server IDs currently running requests
    - Worker tasks: number of worker tasks currently running (but not necessarily busy)
    - Task queue: number of items waiting to be processed
    - Deleted servers: number of servers that have been deleted
    """

    try:
        while True:
            num_workers = len(app.state.worker_tasks)
            num_available_servers = app.state.server_ids.qsize()
            num_total_servers = NUM_SERVERS
            busy_servers = max(num_total_servers - num_available_servers, 0)
            num_tasks_waiting = app.state.task_queue.qsize()
            num_deleted_servers = len(app.state.deleted_server_ids)
            num_errors_since_startup = app.state.num_errors_since_startup
            num_requests_since_startup = app.state.num_requests_since_startup

            print(
                f"[STATUS] busy_servers={busy_servers:,}/{num_total_servers:,}"
                f" worker_tasks={num_workers:,}/{num_total_servers:,}"
                f" task_queue={num_tasks_waiting:,}"
                f" num_deleted_servers={num_deleted_servers:,}"
                f" num_errors_since_startup={num_errors_since_startup:,}"
                f" num_requests_since_startup={num_requests_since_startup:,}"
            )

            await asyncio.sleep(10)
    except asyncio.CancelledError:
        # normal during shutdown
        pass


# ---------------------------------------------------------------------------
# Internal helpers (execution / retry / worker)
# ---------------------------------------------------------------------------


async def _execute_with_timeout_and_retry(
    client: SandboxHTTPClient,
    item: ItemToRun,
    do_edit: bool = True,
):
    """Execute operation with timeout and basic retry/backoff logic."""

    for attempt in range(MAX_RETRIES):
        try:
            server_id: int
            while True:
                server_id = await app.state.server_ids.get()
                # We remove deleted servers from the queue here: they are popped and will not be inserted back into the queue
                if server_id in app.state.deleted_server_ids:
                    continue
                else:
                    break

            try:
                # asyncio.wait_for will raise TimeoutError if exceeded
                return await asyncio.wait_for(
                    _run_single_task(client, server_id, item, do_edit),
                    timeout=RUN_SINGLE_TASK_TIMEOUT,
                )
            finally:
                try:
                    await client.reset_project(
                        server_id=server_id, reset_timeout=RUN_SINGLE_TASK_TIMEOUT
                    )
                    print("Successfully reset project")
                except Exception as e:
                    print(f"[CRITICAL] Error resetting project: {e}")
                    # Mark server as unhealthy and kill it to avoid reusing a bad environment
                    app.state.deleted_server_ids.add(server_id)
                    try:
                        await client.stop_server(namespace=NAMESPACE, server_id=server_id)
                        print(f"Stopped server {server_id} due to failed reset")
                    except Exception as stop_e:
                        print(
                            f"[CRITICAL] Failed to stop server {server_id} after reset failure: {stop_e}"
                        )
                finally:
                    # Return server to pool only if it's not marked as deleted
                    if server_id not in app.state.deleted_server_ids:
                        await app.state.server_ids.put(server_id)
        except (
            SandboxException,
            asyncio.TimeoutError,
            ConnectionError,
            Exception,
        ):
            # On last attempt propagate the error
            if attempt == MAX_RETRIES - 1:
                raise

            # Small back-off before retrying
            await asyncio.sleep(RETRY_DELAY)


async def _run_single_task(
    client: SandboxHTTPClient,
    server_id: int,
    item: ItemToRun,
    do_edit: bool = True,
):
    """Core logic that edits a file (optional), runs tests, and returns outputs."""

    # Compose test command (expects absolute test file paths in item.tests)
    tests_str = " ".join(item.tests)
    if REPO == "sympy":
        test_cmd = f"python bin/test {tests_str}"
    elif REPO == "django":
        test_cmd = f"python tests/runtests.py {tests_str} --verbosity 0 --noinput"
    else:
        raise ValueError(f"Unknown repo {REPO}. Must be either 'django' or 'sympy'")

    print(f"Running test command: {test_cmd}")
    # Apply patch if requested
    if do_edit:
        await client.edit_file(
            server_id=server_id,
            file_path=item.file_path,
            start_line=item.start_line,
            end_line=item.end_line,
            new_content=item.replace_content,
        )

        print("Successfully edited file")

    cat_res: tuple[BashCommandResponse] = await asyncio.gather(
        client.execute_bash(
            server_id=server_id, script=f"cat {item.file_path}", command_timeout=10
        ),
    )
    test_res: tuple[BashCommandResponse] = await asyncio.gather(
        client.execute_bash(
            server_id=server_id, script=test_cmd, command_timeout=BASH_COMMAND_TIMEOUT
        ),
    )

    print("Successfully ran test command")
    print(f"test_res: {test_res}")
    edited_file = _extract_output(cat_res)
    test_output = _extract_output(test_res)

    return {
        "datapoint": {k: v for k, v in item.__dict__.items()},
        "test_output": test_output,
        "edited_file": edited_file,
    }


async def _worker(app: FastAPI):
    """Background worker that sequentially processes queued tasks."""

    client: SandboxHTTPClient = app.state.client  # type: ignore[attr-defined]
    task_queue: asyncio.Queue = app.state.task_queue  # type: ignore[attr-defined]

    while True:
        item, do_edit, fut = await task_queue.get()

        # Sentinel => shutdown
        if item is None:
            task_queue.task_done()
            break

        try:
            result = await _execute_with_timeout_and_retry(client, item, do_edit)
            fut.set_result(result)
        except Exception as e:
            app.state.num_errors_since_startup += 1
            fut.set_exception(e)
        finally:
            task_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler that starts/stops a single Sandbox server."""

    # ---------------------------------------------------------------------
    # Startup
    # ---------------------------------------------------------------------
    username = os.getenv("BASIC_AUTH_USERNAME")
    password = os.getenv("BASIC_AUTH_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Environment variables BASIC_AUTH_USERNAME and BASIC_AUTH_PASSWORD must be provided."
        )

    print(f"Starting Sandbox {REPO} API with {NUM_SERVERS} servers")

    client = SandboxHTTPClient(
        # TODO Restore
        orchestrator_url="orchestrator_url.com",
        namespace=NAMESPACE,
        auth=BasicAuth(username=username, password=password),
        heartbeat_interval_in_seconds=HEARTBEAT_INTERVAL,
    )

    # Health-check and register this client with the orchestrator
    print("Checking orchestrator health")
    await client.health_check()
    print("Registering a new client")
    await client.register_client(name=SERVER_NAME)

    # ------------------------------------------------------------------
    # Start multiple Sandbox server instances in parallel
    # ------------------------------------------------------------------
    async def _start_server() -> tuple[StartServerResponse, asyncio.Task]:
        server_name = SERVER_NAME
        print(f"Starting server {server_name}")
        try:
            server_response: StartServerResponse | ErrorResponse = await client.start_server(
                image_tag=IMAGE_TAG,
                server_name=server_name,
                namespace=NAMESPACE,
                resources=RESOURCES,
                server_start_wait_timeout_in_seconds=SERVER_START_TIMEOUT,
            )
        except Exception as exc:
            print(f"[STARTUP] Exception while starting server {server_name}: {exc}")
            raise

        # If the client response is an ErrorResponse, it will raise an SandboxException
        # If we got a StartServerResponse, it will return the server info
        server_info = _server_response_or_error(
            server_response, f"Failed to start server {server_name}"
        )

        print(f"Successfully started server {server_name}: {server_info}")
        keepalive_task = asyncio.create_task(_keepalive_loop(app, client, server_info.server_id))
        return server_info, keepalive_task

    print(f"Starting {NUM_SERVERS} servers...")
    servers_info: list[tuple[StartServerResponse, asyncio.Task]] = await asyncio.gather(
        *[_start_server() for i in range(NUM_SERVERS)]
    )
    server_ids: list[int] = [info.server_id for info, _ in servers_info]
    keepalive_tasks = [task for _, task in servers_info]
    print(f"Started servers with IDs: {server_ids}")

    # Store for request handlers
    app.state.client = client
    app.state.server_ids = asyncio.Queue()
    app.state.deleted_server_ids = set()

    # Add all servers to the queue
    for server_id in server_ids:
        await app.state.server_ids.put(server_id)

    # ------------------------------------------------------------------
    # Queues & workers
    # ------------------------------------------------------------------
    task_queue: asyncio.Queue = asyncio.Queue()

    app.state.task_queue = task_queue
    app.state.num_errors_since_startup = 0
    app.state.num_requests_since_startup = 0

    # Spin up worker tasks (one per server)
    worker_tasks = [asyncio.create_task(_worker(app)) for _ in server_ids]
    app.state.worker_tasks = worker_tasks

    # Start monitor task
    status_task = asyncio.create_task(_status_loop(app))
    app.state.status_task = status_task

    print(f"Sandbox {REPO} API startup complete")

    try:
        yield  # ---- application is running ----
    finally:
        # ------------------------------------------------------------------
        # Shutdown
        # ------------------------------------------------------------------
        print(f"Shutting down Sandbox {REPO} API")
        # Stop all servers - drain the queue to get all server IDs
        while (sid := await app.state.server_ids.get()) is not None:
            if sid in app.state.deleted_server_ids:
                continue
            try:
                print(f"Stopping server {sid}")
                await client.stop_server(namespace=NAMESPACE, server_id=sid)
            except Exception as e:
                print(f"Error stopping server {sid}: {e}")

        # Signal workers to shut down and wait for completion
        for _ in app.state.worker_tasks:
            await app.state.task_queue.put((None, None, None))

        await asyncio.gather(*app.state.worker_tasks)
        # We need to cancel the tasks gracefully or they will be left hanging
        for task in keepalive_tasks:
            task.cancel()

        await asyncio.gather(*keepalive_tasks)

        # Stop monitor task
        app.state.status_task.cancel()
        await asyncio.gather(app.state.status_task)

        print(f"Sandbox {REPO} API shutdown complete")


app = FastAPI(title=f"Sandbox {REPO} API", lifespan=lifespan)

Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _extract_output(cmd_output: tuple[BashCommandResponse]) -> str:
    """Extracts and combines stdout and stderr from the command output dictionary."""
    stdout = cmd_output[0].stdout
    stderr = cmd_output[0].stderr

    return f"{stdout}\n{stderr}".strip()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/run_item")
async def run_item(item: dict[str, Any] = Body(...), do_edit: bool = True):
    """Enqueue *item* for execution and await its result synchronously."""

    task_queue: asyncio.Queue = app.state.task_queue  # type: ignore[attr-defined]
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # Convert dict to ItemToRun dataclass
    item_to_run = ItemToRun(
        idx=item["idx"],
        dp_id=item["dp_id"],
        file_path=item["file_path"],
        replace_content=item["replace_content"],
        method_name=item["method_name"],
        start_line=item["start_line"],
        end_line=item["end_line"],
        tests=item.get("tests", []),
    )

    print(f"Running item: {item_to_run}")
    app.state.num_requests_since_startup += 1

    # Put the work in the queue
    await task_queue.put((item_to_run, do_edit, future))

    # Await result. If the future holds an exception, `await` will re-raise it.
    # FastAPI's default exception handler will then catch it, log the full
    # traceback, and return a standard 500 Internal Server Error response.
    result = await future
    return result
