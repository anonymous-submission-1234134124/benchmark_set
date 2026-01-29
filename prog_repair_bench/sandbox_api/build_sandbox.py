from pathlib import Path

from dotenv import load_dotenv
from private_sandbox.client import SandboxHTTPClient

load_dotenv()


from fire import Fire


async def main(client_name="sympy-test", yaml_path="sandbox-images-sympy.yaml"):
    async with SandboxHTTPClient(
        # TODO Restore
        orchestrator_url="orchestrator_url.com",
        namespace="sandbox-namespace",
    ) as client:
        await client.register_client(name=client_name)
        return await client.build_and_push_images(
            path=Path(yaml_path),
            poll_interval=30,
        )


if __name__ == "__main__":
    Fire(main)
