import os
import sys

import uvicorn
from fire import Fire


def main(
    num_servers: int = 4,
    repo: str = "django",
    host: str = "0.0.0.0",
    port: int = 8000,
    image_tag: str | None = None,
):
    # pass values to your module via env vars
    if not os.getenv("NUM_SERVERS"):
        os.environ["NUM_SERVERS"] = str(num_servers)

    if not os.getenv("REPO"):
        os.environ["REPO"] = repo.lower()

    if not os.getenv("IMAGE_TAG"):
        if image_tag:
            os.environ["IMAGE_TAG"] = image_tag

    print(f'Num servers to run: {os.environ["NUM_SERVERS"]}')
    print(f'Repo: {os.environ["REPO"]}')

    sys.path.insert(0, os.path.abspath("../../"))
    uvicorn.run("prog_repair_bench.sandbox_api.api:app", factory=False, host=host, port=port)


if __name__ == "__main__":
    Fire(main)
