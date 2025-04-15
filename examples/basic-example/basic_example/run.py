from pathlib import Path

from flwr.cli.run.run import run

if __name__ == "__main__":
    run(app=Path("."), federation="basic-example", stream=True)
