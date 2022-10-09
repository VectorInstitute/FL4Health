import os

NUM_CLIENTS = int(os.getenv("NUM_CLIENTS"))  # type: ignore
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS"))  # type: ignore
SERVER_INTERNAL_HOST = os.getenv("SERVER_INTERNAL_HOST")  # type: ignore
SERVER_INTERNAL_PORT = os.getenv("SERVER_INTERNAL_PORT")  # type: ignore
