!/bin/sh

source .venv/bin/activate
mypy --config-file ./mypy.ini .
