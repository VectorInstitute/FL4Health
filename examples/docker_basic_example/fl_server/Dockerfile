# syntax=docker/dockerfile:latest

ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}-slim
RUN mkdir /src
COPY src/ src/

LABEL Maintainer="Vector Institute"

USER root

RUN pip install -r src/examples/docker_basic_example/fl_server/requirements.txt

CMD ["src/examples/docker_basic_example/fl_server/entrypoint.sh"]
