# syntax=docker/dockerfile:latest

ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}-slim
RUN mkdir /src
COPY src/ src/

LABEL Maintainer="Vector Institute"

USER root

RUN pip install -r examples/docker_basic_example/fl_client/requirements.txt

CMD ["examples/docker_basic_example/fl_client/entrypoint.sh"]
