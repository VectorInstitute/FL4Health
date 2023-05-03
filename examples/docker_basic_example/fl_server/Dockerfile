# syntax=docker/dockerfile:latest

ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}-slim
RUN mkdir /src
COPY . src/

LABEL Maintainer="Vector Institute"

USER root

WORKDIR src/
RUN pip install -r examples/docker_basic_example/fl_server/requirements.txt
CMD ["examples/docker_basic_example/fl_server/entrypoint.sh"]
