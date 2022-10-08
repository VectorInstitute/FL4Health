# syntax=docker/dockerfile:latest

ARG PYTHON_VERSION=3.8

FROM python:${PYTHON_VERSION}-slim
RUN mkdir /src
COPY . /src

LABEL Maintainer="Vector Institute"

USER root

RUN pip install -r src/requirements.txt

CMD ["src/entrypoint.sh"]