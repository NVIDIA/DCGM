FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN set -ex; \
 apt-get update -q; \
 apt-get full-upgrade -qy; \
 apt-get install -qy apt-transport-https ca-certificates curl gnupg2 software-properties-common python3-pip apt-utils; \
 curl -fsSL get.docker.com | sh; \
 pip install -q docker-squash; \
 rm -rf /var/lib/apt/lists/*
