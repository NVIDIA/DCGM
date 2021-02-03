FROM ubuntu:18.04

RUN export DEBIAN_FRONTEND=noninteractive; \
 set -ex; \
 apt-get update -q; \
 apt-get full-upgrade -qy; \
 apt-get install -qy apt-transport-https ca-certificates curl gnupg2 software-properties-common python python-pip apt-utils; \
 curl -fsSL get.docker.com | sh; \
 pip install -q docker-squash; \
 rm -rf /var/lib/apt/lists/*
