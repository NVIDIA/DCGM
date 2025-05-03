# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# See https://docs.docker.com/buildx/working-with-buildx/ for instructions on
# setting up a buildx builder for multi-arch
# NOTE: To create an image for local debugging use "make local"

REGISTRY ?= nvidia

# Sync with latest DCGM version available in the public cuda repo
DCGM_VERSION   := 4.1.0
FULL_VERSION   := $(DCGM_VERSION)-1
OUTPUT         := type=image
PLATFORMS      := linux/amd64,linux/arm64
DOCKERCMD      := docker buildx build

.PHONY: all local
all: ubuntu22.04 ubi9

local:
ifeq ($(shell uname -p),aarch64)
	$(MAKE) PLATFORMS=linux/arm64 OUTPUT=type=docker DOCKERCMD='docker build'
else
	$(MAKE) PLATFORMS=linux/amd64 OUTPUT=type=docker DOCKERCMD='docker build'
endif

push:
	$(MAKE) ubuntu22.04 OUTPUT=type=registry
	$(MAKE) ubi9 OUTPUT=type=registry

ubuntu22.04:
	$(DOCKERCMD) --pull \
		--output $(OUTPUT) \
		--platform $(PLATFORMS) \
		--build-arg "DCGM_VERSION=$(DCGM_VERSION)" \
		--build-arg "VERSION=$(FULL_VERSION)" \
		--tag "$(REGISTRY)/dcgm:$(FULL_VERSION)-ubuntu22.04" \
		--file Dockerfile.ubuntu22.04 .

ubi9:
	$(DOCKERCMD) --pull \
		--output $(OUTPUT) \
		--platform $(PLATFORMS) \
		--build-arg "DCGM_VERSION=$(DCGM_VERSION)" \
		--build-arg "VERSION=$(FULL_VERSION)" \
		--tag "$(REGISTRY)/dcgm:$(FULL_VERSION)-ubi9" \
		--file Dockerfile.ubi9 .
