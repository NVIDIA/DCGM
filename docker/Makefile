# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

DCGM_VERSION   := 2.4.5
FULL_VERSION   := $(DCGM_VERSION)-1
OUTPUT         := type=image
PLATFORMS      := linux/amd64,linux/arm64
DOCKERCMD      := docker buildx build

.PHONY: all local
all: ubuntu20.04 ubi8

local:
ifeq ($(shell uname -p),aarch64)
	$(MAKE) PLATFORMS=linux/arm64 OUTPUT=type=docker DOCKERCMD='docker build'
else
	$(MAKE) PLATFORMS=linux/amd64 OUTPUT=type=docker DOCKERCMD='docker build'
endif

push:
	$(MAKE) ubuntu20.04 OUTPUT=type=registry
	$(MAKE) ubi8 OUTPUT=type=registry

ubuntu20.04:
	$(DOCKERCMD) --pull \
		--output $(OUTPUT) \
		--platform $(PLATFORMS) \
		--build-arg "DCGM_VERSION=$(DCGM_VERSION)" \
		--build-arg "VERSION=$(FULL_VERSION)" \
		--tag "$(REGISTRY)/dcgm:$(FULL_VERSION)-ubuntu20.04" \
		--file Dockerfile.ubuntu20.04 .

ubi8:
	$(DOCKERCMD) --pull \
		--output $(OUTPUT) \
		--platform $(PLATFORMS) \
		--build-arg "DCGM_VERSION=$(DCGM_VERSION)" \
		--build-arg "VERSION=$(FULL_VERSION)" \
		--tag "$(REGISTRY)/dcgm:$(FULL_VERSION)-ubi8" \
		--file Dockerfile.ubi8 .
