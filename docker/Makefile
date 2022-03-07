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

DOCKER   ?= docker
REGISTRY ?= nvidia

DCGM_VERSION   := 2.3.4

all: ubuntu20.04 ubi8

push:
	$(DOCKER) push "$(REGISTRY)/dcgm:$(DCGM_VERSION)-ubuntu20.04"
	$(DOCKER) push "$(REGISTRY)/dcgm:$(DCGM_VERSION)-ubi8"

ubuntu20.04:
	$(DOCKER) build --pull \
		--build-arg "DCGM_VERSION=$(DCGM_VERSION)" \
		--tag "$(REGISTRY)/dcgm:$(DCGM_VERSION)-ubuntu20.04" \
		--file Dockerfile.ubuntu20.04 .

ubi8:
	$(DOCKER) build --pull \
		--build-arg "DCGM_VERSION=$(DCGM_VERSION)" \
		--build-arg "VERSION=$(DCGM_VERSION)" \
		--tag "$(REGISTRY)/dcgm:$(DCGM_VERSION)-ubi8" \
		--file Dockerfile.ubi8 .
