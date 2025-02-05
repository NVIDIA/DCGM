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

import pydcgm
import dcgm_structs
import dcgm_structs_internal
import dcgm_agent_internal
import dcgm_fields
import argparse
import sys

def create_fake_gpus(handle, gpuCount):
    cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
    cfe.numToCreate = 0
    fakeGpuList = []

    for i in range(0, gpuCount):
        cfe.entityList[cfe.numToCreate].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU
        cfe.numToCreate += 1

    updated = dcgm_agent_internal.dcgmCreateFakeEntities(handle, cfe)
    for i in range(0, updated.numToCreate):
        if updated.entityList[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU:
            fakeGpuList.append(updated.entityList[i].entity.entityId)

    return fakeGpuList

def create_fake_gpu_instances(handle, gpuIds, instanceCount):
    cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
    cfe.numToCreate = 0
    fakeInstanceMap = {}

    if instanceCount > 0:
        for i in range(0, instanceCount):
            cfe.entityList[cfe.numToCreate].parent.entityGroupId = dcgm_fields.DCGM_FE_GPU
            gpuListIndex = cfe.numToCreate % len(gpuIds)
            cfe.entityList[cfe.numToCreate].parent.entityId = gpuIds[gpuListIndex]
            cfe.entityList[cfe.numToCreate].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
            cfe.numToCreate += 1

        # Create the instances first so we can control which GPU the compute instances are placed on
        updated = dcgm_agent_internal.dcgmCreateFakeEntities(handle, cfe)
        for i in range(0, updated.numToCreate):
            if updated.entityList[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_I:
                fakeInstanceMap[updated.entityList[i].entity.entityId] = updated.entityList[i].parent.entityId

    return fakeInstanceMap

def create_fake_compute_instances(handle, parentIds, ciCount):
    fakeCIMap = {}
    if ciCount > 0:
        cfe = dcgm_structs_internal.c_dcgmCreateFakeEntities_v2()
        instanceIndex = 0
        for i in range(0, ciCount):
            cfe.entityList[cfe.numToCreate].parent.entityGroupId = dcgm_fields.DCGM_FE_GPU_I
            if instanceIndex > len(parentIds):
                instanceIndex = 0
            cfe.entityList[cfe.numToCreate].parent.entityId = parentIds[instanceIndex]
            instanceIndex = instanceIndex + 1
            cfe.entityList[cfe.numToCreate].entity.entityGroupId = dcgm_fields.DCGM_FE_GPU_CI
            cfe.numToCreate += 1

        updated = dcgm_agent_internal.dcgmCreateFakeEntities(handle, cfe)
        for i in range(0, updated.numToCreate):
            if updated.entityList[i].entity.entityGroupId == dcgm_fields.DCGM_FE_GPU_CI:
                fakeCIMap[updated.entityList[i].entity.entityId] = updated.entityList[i].parent.entityId

    return fakeCIMap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu-count', type=int, default=1, dest='gpuCount',
            help='Specify the number of fake GPUs to create')
    parser.add_argument('-i', '--gpu-instance-count', type=int, default=2, dest='gpuInstanceCount',
            help='Specify the number of fake GPU instances to create')
    parser.add_argument('-c', '--compute-instance-count', type=int, default=2, dest='ciCount',
            help='Specify the number of fake compute instances to create')
    args = parser.parse_args()

    if args.gpuCount < 1:
        print("GPU count must be 1 or larger.")
        sys.exit(1)

    if args.ciCount > 0 and args.gpuInstanceCount < 1:
        print("GPU instance count must be greater than 1 if compute instance count is greater than 1")
        sys.exit(1)

    handle = pydcgm.DcgmHandle(None, "localhost", dcgm_structs.DCGM_OPERATION_MODE_AUTO)
    gpuIds = create_fake_gpus(handle.handle, args.gpuCount)
    if args.gpuInstanceCount > 0:
        instanceMap = create_fake_gpu_instances(handle.handle, gpuIds, args.gpuInstanceCount)
        if args.ciCount > 0:
            create_fake_compute_instances(handle.handle, list(instanceMap.keys()), args.ciCount)

    print("Created {} fake GPUs, {} fake GPU instances, and {} fake compute instances".format(args.gpuCount, args.gpuInstanceCount, args.ciCount))

if __name__ == "__main__":
    main()

