/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
extern "C" __global__ void make_gpu_busy(int* buf, size_t size, int iterations)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t step = blockDim.x * gridDim.x;

    for (size_t i = idx; i < size; i += step)
    {
        float f = buf[i];
        double f2 = buf[i];
        for (int j = 0; j < iterations; j++)
        {
            if (buf[i] % 2)
                buf[i] = buf[i] * 3 + 1;
            else
                buf[i] /= 2;
            // Add more calculations to burn more power
            f2 = f2 * 0.5 + buf[i];
            f = f * 0.5 + sqrtf(buf[i] + f);
        }
        buf[i] += (int) f + f2;
    }
}
