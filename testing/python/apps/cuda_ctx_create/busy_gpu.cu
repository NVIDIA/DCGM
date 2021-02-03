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
