#include <cassert>

// Intentionally doing a cuda assert to generate xid error 43
extern "C" __global__ void make_assert(int* buf, size_t size, int iterations)
{
    assert(false);
}
