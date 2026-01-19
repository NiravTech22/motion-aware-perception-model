#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


__global__ void normalize(uchar4 *in, float4 *out, int pixel_count);
