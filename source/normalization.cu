#include <cuda_runtime.h>

__global__ void normalize(uchar4 *in, float4 *out, int pixel_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixel_count)
    return;

  uchar4 p = in[i];

  out[i].x = p.x / 255.0f;
  out[i].y = p.y / 255.0f;
  out[i].z = p.z / 255.0f;

  return;
}
