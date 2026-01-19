#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.h"
#include "normalization.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


Image *image_normalize(const char *path) {

  int w, h, channels;
  unsigned char *image = stbi_load(path, &w, &h, &channels, 4);
  if (!image) {
    printf("Failed to load image: %s\n", path);
    exit(1);
  }

  uchar4 *pixels = reinterpret_cast<uchar4 *>(image);

  uchar4 *d_in = nullptr;
  float4 *d_out = nullptr;

  cudaMalloc(&d_in, w * h * sizeof(uchar4));
  cudaMalloc(&d_out, w * h * sizeof(float4));

  cudaMemcpy(d_in, pixels, w * h * sizeof(uchar4), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (w * h + threads - 1) / threads;

  normalize<<<blocks, threads>>>(d_in, d_out, w * h);
  cudaDeviceSynchronize();

  float4 *normalized_image = new float4[w * h];
  cudaMemcpy(normalized_image, d_out, w * h * sizeof(float4),
             cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  stbi_image_free(image);

  Image *img = new Image(h, w);

  for (int i = 0; i < w * h; i++) {
    img->R[i] = normalized_image[i].x;
    img->G[i] = normalized_image[i].y;
    img->B[i] = normalized_image[i].z;
  }

  delete[] normalized_image;

  return img;
}
