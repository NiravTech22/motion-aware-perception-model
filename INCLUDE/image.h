#pragma once

class Image {
public:
  int height;
  int width;
  int pixels;

  float *R;
  float *G;
  float *B;

  Image(int h, int w) {
    height = h;
    width = w;
    pixels = height * width;

    R = new float[pixels];
    G = new float[pixels];
    B = new float[pixels];
  }

  ~Image() {
    delete[] R;
    delete[] G;
    delete[] B;
  }
};
