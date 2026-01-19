#include "image.h"
#include "image_normalize.h"
#include <fstream>
#include <iostream>

void create_test_ppm(const char *filename, int width, int height) {
  std::ofstream ofs(filename, std::ios::binary);
  ofs << "P6\n" << width << " " << height << "\n255\n";
  for (int i = 0; i < width * height; ++i) {
    unsigned char r = (i % 256);
    unsigned char g = ((i * 2) % 256);
    unsigned char b = 255;
    ofs << r << g << b;
  }
  ofs.close();
}

int main() {
  const char *filename = "test.ppm";
  int width = 4;
  int height = 4;
  create_test_ppm(filename, width, height);

  std::cout << "Created " << filename << std::endl;

  Image *img = image_normalize(filename);

  if (img) {
    std::cout << "Image was able to normalize successfully." << std::endl;
    std::cout << "Dimensions: " << img->width << "x" << img->height
              << std::endl;
    std::cout << "First pixel values (0-1):" << std::endl;
    std::cout << "R: " << img->R[0] << std::endl;
    std::cout << "G: " << img->G[0] << std::endl;
    std::cout << "B: " << img->B[0] << std::endl;

    // simple check
    float expected_r = 0.0f / 255.0f;   // i=0 -> 0
    float expected_g = 0.0f / 255.0f;   // i=0 -> 0
    float expected_b = 255.0f / 255.0f; // 1.0

    std::cout << "Expected: R=" << expected_r << " G=" << expected_g
              << " B=" << expected_b << std::endl;

    delete img;
  } else {
    std::cerr << "Failed to normalize image." << std::endl;
  }

  return 0;
}
