#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
extern "C" void fast_nms_kernel(const float *objectness, float *output_mask,
                                int batch_size, int height, int width,
                                float threshold);

extern "C" void coordinate_transform_kernel(
    const float *bboxes, const float *depth_map, float *world_coords,
    int batch_size, int height, int width, float focal_x, float focal_y,
    float center_x, float center_y, float grid_scale_x, float grid_scale_y);

// Wrapper for NMS
torch::Tensor fast_nms(torch::Tensor objectness, float threshold) {
  TORCH_CHECK(objectness.is_cuda(), "objectness must be a CUDA tensor");

  // Ensure the tensor is contiguous for raw pointer indexing in the kernel
  auto obj_contig = objectness.contiguous();

  auto batch_size = obj_contig.size(0);
  auto height = obj_contig.size(2);
  auto width = obj_contig.size(3);
  auto output_mask = torch::zeros_like(obj_contig);

  // 3D Grid: x -> width, y -> height, z -> batch
  dim3 threads(16, 16, 1);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y, batch_size);

  fast_nms_kernel<<<blocks, threads>>>(obj_contig.data_ptr<float>(),
                                       output_mask.data_ptr<float>(),
                                       batch_size, height, width, threshold);

  return output_mask;
}

// Wrapper for Coordinate Transform
torch::Tensor coordinate_transform(torch::Tensor bboxes,
                                   torch::Tensor depth_map, float focal_x,
                                   float focal_y, float center_x,
                                   float center_y, float grid_scale_x,
                                   float grid_scale_y) {
  TORCH_CHECK(bboxes.is_cuda(), "bboxes must be a CUDA tensor");
  TORCH_CHECK(depth_map.is_cuda(), "depth_map must be a CUDA tensor");

  // Always ensure contiguity when using raw pointers for multi-dimensional data
  auto bboxes_contig = bboxes.contiguous();
  auto depth_contig = depth_map.contiguous();

  auto batch_size = bboxes_contig.size(0);
  auto height = bboxes_contig.size(2);
  auto width = bboxes_contig.size(3);
  auto world_coords =
      torch::zeros({batch_size, 3, height, width}, bboxes_contig.options());

  dim3 threads(16, 16, 1);
  dim3 blocks((width + threads.x - 1) / threads.x,
              (height + threads.y - 1) / threads.y, batch_size);

  coordinate_transform_kernel<<<blocks, threads>>>(
      bboxes_contig.data_ptr<float>(), depth_contig.data_ptr<float>(),
      world_coords.data_ptr<float>(), batch_size, height, width, focal_x,
      focal_y, center_x, center_y, grid_scale_x, grid_scale_y);

  return world_coords;
}

// Bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_nms", &fast_nms, "Fast CUDA NMS (Heatmap-based)");
  m.def("coordinate_transform", &coordinate_transform,
        "CUDA-accelerated coordinate transformation");
}
