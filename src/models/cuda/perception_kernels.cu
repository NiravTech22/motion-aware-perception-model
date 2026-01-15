#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// --- Fast NMS Kernel (Heatmap-based) ---
// Uses 3D indexing: x -> width, y -> height, z -> batch
extern "C" __global__ void fast_nms_kernel(const float *__restrict__ objectness,
                                           float *__restrict__ output_mask,
                                           int batch_size, int height,
                                           int width, float threshold) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;

  if (w >= width || h >= height || b >= batch_size)
    return;

  int spatial_size = height * width;
  int idx = b * spatial_size + h * width + w;

  float score = objectness[idx];
  if (score < threshold) {
    output_mask[idx] = 0.0f;
    return;
  }

  // Local suppression: 3x3 window check for local maxima
  bool is_max = true;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      if (dx == 0 && dy == 0)
        continue;
      int ny = h + dy;
      int nx = w + dx;
      if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
        int n_idx = b * spatial_size + ny * width + nx;
        float n_score = objectness[n_idx];
        if (n_score > score) {
          is_max = false;
          break;
        }
        // Tie-breaking by index to ensure deterministic NMS
        if (n_score == score && n_idx < idx) {
          is_max = false;
          break;
        }
      }
    }
    if (!is_max)
      break;
  }

  output_mask[idx] = is_max ? 1.0f : 0.0f;
}

// --- Coordinate Transform Kernel ---
// Maps (grid_x, grid_y) + (dx, dy) and depth -> (X, Y, Z)
// Uses 3D indexing: x -> width, y -> height, z -> batch
extern "C" __global__ void coordinate_transform_kernel(
    const float *__restrict__ bboxes,    // (B, 4, H, W) -> [dx, dy, dw, dh]
    const float *__restrict__ depth_map, // (B, 1, H, W)
    float *__restrict__ world_coords,    // (B, 3, H, W) -> [X, Y, Z]
    int batch_size, int height, int width, float focal_x, float focal_y,
    float center_x, float center_y, float grid_scale_x, float grid_scale_y) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;

  if (w >= width || h >= height || b >= batch_size)
    return;

  int spatial_size = height * width;
  int grid_idx = h * width + w;
  int b_offset = b * 4 * spatial_size;
  int w_offset = b * 3 * spatial_size;

  // Center offsets from bboxes (B, 4, H, W)
  // Channel 0: dx, Channel 1: dy
  float dx = bboxes[b_offset + 0 * spatial_size + grid_idx];
  float dy = bboxes[b_offset + 1 * spatial_size + grid_idx];

  // Pixel coordinates in input image (e.g. 256x256)
  // Centering: grid center (w+0.5) + regression offset (dx)
  float px = ((float)w + 0.5f) * grid_scale_x + dx;
  float py = ((float)h + 0.5f) * grid_scale_y + dy;

  // Depth (Z) in world units
  // Indexing into (B, 1, H, W) contiguous tensor
  int depth_idx = b * spatial_size + grid_idx;
  float Z = depth_map[depth_idx];

  // Simple pinhole camera back-projection
  float X = (px - center_x) * Z / focal_x;
  float Y = (py - center_y) * Z / focal_y;

  // Store world coordinates (B, 3, H, W)
  world_coords[w_offset + 0 * spatial_size + grid_idx] = X;
  world_coords[w_offset + 1 * spatial_size + grid_idx] = Y;
  world_coords[w_offset + 2 * spatial_size + grid_idx] = Z;
}
