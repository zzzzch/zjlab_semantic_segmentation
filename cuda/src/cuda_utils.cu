#include <cuda_utils.h>

#include <cuda.h>
#include <glog/logging.h>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"
namespace zjlab {
  namespace pose {

__global__ void kernelGenerateRoadMask(const cv::cuda::PtrStepSz<uchar3> seg,
                                       cv::cuda::PtrStepSzb mask) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  const auto& color = seg(y, x);
  if ((color.x == 0 && color.y == 0 && color.z == 0) |
      (color.x == 32 && color.y == 64 && color.z == 64) |
      (color.x == 0 && color.y == 0 && color.z == 128)) {
    mask(y, x) = 0;
  } else if ((color.x == 128 && color.y == 64 && color.z == 128) |
             (color.x == 192 && color.y == 128 && color.z == 128) |
             (color.x == 0 && color.y == 69 && color.z == 255) |
             (color.x == 255 && color.y == 0 && color.z == 0) |
             (color.x == 139 && color.y == 139 && color.z == 0) |
             (color.x == 203 && color.y == 192 && color.z == 255) |
             (color.x == 139 && color.y == 0 && color.z == 139) |
             (color.x == 192 && color.y == 128 && color.z == 32) |
             (color.x == 255 && color.y == 128 && color.z == 192) |
             (color.x == 64 && color.y == 128 && color.z == 255) |
             (color.x == 0 && color.y == 128 && color.z == 128) |
             (color.x == 125 && color.y == 0 && color.z == 0) |
             (color.x == 0 && color.y == 255 && color.z == 0)) {
    mask(y, x) = 1;
  } else {
    mask(y, x) = 0;
  }
}

__host__ void GenerateRoadMask(const cv::Mat& seg, cv::cuda::GpuMat& mask,
                               cv::cuda::Stream& stream) {
  cv::cuda::GpuMat cuda_seg(seg.size(), CV_8UC3);
  cuda_seg.upload(seg, stream);

  mask.create(seg.size(), CV_8UC1);

  dim3 cthreads(16, 16);
  dim3 cblocks(static_cast<unsigned int>(std::ceil(
                   seg.size().width / static_cast<double>(cthreads.x))),
               static_cast<unsigned int>(std::ceil(
                   seg.size().height / static_cast<double>(cthreads.y))));

  //  size_t N = static_cast<size_t> (1 * sizeof(CudaFeatureTrack));
  //  CudaFeatureTrack *tracksHost;
  //  cudaMallocHost((void **)&tracksHost, N);
  //  CudaFeatureTrack *tracks;
  //  cudaMalloc((void **)&tracks, N);
  //  cudaMemcpy(tracks, tracksHost, N, cudaMemcpyHostToDevice);

  cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

  kernelGenerateRoadMask<<<cblocks, cthreads, 0, cuda_stream>>>(cuda_seg, mask);
  //  stream.waitForCompletion();

  //  cudaMemcpy(tracksHost, tracks, N, cudaMemcpyDeviceToHost);
  //  stream.waitForCompletion();

  //  cudaFree(tracks);
  //  cudaFreeHost(tracksHost);
  //  tracks = nullptr;
  //  tracksHost = nullptr;

  cudaSafeCall(cudaGetLastError());
}

__global__ void nodiag_normalize(double* A, double* I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n)
    if (x == i && x != y) {
      I[x * n + y] /= A[i * n + i];
      A[x * n + y] /= A[i * n + i];
    }
}

__global__ void diag_normalize(double* A, double* I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n)
    if (x == y && x == i) {
      I[x * n + y] /= A[i * n + i];
      A[x * n + y] /= A[i * n + i];
    }
}

__global__ void gaussjordan(double* A, double* I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n) {
    if (x != i) {
      I[x * n + y] -= I[i * n + y] * A[x * n + i];
      if (y != i) {
        A[x * n + y] -= A[i * n + y] * A[x * n + i];
      }
    }
  }
}

__global__ void set_eye(double* A, double* I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x == y)
    A[x * n + y] = 1;
  else
    A[x * n + y] = 0;
}

__global__ void set_zero(double* A, double* I, int n, int i) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < n && y < n) {
    if (x != i) {
      if (y == i) {
        A[x * n + y] = 0;
      }
    }
  }
}

__host__ void inverse(double* L, int n, double* iL) {
  //  TicToc t;
  double *d_A, *dI;
  size_t ddsize = n * n * sizeof(double);

  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  // memory allocation
  cudaMalloc((void**)&d_A, ddsize);
  cudaMalloc((void**)&dI, ddsize);

  //  LOG(INFO) << t.toc() << " cuda malloc";
  //  I = new double[n*n];

  //  for (int i = 0; i<n; i++){
  //    for (int j = 0; j<n; j++){
  //      if (i == j) I[i*n + i] = 1.0;
  //      else I[i*n + j] = 0.0;
  //    }
  //  }
  cv::Mat cvEye = cv::Mat::eye(cv::Size(n, n), CV_64FC1);
  //  LOG(INFO) << t.toc() << " opencv eye";

  // copy data from CPU to GPU
  cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
  cudaMemcpy(dI, cvEye.data, ddsize, cudaMemcpyHostToDevice);

  //  LOG(INFO) << t.toc() << " cuda memcpy";

  // L^(-1)
  for (int i = 0; i < n; i++) {
    nodiag_normalize<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    diag_normalize<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    gaussjordan<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
    set_zero<<<numBlocks, threadsPerBlock>>>(d_A, dI, n, i);
  }

  //  LOG(INFO) << t.toc() << " cuda calcul";

  // copy data from GPU to CPU
  cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
  //  cudaMemcpy(I.data, d_A, ddsize, cudaMemcpyDeviceToHost);

  //  cout << "Cuda Time - inverse: " << time << "ms\n";
  //  savetofile(iL, "inv.txt", n, n);
  // savetofile(I, "I.txt", n, n);
  //  LOG(INFO) << t.toc() << " cuda memcpy back";
  cudaFree(d_A);
  cudaFree(dI);

  //  LOG(INFO) << t.toc() << " cuda free";
  //  double *c = new double[n*n];
  //  for (int i = 0; i<n; i++)
  //    for (int j = 0; j<n; j++)
  //    {
  //      c[i*n+j] = 0;  //put the initial value to zero
  //      for (int x = 0; x<n; x++)
  //        c[i*n + j] = c[i*n + j] + L[i*n+x] * iL[x*n + j];  //matrix
  //        multiplication
  //    }
  //  savetofile(c, "c.txt", n, n);

  //  delete[]I;
  //  delete[]L;
  //  delete[]iL;

  //  system("Pause");
  cudaSafeCall(cudaGetLastError());
}

/**
 * Judge whether a point in the raw image is the road surface point.
 * @param x_pixel : Pixel coordinate x in the raw iamge.
 * @param y_pixel : Pixel coordinate y in the raw iamge.
 * @param seg : Segmentation result of img.
 * @return :
 */
__device__ bool isRoadPoint(const int& x_pixel, const int& y_pixel,
                            const cv::cuda::PtrStepSz<uchar3> seg) {
  const auto& color = seg(y_pixel, x_pixel);
  if ((color.x == 0 && color.y == 0 && color.z == 0) ||
      (color.x == 32 && color.y == 64 && color.z == 64) ||
      (color.x == 0 && color.y == 0 && color.z == 128)) {
    return false;
  } else if ((color.x == 128 && color.y == 64 && color.z == 128) ||
             (color.x == 192 && color.y == 128 && color.z == 128) ||
             (color.x == 0 && color.y == 69 && color.z == 255) ||
             (color.x == 255 && color.y == 0 && color.z == 0) ||
             (color.x == 139 && color.y == 139 && color.z == 0) ||
             (color.x == 203 && color.y == 192 && color.z == 255) ||
             (color.x == 139 && color.y == 0 && color.z == 139) ||
             (color.x == 192 && color.y == 128 && color.z == 32) ||
             (color.x == 255 && color.y == 128 && color.z == 192) ||
             (color.x == 64 && color.y == 128 && color.z == 255) ||
             (color.x == 0 && color.y == 128 && color.z == 128) ||
             (color.x == 125 && color.y == 0 && color.z == 0) ||
             (color.x == 0 && color.y == 255 && color.z == 0)) {
    return true;
  } else {
    return false;
  }
}

/**
 *
 * @param A : Input matrix
 * @param inv_A : Inverse matrix of A.
 */
__device__ void invMat33d(const double A[3][3], double inv_A[3][3]) {
  inv_A[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
  inv_A[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) /
                (A[0][0] * A[1][1] * A[2][2] - A[0][0] * A[1][2] * A[2][1] -
                 A[0][1] * A[1][0] * A[2][2] + A[0][1] * A[1][2] * A[2][0] +
                 A[0][2] * A[1][0] * A[2][1] - A[0][2] * A[1][1] * A[2][0]);
}

/**
 * C = A * B
 * @param A : Input matrix.
 * @param B : Input vector.
 * @param C : Output vector.
 */
__device__ void dotMat33dVec3d(const double A[3][3], const double B[3], double C[3]) {
  C[0] = A[0][0] * B[0] + A[0][1] * B[1] + A[0][2] * B[2];
  C[1] = A[1][0] * B[0] + A[1][1] * B[1] + A[1][2] * B[2];
  C[2] = A[2][0] * B[0] + A[2][1] * B[1] + A[2][2] * B[2];
}

/**
 *
 * @param KR : Result of K*R.
 * @param Kt : Result of K*t.
 * @param n_plane_w : Normal of the road plane.
 * @param d_plane_w : A parameter of Plane Equation which is: n_plane_w*[x, y,
 * z] + d_plane_w = 0.
 * @param x_pixel : Pixel coordinate x in the raw iamge.
 * @param y_pixel : Pixel coordinate y in the raw iamge.
 * @param p_w : Position in the world.
 */
__device__ void calculatePixelLocation(
    const cv::cuda::PtrStepSz<double1> KR,
    const cv::cuda::PtrStepSz<double1> Kt,
    const cv::cuda::PtrStepSz<double1> n_plane_w, double d_plane_w,
    const int x_pixel, const int y_pixel, double* p_w) {
  // clang-format off
  double A[3][3];
  A[0][0] = KR(0, 0).x - KR(2, 0).x * x_pixel;
  A[0][1] = KR(0, 1).x - KR(2, 1).x * x_pixel;
  A[0][2] = KR(0, 2).x - KR(2, 2).x * x_pixel;
  A[1][0] = KR(1, 0).x - KR(2, 0).x * y_pixel;
  A[1][1] = KR(1, 1).x - KR(2, 1).x * y_pixel;
  A[1][2] = KR(1, 2).x - KR(2, 2).x * y_pixel;
  A[2][0] = n_plane_w(0, 0).x;
  A[2][1] = n_plane_w(1, 0).x;
  A[2][2] = n_plane_w(2, 0).x;
  double B[3];
  B[0] = -Kt(0, 0).x + Kt(2, 0).x * double(x_pixel);
  B[1] = -Kt(1, 0).x + Kt(2, 0).x * double(y_pixel);
  B[2] = -d_plane_w;

  double inv_A[3][3];
  invMat33d(A, inv_A);

  dotMat33dVec3d(inv_A, B, p_w);
  // clang-format on
}

__device__ double distToTrack(
    double* p_w, const cv::cuda::PtrStepSz<double1> camera_position,
    const cv::cuda::PtrStepSz<double1> head_vector) {
  double lateral_x = -head_vector(1, 0).x;
  double lateral_y = head_vector(0, 0).x;

  double ray_x = p_w[0] - camera_position(0, 0).x;
  double ray_y = p_w[1] - camera_position(1, 0).x;

  return ray_x * lateral_x + ray_y * lateral_y;
}

__device__ double distToCamera(
    double* p_w, const cv::cuda::PtrStepSz<double1> camera_position) {
  double ray_x = p_w[0] - camera_position(0, 0).x;
  double ray_y = p_w[1] - camera_position(1, 0).x;

  return ray_x * ray_x + ray_y * ray_y;
}

/**
 *
 * @param img : Raw image.
 * @param seg : Segmentation result of img.
 * @param KR  : Result of K*R.
 * @param Kt  : Result of K*t.
 * @param n_plane_w : Normal of the road plane.
 * @param d_plane_w : A parameter of Plane Equation which is: n_plane_w*[x, y,
 * z] + d_plane_w = 0.
 * @param mask_factor :
 * @param scale_factor: Pixels per meter.
 * @param x_min : minimum x of the top-down image area in the world.
 * @param y_min : minimum y of the top-down image area in the world.
 * @param x_max : maximum x of the top-down image area in the world.
 * @param y_max : maximum y of the top-down image area in the world.
 * @param top_down_image : The result.
 */
__global__ void kernelCalculateTopDownPixel(
    const cv::cuda::PtrStepSz<uchar3> img,
    const cv::cuda::PtrStepSz<uchar3> seg,
    const cv::cuda::PtrStepSz<double1> KR,
    const cv::cuda::PtrStepSz<double1> Kt,
    const cv::cuda::PtrStepSz<double1> camera_position,
    const cv::cuda::PtrStepSz<double1> head_vector,
    const cv::cuda::PtrStepSz<double1> n_plane_w, const double d_plane_w,
    const double mask_factor, const double scale_factor, const double x_min,
    const double y_min, const double x_max, const double y_max,
    cv::cuda::PtrStepSz<uchar3> top_down_image) {
  // Get the exact pixel in the img_gpu
  int x_pixel = blockIdx.x * blockDim.x + threadIdx.x;
  int y_pixel = blockIdx.y * blockDim.y + threadIdx.y;

  if (isRoadPoint(x_pixel, y_pixel, seg)) {
    double p_w[3];
    calculatePixelLocation(KR, Kt, n_plane_w, d_plane_w, x_pixel, y_pixel, p_w);

    if (fabsf(distToTrack(p_w, camera_position, head_vector)) < 10) {
      if (distToCamera(p_w, camera_position) < 300) {
        int r_img = top_down_image.rows - static_cast<int>((p_w[1] - y_min) * scale_factor);
        int c_img = static_cast<int>((p_w[0] - x_min) * scale_factor);
        if ((r_img >= 0 && r_img < top_down_image.rows) &&
            (c_img < top_down_image.cols && c_img > 0)) {
          top_down_image(r_img, c_img) = img(y_pixel, x_pixel);
        }
      }
    }
  }
}

/**
 * Project pixels in the img_cv into the top_down_image_gpu.
 * @param img_cv : Raw image.
 * @param seg_img_cv : Segmentation image.
 * @param KR_cv : K*R.
 * @param Kt_cv : K*t.
 * @param n_plane_w_cv : Normalized normal vector of the plane.
 * @param d_plane_w :
 * @param mask_factor : Indicate which area of the image is not to be projected.
 * @param scale_factor : Pixels per meter.
 * @param x_min : minimum x of the top-down image area in the world.
 * @param y_min : minimum y of the top-down image area in the world.
 * @param x_max : maximum x of the top-down image area in the world.
 * @param y_max : maximum y of the top-down image area in the world.
 * @param top_down_image_gpu : Result top-down image
 */
__host__ void projectPixelsCUDA(
    const cv::Mat& img_cv, const cv::Mat& seg_img_cv, const cv::Mat& KR_cv,
    const cv::Mat& Kt_cv, const cv::Mat& camera_position,
    const cv::Mat& head_vector, const cv::Mat& n_plane_w_cv,
    const double& d_plane_w, const double& mask_factor,
    const double& scale_factor, const double& x_min, const double& y_min,
    const double& x_max, const double& y_max,
    cv::cuda::GpuMat& top_down_image_gpu) {  //, cv::cuda::Stream &stream

  // Upload data to GPU
  cv::cuda::GpuMat img_gpu, seg_img_gpu, KR_gpu, Kt_gpu, n_plane_w_gpu;
  cv::cuda::GpuMat camera_position_gpu, head_vector_gpu;
  img_gpu.upload(img_cv);
  seg_img_gpu.upload(seg_img_cv);
  KR_gpu.upload(KR_cv);
  Kt_gpu.upload(Kt_cv);
  camera_position_gpu.upload(camera_position);
  head_vector_gpu.upload(head_vector);
  n_plane_w_gpu.upload(n_plane_w_cv);

  // Config GPU
  dim3 block_size(16, 16);
  dim3 grid_size(static_cast<unsigned int>(std::ceil(
      img_gpu.cols / static_cast<double>(block_size.x))),
                 static_cast<unsigned int>(std::ceil(
                     img_gpu.rows / static_cast<double>(block_size.y))));
  // Start kernel
  kernelCalculateTopDownPixel<<<grid_size, block_size>>>(
      img_gpu, seg_img_gpu, KR_gpu, Kt_gpu, camera_position_gpu,
      head_vector_gpu, n_plane_w_gpu, d_plane_w, mask_factor, scale_factor,
      x_min, y_min, x_max, y_max, top_down_image_gpu);
}

}
}