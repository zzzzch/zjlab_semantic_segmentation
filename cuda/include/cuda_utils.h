#pragma once

#include <cuda_runtime.h>
#include <opencv2/core/mat.hpp>
#define BLOCK_SIZE 8

namespace kd {
namespace mapping {

struct CudaFrameFeature {
  CudaFrameFeature() = default;

  size_t feature_id = 0;

  size_t frame_id = 0;

  int key_point_idx = -1;

  float pt2f_x = 0.0f;

  float pt2f_y = 0.0f;
};

static constexpr size_t MAX_TRACK_LENGTH = 20;

struct CudaFeatureTrack {
  CudaFeatureTrack() = default;

  explicit CudaFeatureTrack(size_t featureId) : feature_id(featureId) {}

  size_t feature_id = 0;

  size_t features_count = 0;

  float avg_reproj_err = 0.0f;

  double global_pt3d_x = 0.0;

  double global_pt3d_y = 0.0;

  double global_pt3d_z = 0.0;

  CudaFrameFeature features[MAX_TRACK_LENGTH];
};

/**
 * MeasureImage is used in class "top_down_viewer"
 */
struct MeasureImage {
  MeasureImage() = default;

  MeasureImage(cv::Mat& image, cv::Mat& R, cv::Mat& t, cv::Mat& camera_position,
               cv::Mat& head_vector, cv::Mat& n_plane_w, double d_plane_w,
               cv::Mat& seg_img)
      : image_(image),
        R_(R),
        t_(t),
        camera_position_(camera_position),
        head_vector_(head_vector),
        n_plane_w_(n_plane_w),
        d_plane_w_(d_plane_w),
        seg_img_(seg_img) {}

  cv::Mat image_;

  cv::Mat R_;

  cv::Mat t_;

  cv::Mat camera_position_;

  cv::Mat head_vector_;

  cv::Mat n_plane_w_;

  double d_plane_w_;

  cv::Mat seg_img_;
};

__host__ void GenerateRoadMask(const cv::Mat& seg, cv::cuda::GpuMat& mask,
                               cv::cuda::Stream& stream);

__host__ void inverse(double* L, int dim, double* iL);

/**
 *
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
    cv::cuda::GpuMat& top_down_image_gpu);
}
}