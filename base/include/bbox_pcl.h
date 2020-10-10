#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace zjlab {
    namespace pose {

        class CloudBbox {
        public:
            /// save pca main director
            void saveMainDirector();

            /// Calculate eight fixed points for bbox
            std::vector<pcl::PointXYZ> computeEightPolePoints();

            /// set three plane points from bbox cube
            void setThreePCAPlanePoints();

            Eigen::Vector3f getPosition() const { return pole_trans_; }

        public:
            /// PCA centroid point
            Eigen::Vector4f pca_centroid_;

            /// bbox pose
            Eigen::Quaternionf pole_quaternion_;  // pole bounding box rotation pose
            Eigen::Vector3f pole_trans_;  // pole bounding box centrol point position

            /// bounding box cube params
            Eigen::Vector3f cube_whd_;
            Eigen::Vector3f min_point_;
            Eigen::Vector3f max_point_;

            /// intersect point between line and plane
            Eigen::Vector3f point_intersect_;

            /// main eigen-direction
            std::string maindirector_;

            /// X Y Z  plane points from PCA direction
            pcl::PointXYZ pcax_;
            pcl::PointXYZ pcay_;
            pcl::PointXYZ pcaz_;

            /// eigen vector and eigen value
            Eigen::Matrix3f pca_eigen_vectors_;
            Eigen::Vector3f pca_eigen_values_;
            Eigen::Vector3f main_eigenPCA;

            double main_scalar_ = 0.0;

            Eigen::Vector3d localutmcenter_;

            Eigen::Vector3f top_point_;
            Eigen::Vector3f bottom_point_;

            std::vector<Eigen::Vector3f> eight_points_;
            std::vector<Eigen::Vector3f> four_points_;

            bool flag_ = false;

            bool has_road_ = false;
        };

        template <typename PointT>
        bool computeBoundingBbox(
                const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                CloudBbox &pole_bbox, bool debug = false);
    }
}