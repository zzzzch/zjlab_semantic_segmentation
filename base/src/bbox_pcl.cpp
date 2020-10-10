#include "bbox_pcl.h"

#include <bits/stl_algobase.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <cmath>
#include <iomanip>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

namespace zjlab {
    namespace pose {

        void CloudBbox::saveMainDirector() {
            if (pca_eigen_values_[0] >= pca_eigen_values_[1] &&
                pca_eigen_values_[0] >= pca_eigen_values_[2]) {
                maindirector_ = "X";
                main_eigenPCA = pca_eigen_vectors_.col(0);
            } else if (pca_eigen_values_[1] >= pca_eigen_values_[2]) {
                maindirector_ = "Y";
                main_eigenPCA = pca_eigen_vectors_.col(1);
            } else {
                maindirector_ = "Z";
                main_eigenPCA = pca_eigen_vectors_.col(2);
            }
            main_scalar_ = std::fmax(cube_whd_[0], std::fmax(cube_whd_[1], cube_whd_[2]));
        }

        std::vector<pcl::PointXYZ> CloudBbox::computeEightPolePoints() {
            Eigen::Vector3f Point_0 = cube_whd_(0) / 2.0 * pca_eigen_vectors_.col(0);
            Eigen::Vector3f Point_1 = cube_whd_(1) / 2.0 * pca_eigen_vectors_.col(1);
            Eigen::Vector3f Point_2 = cube_whd_(2) / 2.0 * pca_eigen_vectors_.col(2);

            Eigen::Vector3f point_0, point_1, point_2, point_3, point_4, point_5, point_6,
                    point_7;

            point_0 = Point_0 + Point_1 + Point_2 + pole_trans_;
            point_1 = -Point_0 + Point_1 + Point_2 + pole_trans_;
            point_2 = Point_0 - Point_1 + Point_2 + pole_trans_;
            point_3 = -Point_0 - Point_1 + Point_2 + pole_trans_;
            point_4 = Point_0 + Point_1 - Point_2 + pole_trans_;
            point_5 = -Point_0 + Point_1 - Point_2 + pole_trans_;
            point_6 = Point_0 - Point_1 - Point_2 + pole_trans_;
            point_7 = -Point_0 - Point_1 - Point_2 + pole_trans_;

            eight_points_.push_back(point_0);
            eight_points_.push_back(point_1);
            eight_points_.push_back(point_2);
            eight_points_.push_back(point_3);
            eight_points_.push_back(point_4);
            eight_points_.push_back(point_5);
            eight_points_.push_back(point_6);
            eight_points_.push_back(point_7);

            std::vector<pcl::PointXYZ> eight_points;
            for (const Eigen::Vector3f &p : eight_points_) {
                pcl::PointXYZ point;
                point.x = p(0);
                point.y = p(1);
                point.z = p(2);

                eight_points.push_back(point);
            }
            return eight_points;
        }

        void CloudBbox::setThreePCAPlanePoints() {
            pcax_.x = cube_whd_(0) / 2.0 * pca_eigen_vectors_(0, 0) + pole_trans_(0);
            pcax_.y = cube_whd_(0) / 2.0 * pca_eigen_vectors_(1, 0) + pole_trans_(1);
            pcax_.z = cube_whd_(0) / 2.0 * pca_eigen_vectors_(2, 0) + pole_trans_(2);

            pcay_.x = cube_whd_(1) / 2.0 * pca_eigen_vectors_(0, 1) + pole_trans_(0);
            pcay_.y = cube_whd_(1) / 2.0 * pca_eigen_vectors_(1, 1) + pole_trans_(1);
            pcay_.z = cube_whd_(1) / 2.0 * pca_eigen_vectors_(2, 1) + pole_trans_(2);

            pcaz_.x = cube_whd_(2) / 2.0 * pca_eigen_vectors_(0, 2) + pole_trans_(0);
            pcaz_.y = cube_whd_(2) / 2.0 * pca_eigen_vectors_(1, 2) + pole_trans_(1);
            pcaz_.z = cube_whd_(2) / 2.0 * pca_eigen_vectors_(2, 2) + pole_trans_(2);
        }

        template <typename PointT>
        bool computeBoundingBbox(
                const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                CloudBbox &pole_bbox, bool debug) {
            /// 7.1 compute PCA centrol point
            pcl::compute3DCentroid(*cloud, pole_bbox.pca_centroid_);
            /// 7.2 compute cavariance
            Eigen::Matrix3f covariance;
            pcl::computeCovarianceMatrixNormalized(*cloud, pole_bbox.pca_centroid_,
                                                   covariance);
            /// 7.3 compute eigen vector of cavariance
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(
                    covariance, Eigen::ComputeEigenvectors);
            pole_bbox.pca_eigen_vectors_ = eigen_solver.eigenvectors();
            pole_bbox.pca_eigen_values_ = eigen_solver.eigenvalues();

            pole_bbox.pca_eigen_vectors_.col(2) =
                    pole_bbox.pca_eigen_vectors_.col(0).cross(
                            pole_bbox.pca_eigen_vectors_.col(1));
            pole_bbox.pca_eigen_vectors_.col(0) =
                    pole_bbox.pca_eigen_vectors_.col(1).cross(
                            pole_bbox.pca_eigen_vectors_.col(2));
            pole_bbox.pca_eigen_vectors_.col(1) =
                    pole_bbox.pca_eigen_vectors_.col(2).cross(
                            pole_bbox.pca_eigen_vectors_.col(0));

            /// 7.4 construct translation Matrix
            Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f tm_inv = Eigen::Matrix4f::Identity();
            tm.topLeftCorner<3, 3>() = pole_bbox.pca_eigen_vectors_.transpose();  // R.
            tm.topRightCorner<3, 1>() = -1.0f *
                                        (pole_bbox.pca_eigen_vectors_.transpose()) *
                                        (pole_bbox.pca_centroid_.head<3>());  //  -R*t
            tm_inv = tm.inverse();

            /// 7.5 translate origin pointscloud into new coordination
            typename pcl::PointCloud<PointT>::Ptr transformedCloud(
                    new pcl::PointCloud<PointT>);
            pcl::transformPointCloud<PointT, float>(*cloud, *transformedCloud, tm);

            /// 7.6 Get the minimum and maximum values on each of the 3 (x-y-z) dimensions
            /// in a given pointcloud.
            PointT min_point, max_point;
            Eigen::Vector3f centrol_point;
            pcl::getMinMax3D<PointT>(*transformedCloud, min_point, max_point);
            pole_bbox.min_point_ << min_point.x, min_point.y, min_point.z;
            pole_bbox.max_point_ << max_point.x, max_point.y, max_point.z;
            centrol_point =
                    0.5f * (min_point.getVector3fMap() + max_point.getVector3fMap());

            /// 7.7 Get the params: pole_trans cube_whd pole_quaternion
            Eigen::Affine3f tm_inv_aff(tm_inv);
            pcl::transformPoint(centrol_point, pole_bbox.pole_trans_, tm_inv_aff);
            pole_bbox.cube_whd_ = max_point.getVector3fMap() - min_point.getVector3fMap();
            pole_bbox.pole_quaternion_ = tm_inv.topLeftCorner<3, 3>();
            float sc1 = std::fmax(pole_bbox.cube_whd_(0),
                                  fmax(pole_bbox.cube_whd_(1), pole_bbox.cube_whd_(2))) /
                        2.0;
            pole_bbox.saveMainDirector();

            /// 7.8 Compute three pca plane points for every bbox
            pole_bbox.setThreePCAPlanePoints();

            /// 7.9 Compute eight pole points for every bbox
            typename std::vector<pcl::PointXYZ> eight_points =
                    pole_bbox.computeEightPolePoints();
            return true;
        }

        template bool computeBoundingBbox<pcl::PointXYZ>(
                const typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr &cloud,
                CloudBbox &pole_bbox, bool debug);

        template bool computeBoundingBbox<pcl::PointXYZI>(
                const typename pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                CloudBbox &pole_bbox, bool debug);

        template bool computeBoundingBbox<pcl::PointXYZRGB>(
                const typename pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &cloud,
                CloudBbox &pole_bbox, bool debug);

    }
}