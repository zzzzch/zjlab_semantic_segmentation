#include "pcl_tools.h"
#include "voxel_grid_indices.h"

#include <glog/logging.h>
#include <unordered_set>
#include <vector>
// type
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// filter
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/morphological_filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>

// search
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>
#include <pcl/search/search.h>
// segment
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
// feature
#include <pcl/features/boundary.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/multiscale_feature_persistence.h>
// pca
#include <pcl/common/pca.h>
// project
#include <pcl/filters/project_inliers.h>


namespace zjlab {
    namespace pose {

        /// Voxel filter

/**
 * If PCL voxel grid fails with given leaf_size, will retry with leaf_size *=
 * leaf_sz_scale until success.
 * @tparam pointT
 * @param input_cloud
 * @param leaf_size
 * @param leaf_sz_scale
 * @return
 */
        template <typename pointT>
        typename std::vector<int> voxelFilterIndices(
                const typename pcl::PointCloud<pointT>::ConstPtr& input_cloud,
                const float leaf_size) {
            std::vector<int> indices;
//            pcl::VoxelGridIndices<pointT> sor;
//            sor.setInputCloud(input_cloud);
//            sor.setLeafSize(leaf_size, leaf_size, leaf_size);
//            sor.filter(indices);
//            // TODO move order to pcl?
//            std::sort(indices.begin(), indices.end());
            return indices;
        }

//#define PCL_INSTANTIATE_voxelFilterIndices(PointType)                  \
//  template PCL_EXPORTS std::vector<int> voxelFilterIndices<PointType>( \
//      const pcl::PointCloud<PointType>::ConstPtr&, const float);
//        PCL_INSTANTIATE(voxelFilterIndices, BASE_PCL_POINT_TYPES)

        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr processByVoxelFilter(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_cloud,
                const float leaf_size) {
            typename pcl::PointCloud<PointT>::Ptr output_cloud(
                    new pcl::PointCloud<PointT>);
            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            indices->indices = voxelFilterIndices<PointT>(input_cloud, leaf_size);
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(input_cloud);
            extract.setIndices(indices);
            extract.filter(*output_cloud);
            return output_cloud;
        }

#define PCL_INSTANTIATE_processByVoxelFilter(PointType)                        \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                         \
  processByVoxelFilter<PointType>(const pcl::PointCloud<PointType>::ConstPtr&, \
                                  const float);
        PCL_INSTANTIATE(processByVoxelFilter, BASE_PCL_POINT_TYPES)

/// Recursive Voxel filter
/**
 * Recursive voxel filter with target number of points,
 * will increase leaf_size until output cloud has less points than @points_num
 * @tparam PointT
 * @param input_cloud
 * @param points_num
 * @param leaf_size initial leaf_size
 * @param leaf_sz_scale leaf_size multiplier in each iteration
 * @return
 */
        template <typename PointT>
        typename std::vector<int> recursiveVoxelGridIndices(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_cloud,
                const int points_num, float leaf_size, float leaf_sz_scale) {
            std::vector<int> indices;

            if (input_cloud->size() > points_num) {
                do {
                    indices = voxelFilterIndices<PointT>(input_cloud, leaf_size);
                    leaf_size *= leaf_sz_scale;
                } while (indices.size() > points_num && leaf_size < 10.0f);
                if (leaf_size >= 10.0f)
                    LOG(WARNING) << "The use leaf_size is a little big(" << leaf_size
                                 << "), stopping iteration !";

                LOG(INFO) << "Recursive voxel filter with target nb points = " << points_num
                          << ", final number of points, " << indices.size()
                          << ", final leaf size: " << leaf_size;
                if (leaf_size > 0.15)
                    LOG(WARNING) << "The use leaf_size is a little big! --" << leaf_size;

            } else {
                LOG(INFO) << "Input cloud points count does not exceed given points_num ("
                          << points_num << "), returning all points indices";
                indices.reserve(input_cloud->points.size());
                for (size_t i = 0; i < input_cloud->points.size(); ++i) {
                    indices.emplace_back(i);
                }
            }
            return indices;
        }

#define PCL_INSTANTIATE_recursiveVoxelGridIndices(PointType)                  \
  template PCL_EXPORTS std::vector<int> recursiveVoxelGridIndices<PointType>( \
      const pcl::PointCloud<PointType>::ConstPtr&, const int, float, float);
        PCL_INSTANTIATE(recursiveVoxelGridIndices, BASE_PCL_POINT_TYPES)

/**
 * Recursive voxel filter with target number of points,
 * will increase leaf_size until output cloud has less points than @points_num
 * @tparam PointT
 * @param input_cloud
 * @param points_num
 * @param leaf_size initial leaf_size
 * @param leaf_sz_scale leaf_size multiplier in each iteration
 * @return
 */
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr recursiveVoxelFilter(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_cloud,
                const int points_num, float leaf_size, const float leaf_sz_scale) {
            typename pcl::PointCloud<PointT>::Ptr output_cloud(
                    new pcl::PointCloud<PointT>);
            pcl::PointIndices::Ptr indices(new pcl::PointIndices);
            indices->indices = recursiveVoxelGridIndices<PointT>(
                    input_cloud, points_num, leaf_size, leaf_sz_scale);
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(input_cloud);
            extract.setIndices(indices);
            extract.filter(*output_cloud);
            return output_cloud;
        }

#define PCL_INSTANTIATE_recursiveVoxelFilter(PointType)                        \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                         \
  recursiveVoxelFilter<PointType>(const pcl::PointCloud<PointType>::ConstPtr&, \
                                  const int, float, const float);
        PCL_INSTANTIATE(recursiveVoxelFilter, BASE_PCL_POINT_TYPES)

        template <typename pointT>
        typename pcl::PointCloud<pointT>::Ptr processByRecurseVoxelFilterInGrounRead(
                const typename pcl::PointCloud<pointT>::Ptr& input_cloud,
                const int points_num, const float leaf_size) {
            if (input_cloud->size() < points_num) return input_cloud;
            if (leaf_size > 0.15) return input_cloud;
            typename pcl::PointCloud<pointT>::Ptr output_cloud(
                    new pcl::PointCloud<pointT>);
            pcl::VoxelGrid<pointT> sor;
            sor.setInputCloud(input_cloud);
            sor.setLeafSize(leaf_size, leaf_size, leaf_size);
            sor.filter(*output_cloud);
            if (output_cloud->size() < points_num) {
                return output_cloud;
            }

            return processByRecurseVoxelFilterInGrounRead<pointT>(
                    output_cloud, points_num, leaf_size + 0.01);
        }

#define PCL_INSTANTIATE_processByRecurseVoxelFilterInGrounRead(PointType) \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                    \
  processByRecurseVoxelFilterInGrounRead<PointType>(                      \
      const pcl::PointCloud<PointType>::Ptr&, const int, const float);
        PCL_INSTANTIATE(processByRecurseVoxelFilterInGrounRead, BASE_PCL_POINT_TYPES)

/// pass filter
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr processByConditionFilter(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_cloud,
                const std::string& field, const double& min_val, const double& max_val) {
            typename pcl::PointCloud<PointT>::Ptr output_cloud(
                    new pcl::PointCloud<PointT>);
            if ("x" != field && "y" != field && "z" != field && "intensity" != field)
                return output_cloud;
            pcl::PassThrough<PointT> pass;
            pass.setInputCloud(input_cloud);
            pass.setFilterFieldName(field);
            pass.setFilterLimits(min_val, max_val);
            // pass.setFilterLimitsNegative (true);   //设置保留范围内还是过滤掉范围内
            pass.filter(*output_cloud);
            return output_cloud;
        }

#define PCL_INSTANTIATE_processByConditionFilter(PointType)            \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                 \
  processByConditionFilter<PointType>(                                 \
      const pcl::PointCloud<PointType>::ConstPtr&, const std::string&, \
      const double&, const double&);
        PCL_INSTANTIATE(processByConditionFilter, BASE_PCL_POINT_TYPES)

/// statistic filter
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr processStatisticalOutlierRemoval(
                const typename pcl::PointCloud<PointT>::ConstPtr& in_cloud, size_t meank,
                double devmulth) {
            typename pcl::PointCloud<PointT>::Ptr output_cloud(
                    new pcl::PointCloud<PointT>);
            pcl::StatisticalOutlierRemoval<PointT> Statistical;
            Statistical.setInputCloud(in_cloud);
            Statistical.setMeanK(meank);
            Statistical.setStddevMulThresh(devmulth);
            Statistical.filter(*output_cloud);
            return output_cloud;
        }

#define PCL_INSTANTIATE_processStatisticalOutlierRemoval(PointType) \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr              \
  processStatisticalOutlierRemoval<PointType>(                      \
      const pcl::PointCloud<PointType>::ConstPtr&, size_t, double);
        PCL_INSTANTIATE(processStatisticalOutlierRemoval, BASE_PCL_POINT_TYPES)

/// radius filter
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr radiusOutlierRemoval(
                const typename pcl::PointCloud<PointT>::Ptr& cloud, double radius,
                int min_neighbors) {
            typename pcl::PointCloud<PointT>::Ptr output_cloud(
                    new pcl::PointCloud<PointT>);
            pcl::RadiusOutlierRemoval<PointT> ror;
            ror.setInputCloud(cloud);
            ror.setRadiusSearch(radius);
            ror.setMinNeighborsInRadius(min_neighbors);
            ror.filter(*output_cloud);
            return output_cloud;
        }

#define PCL_INSTANTIATE_radiusOutlierRemoval(PointType)                   \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                    \
  radiusOutlierRemoval<PointType>(const pcl::PointCloud<PointType>::Ptr&, \
                                  double, int);
        PCL_INSTANTIATE(radiusOutlierRemoval, BASE_PCL_POINT_TYPES)

        template <typename pointT>
        typename pcl::search::KdTree<pointT>::Ptr kdtree(
                const typename pcl::PointCloud<pointT>::Ptr& cloud, float epsilon) {
            typename pcl::search::KdTree<pointT>::Ptr kdtree(
                    new pcl::search::KdTree<pointT>);
            kdtree->setEpsilon(epsilon);
            kdtree->setInputCloud(cloud);
            return kdtree;
        }

#define PCL_INSTANTIATE_kdtree(PointType)                                     \
  template PCL_EXPORTS pcl::search::KdTree<PointType>::Ptr kdtree<PointType>( \
      const pcl::PointCloud<PointType>::Ptr&, float);
        PCL_INSTANTIATE(kdtree, BASE_PCL_POINT_TYPES)

/// build octree
        template <typename PointT>
        typename pcl::search::Octree<PointT>::Ptr buildOctree(
                const typename pcl::PointCloud<PointT>::ConstPtr& cloud, float resolution,
                const pcl::PointIndices::ConstPtr& indices) {
            typename pcl::search::Octree<PointT>::Ptr octree(
                    new pcl::search::Octree<PointT>(resolution));
            if (indices == nullptr)
                octree->setInputCloud(cloud);
            else {
                auto indices_ =
                        boost::make_shared<const std::vector<int>>(indices->indices);
                octree->setInputCloud(cloud, indices_);
            }
            return octree;
        }

#define PCL_INSTANTIATE_buildOctree(PointType)                               \
  template PCL_EXPORTS pcl::search::Octree<PointType>::Ptr                   \
  buildOctree<PointType>(const pcl::PointCloud<PointType>::ConstPtr&, float, \
                         const pcl::PointIndices::ConstPtr&);
        PCL_INSTANTIATE(buildOctree, BASE_PCL_POINT_TYPES)

/// compute pointscloud normal
        template <typename PointT>
        pcl::PointCloud<pcl::Normal>::Ptr computeNormal(
                const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                typename pcl::search::Octree<PointT>::Ptr& octree, int K, double radius) {
            pcl::NormalEstimation<PointT, pcl::Normal> ne;
            ne.setInputCloud(cloud);
            ne.setSearchMethod(octree);
            ne.setViewPoint(std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max());

            if (K > 0)
                ne.setKSearch(K);
            else if (radius > 0.0)
                ne.setRadiusSearch(radius);
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
            ne.compute(*normals);
            return normals;
        }

#define PCL_INSTANTIATE_computeNormal(PointType)                        \
  template PCL_EXPORTS pcl::PointCloud<pcl::Normal>::Ptr                \
  computeNormal<PointType>(const pcl::PointCloud<PointType>::ConstPtr&, \
                           pcl::search::Octree<PointType>::Ptr&, int, double);
        PCL_INSTANTIATE(computeNormal, BASE_PCL_POINT_TYPES)

        template <typename PointT>
        void computeNormalInPoint(
                const typename pcl::PointCloud<PointT>::Ptr& cloud,
                const typename pcl::search::Octree<PointT>::Ptr& octree, int K,
                double radius) {
            pcl::NormalEstimation<PointT, PointT> ne;
            ne.setInputCloud(cloud);
            ne.setSearchMethod(octree);
            ne.setViewPoint(std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max());
            if (K > 0)
                ne.setKSearch(K);
            else if (radius > 0.0)
                ne.setRadiusSearch(radius);
            ne.compute(*cloud);
        }

#define PCL_INSTANTIATE_computeNormalInPoint(PointType)      \
  template PCL_EXPORTS void computeNormalInPoint<PointType>( \
      const pcl::PointCloud<PointType>::Ptr&,                \
      const pcl::search::Octree<PointType>::Ptr&, int, double);
        PCL_INSTANTIATE(computeNormalInPoint, BASE_PCL_NORMAL_POINT_TYPES)

// add by xiaoqi [07/02/2019]
/// compute pointscloud shape class
        template <typename PointT>
        void computePCA(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                        std::vector<PointClassType>& point_class,
                        std::vector<float>& point_verticality, double radius) {
            if (cloud->size() == 0) return;

            /// find the min_x and min_y
            double min_x = std::numeric_limits<double>::max();
            double min_y = std::numeric_limits<double>::max();
            typename pcl::PointCloud<PointT>::Ptr local_cloud(
                    new pcl::PointCloud<PointT>);
            for (size_t i = 0; i != cloud->size(); ++i) {
                if (cloud->points[i].x < min_x) min_x = cloud->points[i].x;
                if (cloud->points[i].y < min_y) min_y = cloud->points[i].y;
            }
            /// Prevent 0;
            min_x = min_x - 1;
            min_y = min_y - 1;
            /// Convert to local coordinate system
            for (size_t i = 0; i != cloud->size(); ++i) {
                PointT pt;
                pt.x = cloud->points[i].x - min_x;
                pt.y = cloud->points[i].y - min_y;
                pt.z = cloud->points[i].z;
                local_cloud->push_back(pt);
            }

            pcl::KdTreeFLANN<PointT> kdtree;
            kdtree.setInputCloud(local_cloud);
            for (int i = 0; i != local_cloud->size(); ++i) {
                std::vector<int> point_idx;
                std::vector<float> point_dis;
                if (kdtree.radiusSearch(i, radius, point_idx, point_dis) > 3) {
                    typename pcl::PointCloud<PointT>::Ptr neighboring_cloud(
                            new pcl::PointCloud<PointT>);
                    for (int j = 0; j != point_idx.size(); j++)
                        neighboring_cloud->points.push_back(local_cloud->points[point_idx[j]]);
                    // compute the PCA for each point
                    pcl::PCA<PointT> PCAer;
                    PCAer.setInputCloud(neighboring_cloud);
                    // extract the eigenvalues from Eigen-decomposition of the covariance
                    // matrix
                    Eigen::Vector3f eigen_values = PCAer.getEigenValues();
                    // compute shape of neighboring geometry
                    eigen_values[0] = sqrt(eigen_values[0]);
                    eigen_values[1] = sqrt(eigen_values[1]);
                    eigen_values[2] = sqrt(eigen_values[2]);

                    ///--- compute the verticality feature---
                    Eigen::Matrix3f eigen_vectors = PCAer.getEigenVectors();
                    std::vector<float> v1 = {eigen_vectors(0, 0), eigen_vectors(0, 1),
                                             eigen_vectors(0, 2)};
                    std::vector<float> v2 = {eigen_vectors(1, 0), eigen_vectors(1, 1),
                                             eigen_vectors(1, 2)};
                    std::vector<float> v3 = {eigen_vectors(2, 0), eigen_vectors(2, 1),
                                             eigen_vectors(2, 2)};
                    std::vector<float> unary_vector = {
                            eigen_values[0] * fabsf(v1[0]) + eigen_values[1] * fabsf(v2[0]) +
                            eigen_values[2] * fabsf(v3[0]),
                            eigen_values[0] * fabsf(v1[1]) + eigen_values[1] * fabsf(v2[1]) +
                            eigen_values[2] * fabsf(v3[1]),
                            eigen_values[0] * fabsf(v1[2]) + eigen_values[1] * fabsf(v2[2]) +
                            eigen_values[2] * fabsf(v3[2])};

                    float norm = sqrt(unary_vector[0] * unary_vector[0] +
                                      unary_vector[1] * unary_vector[1] +
                                      unary_vector[2] * unary_vector[2]);

                    float verticality = 0.0;
                    if (fabsf(norm - 0.0) > 0.000001) verticality = unary_vector[2] / norm;

                    point_verticality[i] = verticality;

                    ///---- compute the shape classification ----
                    float denominator = eigen_values[0] + eigen_values[1] + eigen_values[2];
                    double alpha1 = (eigen_values[0] - eigen_values[1]) / denominator;
                    double alpha2 = (eigen_values[1] - eigen_values[2]) / denominator;
                    double alpha3 = eigen_values[2] / denominator;

                    if (alpha1 > alpha2 && alpha1 > alpha3)
                        point_class[i] = PointClassType::LINE;
                    else if (alpha2 > alpha3 && alpha2 > alpha1) {
                        point_class[i] = PointClassType::PLANE;
                    } else
                        point_class[i] = PointClassType::SPHERE;
                } else {
                    point_class[i] = PointClassType::OTHER;
                    point_verticality[i] = 0.0;
                }
            }
        }

#define PCL_INSTANTIATE_computePCA(PointType)                               \
  template PCL_EXPORTS void computePCA<PointType>(                          \
      const pcl::PointCloud<PointType>::Ptr&, std::vector<PointClassType>&, \
      std::vector<float>&, double);
        PCL_INSTANTIATE(computePCA, BASE_PCL_POINT_TYPES)

/// When the coordinate value is large, the general integer part exceeds three
/// digits.
/// Please use this method.
        template <typename PointT>
        void computeNormalTLocal(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                                 pcl::PointCloud<pcl::Normal>::Ptr& normals, int K,
                                 double radius) {
            /// find the min_x and min_y
            double min_x = std::numeric_limits<double>::max();
            double min_y = std::numeric_limits<double>::max();
            typename pcl::PointCloud<PointT>::Ptr local_cloud(
                    new pcl::PointCloud<PointT>);
            for (size_t i = 0; i != cloud->size(); ++i) {
                if (cloud->points[i].x < min_x) min_x = cloud->points[i].x;
                if (cloud->points[i].y < min_y) min_y = cloud->points[i].y;
            }
            /// Prevent 0;
            min_x = min_x - 1;
            min_y = min_y - 1;
            /// Convert to local coordinate system
            for (size_t i = 0; i != cloud->size(); ++i) {
                PointT pt;
                pt.x = cloud->points[i].x - min_x;
                pt.y = cloud->points[i].y - min_y;
                pt.z = cloud->points[i].z;
                local_cloud->push_back(pt);
            }
            typename pcl::search::KdTree<PointT>::Ptr tree(
                    new pcl::search::KdTree<PointT>());
            pcl::NormalEstimationOMP<PointT, pcl::Normal> ne(1);
            ne.setInputCloud(local_cloud);
            ne.setSearchMethod(tree);
            ne.setViewPoint(std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max());

            if (radius != 0.0) ne.setRadiusSearch(radius);
            if (K != 0) ne.setKSearch(K);
            ne.compute(*normals);
        }

#define PCL_INSTANTIATE_computeNormalTLocal(PointType)      \
  template PCL_EXPORTS void computeNormalTLocal<PointType>( \
      const pcl::PointCloud<PointType>::Ptr&,               \
      pcl::PointCloud<pcl::Normal>::Ptr&, int, double);
        PCL_INSTANTIATE(computeNormalTLocal, BASE_PCL_POINT_TYPES)

        template <typename PointT>
        void computeNormalT(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                            const typename pcl::search::Octree<PointT>::Ptr& octree,
                            pcl::PointCloud<pcl::Normal>::Ptr& normals, int K,
                            double radius) {
            pcl::NormalEstimationOMP<PointT, pcl::Normal> ne(1);
            ne.setInputCloud(cloud);
            if (octree != nullptr) ne.setSearchMethod(octree);
            ne.setViewPoint(std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max(),
                            std::numeric_limits<float>::max());

            if (radius > 0.0) ne.setRadiusSearch(radius);
            if (K != 0) ne.setKSearch(K);
            ne.compute(*normals);
        }

#define PCL_INSTANTIATE_computeNormalT(PointType)      \
  template PCL_EXPORTS void computeNormalT<PointType>( \
      const pcl::PointCloud<PointType>::Ptr&,          \
      const pcl::search::Octree<PointType>::Ptr&,      \
      pcl::PointCloud<pcl::Normal>::Ptr&, int, double);
        PCL_INSTANTIATE(computeNormalT, BASE_PCL_POINT_TYPES)

/// euclidean cluster
        template <typename PointT>
        std::vector<pcl::PointIndices> euclideanClustertool(
                const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                const typename pcl::search::Octree<PointT>::Ptr& octree,
                const double cluster_tolerance, const int min_cluster_size,
                const int max_cluster_size, const pcl::PointIndices::ConstPtr& indices) {
            std::vector<pcl::PointIndices> clusters;
            pcl::EuclideanClusterExtraction<PointT> ec;
            ec.setClusterTolerance(cluster_tolerance);
            ec.setMinClusterSize(min_cluster_size);
            ec.setMaxClusterSize(max_cluster_size);
            ec.setSearchMethod(octree);
            ec.setInputCloud(cloud);
            if (indices != nullptr) ec.setIndices(indices);
            ec.extract(clusters);
            return clusters;
        }

#define PCL_INSTANTIATE_euclideanClustertool(PointType)                        \
  template PCL_EXPORTS std::vector<pcl::PointIndices>                          \
  euclideanClustertool<PointType>(const pcl::PointCloud<PointType>::ConstPtr&, \
                                  const pcl::search::Octree<PointType>::Ptr&,  \
                                  const double, const int, const int,          \
                                  const pcl::PointIndices::ConstPtr&);
        PCL_INSTANTIATE(euclideanClustertool, BASE_PCL_POINT_TYPES)

        template <typename PointT>
        float judgePointSide(PointT& head_point, PointT& tail_point,
                             PointT& query_point) {
            float direction = 0.0;
            /// Construct vectors according to three points
            Eigen::Vector3f start_query(head_point.x - query_point.x,
                                        head_point.y - query_point.y, 0.0);
            Eigen::Vector3f end_query(tail_point.x - query_point.x,
                                      tail_point.y - query_point.y, 0.0);
            /// Judging the left and right of a point according to the direction of the
            /// vector
            Eigen::Vector3f cross_vector = start_query.cross(end_query);
            // -1  is right & 1 is left
            if (cross_vector(2) < 0)
                direction = -1.0;
            else if (cross_vector(2) > 0)
                direction = 1.0;
            return direction;
        }

#define PCL_INSTANTIATE_judgePointSide(PointType)                              \
  template PCL_EXPORTS float judgePointSide<PointType>(PointType&, PointType&, \
                                                       PointType&);
        PCL_INSTANTIATE(judgePointSide, BASE_PCL_POINT_TYPES)

/// sac segmentation
        template <typename PointT>
        void sacSegmentationtool(
                const typename pcl::PointCloud<PointT>::ConstPtr& in_cloud,
                const pcl::SacModel& mode, const int& method, const int& maxiteration,
                const double& distance, pcl::PointIndices::Ptr& inliers,
                pcl::ModelCoefficients::Ptr& coefficients, const bool flagoptimazation) {
            pcl::SACSegmentation<PointT> seg;
            seg.setOptimizeCoefficients(flagoptimazation);
            seg.setModelType(mode);
            seg.setMethodType(method);
            //  seg.setOptima
            seg.setMaxIterations(maxiteration);
            seg.setDistanceThreshold(distance);
            seg.setInputCloud(in_cloud);
            seg.segment(*inliers, *coefficients);
        }

#define PCL_INSTANTIATE_sacSegmentationtool(PointType)                   \
  template PCL_EXPORTS void sacSegmentationtool<PointType>(              \
      const pcl::PointCloud<PointType>::ConstPtr&, const pcl::SacModel&, \
      const int&, const int&, const double&, pcl::PointIndices::Ptr&,    \
      pcl::ModelCoefficients::Ptr&, const bool);
        PCL_INSTANTIATE(sacSegmentationtool, BASE_PCL_POINT_TYPES)

/**
 * @brief Need to use data type with Normal estimated from pcl::NormalEstimation
 *        While adding normals, wrong choice of radius will result in "nan" on
 *        many points.
 * @tparam PointTNormal points with member "curvature"
 * @param cloud
 * @param curvature_threshold : curvature of normal plan can be around 0.002
 */
        template <typename PointTNormal>
        void curvatureFilter(const typename pcl::PointCloud<PointTNormal>::Ptr& cloud,
                             double curvature_threshold) {
            cloud->points.erase(std::remove_if(cloud->points.begin(), cloud->points.end(),
                                               [&](const PointTNormal& point) {
                                                   return point.curvature >
                                                          curvature_threshold ||
                                                          std::isnan(point.curvature);
                                               }),
                                cloud->points.end());
            if (cloud->width != 0) {
                cloud->width = 0;
                cloud->height = 0;
            }
        }

#define PCL_INSTANTIATE_curvatureFilter(PointType)      \
  template PCL_EXPORTS void curvatureFilter<PointType>( \
      const pcl::PointCloud<PointType>::Ptr&, double);
        PCL_INSTANTIATE(curvatureFilter, BASE_PCL_NORMAL_POINT_TYPES)

/**
 * get clouds by given indices in original cloud
 * @tparam pointT
 * @param ori_cloud
 * @param indices
 * @return
 */
        template <typename pointT>
        typename pcl::PointCloud<pointT>::Ptr getChildCloudByIndicesFromOriginal(
                const typename pcl::PointCloud<pointT>::Ptr& ori_cloud,
                const pcl::PointIndices& indices) {
            typename pcl::PointCloud<pointT>::Ptr output_cloud(
                    new typename pcl::PointCloud<pointT>);
            pcl::ExtractIndices<pointT> extract;
            extract.setInputCloud(ori_cloud);
            extract.setIndices(boost::make_shared<pcl::PointIndices const>(indices));
            extract.setNegative(false);
            extract.filter(*output_cloud);
            //  for (const auto& idx : indices.indices) {
            //    output_cloud->push_back(ori_cloud->at(idx));
            //  }
            return output_cloud;
        }

#define PCL_INSTANTIATE_getChildCloudByIndicesFromOriginal(PointType) \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                \
  getChildCloudByIndicesFromOriginal<PointType>(                      \
      const pcl::PointCloud<PointType>::Ptr&, const pcl::PointIndices&);
        PCL_INSTANTIATE(getChildCloudByIndicesFromOriginal, BASE_PCL_POINT_TYPES)

/**
 * project clouds to given coeff plane
 * @tparam pointT
 * @param ori_cloud
 * @param coeff
 * @return
 */
        template <typename pointT>
        typename pcl::PointCloud<pointT>::Ptr getProjectCloudByCoeff(
                const typename pcl::PointCloud<pointT>::Ptr& ori_cloud,
                const pcl::ModelCoefficients::Ptr& coeff) {
            typename pcl::PointCloud<pointT>::Ptr output_cloud(
                    new typename pcl::PointCloud<pointT>);
            pcl::ProjectInliers<pointT> proj;
            proj.setModelType(pcl::SACMODEL_PLANE);
            proj.setInputCloud(ori_cloud);
            proj.setModelCoefficients(coeff);
            proj.filter(*output_cloud);
            return output_cloud;
        }

#define PCL_INSTANTIATE_getProjectCloudByCoeff(PointType)                   \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                      \
  getProjectCloudByCoeff<PointType>(const pcl::PointCloud<PointType>::Ptr&, \
                                    const pcl::ModelCoefficients::Ptr&);
        PCL_INSTANTIATE(getProjectCloudByCoeff, BASE_PCL_POINT_TYPES)

/**
 * get clouds by given indices in original cloud
 * @tparam pointT
 * @param ori_cloud
 * @param indices
 * @return
 */
        template <typename pointT>
        typename pcl::PointCloud<pointT>::Ptr getChildCloudByIndicesFromOriginal2(
                const typename pcl::PointCloud<pointT>::Ptr& ori_cloud,
                const std::vector<int>& indices) {
            //  typename pcl::PointCloud<pointT>::Ptr output_cloud(
            //      new typename pcl::PointCloud<pointT>);
            pcl::PointIndices pcl_indices;
            pcl_indices.indices.assign(indices.begin(), indices.end());
            //  pcl::ExtractIndices<pointT> extract;
            //  extract.setInputCloud(ori_cloud);
            //  extract.setIndices(pcl_indices);
            //  extract.setNegative(false);
            //  extract.filter(*output_cloud);
            //  output_cloud = getChildCloudByIndicesFromOriginal<pointT>(ori_cloud,
            //  pcl_indices);
            //  for (const auto& idx : indices) {
            //    output_cloud->push_back(ori_cloud->at(idx));
            //  }
            return getChildCloudByIndicesFromOriginal<pointT>(ori_cloud, pcl_indices);
        }

#define PCL_INSTANTIATE_getChildCloudByIndicesFromOriginal2(PointType) \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr                 \
  getChildCloudByIndicesFromOriginal2<PointType>(                      \
      const pcl::PointCloud<PointType>::Ptr&, const std::vector<int>&);
        PCL_INSTANTIATE(getChildCloudByIndicesFromOriginal2, BASE_PCL_POINT_TYPES)

/**
 * get the clouds from original cloud by given indices for voxel cloud
 *@param ori_cloud: origin point cloud
 * @param voxel_point_indexes: each voxel with the point indexes in origin cloud
 * @param indices
 */
        template <typename pointT>
        typename pcl::PointCloud<pointT>::Ptr getChildCloudByIndicesFromVoxel(
                const typename pcl::PointCloud<pointT>::Ptr& ori_cloud,
                const std::vector<std::vector<int>>& voxel_point_indexes,
                const pcl::PointIndices& indices) {
            pcl::PointIndices pcl_indices;
            for (size_t i = 0; i != indices.indices.size(); ++i) {
                std::vector<int> pt_indice;
                pt_indice = voxel_point_indexes[indices.indices[i]];
                pcl_indices.indices.insert(pcl_indices.indices.end(), pt_indice.begin(),
                                           pt_indice.end());
            }
            return getChildCloudByIndicesFromOriginal<pointT>(ori_cloud, pcl_indices);
        }

#define PCL_INSTANTIATE_getChildCloudByIndicesFromVoxel(PointType) \
  template PCL_EXPORTS pcl::PointCloud<PointType>::Ptr             \
  getChildCloudByIndicesFromVoxel<PointType>(                      \
      const pcl::PointCloud<PointType>::Ptr&,                      \
      const std::vector<std::vector<int>>&, const pcl::PointIndices&);
        PCL_INSTANTIATE(getChildCloudByIndicesFromVoxel, BASE_PCL_POINT_TYPES)

/**
 * get the index in original clouds from voxel cloud
 * @param voxel_point_indexes: each voxel with the point indexes in origin cloud
 * @param indices
 */
        std::vector<pcl::PointIndices> getIndicesInOriCloudFromVoxel(
                const std::vector<std::vector<int>>& voxel_point_indexes,
                const std::vector<pcl::PointIndices>& indices) {
            std::vector<pcl::PointIndices> output_clusters;
            for (size_t i = 0; i != indices.size(); ++i) {
                pcl::PointIndices pcl_indices;
                for (size_t j = 0; j != indices[i].indices.size(); ++j) {
                    std::vector<int> pt_indice;
                    pt_indice = voxel_point_indexes[indices[i].indices[j]];
                    pcl_indices.indices.insert(pcl_indices.indices.end(), pt_indice.begin(),
                                               pt_indice.end());
                }
                output_clusters.push_back(pcl_indices);
            }
            return output_clusters;
        }

        template <typename pointT>
        void keypointsEstimation(
                const typename pcl::PointCloud<pointT>::ConstPtr& cloud,
                pcl::PointCloud<pcl::Normal>::ConstPtr& normal) {
            typename pcl::FPFHEstimation<pointT, pcl::Normal, pcl::FPFHSignature33>::Ptr
                    fest(
                    new pcl::FPFHEstimation<pointT, pcl::Normal, pcl::FPFHSignature33>());
            fest->setInputCloud(cloud);
            fest->setInputNormals(normal);

            typename pcl::MultiscaleFeaturePersistence<pointT, pcl::FPFHSignature33> fper;
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr features;
            std::vector<int> keypoints;
            std::vector<float> scale_values = {0.5f, 1.0f, 1.5f};
            fper.setScalesVector(scale_values);
            fper.setAlpha(1.3f);
            fper.setFeatureEstimator(fest);
            fper.setDistanceMetric(pcl::CS);
            fper.determinePersistentFeatures(*features, keypoints);
        }

//#define PCL_INSTANTIATE_keypointsEstimation(PointType) \
//  template PCL_EXPORTS void keypointsEstimation<PointType>( \
//      const pcl::PointCloud<PointType>::ConstPtr&, pcl::PointCloud<pcl::Normal>::ConstPtr&);
// PCL_INSTANTIATE(keypointsEstimation, BASE_PCL_POINT_TYPES)

        template <typename PointTNormal>
        void addNormalsToCloud(
                const typename pcl::PointCloud<PointTNormal>::Ptr cloud,
                const typename pcl::search::KdTree<PointTNormal>::Ptr& in_kdtree,
                double radius, int K) {
            pcl::NormalEstimation<PointTNormal, PointTNormal> normal_estimation;
            normal_estimation.setInputCloud(cloud);
            normal_estimation.setViewPoint(std::numeric_limits<float>::max(),
                                           std::numeric_limits<float>::max(),
                                           std::numeric_limits<float>::max());
            if (in_kdtree != nullptr) normal_estimation.setSearchMethod(in_kdtree);
            if (radius > 0.0) normal_estimation.setRadiusSearch(radius);
            if (K != 0) normal_estimation.setKSearch(K);

            normal_estimation.compute(*cloud);
        }

#define PCL_INSTANTIATE_addNormalsToCloud(PointType)      \
  template PCL_EXPORTS void addNormalsToCloud<PointType>( \
      const pcl::PointCloud<PointType>::Ptr,              \
      const pcl::search::KdTree<PointType>::Ptr&, double, int);
        PCL_INSTANTIATE(addNormalsToCloud, BASE_PCL_NORMAL_POINT_TYPES)

/**
 * @brief Use Pcl build-in multithread method, add normal with muiltithreads
 */
        template <typename PointTNormal>
        void addNormalsToCloudOMP(
                const typename pcl::PointCloud<PointTNormal>::Ptr cloud,
                const typename pcl::search::KdTree<PointTNormal>::Ptr& in_kdtree,
                int num_threads, double radius, int K) {
            pcl::NormalEstimationOMP<PointTNormal, PointTNormal> normal_estimation(
                    num_threads);
            normal_estimation.setInputCloud(cloud);
            normal_estimation.setViewPoint(std::numeric_limits<float>::max(),
                                           std::numeric_limits<float>::max(),
                                           std::numeric_limits<float>::max());
            if (in_kdtree != nullptr) normal_estimation.setSearchMethod(in_kdtree);
            if (radius > 0.0) normal_estimation.setRadiusSearch(radius);
            if (K != 0) normal_estimation.setKSearch(K);

            normal_estimation.compute(*cloud);
        }

#define PCL_INSTANTIATE_addNormalsToCloudOMP(PointType)      \
  template PCL_EXPORTS void addNormalsToCloudOMP<PointType>( \
      const pcl::PointCloud<PointType>::Ptr,                 \
      const pcl::search::KdTree<PointType>::Ptr&, int, double, int);
        PCL_INSTANTIATE(addNormalsToCloudOMP, BASE_PCL_NORMAL_POINT_TYPES)

/**
 *
 * @tparam PointT
 * @param in_cloud
 * @param win_sz size of window to apply morphological operator
 * @note a larger window size be able to remove larger objects, but also much
 * slower execution
 * @param height_thres the approximate ground depth
 * @return
 */
        template <typename PointT>
        pcl::PointIndicesPtr removeObjectsAboveGround(
                typename pcl::PointCloud<PointT>::ConstPtr in_cloud, float win_sz,
                float height_thres) {
            pcl::PointIndicesPtr indices(new pcl::PointIndices);

            typename pcl::PointCloud<PointT>::Ptr cloud_filtered(
                    new pcl::PointCloud<PointT>);

            pcl::applyMorphologicalOperator<PointT>(in_cloud, win_sz, pcl::MORPH_OPEN,
                                                    *cloud_filtered);

            std::vector<int> pt_indices;
            for (size_t p_idx = 0; p_idx < in_cloud->points.size(); ++p_idx) {
                float diff = in_cloud->points[p_idx].z - cloud_filtered->points[p_idx].z;
                if (std::fabs(diff) < height_thres) indices->indices.push_back(p_idx);
            }
            return indices;
        }

#define PCL_INSTANTIATE_removeObjectsAboveGround(PointType)                 \
  template PCL_EXPORTS pcl::PointIndicesPtr                                 \
  removeObjectsAboveGround<PointType>(pcl::PointCloud<PointType>::ConstPtr, \
                                      float, float);
        PCL_INSTANTIATE(removeObjectsAboveGround, BASE_PCL_POINT_TYPES)

/**
 *voxelization the cloud with the cloud points index
 * @param cloud: the input cloud
 * @param center_cloud: the voxel cloud points
 * @param center_point_idx: each voxel with the point indexes
 * @param voxel_length: voxel length;
 *
 */
        template <typename PointT>
        void voxelizationWithIdx(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                                 std::vector<std::vector<int>>& center_point_idx,
                                 typename pcl::PointCloud<PointT>::Ptr& center_cloud,
                                 float voxel_length) {
            if (!cloud->size() || cloud->empty()) return;
            pcl::PointXYZ max_pt;
            max_pt.x = -std::numeric_limits<double>::max();
            max_pt.y = -std::numeric_limits<double>::max();
            max_pt.z = -std::numeric_limits<double>::max();
            pcl::PointXYZ min_pt;
            min_pt.x = std::numeric_limits<double>::max();
            min_pt.y = std::numeric_limits<double>::max();
            min_pt.z = std::numeric_limits<double>::max();
            //
            for (int i = 0; i != cloud->points.size(); ++i) {
                if (cloud->points[i].x > max_pt.x) max_pt.x = cloud->points[i].x;
                if (cloud->points[i].x < min_pt.x) min_pt.x = cloud->points[i].x;
                if (cloud->points[i].y > max_pt.y) max_pt.y = cloud->points[i].y;
                if (cloud->points[i].y < min_pt.y) min_pt.y = cloud->points[i].y;
                if (cloud->points[i].z > max_pt.z) max_pt.z = cloud->points[i].z;
                if (cloud->points[i].z < min_pt.z) min_pt.z = cloud->points[i].z;
            }

            float x_length = max_pt.x - min_pt.x;
            float y_length = max_pt.y - min_pt.y;
            uint32_t x_voxel_num = ceil(x_length / voxel_length) + 1;
            uint32_t y_voxel_num = ceil(y_length / voxel_length) + 1;

            /// save the points indexes in each voxel
            //  std::vector<std::vector<int>> voxels_point_idx;
            //  std::vector<int> one_voxels;

            //  voxels_point_idx.resize(x_voxel_num * y_voxel_num);

            std::map<int, std::vector<int>> voxels_point_idx;
            //  PointT one_pt;
            //  for (size_t i = 0; i < x_voxel_num * y_voxel_num; ++i) {
            //    voxels_point_idx.push_back(one_voxels);
            //    voxels_centers.push_back(one_pt);
            //  }
            /// compute each voxel indexes
            for (size_t i = 0; i != cloud->points.size(); ++i) {
                int x_num = uint32_t((cloud->points[i].x - min_pt.x) / voxel_length);
                int y_num = uint32_t((cloud->points[i].y - min_pt.y) / voxel_length);
                voxels_point_idx[y_num * x_voxel_num + x_num].push_back(i);
            }
            /// for each voxel
            for (auto voxels = voxels_point_idx.begin(); voxels != voxels_point_idx.end();
                 ++voxels) {
                float maxz_voxle = -std::numeric_limits<double>::max();
                if (!voxels->second.size()) continue;
                for (size_t j = 0; j != voxels->second.size(); ++j) {
                    if (cloud->points[voxels->second[j]].z > maxz_voxle)
                        maxz_voxle = cloud->points[voxels->second[j]].z;
                }

                //  }
                //  for (size_t i = 0; i != voxels_point_idx.size(); ++i) {
                /// find the maximum of elevation value in each voxel
                //    float maxz_voxle = -std::numeric_limits<double>::max();
                //    if (!voxels_point_idx[i].size()) continue;
                //    for (size_t j = 0; j != voxels_point_idx[i].size(); ++j) {
                //      if (cloud->points[voxels_point_idx[i][j]].z > maxz_voxle)
                //        maxz_voxle = cloud->points[voxels_point_idx[i][j]].z;
                //    }
                /// compute in the Z, how many voxels there should be have
                float z_length = maxz_voxle - min_pt.z;
                uint32_t z_voxel_num = ceil(z_length / voxel_length) + 1;

                /// construct a vector to save cells which are arranged under elevation
                /// value
                std::vector<std::vector<int>> elevation_idx;
                elevation_idx.resize(z_voxel_num);

                //    for (size_t j = 0; j != z_voxel_num; ++j)
                //      elevation_idx.push_back(one_voxels);

                /// put each point in current voxel into corresponding elevation cell
                for (size_t j = 0; j != voxels->second.size(); ++j) {
                    int z_num = uint32_t((cloud->points[voxels->second[j]].z - min_pt.z) /
                                         voxel_length);  /// uint32_t for bottom boundry
                    elevation_idx[z_num].push_back(voxels->second[j]);
                }

                /// set non-empty cell as a cube voxel
                for (size_t j = 0; j != elevation_idx.size(); ++j) {
                    if (elevation_idx[j].size()) {
                        center_point_idx.push_back(elevation_idx[j]);

                        /// Computing centroid
                        PointT one_pt;
                        one_pt.x = 0.0;
                        one_pt.y = 0.0;
                        one_pt.z = 0.0;
                        for (size_t p = 0; p != elevation_idx[j].size(); ++p) {
                            one_pt.x += cloud->points[elevation_idx[j][p]].x;
                            one_pt.y += cloud->points[elevation_idx[j][p]].y;
                            one_pt.z += cloud->points[elevation_idx[j][p]].z;
                        }
                        one_pt.x /= elevation_idx[j].size();
                        one_pt.y /= elevation_idx[j].size();
                        one_pt.z /= elevation_idx[j].size();
                        center_cloud->push_back(one_pt);
                    }
                }
            }
        }

#define PCL_INSTANTIATE_voxelizationWithIdx(PointType)                        \
  template PCL_EXPORTS void voxelizationWithIdx<PointType>(                   \
      const pcl::PointCloud<PointType>::Ptr&, std::vector<std::vector<int>>&, \
      pcl::PointCloud<PointType>::Ptr&, float);
        PCL_INSTANTIATE(voxelizationWithIdx, BASE_PCL_POINT_TYPES)

/**
 * Split point clouds by distance
 * The algorithm is similar to the segmentation in vectorization. Later, we need
 * to modify and use this in the segmentation algorithm in Vectorization
 * in order to reduce unnecessary code
 *
 * */
        template <typename PointT>
        std::vector<pcl::PointIndices> segmentCloudByDistance(
                const typename pcl::PointCloud<PointT>::ConstPtr& in_cloud,
                const float diantance_seg) {
            float x_min = std::numeric_limits<float>::max();
            float x_max = -std::numeric_limits<float>::max();
            float y_min = std::numeric_limits<float>::max();
            float y_max = -std::numeric_limits<float>::max();

            for (const auto& pt : in_cloud->points) {
                if (pt.x > x_max) x_max = pt.x;

                if (pt.y > y_max) y_max = pt.y;

                if (pt.x < x_min) x_min = pt.x;

                if (pt.y < y_min) y_min = pt.y;
            }

            float delta_x = x_max - x_min;
            float delta_y = y_max - y_min;
            std::string flag_xy = "x";
            if (delta_x < delta_y) {
                flag_xy = "y";
            }

            float delta_length = diantance_seg;

            std::map<int, pcl::PointIndices> xy_group;
            if (flag_xy == "x") {
                for (int i = 0; i < in_cloud->size(); ++i) {
                    auto& pt = in_cloud->points[i];
                    int index_xy = int((pt.x - x_min) / delta_length);
                    if (xy_group.find(index_xy) == xy_group.end())
                        xy_group[index_xy].indices = {i};
                    else
                        xy_group[index_xy].indices.push_back(i);
                }
            } else {
                for (int i = 0; i < in_cloud->size(); ++i) {
                    auto& pt = in_cloud->points[i];
                    int index_xy = int((pt.y - y_min) / delta_length);
                    if (xy_group.find(index_xy) == xy_group.end())
                        xy_group[index_xy].indices = {i};
                    else
                        xy_group[index_xy].indices.push_back(i);
                }
            }
            std::vector<pcl::PointIndices> output_clusters;
            for (auto& pair : xy_group) {
                output_clusters.emplace_back(std::move(pair.second));
            }
            return output_clusters;
        }

#define PCL_INSTANTIATE_segmentCloudByDistance(PointType) \
  template PCL_EXPORTS std::vector<pcl::PointIndices>     \
  segmentCloudByDistance<PointType>(                      \
      const pcl::PointCloud<PointType>::ConstPtr&, const float);
        PCL_INSTANTIATE(segmentCloudByDistance, BASE_PCL_POINT_TYPES)

/**
 * add point between two vector point by given step distance
 * @param input_vector: input vertex data
 * @param output_vector: output vertex data
 * @param step_distance: step distance
 *
 **/
        template <typename PointT>
        void addPointWithVector(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_vector,
                typename pcl::PointCloud<PointT>::Ptr& output_vector, float step_distance) {
            for (size_t i = 0; i != input_vector->size() - 1; ++i) {
                Eigen::Vector3d add_points;
                Eigen::Vector3d begin;
                begin << input_vector->points[i].x, input_vector->points[i].y,
                        input_vector->points[i].z;
                add_points << input_vector->points[i + 1].x - input_vector->points[i].x,
                        input_vector->points[i + 1].y - input_vector->points[i].y,
                        input_vector->points[i + 1].z - input_vector->points[i].z;

                auto distance = add_points.norm();
                auto numbers = floor(distance / step_distance);
                for (size_t j = 0; j < numbers; j++) {
                    Eigen::Vector3d result_points;
                    result_points = begin + add_points / distance * j * step_distance;
                    PointT add_point;
                    add_point.x = result_points[0];
                    add_point.y = result_points[1];
                    add_point.z = result_points[2];
                    output_vector->push_back(add_point);
                }
            }
        }
#define PCL_INSTANTIATE_addPointWithVector(PointType)      \
  template PCL_EXPORTS void addPointWithVector<PointType>( \
      const pcl::PointCloud<PointType>::ConstPtr&,         \
      pcl::PointCloud<PointType>::Ptr&, float);
        PCL_INSTANTIATE(addPointWithVector, BASE_PCL_POINT_TYPES)

    }  // pose
}  // zjlab