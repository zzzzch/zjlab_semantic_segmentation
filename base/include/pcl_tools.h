#pragma once

#include <vector>

#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/octree.h>
#include <pcl/point_types.h>
#include <pcl/impl/instantiate.hpp>

namespace zjlab {
    namespace pose {

#define BASE_PCL_POINT_TYPES                                               \
  (pcl::PointXYZ)(pcl::PointXYZI)(pcl::PointXYZL)(pcl::PointXYZINormal)(pcl::PointNormal)


#define BASE_PCL_NORMAL_POINT_TYPES (pcl::PointXYZINormal)(pcl::PointNormal)

        /// Voxel filter
        enum class PointClassType : uint8_t {

            OTHER = 0,
            LINE = 1,
            PLANE = 2,
            SPHERE = 3

        };

/**
 * If PCL voxel grid fails with given leaf_size, will retry with leaf_size *=
 * leaf_sz_scale until success.
 * @tparam pointT
 * @param input_cloud
 * @param leaf_size
 * @param leaf_sz_scale
 * @return
 */
//        template <typename pointT>
//        typename std::vector<int> voxelFilterIndices(
//                const typename pcl::PointCloud<pointT>::ConstPtr& input_cloud,
//                const float leaf_size);

        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr processByVoxelFilter(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_cloud,
                const float leaf_size);

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
                const int points_num, float leaf_size, float leaf_sz_scale = 1.2f);

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
                const int points_num, float leaf_size, const float leaf_sz_scale = 1.2f);

        template <typename pointT>
        typename pcl::PointCloud<pointT>::Ptr processByRecurseVoxelFilterInGrounRead(
                const typename pcl::PointCloud<pointT>::Ptr& input_cloud,
                const int points_num, const float leaf_size);

/// pass filter
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr processByConditionFilter(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_cloud,
                const std::string& field, const double& min_val, const double& max_val);

/// statistic filter
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr processStatisticalOutlierRemoval(
                const typename pcl::PointCloud<PointT>::ConstPtr& in_cloud, size_t meank,
                double devmulth);

/// radius filter
        template <typename PointT>
        typename pcl::PointCloud<PointT>::Ptr radiusOutlierRemoval(
                const typename pcl::PointCloud<PointT>::Ptr& cloud, double radius,
                int min_neighbors);

        template <typename pointT>
        typename pcl::search::KdTree<pointT>::Ptr kdtree(
                const typename pcl::PointCloud<pointT>::Ptr& cloud, float epsilon);

/// build octree
        template <typename PointT>
        typename pcl::search::Octree<PointT>::Ptr buildOctree(
                const typename pcl::PointCloud<PointT>::ConstPtr& cloud, float resolution,
                const pcl::PointIndices::ConstPtr& indices = nullptr);

/// compute pointscloud normal
        template <typename PointT>
        pcl::PointCloud<pcl::Normal>::Ptr computeNormal(
                const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                typename pcl::search::Octree<PointT>::Ptr& octree, int K, double radius);

        template <typename PointT>
        void computeNormalInPoint(
                const typename pcl::PointCloud<PointT>::Ptr& cloud,
                const typename pcl::search::Octree<PointT>::Ptr& octree, int K,
                double radius);

// add by xiaoqi [07/02/2019]
/// compute pointscloud shape class
        template <typename PointT>
        void computePCA(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                        std::vector<PointClassType>& point_class,
                        std::vector<float>& point_verticality, double radius);

/// When the coordinate value is large, the general integer part exceeds three
/// digits.
/// Please use this method.
        template <typename PointT>
        void computeNormalTLocal(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                                 pcl::PointCloud<pcl::Normal>::Ptr& normals, int K,
                                 double radius);

        template <typename PointT>
        void computeNormalT(const typename pcl::PointCloud<PointT>::Ptr& cloud,
                            const typename pcl::search::Octree<PointT>::Ptr& octree,
                            pcl::PointCloud<pcl::Normal>::Ptr& normals, int K,
                            double radius);

/// euclidean cluster
        template <typename PointT>
        std::vector<pcl::PointIndices> euclideanClustertool(
                const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                const typename pcl::search::Octree<PointT>::Ptr& octree,
                const double cluster_tolerance, const int min_cluster_size,
                const int max_cluster_size, const pcl::PointIndices::ConstPtr& indices = nullptr);

        template <typename PointT>
        float judgePointSide(PointT& head_point, PointT& tail_point,
                             PointT& query_point);

/// sac segmentation
        template <typename PointT>
        void sacSegmentationtool(
                const typename pcl::PointCloud<PointT>::ConstPtr& in_cloud,
                const pcl::SacModel& mode, const int& method, const int& maxiteration,
                const double& distance, pcl::PointIndices::Ptr& inliers,
                pcl::ModelCoefficients::Ptr& coefficients,
                const bool flagoptimazation = true);

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
                             double curvature_threshold);

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
                const pcl::PointIndices& indices);

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
                const pcl::ModelCoefficients::Ptr& coeff);

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
                const std::vector<int>& indices);

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
                const pcl::PointIndices& indices);

/**
 * get the index in original clouds from voxel cloud
 * @param voxel_point_indexes: each voxel with the point indexes in origin cloud
 * @param indices
 */
        std::vector<pcl::PointIndices> getIndicesInOriCloudFromVoxel(
                const std::vector<std::vector<int>>& voxel_point_indexes,
                const std::vector<pcl::PointIndices>& indices);

        template <typename pointT>
        void keypointsEstimation(
                const typename pcl::PointCloud<pointT>::ConstPtr& cloud,
                pcl::PointCloud<pcl::Normal>::ConstPtr& normal);

        template <typename PointTNormal>
        void addNormalsToCloud(
                const typename pcl::PointCloud<PointTNormal>::Ptr cloud,
                const typename pcl::search::KdTree<PointTNormal>::Ptr& in_kdtree = nullptr,
                double radius = 0.35, int K = 0);
/**
 * @brief Use Pcl build-in multithread method, add normal with muiltithreads
 */
        template <typename PointTNormal>
        void addNormalsToCloudOMP(
                const typename pcl::PointCloud<PointTNormal>::Ptr cloud,
                const typename pcl::search::KdTree<PointTNormal>::Ptr& in_kdtree = nullptr,
                int num_threads = 1, double radius = 0.35, int K = 0);

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
                typename pcl::PointCloud<PointT>::ConstPtr in_cloud, float win_sz = 0.4f,
                float height_thres = 0.05f);

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
                                 float voxel_length = 0.1f);

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
                const float diantance_seg);

/**
 * add point between two vector point by given step distance
 * example: input vector is [(3,5,0),(5,5,0)] step_distance is 0.5
 *          output vector is [(3,5,0),(3.5,5,0),(4,5,0),(4.5,5,0),(5,5,0)]
 * @param input_vector: input vertex data
 * @param output_vector: output vertex data
 * @param step_distance: step distance
 *
 **/
        template <typename PointT>
        void addPointWithVector(
                const typename pcl::PointCloud<PointT>::ConstPtr& input_vector,
                typename pcl::PointCloud<PointT>::Ptr& output_vector,
                float step_distance = 0.5f);

    }  // mapping
}  // kd