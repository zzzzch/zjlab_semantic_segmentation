//
// Created by zhachanghai on 20-10-15.
//

#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/types_c.h"
//pcl
#include <pcl/ModelCoefficients.h>
#include <pcl/PointIndices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/board.h>
#include <pcl/features/vfh.h>
#include <pcl/features/range_image_border_extractor.h>

#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>

#include "pcl_tools.h"
#include "base_io.h"

#include "coordinate_transformation.h"

using namespace zjlab;
using namespace pose;
using namespace std;

#define PI 3.14159265358979323846

DEFINE_string(scene_path,
              "/home/zhachanghai/bottle/pcl_cloud/", "scene pcd file");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

    Eigen::Vector4f centroid_origin;

    for (int count = 0; count < 38; count++) {
      string scene_file_path =
          FLAGS_scene_path + to_string(count) + "_point.pcd";
      //        string scene_file_path =
      //        "/home/zhachanghai/bottle/grap_bottle.pcd";

      pcl::PointCloud<pcl::PointXYZ>::Ptr origin_input(
          new pcl::PointCloud<pcl::PointXYZ>);
      LOG(INFO) << "Begin estimate pose !!";
      pcl::io::loadPCDFile(scene_file_path, *origin_input);

      // remove far points
      pcl::PointCloud<pcl::PointXYZ>::Ptr scene(
          new pcl::PointCloud<pcl::PointXYZ>);
      for (int i = 0; i < origin_input->size(); i++) {
        float distance = pow(origin_input->points[i].x, 2) +
                         pow(origin_input->points[i].y, 2) +
                         pow(origin_input->points[i].z, 2);
        //        LOG_EVERY_N(INFO,100) << "distance is " << distance ;
        if (distance < 0.6) {
          scene->push_back(origin_input->points[i]);
        }
      }

      pcl::VoxelGrid<pcl::PointXYZ> grid;
      float leaf = 0.008f;
      grid.setLeafSize(leaf, leaf, leaf);
      grid.setInputCloud(scene);
      grid.filter(*scene);

      LOG(INFO) << "grid filter over point size is " << scene->size();
      //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/scene.pcd",
      //        *scene);

      // transform coordinate from camera(right ground forward) to robot(forward
      // left up)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation2(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation3(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PointCloud<pcl::PointXYZ>::Ptr origin_rotation(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr origin_rotation2(
            new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr origin_rotation3(
            new pcl::PointCloud<pcl::PointXYZ>);

        // yaw(z) pitch(y) roll(x)
        Eigen::Matrix3d first_step_rotation =
            yprDegrees2RotationZYX(0, -41.1, -51);
        pcl::transformPointCloud(*scene, *scene_rotation,
                                 R2T(first_step_rotation));
        pcl::transformPointCloud(*origin_input, *origin_rotation,
                                 R2T(first_step_rotation));

        Eigen::Matrix3d second_step_rotation;
        second_step_rotation << 0, 0, 1, -1, 0, 0, 0, -1, 0;

        pcl::transformPointCloud(*scene_rotation, *scene_rotation2,
                                 R2T(second_step_rotation));
        pcl::transformPointCloud(*origin_rotation, *origin_rotation2,
                                 R2T(second_step_rotation));

        Eigen::Matrix4d thrid_step_trans;
        thrid_step_trans << 1, 0, 0, 0.05, 0, 1, 0, 0, 0, 0, 1, 1.8945, 0, 0, 0,
            1;

        pcl::transformPointCloud(*scene_rotation2, *scene_rotation3,
                                 thrid_step_trans);
        pcl::transformPointCloud(*origin_rotation2, *origin_rotation3,
                                 thrid_step_trans);

        //            pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/scene_rotation2.pcd",
        //            *scene_rotation3);
        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/origin_rotation/" +
                                      to_string(count) + "_origin.pcd",
                                  *origin_rotation3);
        origin_input.swap(origin_rotation3);
        scene.swap(scene_rotation3);
      }

      //估计点的法线，50为一组/
      // Estimate point normals
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
          new pcl::search::KdTree<pcl::PointXYZ>());
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
          new pcl::PointCloud<pcl::Normal>());
      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setSearchMethod(tree);
      ne.setInputCloud(scene);
      ne.setKSearch(50);
      ne.compute(*cloud_normals);

      ///分割平面///
      // Create the segmentation object for the planar model and set all the
      // parameters
      pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
      pcl::ModelCoefficients::Ptr coefficients_plane(
          new pcl::ModelCoefficients),
          coefficients_cylinder(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices),
          inliers_cylinder(new pcl::PointIndices);

      //把平面去掉，提取剩下的///
      // Remove the planar inliers, extract the rest
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2(
          new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(
          new pcl::PointCloud<pcl::Normal>);
      pcl::ExtractIndices<pcl::Normal> extract_normals;

      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
      seg.setNormalDistanceWeight(0.1);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setMaxIterations(100);
      seg.setDistanceThreshold(0.03);
      seg.setInputCloud(scene);
      seg.setInputNormals(cloud_normals);

      // 根据上面的输入参数执行分割获取平面模型系数和处在平面上的内点
      // Obtain the plane inliers and coefficients
      seg.segment(*inliers_plane, *coefficients_plane);
      LOG(INFO) << "Plane coefficients: " << *coefficients_plane;

      // 利用extract提取平面
      // Extract the planar inliers from the input cloud
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(
          new pcl::PointCloud<pcl::PointXYZ>());
      extract.setInputCloud(scene);  //设置输入点云
      extract.setIndices(inliers_plane);  //设置分割后的内点为需要提取的点集
      extract.setNegative(false);    //设置提取内点而非外点
      extract.filter(*cloud_plane);  //提取输出存储到cloud_plane

      //存储分割得到的平面上的点到点云文件/
      // Write the planar inliers to disk
      LOG(INFO) << "PointCloud representing the planar component: "
                << cloud_plane->points.size() << " data points.";
      //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/table_plane.pcd",
      //        *cloud_plane);

      float average_height = 0;
      int points_num = 0;
      for (int i = 0; i < cloud_plane->size(); i = i + 10) {
        average_height += cloud_plane->points[i].z;
        points_num++;
      }
      average_height = average_height / points_num;
      LOG(INFO) << "table plane average height is " << average_height;

      extract.setNegative(true);         //设置提取外点
      extract.filter(*cloud_filtered2);  //提取输出存储到cloud_filtered2
      extract_normals.setNegative(true);
      extract_normals.setInputCloud(cloud_normals);
      extract_normals.setIndices(inliers_plane);
      extract_normals.filter(*cloud_normals2);

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_height(
          new pcl::PointCloud<pcl::PointXYZ>());
      for (int i = 0; i < cloud_filtered2->size(); i++) {
        if (cloud_filtered2->points[i].z - 0.03 > average_height) {
          cloud_filtered_height->push_back(cloud_filtered2->points[i]);
        }
      }

      //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/up_table_point.pcd",
      //        *cloud_filtered_height);

      LOG(INFO) << "get up table point over !";

      //        //开始找圆柱///
      //        // Create the segmentation object for cylinder segmentation and
      //        set all the parameters
      //        seg.setOptimizeCoefficients(true);
      //        //设置对估计的模型系数需要进行优化
      //        seg.setModelType(pcl::SACMODEL_PARALLEL_LINE);
      //        //设置分割模型为圆柱型
      //        seg.setMethodType(pcl::SAC_RANSAC);
      //        //设置采用RANSAC作为算法的参数估计方法
      //        seg.setNormalDistanceWeight(0.4);
      //        //设置表面法线权重系数
      //        seg.setMaxIterations(10000);
      //        //设置迭代的最大次数10000
      //        seg.setDistanceThreshold(0.05);
      //        //设置内点到模型的距离允许最大值
      //        seg.setRadiusLimits(0, 0.3);
      //        //设置估计出的圆柱模型的半径范围
      //        seg.setInputCloud(cloud_filtered2);
      //        seg.setInputNormals(cloud_normals2);
      //
      //        //获得圆柱层和内点///
      //        // Obtain the cylinder inliers and coefficients
      //        seg.segment (*inliers_cylinder, *coefficients_cylinder);
      //
      //        pcl::PointCloud<pcl::Normal>::Ptr pipline_normals (new
      //        pcl::PointCloud<pcl::Normal>);
      //        //存储分割得到的平面上的点到点云文件///
      //        // Write the cylinder inliers to disk
      //        extract.setInputCloud (cloud_filtered2);
      //        extract.setIndices (inliers_cylinder);
      //        extract.setNegative (false);
      //
      //
      //        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder (new
      //        pcl::PointCloud<pcl::PointXYZ> ());
      //        extract.filter (*cloud_cylinder);
      //        if (cloud_cylinder->points.empty ())
      //            LOG(INFO) << "Can't find the cylindrical component." ;
      //        else
      //        {
      //            LOG(INFO) << "PointCloud representing the cylindrical
      //            component: " << cloud_cylinder->points.size () << " data
      //            points." ;
      //            pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/table_scene_mug_stereo_textured_cylinder.pcd",
      //            *cloud_cylinder);
      //        }

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
      ec.setClusterTolerance(0.02);
      ec.setMinClusterSize(10);
      ec.setMaxClusterSize(100000);
      ec.setSearchMethod(tree);
      ec.setInputCloud(cloud_filtered_height);
      ec.extract(cluster_indices);

      pcl::PointCloud<pcl::PointXYZ>::Ptr process_scene(
          new pcl::PointCloud<pcl::PointXYZ>);

      if (cluster_indices.size() > 0) {
        process_scene = getChildCloudByIndicesFromOriginal<pcl::PointXYZ>(
            cloud_filtered_height, cluster_indices[0]);
      } else {
        process_scene = cloud_filtered_height;
      }

      //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/bottle.pcd",
      //        *process_scene);

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(
          new pcl::PointCloud<pcl::PointXYZ>);
      // Create the filtering object
      pcl::ProjectInliers<pcl::PointXYZ> proj;
      proj.setModelType(pcl::SACMODEL_PLANE);
      proj.setInputCloud(process_scene);
      proj.setModelCoefficients(coefficients_plane);
      proj.filter(*cloud_projected);

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud_projected, centroid);

      if (!centroid_origin.data()) {
        centroid_origin = centroid;
      } else {
        auto distance = sqrt(pow((centroid[0] - centroid_origin[0]), 2) +
                             pow((centroid[1] - centroid_origin[1]), 2) +
                             pow((centroid[1] - centroid_origin[1]), 2));
        LOG(INFO) << "distance is " << distance;
        if (distance > 0.04) {
          centroid_origin = centroid;
        } else {
          continue;
        }
      }

      LOG(INFO) << "centroid is " << centroid;
      //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/bottle_project.pcd",
      //        *cloud_projected);

      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_src;
      ne_src.setInputCloud(cloud_projected);
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src(
          new pcl::search::KdTree<pcl::PointXYZ>());
      ne_src.setSearchMethod(tree_src);
      pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(
          new pcl::PointCloud<pcl::Normal>);
      ne_src.setRadiusSearch(0.02);
      ne_src.compute(*cloud_src_normals);
      //
      seg.setOptimizeCoefficients(true);  //设置对估计的模型系数需要进行优化
      seg.setModelType(pcl::SACMODEL_CIRCLE3D);  //设置分割模型为圆柱型
      seg.setMethodType(
          pcl::SAC_RANSAC);  //设置采用RANSAC作为算法的参数估计方法
      seg.setNormalDistanceWeight(0.2);  //设置表面法线权重系数
      seg.setMaxIterations(10000);       //设置迭代的最大次数10000
      seg.setDistanceThreshold(0.005);  //设置内点到模型的距离允许最大值
      seg.setRadiusLimits(0, 0.4);  //设置估计出的圆柱模型的半径范围
      seg.setInputCloud(cloud_projected);
      seg.setInputNormals(cloud_src_normals);

      pcl::PointIndices::Ptr inliers_bottle(new pcl::PointIndices);
      pcl::ModelCoefficients::Ptr coefficients_bottle(
          new pcl::ModelCoefficients);

      seg.segment(*inliers_bottle, *coefficients_bottle);
      LOG(INFO) << "bottle plane coefficients: " << *coefficients_bottle;

      pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_origin(
          new pcl::PointCloud<pcl::PointXYZ>());

      for (float z = average_height; z < average_height + 0.2; z = z + 0.01) {
        for (float ang = 0.0; ang <= 360; ang += 5.0) {
          pcl::PointXYZ basic_point;
          basic_point.x =
              coefficients_bottle->values[0] + cosf(pcl::deg2rad(ang)) * 0.04;
          basic_point.y =
              coefficients_bottle->values[1] + sinf(pcl::deg2rad(ang)) * 0.04;
          basic_point.z = z;
          cylinder_origin->push_back(basic_point);
        }
      }
      cylinder_origin->width = (int)cylinder_origin->points.size();
      cylinder_origin->height = 1;
      LOG(INFO) << "cylinder_origin size is " << cylinder_origin->points.size();

      //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/bottle1.pcd",
      //        *cylinder_origin);

      LOG(INFO) << " Number " << count << " result is  "
                << coefficients_bottle->values[0] << " "
                << coefficients_bottle->values[1] << " "
                << coefficients_bottle->values[2];

      pcl::PointCloud<pcl::PointXYZ>::Ptr line(
          new pcl::PointCloud<pcl::PointXYZ>());

      for (float z = average_height; z < average_height + 0.2; z = z + 0.01) {
        pcl::PointXYZ basic_point;
        float x = (z - coefficients_bottle->values[2]) /
                      coefficients_bottle->values[6] *
                      coefficients_bottle->values[4] +
                  coefficients_bottle->values[0];
        float y = (z - coefficients_bottle->values[2]) /
                      coefficients_bottle->values[6] *
                      coefficients_bottle->values[5] +
                  coefficients_bottle->values[1];
        basic_point.x = x;
        basic_point.y = y;
        basic_point.z = z;
        line->push_back(basic_point);
      }

      pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/result/" +
                                    to_string(count) + "_bottle_line.pcd",
                                *line);
    }
    LOG(INFO) << "RUN grap bottle success !";
    return 0;
}