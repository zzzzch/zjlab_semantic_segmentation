#include <Python.h>
#include <iostream>
//#include <fstream>
//#include <nlohmann/json.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/imgproc/types_c.h"
//#include <opencv2/highgui/highgui_c.h>
//
//#include <pcl/features/normal_3d.h>
//#include <pcl/features/normal_3d_omp.h>
//
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include "base_io.h"
#include "bbox_pcl.h"
#include "coordinate_transformation.h"

//#include "coordinate_transformation.h"
#include <random>
#include "tictoc.hpp"

using namespace zjlab;
using namespace pose;
using namespace std;
// using json = nlohmann::json;

// void readJsonFile(const string& json_path){
//    json j;
//    j["x"] = "100";
//    j["y"] = "200";
//    ofstream json_file(json_path);
//    json_file << j;
//    json_file.close();
//}

// inline Eigen::Matrix4d R2T(const Eigen::Matrix3d& R) {
//    Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
//    tf.topLeftCorner<3, 3>() = R;
//    return tf;
//}

int main(int argc, char** argv)
{
  CloudBbox box;
  cv::Mat input_image;
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(
      new pcl::PointCloud<pcl::PointXYZ>());

  pcl::io::loadPCDFile("/home/zhachanghai/show_test/bottle1.pcd", *input_cloud);
  input_image =
      cv::imread("/home/zhachanghai/show_test/rgb_image/37_image.jpg");

  computeBoundingBbox<pcl::PointXYZ>(input_cloud, box);

  Eigen::Matrix3f k_depth_camera;
  k_depth_camera << 634.2861328125, 0, 633.5665893554688, 0, 632.8214111328125,
      404.9833679199219, 0, 0, 1;

  float fx = 380.572;
  float fy = 379.693;
  float cx = 316.14;
  float cy = 242.99;

  for (auto i = 0; i < box.eight_points_.size(); i++) {
    pcl::PointXYZ pt;
    pt.x = box.eight_points_[i][0];
    pt.y = box.eight_points_[i][1];
    pt.z = box.eight_points_[i][2];

    output_cloud->push_back(pt);
    //    LOG(INFO) << "point " << i << " is " << box.eight_points_[i][0] << " "
    //    << box.eight_points_[i][1] << " " << box.eight_points_[i][2];
  }

  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation2(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation3(
        new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Matrix4d first_step_rotation;
    first_step_rotation << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

    //              first_step_rotation = first_step_rotation.inverse();
    pcl::transformPointCloud(*output_cloud, *scene_rotation,
                             first_step_rotation);
    //        for(int i = 0;i<scene_rotation->size();i++){
    //            LOG(INFO) << "cube points1 " << i << " is " <<
    //            scene_rotation->points[i].x << " " <<
    //            scene_rotation->points[i].y << " " <<
    //            scene_rotation->points[i].z;
    //        }

    Eigen::Matrix3d second_step_rotation;
    second_step_rotation << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    //              second_step_rotation = second_step_rotation.inverse();
    pcl::transformPointCloud(*scene_rotation, *scene_rotation2,
                             R2T(second_step_rotation.inverse()));

    //        for(int i = 0;i<scene_rotation2->size();i++){
    //            LOG(INFO) << "cube points2 " << i << " is " <<
    //            scene_rotation2->points[i].x << " " <<
    //            scene_rotation2->points[i].y << " " <<
    //            scene_rotation2->points[i].z;
    //        }

    // yaw(z) pitch(y) roll(x)
    Eigen::Matrix3d third_step_rotation = yprDegrees2RotationZYX(0, 0, -30);
    pcl::transformPointCloud(*scene_rotation2, *scene_rotation3,
                             R2T(third_step_rotation.inverse()));

    //        for(int i = 0;i<scene_rotation3->size();i++){
    //            LOG(INFO) << "cube points3 " << i << " is " <<
    //            scene_rotation3->points[i].x << " " <<
    //            scene_rotation3->points[i].y << " " <<
    //            scene_rotation3->points[i].z;
    //        }

    output_cloud.swap(scene_rotation3);
  }

  //    for(int i = 0;i<output_cloud->size();i++){
  //        LOG(INFO) << "cube points4 " << i << " is " <<
  //        output_cloud->points[i].x << " " << output_cloud->points[i].y << " "
  //        << output_cloud->points[i].z;
  //    }

  std::vector<cv::Point> cube_points;
  for (int i = 0; i < output_cloud->size(); i++) {
    auto pt = output_cloud->points[i];
    cv::Point cube_pt;
    //        pt.z /= 1000;
    //        pt.x = (x - cx) * pt.z / fx;
    //        pt.y = (y - cy) * pt.z / fy;
    //        LOG(INFO) << "pt x is " << pt.x << " y is " << pt.y << " z is " <<
    //        pt.z;
    //        LOG(INFO) << "point " << i << " is " << 1.0f * pt.x * fx / pt.z +
    //        cx << " " << 1.0f * pt.y * fy / pt.z + cy;

    cube_pt.x = int(1.0f * pt.x * fx / pt.z + cx);
    cube_pt.y = int(1.0f * pt.y * fy / pt.z + cy);

    cube_points.push_back(cube_pt);
  }
  for (int i = 0; i < cube_points.size(); i++) {
    LOG(INFO) << "cube points " << i << " is " << cube_points[i].x << " "
              << cube_points[i].y;
  }

  if (cube_points.size() == 8) {
    LOG(INFO) << "cube points is drawing ! ";
    int thickness = 2;
    cv::line(input_image, cube_points[0], cube_points[1], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[1], cube_points[3], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[3], cube_points[2], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[0], cube_points[2], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[0], cube_points[4], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[1], cube_points[5], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[2], cube_points[6], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[3], cube_points[7], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[4], cube_points[5], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[5], cube_points[7], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[7], cube_points[6], cv::Scalar(0, 0, 255),
             thickness);
    cv::line(input_image, cube_points[4], cube_points[6], cv::Scalar(0, 0, 255),
             thickness);
  }

  cv::imshow("cube_image", input_image);
  cv::waitKey();
  //    pcl::io::savePCDFile("/home/zhachanghai/bottle/bottle_boundbox.pcd",*output_cloud);
  return 0;
}
