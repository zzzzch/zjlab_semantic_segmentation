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
//#include <pcl/io/pcd_io.h>
//#include "base_io.h"
//#include "coordinate_transformation.h"
//#include "tictoc.hpp"

// using namespace zjlab;
// using namespace pose;
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

int main(int argc, char** argv)
{
  //    string infile_pic = "/home/zhachanghai/test_grap/video/";
  //    string outfile_video = "/home/zhachanghai/test_grap/demo_show.avi";
  //    image2Video(infile_pic,outfile_video);
  //    string json_file = "/home/zhachanghai/test.json";
  //    readJsonFile(json_file);

  //     pcl::gpu::DeviceArray<pcl::PointXYZRGB> cloud_device;

  //  TicToc t;
  //
  //  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new
  //  pcl::PointCloud<pcl::PointXYZ>);
  //  string scene_file_path = "/home/zhachanghai/guochen_test/down_scene.pcd";
  //
  //  pcl::io::loadPCDFile(scene_file_path, *cloud);
  //
  //  cout << t.toc() << "first time " << endl;
  //
  //  cout << "cloud size is " << cloud->size() << endl;
  //  pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
  //  ne.setInputCloud(cloud);
  //  //创建一个空的kdtree对象，并把它传递给法线估计对象
  //  //基于给出的输入数据集，kdtree将被建立
  //  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
  //      new pcl::search::KdTree<pcl::PointXYZ>());
  //  ne.setSearchMethod(tree);
  //  //输出数据集
  //  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
  //      new pcl::PointCloud<pcl::Normal>);
  //  //使用半径在查询点周围3厘米范围内的所有邻元素
  //  ne.setRadiusSearch(0.03);
  //  ne.setNumberOfThreads(8);
  //  //计算特征值
  //  ne.compute(*cloud_normals);
  //
  //  cout << t.toc() << "end time omp" << endl;
  //
  //  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne1;
  //  ne1.setInputCloud(cloud);
  //  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(
  //      new pcl::search::KdTree<pcl::PointXYZ>());
  //  ne1.setSearchMethod(tree1);
  //  //输出数据集
  //  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1(
  //      new pcl::PointCloud<pcl::Normal>);
  //  //使用半径在查询点周围3厘米范围内的所有邻元素
  //  ne1.setRadiusSearch(0.03);
  //  //计算特征值
  //  ne1.compute(*cloud_normals1);
  //
  //  cout << t.toc() << "end time normal " << endl;
  int a = 0x61626364;

  int c = 0x65666768;
  int b = 0x68697071;

  char *p = (char *)&a;
  //    cout << "b = " << b << endl;
  cout << "&a " << &a << endl;
  cout << "&b " << &b << endl;
  cout << "&c " << &c << endl;

  cout << "p = " << p << endl;

  return 0;
}
