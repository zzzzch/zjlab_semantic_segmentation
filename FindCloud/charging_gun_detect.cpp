
#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "base_io.h"
#include "tictoc.hpp"

#include "charging_gun_detect.h"
#include "pcl_tools.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Python.h>

using namespace zjlab;
using namespace pose;
using namespace std;

DEFINE_string(image_path,
              "/home/zhachanghai/charging_gun/rgb_image/", "Input rgb image file path");

DEFINE_string(depth_path,
              "/home/zhachanghai/charging_gun/depth_image/", "Input depth image file path");

DEFINE_string(target_path,
              "/home/zhachanghai/charging_gun/target.png", "target image file path");


/**
 * Find Bigest Contour in the image, need trans the background to (0,0,0)
 * @param src : input image.
 * @return contours[imax] : Bigest Contour point vector.
 */
vector<cv::Point> FindBigestContour(const cv::Mat& src){
    int imax = 0;
    int imaxcontour = -1;
    std::vector<std::vector<cv::Point> >contours;
    cv::findContours(src,contours,CV_RETR_LIST,CV_CHAIN_APPROX_SIMPLE);
    int max_num = src.rows * src.cols * 0.9;
    for (int i=0;i<contours.size();i++){
        int itmp =  contourArea(contours[i]);
        if (imaxcontour < itmp && itmp < max_num){
            imax = i;
            imaxcontour = itmp;
        }
    }
    LOG(INFO) << "result contour num is " << imaxcontour;

    return contours[imax];
}

pcl::PointCloud<pcl::PointXYZ>::Ptr depth2cloudarea(cv::Mat depth_image, cv::Rect& rect, float fx, float fy, float cx, float cy) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
    cloud_ptr->width = rect.width;
    cloud_ptr->height = rect.height;
    cloud_ptr->is_dense = false;
    int begin_y = rect.tl().y;
    int begin_x = rect.tl().x;
    int end_y = rect.br().y;
    int end_x = rect.br().x;

    for (int y = begin_y; y < end_y; ++y) {
        for (int x = begin_x; x < end_x; ++x) {
            pcl::PointXYZ pt;
            if (depth_image.at<unsigned short>(y, x) != 0) {
                if (depth_image.at<unsigned short>(y, x) > 2000) {
                    pt.z = 0;
                    pt.x = 0;
                    pt.y = 0;
                } else {
                    pt.z = depth_image.at<unsigned short>(y, x);
                    pt.z /= 1000;
                    pt.x = (x - cx) * pt.z / fx;
                    pt.y = (y - cy) * pt.z / fy;
                }
                cloud_ptr->points.push_back(pt);
            } else {
                pt.z = 0;
                pt.x = 0;
                pt.y = 0;
                cloud_ptr->points.push_back(pt);
            }
        }
    }
    return cloud_ptr;
}

/**
 * Detect target obj rect in the src image , rect size is target image size
 * @param src : input scene image.
 * @param tar : input target image.
 * @param rect : result of the detect rect.
 * @param show_image : show image or not.
 * @param trackbar_method : match template method.
 */
void detectObjRect(const cv::Mat& src, const cv::Mat& tar, cv::Rect &rect, bool show_image, int trackbar_method){
    cv::Mat result;
    result.create(src.rows, src.cols, src.type());

    cv::matchTemplate(src, tar, result,trackbar_method);   //模板匹配
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);       //归一化处理

    //通过minMaxLoc定位最佳匹配位置
    double minValue, maxValue;
    cv::Point minLocation, maxLocation;
    cv::Point matchLocation;
    minMaxLoc(result, &minValue, &maxValue, &minLocation, &maxLocation, cv::Mat());

    //对于方法SQDIFF和SQDIFF_NORMED两种方法来讲，越小的值就有着更高的匹配结果
    //而其余的方法则是数值越大匹配效果越好
    if(trackbar_method==CV_TM_SQDIFF||trackbar_method==CV_TM_SQDIFF_NORMED)
    {
        matchLocation=minLocation;
    }
    else
    {
        matchLocation=maxLocation;
    }

    //TODO zch: rect image need resize
    rect = cv::Rect(matchLocation.x,matchLocation.y,tar.cols,tar.rows);

    if(show_image){
        cv::Mat display;
        src.copyTo(display);
        rectangle(display, rect, cv::Scalar(0,0,255));
        cv::imshow("rect ", display);
    }
}


/**
 * Detect the gun area
 * @param src : input scene image.
 * @param controus : result of point vector.
 * @param show_image : show image or not.
 */
void detectGunArea(const cv::Mat& src, vector<vector<cv::Point> >& controus, bool show_image){

    cv::Mat src_gray;
    cv::Mat src_bin;

    cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

    cv::threshold(src_gray, src_bin, 20, 255, CV_THRESH_BINARY);
    cv::bitwise_not(src_bin, src_bin);

    vector<cv::Point> bigestcontrour = FindBigestContour(src_bin);
    controus.push_back(bigestcontrour);

    if(show_image){
        cv::imshow("scene gray", src_gray);
//        cv::imshow("scene gray threshold", bin);
        cv::drawContours(src, controus, 0, cv::Scalar(0, 0,255), 3);
        cv::imshow("result scene", src);
    }
}

void sortCirclePoint(const vector<cv::Vec3f> &circles){
    vector<Eigen::Vector3f> circles_point_list;
    for(int i = 0; i < circles.size(); i++){
        Eigen::Vector3f pt;
        pt << circles[i][0],circles[i][1],circles[i][2];
        circles_point_list.push_back(pt);
    }

    sort(circles_point_list.begin(), circles_point_list.end(),
              [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
                  return a[2] > b[2];
              });

    vector<Eigen::Vector3f> top_layer,bot_layer;

    top_layer.insert(top_layer.end(),circles_point_list.begin(),circles_point_list.begin()+3);
    bot_layer.insert(bot_layer.end(),circles_point_list.begin()+3,circles_point_list.end());

    sort(top_layer.begin(), top_layer.end(),
         [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
             return a[0] < b[0];
         });
    sort(bot_layer.begin(), bot_layer.end(),
         [](const Eigen::Vector3f& a, const Eigen::Vector3f& b) {
             return a[0] < b[0];
         });



    auto distance_top_1 = sqrt(pow((top_layer[1][1]-top_layer[0][1]),2) + pow((top_layer[1][0]-top_layer[0][0]),2));
    auto distance_top_2 = sqrt(pow((top_layer[2][1]-top_layer[2][1]),2) + pow((top_layer[2][0]-top_layer[1][0]),2));

    auto distance_bot_1 = sqrt(pow((bot_layer[1][1]-bot_layer[0][1]),2) + pow((bot_layer[1][0]-bot_layer[0][0]),2));


//    if(circles_point_list.size() != 5){
//        LOG(WARNING) << "circles not be detected completely(5 circles), please check it !!!";
//        return circles_point_list;
//    }

}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

    CameraInfo camera_info;
    camera_info.init();

    TicToc t;
    for (int count = 200; count < 201; count++){
        cv::Mat scene = cv::imread(FLAGS_image_path + to_string(count) + "_image.jpg");
        cv::Mat depth = cv::imread(FLAGS_depth_path + to_string(count) + "_depth.png",CV_16UC1);

        cv::Mat target = cv::imread(FLAGS_target_path);
        cv::Mat gun_area;

        LOG(INFO) << t.toc() << "begin detect !";
//        cv::imshow("scene", scene);
//        cv::imshow("target", target);

        int trackbar_method = CV_TM_CCOEFF_NORMED;
        cv::Rect rect_gun;
        detectObjRect(scene, target, rect_gun, false, trackbar_method);

        cv::Mat scene_roi = scene(rect_gun);
        vector<vector<cv::Point> > controus;

        detectGunArea(scene_roi,controus ,false);

        // use for show origin area in scene image
//        vector<cv::Point> controu = controus[0];
//        for (int i = 0; i < controu.size(); i++){
//            cv::Point pt = controu[i];
//            pt.x += rect_gun.x;
//            pt.y += rect_gun.y;
//            controu[i] = pt;
//        }
//
//        controus[0] = controu;

        cv::Mat mask = cv::Mat::zeros(cv::Size(scene_roi.cols, scene_roi.rows), CV_8UC3);//黑色图像
        cv::drawContours(mask,controus, 0, cv::Scalar(255,255,255),-1);
//        cv::imshow("mask scene", mask);

        cv::bitwise_and(scene_roi, mask, gun_area);
//        cv::imshow("result gun area", gun_area);

        cv::Mat gun_area_gray;
        cv::Mat gun_area_gray_bin;

        cv::cvtColor(gun_area, gun_area_gray, cv::COLOR_BGR2GRAY);

        cv::threshold(gun_area_gray, gun_area_gray_bin, 40, 255, CV_THRESH_BINARY);
        cv::imshow("result gun area result", gun_area_gray_bin);
        std::vector<cv::Vec3f> circles;
//        cv::HoughCircles(gun_area_gray_bin, circles, CV_HOUGH_GRADIENT, 2, 30, 300, 20, 0, 10);
        cv::HoughCircles(gun_area_gray_bin, circles, CV_HOUGH_GRADIENT, 2, 20, 300, 10, 0, 10);

        LOG(INFO) << "circles size " << circles.size();

//        vector<cv::Point> circles_point_list;
        sortCirclePoint(circles);

        cv::Mat image_show;
        gun_area.copyTo(image_show);
        std::vector<cv::Vec3f>::const_iterator itc = circles.begin();
        while (itc != circles.end()) {
            cv::Point pt((*itc)[0], (*itc)[1]);
            LOG(INFO) << "pt.x is " << pt.x << " pt.y is " << pt.y << " r is " << (*itc)[2];
            cv::circle(image_show, pt, int((*itc)[2]), cv::Scalar(0,0,255), 2); //画圆
            ++itc;
        }

        LOG(INFO) << t.toc() << "end detect !";

        cv::imshow("circle result", image_show);

        cv::waitKey();



//        pcl::PointCloud<pcl::PointXYZ>::Ptr gun_cloud(new pcl::PointCloud<pcl::PointXYZ>());
//        float fx = camera_info.get_rgb_fx();
//        float fy = camera_info.get_rgb_fy();
//        float cx = camera_info.get_rgb_cx();
//        float cy = camera_info.get_rgb_cy();
//
//        gun_cloud = depth2cloudarea(depth,rect_gun,fx,fy,cx,cy);
//
//        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
//        tree->setInputCloud (gun_cloud);
//
//        std::vector<pcl::PointIndices> cluster_indices;
//
//        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
//        ec.setClusterTolerance (0.02);
//        ec.setMinClusterSize (100);
//        ec.setMaxClusterSize (100000);
//        ec.setSearchMethod (tree);
//        ec.setInputCloud (gun_cloud);
//        ec.extract (cluster_indices);
//
//        pcl::PointCloud<pcl::PointXYZ>::Ptr process_scene(new pcl::PointCloud<pcl::PointXYZ>);
//
//        process_scene = getChildCloudByIndicesFromOriginal<pcl::PointXYZ>(gun_cloud,cluster_indices[0]);


//        pcl::io::savePCDFile("/home/zhachanghai/charging_gun/gun.pcd",*process_scene);



    }
    return 0;
}