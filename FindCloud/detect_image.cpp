#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "base_io.h"
#include "tictoc.hpp"

using namespace zjlab;
using namespace pose;
using namespace std;

DEFINE_string(image_path,
              "/home/zhachanghai/charging_gun/rgb_image/", "Input rgb image file path");

DEFINE_string(target_path,
              "/home/zhachanghai/charging_gun/depth_image/", "Input depth image file path");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

//    CameraInfo camera_info;
//    camera_info.init();

    TicToc t;
    for (int count = 200; count < 201; count++) {
        cv::Mat scene = cv::imread(FLAGS_image_path + to_string(count) + "_image.jpg");
//        cv::Mat depth = cv::imread(FLAGS_depth_path + to_string(count) + "_depth.png", CV_16UC1);

        cv::Mat target = cv::imread(FLAGS_target_path);
        cv::Mat gun_area;

        LOG(INFO) << t.toc() << "begin detect !";
//        cv::imshow("scene", scene);
//        cv::imshow("target", target);

        int trackbar_method = CV_TM_CCOEFF_NORMED;
        cv::Rect rect_gun;

        cv::Mat result;
        result.create(scene.rows, scene.cols, scene.type());

        cv::matchTemplate(scene, target, result,trackbar_method);   //模板匹配
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
        rect_gun = cv::Rect(matchLocation.x,matchLocation.y,target.cols,target.rows);

        bool show_image = false;

        if(show_image){
            cv::Mat display;
            scene.copyTo(display);
            rectangle(display, rect_gun, cv::Scalar(0,0,255));
            cv::imshow("rect ", display);
        }

    }
}