////
//// Created by zhachanghai on 20-12-3.
////
#include "iostream"
#include "SemiGlobalMatching.h"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "base_io.h"
#include "tictoc.hpp"
//
using namespace zjlab;
using namespace pose;
using namespace std;

DEFINE_string(left_path,
              "/home/zhachanghai/xiaomi/left/0_image.jpg", "Input left image file path");

DEFINE_string(right_path,
              "/home/zhachanghai/xiaomi/right/0_image.jpg", "Input right image file path");

// opencv library
#include <opencv2/opencv.hpp>

void stereoRectification(cv::Mat& img_left,cv::Mat& img_right,cv::Mat& img_left_result,cv::Mat& img_right_result){

    cv::Size s1,s2;
    s1 = img_left.size();
    s2 = img_right.size();
    cv::Mat intrinsic_left;
    cv::Mat intrinsic_right;

    cv::Mat distCoeffs_left,distCoeffs_right;
    cv::Mat R_L;
    cv::Mat R_R;
    cv::Mat P1;
    cv::Mat P2;

    intrinsic_left = (cv::Mat_<double>(3, 3) << 704.0983276367188, 0.0, 622.9022827148438, 0.0, 703.97314453125, 369.13958740234375, 0.0, 0.0, 1.0);
    distCoeffs_left = (cv::Mat_<double>(5, 1) << -0.33391571044921875, 0.14031600952148438, -0.0005035400390625, -0.000263214111328125, 0.0);

    intrinsic_right = (cv::Mat_<double>(3, 3) << 707.31005859375, 0.0, 666.1992797851562, 0.0, 707.0082397460938, 369.8266906738281, 0.0, 0.0, 1.0);
    distCoeffs_right = (cv::Mat_<double>(5, 1) << -0.3342323303222656, 0.1385040283203125, -2.288818359375e-05, -0.000396728515625, 0.0);



    R_L = (cv::Mat_<double>(3, 3) << 0.9999821186065674, -0.0031054019927978516, -0.005091667175292969, 0.0030974149703979492, 0.9999939203262329, -0.0015758275985717773, 0.005096554756164551, 0.0015600919723510742, 0.9999856948852539);
    R_R = (cv::Mat_<double>(3, 3) << 0.9999693632125854, -0.0029882192611694336, 0.007230162620544434, 0.0029768943786621094, 0.9999942779541016, 0.001578688621520996, -0.007234811782836914, -0.0015571117401123047, 0.9999725818634033);


    P1 = (cv::Mat_<double>(3, 4) << 698.4000244140625, 0.0, 641.236572265625, 0.0, 0.0, 698.4000244140625, 366.6060791015625, 0.0, 0.0, 0.0, 1.0, 0.0);
    P2 = (cv::Mat_<double>(3, 4) << 698.4000244140625, 0.0, 641.236572265625, -78031.546875, 0.0, 698.4000244140625, 366.6060791015625, 0.0, 0.0, 0.0, 1.0, 0.0);

    cv::Mat mapLx, mapLy, mapRx, mapRy;
    initUndistortRectifyMap(intrinsic_left, distCoeffs_left, R_L, P1, s1, CV_16SC2, mapLx, mapLy);
    initUndistortRectifyMap(intrinsic_right, distCoeffs_right, R_R, P2, s1, CV_16SC2, mapRx, mapRy);

    remap(img_left, img_left_result, mapLx, mapLy, cv::INTER_LINEAR);
    remap(img_right, img_right_result, mapRx, mapRy, cv::INTER_LINEAR);

}


void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap)
{
    float fx = 698.4000244140625;
    float baseline = 83; //基线距离b，根据标定的相机外参计算。如果只需要相对深度取1即可

    int height = dispMap.rows;
    int width = dispMap.cols;

    LOG(INFO) << "height " << height <<  " width " << width;
    uchar* dispData = dispMap.data;
    ushort* depthData = (ushort*)depthMap.data;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int id = i * width + j;
            if (!dispData[id])  continue;
//            cout << "id is " << id << endl;
            depthData[id] = ushort( fx * baseline / dispData[id]);
        }
    }
}

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

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

//    cv::Mat img_left_c = cv::imread(FLAGS_left_path, cv::IMREAD_COLOR);
    cv::Mat img_left = cv::imread(FLAGS_left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(FLAGS_right_path, cv::IMREAD_GRAYSCALE);


    if (img_left.data == nullptr || img_right.data == nullptr) {
        LOG(FATAL) << "image read wrong , please check the file path :" << FLAGS_left_path << " & " << FLAGS_right_path;
    }

    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        LOG(FATAL) << "image left & right don't have same width & height !";
    }

    cv::Mat left_result = img_left.clone();
    cv::Mat right_result = img_right.clone();

    stereoRectification(img_left,img_right,left_result,right_result);

    img_left = left_result.clone();
    img_right = right_result.clone();

    const sint32 width = static_cast<uint32>(img_left.cols);
    const sint32 height = static_cast<uint32>(img_right.rows);


    auto bytes_left = new uint8[width * height];
    auto bytes_right = new uint8[width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            bytes_left[i * width + j] = img_left.at<uint8>(i, j);
            bytes_right[i * width + j] = img_right.at<uint8>(i, j);
        }
    }

    LOG(INFO) << "Loading Views...Done!";

    SemiGlobalMatching::SGMOption sgm_option;
    sgm_option.num_paths = 8;
    sgm_option.min_disparity = 0;
    sgm_option.max_disparity = 64;

    sgm_option.census_size = SemiGlobalMatching::Census5x5;
    sgm_option.is_check_lr = true;
    sgm_option.lrcheck_thres = 1.0f;
    sgm_option.is_check_unique = true;
    sgm_option.uniqueness_ratio = 0.99;
    sgm_option.is_remove_speckles = true;
    sgm_option.min_speckle_aera = 50;
    sgm_option.p1 = 10;
    sgm_option.p2_init = 150;

    sgm_option.is_fill_holes = false;

    printf("w = %d, h = %d, d = [%d,%d]\n\n", width, height, sgm_option.min_disparity, sgm_option.max_disparity);

    SemiGlobalMatching sgm;

    if (!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM³õÊŒ»¯Ê§°Ü£¡" << std::endl;
        return -2;
    }

    auto disparity = new float32[uint32(width * height)]();
    if (!sgm.Match(bytes_left, bytes_right, disparity)) {
        std::cout << "SGMÆ¥ÅäÊ§°Ü£¡" << std::endl;
        return -2;
    }

    cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    float min_disp = width, max_disp = -width;
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = disparity[i * width + j];
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (sint32 i = 0; i < height; i++) {
        for (sint32 j = 0; j < width; j++) {
            const float32 disp = disparity[i * width + j];
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }

    cv::Mat disp_color;
    applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);

    std::string disp_map_path = "/home/zhachanghai/xiaomi/sgm_stereo";
    disp_map_path += "_d.png";
    std::string disp_color_map_path ="/home/zhachanghai/xiaomi/sgm_stereo";
    disp_color_map_path += "_c.png";
    cv::imwrite(disp_map_path, disp_mat);
    cv::imwrite(disp_color_map_path, disp_color);

    cv::Mat depth_image = cv::Mat::zeros(disp_mat.rows, disp_mat.cols, CV_16UC1);

    disp2Depth(disp_mat,depth_image);
    std::string depth_image_path ="/home/zhachanghai/xiaomi/depth_image.png";
    cv::imwrite(depth_image_path, depth_image);

    delete[] disparity;
    disparity = nullptr;
    delete[] bytes_left;
    bytes_left = nullptr;
    delete[] bytes_right;
    bytes_right = nullptr;

    return 0;
}