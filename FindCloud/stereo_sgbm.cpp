//
// Created by zhachanghai on 20-12-4.
//

#include "iostream"
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "base_io.h"
#include "tictoc.hpp"

#include <opencv2/opencv.hpp>

using namespace zjlab;
using namespace pose;
using namespace std;

using namespace std::chrono;

DEFINE_string(left_path,
              "/home/zhachanghai/xiaomi/left/0_image.jpg", "Input left image file path");

DEFINE_string(right_path,
              "/home/zhachanghai/xiaomi/right/0_image.jpg", "Input right image file path");


static void prefilterXSobel(const cv::Mat& src, cv::Mat& dstImg, int preFilterCap)
{
    cv::Mat srcImg;
    cv::cvtColor(src, srcImg, cv::COLOR_BGR2GRAY);

//    assert(srcImg.channels() == 1);
    int radius = 1;
    int width = srcImg.cols;
    int height = srcImg.rows;
    uchar *pSrcData = srcImg.data;
    uchar *pDstData = dstImg.data;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int idx = i*width + j;
            if (i >= radius && i < height - radius && j >= radius && j < width - radius)
            {
                int diff0 = pSrcData[(i - 1)*width + j + 1] - pSrcData[(i - 1)*width + j - 1];
                int diff1 = pSrcData[i*width + j + 1] - pSrcData[i*width + j - 1];
                int diff2 = pSrcData[(i + 1)*width + j + 1] - pSrcData[(i + 1)*width + j - 1];

                int value = diff0 + 2 * diff1 + diff2;
                if (value < -preFilterCap)
                {
                    pDstData[idx] = 0;
                }
                else if (value >= -preFilterCap && value <= preFilterCap)
                {
                    pDstData[idx] = uchar(value + preFilterCap);
                }
                else
                {
                    pDstData[idx] = uchar(2 * preFilterCap);
                }
            }
            else
            {
                pDstData[idx] = 0;
            }
        }
    }
}

//static double computeReprojectionErrors(
//        const vector<vector<cv::Point3f> >& objectPoints,
//        const vector<vector<cv::Point2f> >& imagePoints,
//        const vector<cv::Mat>& rvecs, const vector<cv::Mat>& tvecs,
//        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs){
//    vector<cv::Point2f> imagePoints2;
//    int i, totalPoints = 0;
//    double totalErr = 0, err;
//    for (i = 0; i < (int)objectPoints.size(); i++)
//    {
//        projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i],
//                      cameraMatrix, distCoeffs, imagePoints2);
//        err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
//        int n = (int)objectPoints[i].size();
//
//        totalErr += err*err;
//        totalPoints += n;
//    }
//    return std::sqrt(totalErr / totalPoints);
//}

void stereoRectification(cv::Mat& img_left,cv::Mat& img_right,cv::Mat& img_left_result,cv::Mat& img_right_result){

    cv::Size s1,s2;
    s1 = img_left.size();
    s2 = img_right.size();
    LOG(INFO) << " image left size is " << s1;
//    cv::Mat intrinsic_left = cv::Mat(3, 3, CV_32FC1);
//    cv::Mat intrinsic_right = cv::Mat(3, 3, CV_32FC1);
    cv::Mat intrinsic_left;
    cv::Mat intrinsic_right;

    cv::Mat distCoeffs_left,distCoeffs_right;
    cv::Mat R_L;
    cv::Mat R_R;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;
    cv::Rect validROIL, validROIR;
    cv::Mat R_total;
    cv::Mat T_total;
//    intrinsic_left = (cv::Mat_<double>(3, 3) << 726.941958, 0.0, 631.164040, 0.0, 728.944851, 361.788413, 0.0, 0.0, 1.0);
//    distCoeffs_left = (cv::Mat_<double>(5, 1) << -0.284046, 0.065381, 0.000321, -0.000797, 0.0);
//
//    intrinsic_right = (cv::Mat_<double>(3, 3) << 711.109047, 0.0, 662.676498, 0.0, 711.068667, 368.185037, 0.0, 0.0, 1.0);
//    distCoeffs_right = (cv::Mat_<double>(5, 1) << -0.312199, 0.093603, -0.000082, 0.000686, 0.0);

//    R_total = (cv::Mat_<double>(3, 3) << 0.991614,-0.007737,0.129002, 0.007069,0.999959, 0.005636, -0.129040, -0.004677, 0.991628);
//    T_total = (cv::Mat_<double>(3, 1) << -83.630163, 0.00000000000000000, 0.00000000000000000);

//    cv::stereoRectify(intrinsic_left, distCoeffs_left, intrinsic_right, distCoeffs_right, s1, R_total, T_total, R_L, R_R, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, s1, &validROIL, &validROIR);
//    double left_error = computeReprojectionErrors(objectPoints, image_leftPoints, rvecs_l, tvecs_l, intrinsic_left, distCoeffs_left);
//    double right_error = computeReprojectionErrors(objectPoints, image_rightPoints, rvecs_r, tvecs_r, intrinsic_right, distCoeffs_right);

//    LOG(INFO) << "left_Calibration error: " << left_error;
//    LOG(INFO) << "right_Calibration error: " << right_error;

//    LOG(INFO) << "R_L is " << R_L;
//    LOG(INFO) << "R_R is " << R_R;
//    LOG(INFO) << "P1 is " << P1;
//    LOG(INFO) << "P2 is " << P2;

//    R_L = (cv::Mat_<double>(3, 3) << 0.989732, -0.006343, 0.142792, 0.007083, 0.999964, -0.004673, -0.142758, 0.005636, 0.989742);
//    R_R = (cv::Mat_<double>(3, 3) << 0.991614, -0.007737, 0.129002, 0.007069, 0.999959, 0.005636, -0.129040, -0.004677, 0.991628);



//    P1 = (cv::Mat_<double>(3, 4) << 680.179765, 0.000000, 387.484009, 0.000000,
//            0.000000, 680.179765, 366.438118, 0.000000,
//            0.000000, 0.000000, 1.000000, 0.000000);
//
//
//    P2 = (cv::Mat_<double>(3, 4) << 680.179765, 0.000000, 387.484009, -83.630163,
//            0.000000, 680.179765, 366.438118, 0.000000,
//            0.000000, 0.000000, 1.000000, 0.000000);


    intrinsic_left = (cv::Mat_<double>(3, 3) << 704.0983276367188, 0.0, 622.9022827148438, 0.0, 703.97314453125, 369.13958740234375, 0.0, 0.0, 1.0);
    distCoeffs_left = (cv::Mat_<double>(5, 1) << -0.33391571044921875, 0.14031600952148438, -0.0005035400390625, -0.000263214111328125, 0.0);

    intrinsic_right = (cv::Mat_<double>(3, 3) << 707.31005859375, 0.0, 666.1992797851562, 0.0, 707.0082397460938, 369.8266906738281, 0.0, 0.0, 1.0);
    distCoeffs_right = (cv::Mat_<double>(5, 1) << -0.3342323303222656, 0.1385040283203125, -2.288818359375e-05, -0.000396728515625, 0.0);



    R_L = (cv::Mat_<double>(3, 3) << 0.9999821186065674, -0.0031054019927978516, -0.005091667175292969, 0.0030974149703979492, 0.9999939203262329, -0.0015758275985717773, 0.005096554756164551, 0.0015600919723510742, 0.9999856948852539);
    R_R = (cv::Mat_<double>(3, 3) << 0.9999693632125854, -0.0029882192611694336, 0.007230162620544434, 0.0029768943786621094, 0.9999942779541016, 0.001578688621520996, -0.007234811782836914, -0.0015571117401123047, 0.9999725818634033);


    P1 = (cv::Mat_<double>(3, 4) << 698.4000244140625, 0.0, 641.236572265625, 0.0, 0.0, 698.4000244140625, 366.6060791015625, 0.0, 0.0, 0.0, 1.0, 0.0);
    P2 = (cv::Mat_<double>(3, 4) << 698.4000244140625, 0.0, 641.236572265625, -78031.546875, 0.0, 698.4000244140625, 366.6060791015625, 0.0, 0.0, 0.0, 1.0, 0.0);


    LOG(INFO) << "P1 is " << P1;
    LOG(INFO) << "P2 is " << P2;

    cv::Mat mapLx, mapLy, mapRx, mapRy;
    initUndistortRectifyMap(intrinsic_left, distCoeffs_left, R_L, P1, s1, CV_16SC2, mapLx, mapLy);
    initUndistortRectifyMap(intrinsic_right, distCoeffs_right, R_R, P2, s1, CV_16SC2, mapRx, mapRy);

    remap(img_left, img_left_result, mapLx, mapLy, cv::INTER_LINEAR);
    remap(img_right, img_right_result, mapRx, mapRy, cv::INTER_LINEAR);

}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;


    cv::Mat img_left = cv::imread(FLAGS_left_path, cv::IMREAD_COLOR);
    cv::Mat img_right = cv::imread(FLAGS_right_path, cv::IMREAD_COLOR);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        LOG(FATAL) << "image read wrong , please check the file path :" << FLAGS_left_path << " & " << FLAGS_right_path;
    }

    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        LOG(FATAL) << "image left & right don't have same width & height !";
    }

    cv::Mat left_result = img_left.clone();
    cv::Mat right_result = img_right.clone();

    stereoRectification(img_left,img_right,left_result,right_result);

    cv::imshow("stereo Rectification left result",left_result);
    cv::imshow("stereo Rectification right result",right_result);

    img_left = left_result.clone();
    img_right = right_result.clone();
    cv::Mat disp;
    cv::Mat disp_filter;

    int width = img_left.cols;
    int height = img_left.rows;

    cv::Mat img_left_filter = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat img_right_filter = cv::Mat::zeros(height, width, CV_8UC1);


    prefilterXSobel(img_left,img_left_filter,100);
    prefilterXSobel(img_right,img_right_filter,100);

//    cv::imshow("show left sobel",img_left_filter);
//    cv::imshow("show right sobel",img_right_filter);

    int numberOfDisparities = ((width / 8) + 15) & -16;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
    sgbm->setPreFilterCap(63);
    int SADWindowSize = 9;
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = img_left.channels();
    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);

    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

//    int alg = STEREO_SGBM;
//    if (alg == STEREO_HH)
//        sgbm->setMode(cv::StereoSGBM::MODE_HH);
//    else if (alg == STEREO_SGBM)
//        sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
//    else if (alg == STEREO_3WAY)
//        sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    sgbm->compute(img_left, img_right, disp);

    sgbm->compute(img_left_filter, img_right_filter, disp_filter);
//    cv::imshow("show result",disp);
    cv::imwrite("/home/zhachanghai/xiaomi/sgbm_stereo_filter.png",disp_filter);
    cv::imwrite("/home/zhachanghai/xiaomi/sgbm_stereo.png",disp);

    cv::waitKey();
    return 0;
}


