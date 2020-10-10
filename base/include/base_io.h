#pragma once
#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//opengl
#include <GL/glut.h>
//openni
#include <OpenNI.h>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/types_c.h"



namespace zjlab{
    namespace pose{
        using namespace std;

        struct intrinsicsDepth {
            float fx = 1;
            float fy = 1;
            float cx = 0;
            float cy = 0;
        };


        bool readPly(int &pt_num,
                     Eigen::Vector3f *&positions,
                     Eigen::Vector3f *&colors,
                     Eigen::Vector3f *&normals,
                     float *&radii,
                     string &file_path);

        pcl::PointCloud<pcl::PointXYZ>::Ptr depth2cloud(cv::Mat depth_image, float fx, float fy, float cx, float cy);

        void getVideoFrame2Cloud(const string &video_input, const intrinsicsDepth &intrinsics_depth, const string &output_path);

        void image2Video(const string &input_path, const string &output_video_path);

        Eigen::Vector3f rotationAboutX(Eigen::Vector3f &origin, float &theta);

        Eigen::Vector3f rotationAboutY(Eigen::Vector3f &origin, float &theta);

        Eigen::Vector3f rotationAboutZ(Eigen::Vector3f &origin, float &theta);
    }
}
