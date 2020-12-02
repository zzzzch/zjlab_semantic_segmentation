//
// Created by zhachanghai on 20-10-15.
//

#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <nlohmann/json.hpp>

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
#include "realsense_base.h"


using namespace zjlab;
using namespace pose;
using namespace std;
using json = nlohmann::json;

bool grap_on = false;

void writeJsonFile(const string& json_path,const pcl::PointXYZ& pt){
    json j;
    j["x"] = to_string(pt.x);
    j["y"] = to_string(pt.y);
    j["z"] = to_string(pt.z);
    ofstream json_file(json_path);
    json_file << j;
    json_file << '\n';
    json_file.close();
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)//按下esc退出程序
    {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
    else if (key == GLFW_KEY_G && action == GLFW_PRESS)
    {
        if (!grap_on){
            LOG(INFO) << "grap ready to run !";
            grap_on = true;
        }
    }
}

void testGrapPose(pcl::PointCloud<pcl::PointXYZ>::Ptr& origin_input){
    //remove far points
    pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i = 0; i < origin_input->size(); i++){
        float distance = pow(origin_input->points[i].x,2)+pow(origin_input->points[i].y,2)+pow(origin_input->points[i].z,2);
//        LOG_EVERY_N(INFO,100) << "distance is " << distance ;
        if(distance < 1.0){
            scene->push_back(origin_input->points[i]);
        }
    }

    pcl::VoxelGrid<pcl::PointXYZ> grid;
    float leaf = 0.008f;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud (scene);
    grid.filter (*scene);

    LOG(INFO) << "grid filter over point size is " << scene->size();
//    pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/scene.pcd", *scene);

    //transform coordinate from camera(right ground forward) to robot(forward left up)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation2(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene_rotation3(
            new pcl::PointCloud<pcl::PointXYZ>);

        pcl::PointCloud<pcl::PointXYZ>::Ptr origin_rotation(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr origin_rotation2(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr origin_rotation3(
            new pcl::PointCloud<pcl::PointXYZ>);

        //yaw(z) pitch(y) roll(x)
        Eigen::Matrix3d first_step_rotation = yprDegrees2RotationZYX(0,0,-15);
        pcl::transformPointCloud (*scene,*scene_rotation,R2T(first_step_rotation));
        pcl::transformPointCloud (*origin_input,*origin_rotation,R2T(first_step_rotation));


        Eigen::Matrix3d second_step_rotation;
        second_step_rotation << 0,0,1,
                -1,0,0,
                0,-1,0;

        pcl::transformPointCloud (*scene_rotation,*scene_rotation2,R2T(second_step_rotation));
        pcl::transformPointCloud (*origin_rotation,*origin_rotation2,R2T(second_step_rotation));

        Eigen::Matrix4d thrid_step_trans;
        thrid_step_trans << 1, 0, 0, 0.05, 0, 1, 0, 0, 0, 0, 1, 1.8945, 0, 0, 0,
            1;

        pcl::transformPointCloud(*scene_rotation2, *scene_rotation3,
                                 thrid_step_trans);
        pcl::transformPointCloud(*origin_rotation2, *origin_rotation3,
                                 thrid_step_trans);

        //        pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/scene_rotation2.pcd",
        //        *scene_rotation3);
        pcl::io::savePCDFileASCII(
            "/home/zhachanghai/bottle/origin_rotation.pcd", *origin_rotation3);
        origin_input.swap(origin_rotation3);
        scene.swap(scene_rotation3);
    }

    //估计点的法线，50为一组/
    // Estimate point normals
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setSearchMethod (tree);
    ne.setInputCloud (scene);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);


    ///分割平面///
    // Create the segmentation object for the planar model and set all the parameters
    pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

    //把平面去掉，提取剩下的///
    // Remove the planar inliers, extract the rest
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2 (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ExtractIndices<pcl::Normal> extract_normals;

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight (0.1);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.03);
    seg.setInputCloud (scene);
    seg.setInputNormals (cloud_normals);

    // 根据上面的输入参数执行分割获取平面模型系数和处在平面上的内点
    // Obtain the plane inliers and coefficients
    seg.segment (*inliers_plane, *coefficients_plane);

    // 利用extract提取平面
    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    extract.setInputCloud (scene);		 //设置输入点云
    extract.setIndices (inliers_plane);				//设置分割后的内点为需要提取的点集
    extract.setNegative (false);						//设置提取内点而非外点
    extract.filter (*cloud_plane);						//提取输出存储到cloud_plane

    //存储分割得到的平面上的点到点云文件/
    // Write the planar inliers to disk

    float average_height = 0;
    int points_num = 0;
    for(int i = 0;i<cloud_plane->size();i=i+10){
        average_height += cloud_plane->points[i].z;
        points_num ++;
    }
    average_height = average_height/points_num;

    extract.setNegative (true);							//设置提取外点
    extract.filter (*cloud_filtered2);					//提取输出存储到cloud_filtered2
    extract_normals.setNegative (true);
    extract_normals.setInputCloud (cloud_normals);
    extract_normals.setIndices (inliers_plane);
    extract_normals.filter (*cloud_normals2);


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_height (new pcl::PointCloud<pcl::PointXYZ>());
    for(int i = 0; i < cloud_filtered2->size();i++){
        if(cloud_filtered2->points[i].z - 0.03 > average_height){
            cloud_filtered_height->push_back(cloud_filtered2->points[i]);
        }
    }

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize (100000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered_height);
    ec.extract (cluster_indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr process_scene(new pcl::PointCloud<pcl::PointXYZ>);

    if (cluster_indices.size() == 0) {
      LOG(ERROR) << "can't get child cloud , maybe something wrong !";
      return;
    }

    process_scene = getChildCloudByIndicesFromOriginal<pcl::PointXYZ>(cloud_filtered_height,cluster_indices[0]);


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected(new pcl::PointCloud<pcl::PointXYZ>);
    // Create the filtering object
    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(process_scene);
    proj.setModelCoefficients(coefficients_plane);
    proj.filter(*cloud_projected);

    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
    ne_src.setInputCloud(cloud_projected);
    pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
    ne_src.setSearchMethod(tree_src);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
    ne_src.setRadiusSearch(0.02);
    ne_src.compute(*cloud_src_normals);
////
    seg.setOptimizeCoefficients(true);        					 		//设置对估计的模型系数需要进行优化
    seg.setModelType(pcl::SACMODEL_CIRCLE3D); 		//设置分割模型为圆柱型
    seg.setMethodType(pcl::SAC_RANSAC);      				//设置采用RANSAC作为算法的参数估计方法
    seg.setNormalDistanceWeight(0.2);         						//设置表面法线权重系数
    seg.setMaxIterations(10000);              							//设置迭代的最大次数10000
    seg.setDistanceThreshold(0.005);  //设置内点到模型的距离允许最大值
    seg.setRadiusLimits(0, 0.5);  //设置估计出的圆柱模型的半径范围
    seg.setInputCloud(cloud_projected);
    seg.setInputNormals(cloud_src_normals);

    pcl::PointIndices::Ptr inliers_bottle (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients_bottle (new pcl::ModelCoefficients);

    seg.segment (*inliers_bottle, *coefficients_bottle);
//
//    pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_origin (new pcl::PointCloud<pcl::PointXYZ>());
//
//    for(float z=average_height;z < average_height+0.2;z=z+0.01 ){
//        for(float ang = 0.0;ang <= 360;ang += 5.0){
//            pcl::PointXYZ basic_point;
//            basic_point.x = coefficients_bottle->values[0] + cosf(pcl::deg2rad(ang)) * 0.04;
//            basic_point.y = coefficients_bottle->values[1] + sinf(pcl::deg2rad(ang)) * 0.04;
//            basic_point.z = z;
//            cylinder_origin->push_back(basic_point);
//        }
//    }
//    cylinder_origin->width = (int)cylinder_origin->points.size();
//    cylinder_origin->height = 1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr line (new pcl::PointCloud<pcl::PointXYZ>());

    for(float z=average_height;z < average_height+0.2;z=z+0.01 ){
        pcl::PointXYZ basic_point;
        float x = (z - coefficients_bottle->values[2])/coefficients_bottle->values[6] * coefficients_bottle->values[4] + coefficients_bottle->values[0];
        float y = (z - coefficients_bottle->values[2])/coefficients_bottle->values[6] * coefficients_bottle->values[5] + coefficients_bottle->values[1];
        basic_point.x = x;
        basic_point.y = y;
        basic_point.z = z;
        line->push_back(basic_point);
    }


    writeJsonFile("/home/zhachanghai/bottle/position.json",pcl::PointXYZ(coefficients_bottle->values[0],coefficients_bottle->values[1],coefficients_bottle->values[2]));
    pcl::io::savePCDFileASCII("/home/zhachanghai/bottle/bottle_line.pcd", *line);

}

 cv::Mat Frame2Mat(const rs2::frame &frame) {
     auto vf = frame.as<rs2::video_frame>();
     const int w = vf.get_width();
     const int h = vf.get_height();

     if (frame.get_profile().format() == RS2_FORMAT_BGR8)
     {
         return cv::Mat(cv::Size(w, h), CV_8UC3, (void *)frame.get_data(), cv::Mat::AUTO_STEP);
     }

     else if (frame.get_profile().format() == RS2_FORMAT_RGB8)
     {
         auto r = cv::Mat(cv::Size(w, h), CV_8UC3, (void *)frame.get_data(), cv::Mat::AUTO_STEP);
         cv::cvtColor(r, r, CV_RGB2BGR);
         return r;
     }

     else if (frame.get_profile().format() == RS2_FORMAT_Z16)
     {
         return cv::Mat(cv::Size(w, h), CV_16UC1, (void *)frame.get_data(), cv::Mat::AUTO_STEP);
     }

     else if (frame.get_profile().format() == RS2_FORMAT_Y8)
     {
         return cv::Mat(cv::Size(w, h), CV_8UC1, (void *)frame.get_data(), cv::Mat::AUTO_STEP);
     }
 }

void showRealSensePointCloud(){

    window app(1280, 720, "RealSense Pointcloud Example");
    // Construct an object to manage view state
    glfw_state app_state;
    // register callbacks to allow manipulation of the pointcloud
    register_glfw_callbacks(app, app_state);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points, filtered_points;

    // Get camera info from depth & rgb image
    rs2_camera_info camera_info;
    // Declare RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;
    // Start streaming with default recommended configuration
    pipe.start();
    while (app) // Application still alive?
    {
        // Wait for the next set of frames from the camera
        auto frames = pipe.wait_for_frames();

        auto depth = frames.get_depth_frame();

        // Generate the pointcloud and texture mappings
        points = pc.calculate(depth);
        // for(int i = 0; i < points.size(); i++){

        // }

        auto color = frames.get_color_frame();

        // For cameras that don't have RGB sensor, we'll map the pointcloud to infrared instead of color
        if (!color)
            color = frames.get_infrared_frame();

        // Tell pointcloud object to map to this color frame
        pc.map_to(color);

        // Upload the color frame to OpenGL
        app_state.tex.upload(color);

        // Draw the pointcloud
        draw_pointcloud(app.width(), app.height(), app_state, points);
    }

}

void showRealSenseCaptureImage(){
    // Include a short list of convenience functions for rendering
    window app(1280, 720, "RealSense Capture Example");

    // Declare depth colorizer for enhanced color visualization of depth data
    rs2::colorizer color_map;

    // Declare rates printer for showing streaming rates of the enabled streams
    rs2::rates_printer printer;

    // Declare the RealSense pipeline, encapsulating the actual device and sensors
    rs2::pipeline pipe;

    // Start streaming with default recommended configuration
    // The default video configuration contains Depth and Color streams
    // If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
    pipe.start();

    while(app){
        rs2::frameset data = pipe.wait_for_frames().    // Wait for next set of frames from the camera
                apply_filter(printer).     // Print each enabled stream frame rate
                apply_filter(color_map);   // Find and colorize the depth data

        // Show method, when applied on frameset, break it to frames and upload each frame into a gl textures
        // Each texture is displayed on different viewport according to it's stream unique id
        app.show(data);

    }

}

void alignDepthColorImage(){

    // Create and initialize GUI related objects
    window app(1280, 720, "RealSense Align (Advanced) Example"); // Simple window handling
//    window app_cloud(1280, 720, "RealSense Pointcloud Example");
    glfwSetKeyCallback(app, key_callback);

    ImGui_ImplGlfw_Init(app, false);      // ImGui library intializition
    rs2::colorizer c;                     // Helper to colorize depth images
    texture renderer;                     // Helper for renderig images
//    // Construct an object to manage view state
//    glfw_state app_state;
//    // register callbacks to allow manipulation of the pointcloud
//    register_glfw_callbacks(app_cloud, app_state);


    // Create a pipeline to easily configure and start the camera
    rs2::pipeline pipe;
    //Calling pipeline's start() without any additional parameters will start the first device
    // with its default streams.
    //The start function returns the pipeline profile which the pipeline used to start the device
    rs2::pipeline_profile profile = pipe.start();

    // Each depth camera might have different units for depth pixels, so we get it here
    // Using the pipeline's profile, we can retrieve the device that the pipeline uses
    float depth_scale = get_depth_scale(profile.get_device());

    //Pipeline could choose a device that does not have a color stream
    //If there is no color stream, choose to align depth to another stream
    rs2_stream align_to = find_stream_to_align(profile.get_streams());

    // Create a rs2::align object.
    // rs2::align allows us to perform alignment of depth frames to others frames
    //The "align_to" is the stream type to which we plan to align depth frames.
    rs2::align align(align_to);

    // Declare pointcloud object, for calculating pointclouds and texture mappings
    rs2::pointcloud pc;
    // We want the points object to be persistent so we can display the last cloud when a frame drops
    rs2::points points;

    // Define a variable for controlling the distance to clip
    float depth_clipping_distance = 1.8f;

    while (app) // Application still alive?
    {
        // Using the align object, we block the application until a frameset is available
        rs2::frameset frameset = pipe.wait_for_frames();

        // rs2::pipeline::wait_for_frames() can replace the device it uses in case of device error or disconnection.
        // Since rs2::align is aligning depth to some other stream, we need to make sure that the stream was not changed
        //  after the call to wait_for_frames();
        if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
        {
            //If the profile was changed, update the align object, and also get the new device's depth scale
            profile = pipe.get_active_profile();
            align_to = find_stream_to_align(profile.get_streams());
            align = rs2::align(align_to);
            depth_scale = get_depth_scale(profile.get_device());
        }

        //Get processed aligned frame
        auto processed = align.process(frameset);

        // Trying to get both other and aligned depth frames
        rs2::video_frame other_frame = processed.first(align_to);
        rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

        //If one of them is unavailable, continue iteration
        if (!aligned_depth_frame || !other_frame)
        {
            continue;
        }
        // Passing both frames to remove_background so it will "strip" the background
        // NOTE: in this example, we alter the buffer of the other frame, instead of copying it and altering the copy
        //       This behavior is not recommended in real application since the other frame could be used elsewhere
        remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);

        // Taking dimensions of the window for rendering purposes
        float w = static_cast<float>(app.width());
        float h = static_cast<float>(app.height());

        // At this point, "other_frame" is an altered frame, stripped form its background
        // Calculating the position to place the frame in the window
        rect altered_other_frame_rect{ 0, 0, w, h };
        altered_other_frame_rect = altered_other_frame_rect.adjust_ratio({ static_cast<float>(other_frame.get_width()),static_cast<float>(other_frame.get_height()) });

        // Render aligned image
        renderer.render(other_frame, altered_other_frame_rect);

        points = pc.calculate(aligned_depth_frame);
        // Tell pointcloud object to map to this color frame
        pc.map_to(other_frame);

        if(grap_on){

//            cv::Mat pic_rgb =  Frame2Mat(other_frame);
//            cv::Mat pic_depth = Frame2Mat(aligned_depth_frame);
//            cv::imwrite("/home/zhachanghai/bottle/grap.jpg",pic_rgb);
//            cv::imwrite("/home/zhachanghai/bottle/grap_depth.jpg",pic_depth);
//            LOG(INFO) << "save picture over !";
            auto vertices = points.get_vertices();
            pcl::PointCloud<pcl::PointXYZ>::Ptr p_cloud (new pcl::PointCloud<pcl::PointXYZ>());
            for(int i = 0;i < points.size();i++ ){
                if(vertices[i].z){
                    pcl::PointXYZ pt;
                    pt.x = vertices[i].x;
                    pt.y = vertices[i].y;
                    pt.z = vertices[i].z;
                    p_cloud->push_back(pt);
                }
            }
            pcl::io::savePCDFile("/home/zhachanghai/bottle/grap_bottle.pcd", *p_cloud);
            LOG(INFO) << "begin estimation !";

            testGrapPose(p_cloud);

            grap_on = false;
        }

        // The example also renders the depth frame, as a picture-in-picture
        // Calculating the position to place the depth frame in the window
        rect pip_stream{ 0, 0, w / 5, h / 5 };
        pip_stream = pip_stream.adjust_ratio({ static_cast<float>(aligned_depth_frame.get_width()),static_cast<float>(aligned_depth_frame.get_height()) });
        pip_stream.x = altered_other_frame_rect.x + altered_other_frame_rect.w - pip_stream.w - (std::max(w, h) / 25);
        pip_stream.y = altered_other_frame_rect.y + (std::max(w, h) / 25);

        // Render depth (as picture in pipcture)
        renderer.upload(c.process(aligned_depth_frame));
        renderer.show(pip_stream);

//        // Using ImGui library to provide a slide controller to select the depth clipping distance
//        ImGui_ImplGlfw_NewFrame(1);
//        render_slider({ 5.f, 0, w, h }, depth_clipping_distance);
//        ImGui::Render();

    }
}


int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;
//    showRealSensePointCloud();
//    showRealSenseCaptureImage();
    alignDepthColorImage();
    LOG(INFO) << "RUN grap bottle success !";
    return 0;
}