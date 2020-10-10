//
// Created by zhachanghai on 20-9-14.
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
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/board.h>
#include <pcl/features/vfh.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/sample_consensus_prerejective.h>

//openMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include "pcl_tools.h"
#include "base_io.h"
using namespace zjlab;
using namespace pose;
using namespace std;

DEFINE_string(scene_path,
              "/home/zhachanghai/test_grap/pcl_test/", "scene pcd file");

DEFINE_string(target_path,
              "/home/zhachanghai/guochen_test/model.ply", "target ply file");

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr transPointCloud(typename pcl::PointCloud<PointT>::Ptr& input_cloud,Eigen::Matrix4f& rotation){
    typename pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>());
    for(int i = 0; i < input_cloud->size(); i++){
        Eigen::Vector4f origin;
        Eigen::Vector4f process;
        origin << input_cloud->points[i].x,input_cloud->points[i].y,input_cloud->points[i].z,1;
        process =  rotation * origin;
        result->push_back(PointT(process(0),process(1),process(2)));
    }
    return result;
}

void addLine(string& in_file, int count, Eigen::Matrix4f &rotation){

    cv::Mat ori_pic = cv::imread(in_file + to_string(count) + "_image.jpg");

    Eigen::Vector4f origin(0.0264-0.0276, -1.6096 + 1.6534, 40.8759 - 40.5586, 1);
    Eigen::Vector3f ori_x(0.04780 - 0.0264,-1.7689 + 1.6096,40.853199 - 40.8759);
    Eigen::Vector3f ori_y(0.0515 - 0.0264,-1.6401 + 1.6096,41.188702 - 40.8759);

    Eigen::Vector3f ori_z;
    ori_z = ori_x.cross(ori_y);

    for(int i = 0;i<3;i++){
        ori_x(i) = ori_x(i)+ origin(i);
        ori_y(i) = ori_y(i)+ origin(i);
        ori_z(i) = ori_z(i)+ origin(i);
    }


    Eigen::Vector4f origin_x(ori_x(0), ori_x(1), ori_x(2), 1);
    Eigen::Vector4f origin_y(ori_y(0), ori_y(1), ori_y(2), 1);
    Eigen::Vector4f origin_z(ori_z(0), ori_z(1), ori_z(2), 1);


    float fx = 458.91;
    float fy = 459.547;
    float cx = 337.824;
    float cy = 242.959;

    Eigen::Vector4f rst = rotation * origin;
    Eigen::Vector4f rst_x = rotation * origin_x;
    Eigen::Vector4f rst_y = rotation * origin_y;
    Eigen::Vector4f rst_z = rotation * origin_z;


    int x0,x1,x2,x3,y0,y1,y2,y3;
    float pic_x,pic_y;

    pic_x = rst(0) * fx / rst(2) + cx;
    pic_y = rst(1) * fy / rst(2) + cy;

    x0 = pic_x;
    y0 = pic_y ;


    pic_x = rst_x(0) * fx / rst_x(2) + cx;
    pic_y = rst_x(1) * fy / rst_x(2) + cy;

    x1 = pic_x ;
    y1 = pic_y ;

    pic_x = rst_y(0) * fx / rst_y(2) + cx;
    pic_y = rst_y(1) * fy / rst_y(2) + cy;

    x2 = pic_x ;
    y2 = pic_y ;

    pic_x = rst_z(0) * fx / rst_z(2) + cx;
    pic_y = rst_z(1) * fy / rst_z(2) + cy;

    x3 = pic_x ;
    y3 = pic_y ;


    cv::Point p0 = cv::Point(x0,y0);
    cv::Point p1 = cv::Point(x1, y1);
    cv::line(ori_pic, p0, p1, cv::Scalar(0, 0, 255), 3, 4);

    p1.x = x2;
    p1.y = y2;
    cv::line(ori_pic, p0, p1, cv::Scalar(255, 0, 0), 3, 4);


    p1.x = x3;
    p1.y = y3;
    cv::line(ori_pic, p0, p1, cv::Scalar(0, 255, 0), 3, 4);

    cv::imwrite("/home/zhachanghai/test_grap/image_result/" + to_string(count) + "_result.jpg",ori_pic);

}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

//    // get guochen video to pcd file & depth image
//    string guochen_movie_file = "/home/zhachanghai/guochen_test/dmrep_20200913_135548.oni";
//    string output_path = "/home/zhachanghai/guochen_test/";
//
//    intrinsicsDepth intrinsics_depth;
//    intrinsics_depth.fx = 524.361;
//    intrinsics_depth.fy = 525.246;
//    intrinsics_depth.cx = 311.564;
//    intrinsics_depth.cy = 255.234;
//    getVideoFrame2Cloud(guochen_movie_file, intrinsics_depth, output_path);

    ofstream output("/home/zhachanghai/test_grap/rotation.txt");

    for(int count = 240;count < 241; count++){

        string scene_file_path = FLAGS_scene_path + to_string(count) + "_point.pcd";

        string target_file_path = FLAGS_target_path;

        LOG(INFO) << "Begin estimate pose !!";
        pcl::PointCloud<pcl::PointXYZ>::Ptr scene(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr model(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::io::loadPCDFile(scene_file_path,*scene);

        //remove far points
        pcl::PointCloud<pcl::PointXYZ>::Ptr distance_scene(new pcl::PointCloud<pcl::PointXYZ>);
        for(int i = 0; i < scene->size(); i++){
            float distance = pow(scene->points[i].x,2)+pow(scene->points[i].y,2)+pow(scene->points[i].z,2);
//        LOG_EVERY_N(INFO,100) << "distance is " << distance ;
            if(distance < 3){
                distance_scene->push_back(scene->points[i]);
            }
        }
        scene.swap(distance_scene);

        int point_num = 0;
        Eigen::Vector3f *positions = nullptr;
        Eigen::Vector3f *colors = nullptr;
        Eigen::Vector3f *normals = nullptr;
        float *radii = nullptr;

        if (!readPly(point_num, positions, colors, normals, radii, target_file_path)) {
            LOG(FATAL) << "read ply file error !";
            return -1;
        }

        float x,y,z;
        x = positions[0][0];
        y = positions[0][1];
        z = positions[0][2];

        LOG(INFO) << "origin set position is " << x << " " << y << " " << z ;
        for (int i = 0; i < point_num; i++) {
            pcl::PointXYZ pt;
            pt.x = positions[i][0]-x;
            pt.y = positions[i][1]-y;
            pt.z = positions[i][2]-z;
            model->push_back(pt);
        }

        LOG(INFO) << "Downsampling !!";
        pcl::VoxelGrid<pcl::PointXYZ> grid;
        float leaf = 0.005f;
        grid.setLeafSize(leaf, leaf, leaf);
        grid.setInputCloud (scene);
        grid.filter (*scene);

        grid.setInputCloud (model);
        grid.filter (*model);

        LOG(INFO) << "After downsample, scene point number is " << scene->size();
        LOG(INFO) << "After downsample, target point number is " << model->size();

//        pcl::io::savePCDFileASCII("/home/zhachanghai/test_grap/down_scene.pcd", *scene);
//        pcl::io::savePCDFileASCII("/home/zhachanghai/test_grap/down_target.pcd", *model);

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud (scene);

        std::vector<pcl::PointIndices> cluster_indices;

        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance (0.02);
        ec.setMinClusterSize (100);
        ec.setMaxClusterSize (100000);
        ec.setSearchMethod (tree);
        ec.setInputCloud (scene);
        ec.extract (cluster_indices);

        pcl::PointCloud<pcl::PointXYZ>::Ptr process_scene(new pcl::PointCloud<pcl::PointXYZ>);

        if(count > 80){
            process_scene = getChildCloudByIndicesFromOriginal<pcl::PointXYZ>(scene,cluster_indices[2]);
        }else{
            process_scene = getChildCloudByIndicesFromOriginal<pcl::PointXYZ>(scene,cluster_indices[1]);
        }


        LOG(INFO) << "After euclidean cluster, process scene point number is " << process_scene->size();

//        pcl::io::savePCDFileASCII("/home/zhachanghai/test_grap/process_scene.pcd", *process_scene);


        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(0.04,0.04,0.04);
        voxel_grid.setInputCloud(model);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZ>());
        voxel_grid.filter(*cloud_src);
        LOG(INFO) << "down size target from " << model->size() << " to "<< cloud_src->size();

        //计算表面法线
        pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
        ne_src.setInputCloud(cloud_src);
        pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
        ne_src.setSearchMethod(tree_src);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
        ne_src.setRadiusSearch(0.02);
        ne_src.compute(*cloud_src_normals);

        std::vector<int> indices_tgt;
        pcl::removeNaNFromPointCloud(*process_scene,*process_scene, indices_tgt);


        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
        voxel_grid_2.setLeafSize(0.04,0.04,0.04);
        voxel_grid_2.setInputCloud(process_scene);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt (new pcl::PointCloud<pcl::PointXYZ>);
        voxel_grid_2.filter(*cloud_tgt);
        LOG(INFO) << "down size scene from " << process_scene->size() << " to "<< cloud_tgt->size();


        pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_tgt;
        ne_tgt.setInputCloud(cloud_tgt);
        pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZ>());
        ne_tgt.setSearchMethod(tree_tgt);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
        //ne_tgt.setKSearch(20);
        ne_tgt.setRadiusSearch(0.02);
        ne_tgt.compute(*cloud_tgt_normals);
        LOG(INFO) << "normal compute over ";

        //计算FPFH
        pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_src;
        fpfh_src.setInputCloud(cloud_src);
        fpfh_src.setInputNormals(cloud_src_normals);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src_fpfh (new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh_src.setSearchMethod(tree_src_fpfh);
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
        fpfh_src.setRadiusSearch(0.05);
        //fpfh_src.setKSearch(20);
        fpfh_src.compute(*fpfhs_src);
        LOG(INFO) << "compute model cloud FPFH over !";

        pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_tgt;
        fpfh_tgt.setInputCloud(cloud_tgt);
        fpfh_tgt.setInputNormals(cloud_tgt_normals);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_tgt_fpfh (new pcl::search::KdTree<pcl::PointXYZ>);
        fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
        fpfh_tgt.setRadiusSearch(0.05);
        //fpfh_tgt.setKSearch(20);
        fpfh_tgt.compute(*fpfhs_tgt);
        LOG(INFO) << "compute target cloud FPFH over !";


        //SAC配准
        pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
        scia.setInputSource(cloud_src);
        scia.setInputTarget(cloud_tgt);
        scia.setSourceFeatures(fpfhs_src);
        scia.setTargetFeatures(fpfhs_tgt);
        //scia.setMinSampleDistance(1);
        //scia.setNumberOfSamples(2);
        //scia.setCorrespondenceRandomness(20);
        pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result (new pcl::PointCloud<pcl::PointXYZ>);
        scia.align(*sac_result);
        LOG(INFO) << "sac has converged:"<<scia.hasConverged()<<"  score: "<<scia.getFitnessScore();
        Eigen::Matrix4f sac_trans;
        sac_trans = scia.getFinalTransformation();
        LOG(INFO) << sac_trans;
//        pcl::io::savePCDFileASCII("/home/zhachanghai/test_grap/icp_guess.pcd", *sac_result);


        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(sac_result);
        icp.setInputTarget(cloud_tgt);
//    icp.setMaximumIterations (100);
//    icp.setTransformationEpsilon (1e-9);
//    icp.setMaxCorrespondenceDistance (0.05);
//    icp.setEuclideanFitnessEpsilon (1);
//    icp.setRANSACOutlierRejectionThreshold (1.5);

        //icp.setMaxCorrespondenceDistance(0.005);
        pcl::PointCloud<pcl::PointXYZ> Final;
        icp.align(Final);

        LOG(INFO) << "has converged:" << icp.hasConverged() << " score: " <<
                  icp.getFitnessScore();
        Eigen::Matrix4f icp_trans;
        icp_trans = icp.getFinalTransformation();
        LOG(INFO) << icp_trans;

        pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud_(new pcl::PointCloud<pcl::PointXYZ>);

        Eigen::Matrix4f rototranslation =  icp_trans * sac_trans;
        LOG(INFO) << rototranslation;

        Eigen::Matrix3f rotation = rototranslation.block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslation.block<3,1>(0, 3);

        pcl::transformPointCloud(*model,*final_cloud_, rototranslation);


//        output << rotation (0,0) << " " << rotation (0,1) << " " << rotation (0,2) << endl;
//        output << rotation (1,0) << " " << rotation (1,1) << " " << rotation (1,2) << endl;
//        output << rotation (2,0) << " " << rotation (2,1) << " " << rotation (2,2) << endl;
//        output << translation (0) << " " << translation (1) << " " << translation (2) << endl;
//
//        output << endl;

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

        // show demo
//        string infile_pic = "/home/zhachanghai/test_grap/rgb_image/";
//
//        addLine(infile_pic, count, rototranslation);

        pcl::io::savePCDFileASCII("/home/zhachanghai/test_grap/result/" + to_string(count) + "_result.pcd", *final_cloud_);
    }

    output.close();

    return 0;

}

